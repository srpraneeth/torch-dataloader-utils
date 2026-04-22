import heapq
import logging
import random
from collections.abc import Iterator

import pyarrow.parquet as pq

from torch_dataloader_utils.splits.core import DataFileInfo, Shard, Split, RowRange

logger = logging.getLogger(__name__)

_DEFAULT_TARGET_BYTES = 128 * 1024 * 1024  # 128 MiB

_BYTE_UNITS = {
    "tib": 1024**4, "tb": 1000**4,
    "gib": 1024**3, "gb": 1000**3,
    "mib": 1024**2, "mb": 1000**2,
    "kib": 1024,    "kb": 1000,
    "b":   1,
}


def parse_bytes(value: int | str) -> int:
    """Parse a byte size expressed as an int or human-readable string.

    Supported suffixes (case-insensitive): B, KB, KiB, MB, MiB, GB, GiB, TB, TiB.
    Examples: 10485760, "10MiB", "128 MiB", "1GiB", "512mb".
    """
    if isinstance(value, int):
        return value
    s = value.strip().lower()
    for suffix, multiplier in _BYTE_UNITS.items():
        if s.endswith(suffix):
            num = s[: -len(suffix)].strip()
            return int(float(num) * multiplier)
    return int(s)  # bare number string e.g. "10485760"

def _parquet_chunks(
    file: DataFileInfo,
    target_bytes: int | None,
    target_rows: int | None = None,
) -> Iterator[Split]:
    """Yield Split chunks for a Parquet file, aligned to row group boundaries.

    Reads only the file footer (metadata) — no data scan.
    Packs consecutive row groups into chunks until target_bytes or target_rows
    is reached. Each chunk covers a contiguous row range within the file.

    When target_rows is set it takes precedence over target_bytes.
    """
    try:
        meta = pq.read_metadata(file.path)
    except Exception as e:
        logger.warning(
            "Could not read Parquet metadata for %s (%s) — treating as single chunk",
            file.path, e,
        )
        yield Split(file=file, row_range=None)
        return

    num_rg = meta.num_row_groups
    if num_rg == 0:
        yield Split(file=file, row_range=None)
        return

    chunk_start_row = 0
    chunk_bytes = 0
    chunk_rows = 0
    current_row_offset = 0

    for i in range(num_rg):
        rg = meta.row_group(i)
        rg_rows = rg.num_rows
        rg_bytes = rg.total_byte_size

        # Always include at least one row group per chunk to avoid empty chunks
        # when a single row group exceeds the target.
        chunk_bytes += rg_bytes
        chunk_rows += rg_rows

        at_last = (i == num_rg - 1)
        if target_rows is not None:
            chunk_full = chunk_rows >= target_rows
        else:
            chunk_full = chunk_bytes >= target_bytes  # type: ignore[operator]

        if chunk_full or at_last:
            yield Split(
                file=file,
                row_range=RowRange(offset=chunk_start_row, length=chunk_rows),
            )
            logger.debug(
                "Parquet chunk: %s  rows=[%d, %d)  bytes=%d  rg_end=%d",
                file.path, chunk_start_row, chunk_start_row + chunk_rows, chunk_bytes, i,
            )
            chunk_start_row = current_row_offset + rg_rows
            chunk_bytes = 0
            chunk_rows = 0

        current_row_offset += rg_rows


class TargetSizeSplitStrategy:
    """Splits files into target-sized chunks and distributes them across workers.

    For Parquet files, chunks are aligned to row group boundaries — no row group
    is ever split across two chunks, and only footer metadata is read (no data scan)
    when generating splits.

    Two chunking modes:
    - ``target_bytes`` (default 128 MiB): pack row groups until chunk reaches the
      byte target. Good default for mixed schemas or unknown row counts.
    - ``target_rows``: pack row groups until chunk reaches the row count target.
      Produces near-perfectly balanced splits for homogeneous schemas (same
      columns across all files) because row count is the exact measure of work.
      When both are provided, ``target_rows`` takes precedence.

    For non-Parquet files (ORC, CSV, JSONL) the entire file is treated as a
    single unsplittable chunk and distributed whole to one worker.

    Chunks are assigned to workers using a greedy min-heap (LPT scheduling) —
    always assign the next chunk to the least-loaded worker. This is optimal
    for unequal chunk sizes and equivalent to round-robin for equal sizes.

    Satisfies the SplitStrategy protocol.
    """

    def __init__(
        self,
        target_bytes: int | str = _DEFAULT_TARGET_BYTES,
        target_rows: int | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self.target_bytes = parse_bytes(target_bytes)
        self.target_rows = target_rows
        self.shuffle = shuffle
        self.seed = seed

    def generate(
        self,
        files: list[DataFileInfo],
        num_workers: int,
        epoch: int = 0,
    ) -> list[Shard]:
        # Build a flat list of splits across all files
        all_splits: list[Split] = []
        for file in files:
            ext = file.path.rsplit(".", 1)[-1].lower() if "." in file.path else ""
            if ext == "parquet":
                all_splits.extend(_parquet_chunks(file, self.target_bytes, self.target_rows))
            else:
                # Non-Parquet: treat as a single unsplittable chunk
                all_splits.append(Split(file=file, row_range=None))

        if self.shuffle:
            rng = random.Random(self.seed + epoch)
            rng.shuffle(all_splits)
        else:
            # LPT: sort splits largest → smallest so greedy heap gives near-optimal balance
            all_splits.sort(key=lambda s: (
                s.row_range.length if s.row_range is not None
                else (s.file.file_size or 0)
            ), reverse=True)

        # Greedy min-heap: always assign next split to the least-loaded worker
        heap = [(0, i) for i in range(num_workers)]
        heapq.heapify(heap)
        shards = [Shard(id=i) for i in range(num_workers)]
        for split in all_splits:
            split_size = (
                split.row_range.length if split.row_range is not None
                else (split.file.record_count or split.file.file_size or 1)
            )
            total, worker_id = heapq.heappop(heap)
            shards[worker_id].splits.append(split)
            heapq.heappush(heap, (total + split_size, worker_id))

        mode = f"target_rows={self.target_rows}" if self.target_rows else f"target_bytes={self.target_bytes}"
        splits_per_worker = [len(s.splits) for s in shards]
        rows_per_worker = [
            sum(sp.row_range.length if sp.row_range else (sp.file.record_count or 0)
                for sp in s.splits)
            for s in shards
        ]
        logger.info(
            "TargetSizeSplitStrategy: %d file(s) → %d split(s) → %d worker(s)  "
            "%s  epoch=%d  shuffle=%s  splits_per_worker=%s  rows_per_worker=%s",
            len(files), len(all_splits), num_workers,
            mode, epoch, self.shuffle, splits_per_worker, rows_per_worker,
        )
        for shard in shards:
            logger.debug(
                "Shard %d: %d split(s)  %s",
                shard.id,
                len(shard.splits),
                [(s.file.path, s.row_range) for s in shard.splits],
            )

        return shards
