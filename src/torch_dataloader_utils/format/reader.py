from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pad
import pyarrow.fs as pafs
import pyarrow.orc as orc
import pyarrow.parquet as pq

from torch_dataloader_utils.observability import WorkerMetrics
from torch_dataloader_utils.splits.core import RowRange, Shard

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"parquet", "orc", "csv", "json", "jsonl"}

# pyarrow uses "json" for both json and jsonl
_FORMAT_ALIASES = {"jsonl": "json"}


def _parse_hive_partitions(path: str) -> dict[str, str]:
    """Extract key=value Hive partition segments from a file path.

    Example: "/data/region=us/year=2024/part.parquet" → {"region": "us", "year": "2024"}
    Values are always strings — callers cast as needed.
    """
    parts: dict[str, str] = {}
    for segment in path.replace("\\", "/").split("/"):
        if "=" in segment:
            key, _, value = segment.partition("=")
            if key and value:
                parts[key] = value
    return parts


def read_split(
    shard: Shard,
    format: str,
    batch_size: int = 1024,
    columns: list[str] | None = None,
    filters: pc.Expression | None = None,
    storage_options: dict | None = None,
    partitioning: str | None = None,
    metrics: WorkerMetrics | None = None,
    pbar: Any | None = None,
) -> Iterator[pa.RecordBatch]:
    """Read a Shard and yield pyarrow RecordBatches.

    Args:
        shard: The Shard to read — contains one or more Splits.
        format: File format — parquet, orc, csv, json, or jsonl.
        batch_size: Number of rows per RecordBatch.
        columns: Column projection — only these columns are read. None = all columns.
        filters: Predicate pushdown expression via pyarrow.compute. None = no filter.
        storage_options: Passed to fsspec for filesystem construction.
        partitioning: Partition scheme — "hive" decodes key=value directory segments
            and adds them as columns. None = no partitioning (default).
        metrics: Optional WorkerMetrics to increment in-place as batches are yielded.
        pbar: Optional tqdm progress bar; pbar.update(n) called with each batch's
            row count, pbar.reset(total) called at the start of each split.

    Yields:
        pyarrow.RecordBatch

    Raises:
        ValueError: When format is not supported.
    """
    if format not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise ValueError(f"Unsupported format {format!r}. Supported formats: {supported}")

    arrow_format = _FORMAT_ALIASES.get(format, format)
    opts = storage_options or {}

    n_splits = len(shard.splits)
    logger.info(
        "Reading shard %d: %d split(s)  format=%s  batch_size=%d"
        "  columns=%s  filters=%s  partitioning=%s",
        shard.id,
        n_splits,
        format,
        batch_size,
        columns if columns else "all",
        "yes" if filters is not None else "none",
        partitioning or "none",
    )
    for i, sp in enumerate(shard.splits):
        fname = sp.file.path.rsplit("/", 1)[-1]
        rows = (
            sp.row_range.length
            if sp.row_range is not None
            else (sp.file.record_count or sp.file.file_size or "?")
        )
        row_range = (
            f"rows=[{sp.row_range.offset},{sp.row_range.offset + sp.row_range.length})"
            if sp.row_range is not None
            else "full"
        )
        logger.info("  [%d] %s  %s  total=%s", i, fname, row_range, rows)

    total_batches = 0
    for split in shard.splits:
        path = split.file.path
        fname = path.rsplit("/", 1)[-1]
        logger.debug("Opening file: %s  size=%s bytes", path, split.file.file_size)

        # Estimate bytes for this split
        split_bytes = 0
        if split.file.file_size is not None:
            if split.row_range is None:
                split_bytes = split.file.file_size
            elif split.file.record_count:
                split_bytes = int(
                    split.file.file_size * split.row_range.length / split.file.record_count
                )
            else:
                split_bytes = split.file.file_size

        # Determine total rows for pbar (for percentage display)
        split_total_rows: int | None = None
        if split.row_range is not None:
            split_total_rows = split.row_range.length
        elif split.file.record_count is not None:
            split_total_rows = split.file.record_count

        # Reset pbar for this file
        if pbar is not None:
            pbar.reset(total=split_total_rows)
            pbar.set_description(f"W{metrics.worker_id if metrics else '?'} | {fname}")

        arrow_fs, resolved_path = _get_arrow_filesystem(path, opts)

        if split.row_range is not None and arrow_format == "parquet":
            gen = _read_parquet_row_range(
                path,
                resolved_path,
                split.row_range,
                columns,
                filters,
                batch_size,
                arrow_fs,
                partitioning,
            )
        elif split.row_range is not None and arrow_format == "orc":
            gen = _read_orc_row_range(
                path,
                resolved_path,
                split.row_range,
                columns,
                filters,
                batch_size,
                arrow_fs,
                partitioning,
            )
        else:
            ds = pad.dataset(
                resolved_path,
                format=arrow_format,
                filesystem=arrow_fs,
                partitioning=partitioning,
            )
            scanner = ds.scanner(columns=columns, filter=filters, batch_size=batch_size)
            gen = scanner.to_batches()

        file_batches = 0
        file_rows = 0
        t_file = time.perf_counter()
        for batch in gen:
            file_batches += 1
            file_rows += batch.num_rows
            total_batches += 1
            if metrics is not None:
                metrics.rows_read += batch.num_rows
                metrics.batches_read += 1
            if pbar is not None:
                pbar.update(batch.num_rows)
            yield batch

        file_elapsed = time.perf_counter() - t_file
        if metrics is not None:
            metrics.files_read += 1
            metrics.bytes_read += split_bytes

        logger.info(
            "Worker %s file done: %s  rows=%d  batches=%d  bytes_est=%d  elapsed=%.3fs",
            metrics.worker_id if metrics else "?",
            fname,
            file_rows,
            file_batches,
            split_bytes,
            file_elapsed,
            extra={
                "event": "file_done",
                "worker_id": metrics.worker_id if metrics else None,
                "path": path,
                "rows_read": file_rows,
                "batches_read": file_batches,
                "bytes_read": split_bytes,
                "elapsed_sec": file_elapsed,
            },
        )

    logger.info(
        "Shard %d complete: %d batch(es) yielded from %d split(s)",
        shard.id,
        total_batches,
        n_splits,
    )


def _read_parquet_row_range(
    original_path: str,
    resolved_path: str,
    row_range: RowRange,
    columns: list[str] | None,
    filters: pc.Expression | None,
    batch_size: int,
    arrow_fs: pafs.FileSystem | None,
    partitioning: str | None,
) -> Iterator[pa.RecordBatch]:
    """Read a RowRange from a Parquet file using row group random access.

    Finds the row groups that cover [row_range.offset, row_range.offset + row_range.length),
    reads only those row groups (true seek, no full scan), applies filters, and yields
    RecordBatches of batch_size rows.

    When partitioning="hive", parses key=value segments from the original file path and
    attaches them as constant string columns to each batch.
    """
    kwargs = {"filesystem": arrow_fs} if arrow_fs is not None else {}
    pf = pq.ParquetFile(resolved_path, **kwargs)
    meta = pf.metadata

    target_start = row_range.offset
    target_end = row_range.offset + row_range.length

    # Find row group indices that overlap [target_start, target_end)
    rg_indices = []
    cumulative = 0
    for i in range(meta.num_row_groups):
        rg_start = cumulative
        rg_end = cumulative + meta.row_group(i).num_rows
        if rg_end > target_start and rg_start < target_end:
            rg_indices.append(i)
        cumulative = rg_end

    if not rg_indices:
        return

    hive_parts: dict[str, str] = {}
    if partitioning == "hive":
        hive_parts = _parse_hive_partitions(original_path)

    # Use iter_batches for streaming independent RecordBatches.
    # Slices from read_row_groups().to_batches() carry the full table buffer when
    # pickled for DataLoader IPC (97× size overhead). iter_batches produces
    # independently allocated batches that pickle at their actual size.
    #
    # When a filter is set, read full columns and apply the filter per batch
    # via a single-batch Table (small: batch_size rows, not the whole chunk).
    # When a filter is set AND column projection is needed, project after filtering.
    read_columns = None if (filters is not None and columns is not None) else columns

    for batch in pf.iter_batches(batch_size, row_groups=rg_indices, columns=read_columns):
        if filters is not None:
            batch_table = pa.Table.from_batches([batch])
            batch_table = batch_table.filter(filters)
            if columns is not None:
                batch_table = batch_table.select(columns)
            if hive_parts:
                for col_name, col_value in hive_parts.items():
                    if col_name not in batch_table.schema.names:
                        batch_table = batch_table.append_column(
                            col_name,
                            pa.array([col_value] * len(batch_table), type=pa.string()),
                        )
            for b in batch_table.to_batches(max_chunksize=batch_size):
                if b.num_rows > 0:
                    yield b
        elif hive_parts:
            batch_table = pa.Table.from_batches([batch])
            for col_name, col_value in hive_parts.items():
                if col_name not in batch_table.schema.names:
                    batch_table = batch_table.append_column(
                        col_name,
                        pa.array([col_value] * len(batch_table), type=pa.string()),
                    )
            for b in batch_table.to_batches(max_chunksize=batch_size):
                if b.num_rows > 0:
                    yield b
        else:
            if batch.num_rows > 0:
                yield batch


def _read_orc_row_range(
    original_path: str,
    resolved_path: str,
    row_range: RowRange,
    columns: list[str] | None,
    filters: pc.Expression | None,
    batch_size: int,
    arrow_fs: pafs.FileSystem | None,
    partitioning: str | None,
) -> Iterator[pa.RecordBatch]:
    """Read a RowRange from an ORC file using stripe-level random access.

    Recovers the stripe indices from row_range using the same uniform formula used
    by _orc_chunks at split-generation time. Reads only the required stripes via
    read_stripe(); applies filters post-read (ORC has limited predicate pushdown).
    """
    kwargs = {"filesystem": arrow_fs} if arrow_fs is not None else {}
    orf = orc.ORCFile(resolved_path, **kwargs)
    nstripes = orf.nstripes
    nrows = orf.nrows

    if nstripes == 0 or nrows == 0:
        return

    # Recover stripe indices using the same uniform formula as _orc_chunks
    start_stripe = round(row_range.offset * nstripes / nrows)
    end_row = row_range.offset + row_range.length
    end_stripe = nstripes if end_row >= nrows else round(end_row * nstripes / nrows)
    end_stripe = min(nstripes, max(start_stripe + 1, end_stripe))

    if start_stripe >= nstripes:
        return

    stripe_batches = [orf.read_stripe(i, columns=columns) for i in range(start_stripe, end_stripe)]
    if not stripe_batches:
        return

    table = pa.Table.from_batches(stripe_batches)

    if filters is not None:
        table = table.filter(filters)

    if partitioning == "hive":
        hive_parts = _parse_hive_partitions(original_path)
        for col_name, col_value in hive_parts.items():
            if col_name not in table.schema.names:
                table = table.append_column(
                    col_name,
                    pa.array([col_value] * len(table), type=pa.string()),
                )

    for batch in table.to_batches(max_chunksize=batch_size):
        if batch.num_rows > 0:
            yield batch


def _get_arrow_filesystem(path: str, storage_options: dict) -> tuple[pafs.FileSystem | None, str]:
    """Return a (pyarrow FileSystem, resolved path) pair for the given path.

    For local paths returns (None, path) — pyarrow uses its default local fs.
    For remote paths wraps the fsspec filesystem in PyFileSystem.
    """
    fs, resolved = fsspec.url_to_fs(path, **storage_options)

    # Local filesystem — let pyarrow use its native local fs (faster, no wrapping)
    if fs.protocol in ("file", "local") or (
        isinstance(fs.protocol, tuple) and "file" in fs.protocol
    ):
        return None, resolved

    logger.debug("Wrapping %s filesystem in PyFileSystem for path: %s", type(fs).__name__, resolved)
    arrow_fs = pafs.PyFileSystem(pafs.FSSpecHandler(fs))
    return arrow_fs, resolved
