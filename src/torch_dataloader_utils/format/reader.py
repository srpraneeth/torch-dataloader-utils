import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pad
import pyarrow.fs as pafs
import pyarrow.parquet as pq

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

    Yields:
        pyarrow.RecordBatch

    Raises:
        ValueError: When format is not supported.
    """
    if format not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise ValueError(
            f"Unsupported format {format!r}. Supported formats: {supported}"
        )

    arrow_format = _FORMAT_ALIASES.get(format, format)
    opts = storage_options or {}

    n_splits = len(shard.splits)
    logger.info(
        "Reading shard %d: %d split(s)  format=%s  batch_size=%d  columns=%s  filters=%s  partitioning=%s",
        shard.id, n_splits, format, batch_size,
        columns if columns else "all",
        "yes" if filters is not None else "none",
        partitioning or "none",
    )
    for i, sp in enumerate(shard.splits):
        fname = sp.file.path.rsplit("/", 1)[-1]
        rows = sp.row_range.length if sp.row_range is not None else (sp.file.record_count or sp.file.file_size or "?")
        row_range = f"rows=[{sp.row_range.offset},{sp.row_range.offset + sp.row_range.length})" if sp.row_range is not None else "full"
        logger.info("  [%d] %s  %s  total=%s", i, fname, row_range, rows)

    total_batches = 0
    for split in shard.splits:
        path = split.file.path
        logger.debug("Opening file: %s  size=%s bytes", path, split.file.file_size)

        arrow_fs, resolved_path = _get_arrow_filesystem(path, opts)

        if split.row_range is not None and arrow_format == "parquet":
            gen = _read_parquet_row_range(
                path, resolved_path, split.row_range, columns, filters, batch_size,
                arrow_fs, partitioning,
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
        last_batch = None
        for batch in gen:
            file_batches += 1
            total_batches += 1
            last_batch = batch
            yield batch

        logger.debug(
            "Finished file: %s  batches=%d  rows_last_batch=%d",
            path, file_batches, last_batch.num_rows if last_batch is not None else 0,
        )

    logger.info("Shard %d complete: %d batch(es) yielded from %d split(s)", shard.id, total_batches, n_splits)


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

    # Read without column projection first when a filter references columns not in `columns`.
    # Apply filter on the full set of columns, then project down.
    read_columns = None if (filters is not None and columns is not None) else columns
    table = pf.read_row_groups(rg_indices, columns=read_columns)

    if filters is not None:
        table = table.filter(filters)

    if filters is not None and columns is not None:
        table = table.select(columns)

    # Attach Hive partition columns if requested
    if partitioning == "hive":
        hive_parts = _parse_hive_partitions(original_path)
        for col_name, col_value in hive_parts.items():
            if col_name not in table.schema.names:
                table = table.append_column(
                    col_name,
                    pa.array([col_value] * len(table), type=pa.string()),
                )

    # Yield in batch_size chunks
    offset = 0
    while offset < len(table):
        yield table.slice(offset, batch_size).to_batches()[0]
        offset += batch_size


def _get_arrow_filesystem(
    path: str, storage_options: dict
) -> tuple[pafs.FileSystem | None, str]:
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

