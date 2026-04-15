import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

import fsspec
import pyarrow as pa
import pyarrow.dataset as pad
import pyarrow.compute as pc
import pyarrow.fs as pafs

from torch_dataloader_utils.splits.core import Split

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"parquet", "orc", "csv", "json", "jsonl"}

# pyarrow uses "json" for both json and jsonl
_FORMAT_ALIASES = {"jsonl": "json"}


def read_split(
    split: Split,
    format: str,
    batch_size: int = 1024,
    columns: list[str] | None = None,
    filters: pc.Expression | None = None,
    storage_options: dict | None = None,
) -> Iterator[pa.RecordBatch]:
    """Read a Split and yield pyarrow RecordBatches.

    Args:
        split: The Split to read — contains one or more FileSplits.
        format: File format — parquet, orc, csv, json, or jsonl.
        batch_size: Number of rows per RecordBatch.
        columns: Column projection — only these columns are read. None = all columns.
        filters: Predicate pushdown expression via pyarrow.compute. None = no filter.
        storage_options: Passed to fsspec for filesystem construction.

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

    n_files = len(split.file_splits)
    logger.info(
        "Reading split %d: %d file(s)  format=%s  batch_size=%d  columns=%s  filters=%s",
        split.id, n_files, format, batch_size,
        columns if columns else "all",
        "yes" if filters is not None else "none",
    )

    total_batches = 0
    for file_split in split.file_splits:
        path = file_split.file.path
        logger.debug("Opening file: %s  size=%s bytes", path, file_split.file.file_size)

        arrow_fs, resolved_path = _get_arrow_filesystem(path, opts)

        ds = pad.dataset(resolved_path, format=arrow_format, filesystem=arrow_fs)
        scanner = ds.scanner(columns=columns, filter=filters, batch_size=batch_size)

        file_batches = 0
        for batch in scanner.to_batches():
            file_batches += 1
            total_batches += 1
            yield batch

        logger.debug(
            "Finished file: %s  batches=%d  rows_last_batch=%d",
            path, file_batches, batch.num_rows if file_batches > 0 else 0,
        )

    logger.info("Split %d complete: %d batch(es) yielded from %d file(s)", split.id, total_batches, n_files)


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
