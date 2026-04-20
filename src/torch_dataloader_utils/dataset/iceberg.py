import logging
import os
from collections.abc import Callable, Iterator
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from torch.utils.data import DataLoader, IterableDataset

from torch_dataloader_utils.dataset.output import convert_batch
from torch_dataloader_utils.splits.core import IcebergDataFileInfo, Shard, SplitStrategy
from torch_dataloader_utils.splits.round_robin import RoundRobinSplitStrategy
from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

logger = logging.getLogger(__name__)

_SUPPORTED_OUTPUT_FORMATS = {"torch", "numpy", "arrow", "dict"}


def _require_pyiceberg():
    try:
        import pyiceberg  # noqa: F401
    except ImportError:
        raise ImportError(
            "pyiceberg is required for IcebergDataset.\n\n"
            "Install the required backend: pip install torch-dataloader-utils[iceberg]"
        )


def _detect_format(files: list[IcebergDataFileInfo]) -> str:
    """Detect file format from extensions. Raises ValueError for mixed formats."""
    exts = {f.path.rsplit(".", 1)[-1].lower() for f in files if "." in f.path}
    if "parquet" in exts and len(exts) == 1:
        return "parquet"
    if "orc" in exts and len(exts) == 1:
        return "orc"
    if len(exts) > 1:
        raise ValueError(
            f"Mixed file formats in Iceberg table: {exts}. "
            "Only homogeneous Parquet or ORC tables are supported in V1."
        )
    return "parquet"


def _auto_select_strategy(
    files: list[IcebergDataFileInfo], shuffle: bool, shuffle_seed: int
) -> SplitStrategy:
    if files:
        return TargetSizeSplitStrategy(shuffle=shuffle, seed=shuffle_seed)
    return RoundRobinSplitStrategy(shuffle=shuffle, seed=shuffle_seed)


def _resolve_files(
    table_identifier: str,
    catalog_config: dict,
    snapshot_id: int | None,
) -> tuple[list[IcebergDataFileInfo], bool, dict[str, set]]:
    """Connect to catalog, load table, scan, return file metadata.

    Returns:
        files:           IcebergDataFileInfo list — plain picklable data, safe to send to workers
        has_deletes:     True if any task has delete files
        delete_paths:    mapping of data file path → set of delete file paths (for worker reconnect)
    """
    from pyiceberg.catalog import load_catalog

    catalog_type = catalog_config.get("type", "unknown")
    logger.info(
        "Connecting to Iceberg catalog: type=%s  table=%s  snapshot_id=%s",
        catalog_type, table_identifier, snapshot_id,
    )

    catalog = load_catalog(**catalog_config)

    parts = table_identifier.split(".")
    if len(parts) == 3:
        table = catalog.load_table(f"{parts[1]}.{parts[2]}")
    else:
        table = catalog.load_table(table_identifier)

    # Fix #4: call current_snapshot() once, not once per file
    current_snap_id = table.current_snapshot().snapshot_id if snapshot_id is None else snapshot_id

    scan = table.scan(snapshot_id=snapshot_id)
    scan_tasks = list(scan.plan_files())

    files: list[IcebergDataFileInfo] = []
    # Store only picklable path strings, not live FileScanTask objects (Fix #1)
    delete_paths: dict[str, set[str]] = {}

    for task in scan_tasks:
        data_file = task.file
        path = data_file.file_path
        file_size = data_file.file_size_in_bytes
        record_count = data_file.record_count
        partition = dict(data_file.partition) if data_file.partition else None

        info = IcebergDataFileInfo(
            path=path,
            file_size=file_size,
            record_count=record_count,
            partition=partition,
            snapshot_id=current_snap_id,
        )

        # Store delete file paths as strings — picklable, reconnect in worker
        if task.delete_files:
            delete_paths[path] = {df.file_path for df in task.delete_files}

        logger.debug(
            "Resolved file: %s  size=%s  records=%s  partition=%s  deletes=%d",
            path, file_size, record_count, partition, len(task.delete_files),
        )
        files.append(info)

    total_size = sum(f.file_size for f in files if f.file_size is not None)
    total_records = sum(f.record_count for f in files if f.record_count is not None)
    has_deletes = bool(delete_paths)
    logger.info(
        "Iceberg scan complete: table=%s  files=%d  total_size=%d bytes  "
        "total_records=%d  has_delete_files=%s",
        table_identifier, len(files), total_size, total_records, has_deletes,
    )
    return files, has_deletes, delete_paths


def _read_task_with_deletes(
    data_file_path: str,
    table_identifier: str,
    catalog_config: dict,
    snapshot_id: int | None,
    columns: list[str] | None,
    filters: pc.Expression | None,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """Read a single data file with delete files applied using pyiceberg's ArrowScan.

    Reconnects to the catalog inside the worker — all arguments are plain picklable
    values (strings, dicts) so this is safe to call from a DataLoader worker subprocess.

    ArrowScan applies position delete files before yielding batches, so deleted
    rows are never returned.
    """
    from pyiceberg.catalog import load_catalog
    from pyiceberg.io.pyarrow import ArrowScan, schema_to_pyarrow
    from pyiceberg.expressions import AlwaysTrue

    catalog = load_catalog(**catalog_config)
    parts = table_identifier.split(".")
    if len(parts) == 3:
        table = catalog.load_table(f"{parts[1]}.{parts[2]}")
    else:
        table = catalog.load_table(table_identifier)

    # Reconstruct scan tasks from stored path strings
    scan = table.scan(snapshot_id=snapshot_id)
    task = next(
        (t for t in scan.plan_files() if t.file.file_path == data_file_path),
        None,
    )
    if task is None:
        logger.warning("Could not find task for %s — skipping", data_file_path)
        return

    projected_schema = table.schema()
    if columns is not None:
        projected_schema = projected_schema.select(columns)

    arrow_schema = schema_to_pyarrow(projected_schema)

    arrow_scan = ArrowScan(
        table_metadata=table.metadata,
        io=table.io,
        projected_schema=projected_schema,
        row_filter=AlwaysTrue(),
        case_sensitive=True,
        limit=None,
    )

    for batch in arrow_scan.to_record_batches(tasks=[task]):
        batch = batch.cast(arrow_schema)

        if filters is not None:
            tbl = pa.Table.from_batches([batch])
            tbl = tbl.filter(filters)
            if len(tbl) == 0:
                continue
            # Fix #2: guard against empty to_batches() result
            batches = tbl.to_batches()
            if not batches:
                continue
            batch = batches[0]

        offset = 0
        while offset < len(batch):
            yield batch.slice(offset, batch_size)
            offset += batch_size


class IcebergDataset(IterableDataset):
    """PyTorch IterableDataset for Apache Iceberg tables.

    Resolves the table to a list of data files via pyiceberg, then distributes
    those files across DataLoader workers using pre-computed splits — the same
    mechanism as StructuredDataset.

    Delete file handling
    --------------------
    When the Iceberg table contains **position delete files** (written by row-level
    DELETE or MERGE INTO operations), rows marked for deletion must be filtered out
    before returning batches. Reading the Parquet data files directly (as
    StructuredDataset does) would return deleted rows — pyarrow has no knowledge of
    Iceberg delete files.

    IcebergDataset detects whether any scan task carries delete files at construction
    time. When deletes are present, each worker reconnects to the catalog inside
    __iter__ using the stored catalog_config (plain dict — picklable) and uses
    pyiceberg's ArrowScan to apply position deletes before yielding batches.
    When no delete files exist, it uses the direct pyarrow reader (faster).

    Limitations
    -----------
    - **Equality deletes** (written by some engines for schema-evolution-aware deletes)
      are not supported by pyiceberg and will raise NotImplementedError. Compact the
      table with your query engine (e.g. ``ALTER TABLE ... REWRITE MANIFESTS``) to
      convert equality deletes to position deletes, or avoid equality-delete writes.
    - **Sub-file row-range splitting** is disabled for tables with delete files.
      Splits are at file granularity — one FileScanTask per chunk — because row
      offsets in position delete files reference absolute row positions within the
      original data file, not within a sub-range slice. Without delete files,
      TargetSizeSplitStrategy sub-splits Parquet files at row group boundaries.
    - **Partition pruning** via pyarrow ``pc.Expression`` is not supported at the
      Iceberg scan level (pyiceberg uses its own expression type). The ``filters``
      parameter is applied as row-level pushdown only — all files in the partition
      are read and filtered in memory.
    - **Schema evolution**: if columns were renamed after data was written, column
      selection by name may not match across old and new files. Use pyiceberg's
      native column IDs for schema-evolved tables.
    - **TOCTOU**: delete files are detected at construction time. New deletes
      committed between construction and iteration will be missed. Pin snapshot_id
      for strict consistency.

    Usage::

        loader, dataset = IcebergDataset.create_dataloader(
            table="my_db.my_table",
            catalog_config={"type": "rest", "uri": "https://catalog.example.com"},
            num_workers=4,
            batch_size=1024,
        )
        for epoch in range(num_epochs):
            dataset.set_epoch(epoch)
            for batch in loader:
                ...
    """

    def __init__(
        self,
        table: str,
        catalog_config: dict,
        batch_size: int = 1024,
        columns: list[str] | None = None,
        filters: pc.Expression | None = None,
        snapshot_id: int | None = None,
        shuffle: bool = False,
        shuffle_seed: int = 42,
        split_strategy: SplitStrategy | None = None,
        num_workers: int = 1,
        output_format: str = "torch",
        collate_fn: Callable | None = None,
    ) -> None:
        _require_pyiceberg()

        # Fix #6: validate output_format + collate_fn early
        if output_format not in _SUPPORTED_OUTPUT_FORMATS:
            supported = ", ".join(sorted(_SUPPORTED_OUTPUT_FORMATS))
            raise ValueError(
                f"Unsupported output_format {output_format!r}. Supported: {supported}"
            )
        if output_format in ("arrow", "dict") and collate_fn is None:
            raise ValueError(
                f"output_format={output_format!r} requires a collate_fn. "
                "PyTorch's default collate cannot handle this type. "
                "Pass a custom collate_fn or use output_format='torch'."
            )

        self._table_identifier = table
        self._catalog_config = catalog_config
        self._batch_size = batch_size
        self._columns = columns
        self._filters = filters
        self._snapshot_id = snapshot_id
        self._shuffle = shuffle
        self._shuffle_seed = shuffle_seed
        self._num_workers = num_workers
        self._output_format = output_format
        self._collate_fn = collate_fn

        # Fix #1: _resolve_files returns only picklable plain data — no live objects
        files, has_deletes, delete_paths = _resolve_files(table, catalog_config, snapshot_id)
        if not files:
            raise FileNotFoundError(
                f"No data files found in Iceberg table {table!r} "
                f"(snapshot_id={snapshot_id}). Check the table exists and has data."
            )

        self._has_deletes = has_deletes
        # Plain dict of path → set[str] — fully picklable
        self._delete_paths = delete_paths

        fmt = _detect_format(files)
        logger.info(
            "Detected file format: %s  has_delete_files=%s", fmt, self._has_deletes
        )
        if self._has_deletes:
            logger.info(
                "Delete files detected — workers will reconnect to catalog per file "
                "(file-level splits only, no sub-file row-range splitting)"
            )

        self._format = fmt
        self._files = files
        self._strategy = (
            split_strategy
            if split_strategy is not None
            else _auto_select_strategy(files, shuffle, shuffle_seed)
        )
        self._epoch: int = 0
        self._splits: list[Shard] = self._generate_splits()

    def _generate_splits(self) -> list[Shard]:
        n = max(self._num_workers, 1)
        return self._strategy.generate(self._files, num_workers=n, epoch=self._epoch)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for shuffle reproducibility. Call before each epoch when shuffle=True."""
        self._epoch = epoch
        self._splits = self._generate_splits()
        logger.info(
            "Regenerated splits for epoch %d  strategy=%s  num_workers=%d",
            epoch, type(self._strategy).__name__, self._num_workers,
        )

    def __iter__(self) -> Iterator[Any]:
        from torch.utils.data import get_worker_info
        from torch_dataloader_utils.format.reader import read_split

        info = get_worker_info()
        worker_id = info.id if info is not None else 0

        if worker_id >= len(self._splits):
            logger.debug(
                "Worker %d: no split assigned (%d split(s) for %d worker(s))",
                worker_id, len(self._splits), self._num_workers,
            )
            return

        shard = self._splits[worker_id]
        logger.info(
            "Worker %d: assigned shard %d with %d split(s)  has_deletes=%s",
            worker_id, shard.id, len(shard.splits), self._has_deletes,
        )

        if self._has_deletes:
            for split in shard.splits:
                path = split.file.path
                logger.debug(
                    "Worker %d: reading with ArrowScan (deletes) → %s",
                    worker_id, path,
                )
                for batch in _read_task_with_deletes(
                    data_file_path=path,
                    table_identifier=self._table_identifier,
                    catalog_config=self._catalog_config,
                    snapshot_id=self._snapshot_id,
                    columns=self._columns,
                    filters=self._filters,
                    batch_size=self._batch_size,
                ):
                    yield convert_batch(batch, self._output_format)
        else:
            # Fast path: direct pyarrow reader with sub-file splitting support
            for batch in read_split(
                shard,
                format=self._format,
                batch_size=self._batch_size,
                columns=self._columns,
                filters=self._filters,
            ):
                yield convert_batch(batch, self._output_format)

    @classmethod
    def create_dataloader(
        cls,
        table: str,
        catalog_config: dict,
        batch_size: int = 1024,
        columns: list[str] | None = None,
        filters: pc.Expression | None = None,
        snapshot_id: int | None = None,
        shuffle: bool = False,
        shuffle_seed: int = 42,
        split_strategy: SplitStrategy | None = None,
        num_workers: int | None = None,
        output_format: str = "torch",
        collate_fn: Callable | None = None,
    ) -> tuple[DataLoader, "IcebergDataset"]:
        """Create a DataLoader for an Iceberg table.

        Returns (DataLoader, dataset) — keep a reference to dataset to call
        set_epoch(n) at the start of each epoch when shuffle=True.
        """
        _require_pyiceberg()

        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 1) - 1)
            logger.info("Auto-detected num_workers=%d", num_workers)

        logger.info(
            "IcebergDataset.create_dataloader: table=%s  num_workers=%d  "
            "batch_size=%d  output_format=%s  shuffle=%s  snapshot_id=%s",
            table, num_workers, batch_size, output_format, shuffle, snapshot_id,
        )

        dataset = cls(
            table=table,
            catalog_config=catalog_config,
            batch_size=batch_size,
            columns=columns,
            filters=filters,
            snapshot_id=snapshot_id,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            split_strategy=split_strategy,
            num_workers=num_workers,
            output_format=output_format,
            collate_fn=collate_fn,
        )

        effective_collate = collate_fn
        if effective_collate is None and output_format != "torch":
            effective_collate = lambda x: x  # noqa: E731

        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            collate_fn=effective_collate,
        )

        return loader, dataset
