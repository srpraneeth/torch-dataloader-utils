"""
Integration tests for IcebergDataset.

Requires pyiceberg to be installed:
    pip install torch-dataloader-utils[iceberg]

Run with:
    uv run pytest tests/integration/test_iceberg.py -m integration -v
"""

import os

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

pytest.importorskip("pyiceberg", reason="pyiceberg not installed — skipping Iceberg tests")

import torch
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import (
    FloatType,
    IntegerType,
    NestedField,
)

from torch_dataloader_utils.dataset.iceberg import IcebergDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_arrow_table(n_rows: int, row_id_offset: int = 0) -> pa.Table:
    row_ids = list(range(row_id_offset, row_id_offset + n_rows))
    return pa.table(
        {
            "row_id": pa.array(row_ids, type=pa.int32()),
            "feature_a": pa.array([float(i % 10) for i in row_ids], type=pa.float32()),
            "feature_b": pa.array([i % 100 for i in row_ids], type=pa.int32()),
            "label": pa.array([i % 2 for i in row_ids], type=pa.int32()),
        }
    )


def _collect(loader) -> dict[str, list]:
    result: dict[str, list] = {}
    for batch in loader:
        for key, val in batch.items():
            if key not in result:
                result[key] = []
            if isinstance(val, torch.Tensor):
                result[key].extend(val.tolist())
            else:
                result[key].extend(val.tolist() if hasattr(val, "tolist") else list(val))
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def iceberg_catalog(tmp_path):
    """Create an in-process SQLite-backed Iceberg catalog."""
    warehouse = str(tmp_path / "warehouse")
    os.makedirs(warehouse, exist_ok=True)
    catalog = SqlCatalog(
        "test",
        **{
            "uri": f"sqlite:///{tmp_path}/catalog.db",
            "warehouse": f"file://{warehouse}",
        },
    )
    catalog.create_namespace("mydb")
    return catalog, tmp_path


@pytest.fixture()
def iceberg_table_3files(iceberg_catalog):
    """Iceberg table with 3 Parquet files × 100 rows = 300 total rows."""
    catalog, tmp_path = iceberg_catalog

    schema = Schema(
        NestedField(1, "row_id", IntegerType(), required=False),
        NestedField(2, "feature_a", FloatType(), required=False),
        NestedField(3, "feature_b", IntegerType(), required=False),
        NestedField(4, "label", IntegerType(), required=False),
    )

    table = catalog.create_table("mydb.test_table", schema=schema)

    # Append 3 batches so pyiceberg creates 3 data files
    for i in range(3):
        arrow_table = _make_arrow_table(100, row_id_offset=i * 100)
        table.append(arrow_table)

    catalog_config = {
        "name": "test",
        "uri": f"sqlite:///{tmp_path}/catalog.db",
        "warehouse": f"file://{tmp_path}/warehouse",
    }
    return catalog_config, table, tmp_path


# ---------------------------------------------------------------------------
# Scenario: End-to-end Parquet read
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_end_to_end_parquet_read(iceberg_table_3files):
    """3 files × 100 rows → 300 rows, each batch is dict[str, torch.Tensor]."""
    catalog_config, _, _ = iceberg_table_3files

    loader, _ = IcebergDataset.create_dataloader(
        table="mydb.test_table",
        catalog_config=catalog_config,
        num_workers=0,
        batch_size=50,
    )

    rows = _collect(loader)
    assert len(rows["row_id"]) == 300

    # Verify batch type from first iteration
    batch = next(iter(loader))
    assert isinstance(batch, dict)
    assert isinstance(batch["row_id"], torch.Tensor)


# ---------------------------------------------------------------------------
# Scenario: Column projection
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_column_projection(iceberg_table_3files):
    """Only requested columns are returned."""
    catalog_config, _, _ = iceberg_table_3files

    loader, _ = IcebergDataset.create_dataloader(
        table="mydb.test_table",
        catalog_config=catalog_config,
        num_workers=0,
        columns=["feature_a", "label"],
    )

    batch = next(iter(loader))
    assert set(batch.keys()) == {"feature_a", "label"}
    assert "row_id" not in batch
    assert "feature_b" not in batch


# ---------------------------------------------------------------------------
# Scenario: Predicate pushdown
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_predicate_pushdown(iceberg_table_3files):
    """Row-level filter: feature_b >= 50 keeps 50 rows per file → 150 total."""
    catalog_config, _, _ = iceberg_table_3files

    loader, _ = IcebergDataset.create_dataloader(
        table="mydb.test_table",
        catalog_config=catalog_config,
        num_workers=0,
        filters=pc.field("feature_b") >= 50,
    )

    rows = _collect(loader)
    assert all(v >= 50 for v in rows["feature_b"])
    assert len(rows["feature_b"]) == 150


# ---------------------------------------------------------------------------
# Scenario: No rows dropped or duplicated
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_no_rows_dropped_or_duplicated(iceberg_table_3files):
    """All row_ids 0–299 appear exactly once."""
    catalog_config, _, _ = iceberg_table_3files

    loader, _ = IcebergDataset.create_dataloader(
        table="mydb.test_table",
        catalog_config=catalog_config,
        num_workers=0,
        batch_size=32,
    )

    rows = _collect(loader)
    row_ids = rows["row_id"]
    assert len(row_ids) == 300
    assert sorted(row_ids) == list(range(300))


# ---------------------------------------------------------------------------
# Scenario: Snapshot time travel
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_snapshot_time_travel(iceberg_catalog):
    """Reading an old snapshot returns only rows from that point in time."""
    catalog, tmp_path = iceberg_catalog

    schema = Schema(
        NestedField(1, "row_id", IntegerType(), required=False),
    )
    table = catalog.create_table("mydb.snapshot_table", schema=schema)

    # Snapshot 1: 100 rows
    table.append(pa.table({"row_id": pa.array(list(range(100)), type=pa.int32())}))
    snapshot_1_id = table.current_snapshot().snapshot_id

    # Snapshot 2: 100 more rows
    table.append(pa.table({"row_id": pa.array(list(range(100, 200)), type=pa.int32())}))

    catalog_config = {
        "name": "test",
        "uri": f"sqlite:///{tmp_path}/catalog.db",
        "warehouse": f"file://{tmp_path}/warehouse",
    }

    # Reading snapshot 1 should return only the first 100 rows
    loader, _ = IcebergDataset.create_dataloader(
        table="mydb.snapshot_table",
        catalog_config=catalog_config,
        num_workers=0,
        snapshot_id=snapshot_1_id,
    )

    rows = _collect(loader)
    assert len(rows["row_id"]) == 100
    assert all(v < 100 for v in rows["row_id"])


# ---------------------------------------------------------------------------
# Scenario: create_dataloader returns (DataLoader, IcebergDataset)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_create_dataloader_returns_tuple(iceberg_table_3files):
    """Return type is (DataLoader, IcebergDataset)."""
    from torch.utils.data import DataLoader

    catalog_config, _, _ = iceberg_table_3files

    result = IcebergDataset.create_dataloader(
        table="mydb.test_table",
        catalog_config=catalog_config,
        num_workers=0,
    )

    assert isinstance(result, tuple)
    assert len(result) == 2
    loader, dataset = result
    assert isinstance(loader, DataLoader)
    assert isinstance(dataset, IcebergDataset)


# ---------------------------------------------------------------------------
# Scenario: set_epoch works without error
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_set_epoch(iceberg_table_3files):
    """set_epoch can be called before each epoch."""
    catalog_config, _, _ = iceberg_table_3files

    _, dataset = IcebergDataset.create_dataloader(
        table="mydb.test_table",
        catalog_config=catalog_config,
        num_workers=0,
        shuffle=True,
    )

    dataset.set_epoch(0)
    dataset.set_epoch(1)
    dataset.set_epoch(2)  # should not raise


# ---------------------------------------------------------------------------
# Scenario: Missing pyiceberg raises ImportError
# ---------------------------------------------------------------------------


def test_missing_pyiceberg_raises():
    """ImportError with pip install hint when pyiceberg is not installed."""
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"pyiceberg": None}):
        from torch_dataloader_utils.dataset.iceberg import _require_pyiceberg

        with pytest.raises(ImportError, match="pip install torch-dataloader-utils"):
            _require_pyiceberg()


# ---------------------------------------------------------------------------
# Scenario: No delete files — fast path confirmed
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_no_deletes_uses_fast_path(iceberg_table_3files):
    """Append-only tables have no delete files — _has_deletes is False (fast path)."""
    catalog_config, _, _ = iceberg_table_3files

    _, dataset = IcebergDataset.create_dataloader(
        table="mydb.test_table",
        catalog_config=catalog_config,
        num_workers=0,
    )

    assert dataset._has_deletes is False


# ---------------------------------------------------------------------------
# Scenario: Delete file detection — has_deletes flag set correctly
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_has_deletes_flag_set_when_delete_files_present(iceberg_catalog):
    """When _resolve_files signals has_deletes=True, _has_deletes is set on the dataset."""
    from unittest.mock import patch

    catalog, tmp_path = iceberg_catalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import IntegerType, NestedField

    schema = Schema(NestedField(1, "row_id", IntegerType(), required=False))
    table = catalog.create_table("mydb.mock_delete_table", schema=schema)
    table.append(pa.table({"row_id": pa.array(list(range(10)), type=pa.int32())}))

    catalog_config = {
        "name": "test",
        "uri": f"sqlite:///{tmp_path}/catalog.db",
        "warehouse": f"file://{tmp_path}/warehouse",
    }

    original_resolve = __import__(
        "torch_dataloader_utils.dataset.iceberg", fromlist=["_resolve_files"]
    )._resolve_files

    def patched_resolve(table_id, config, snap_id, scan_filter=None):
        files, _, delete_paths = original_resolve(table_id, config, snap_id, scan_filter)
        # Force has_deletes=True to simulate a table with delete files
        return files, True, delete_paths

    with patch("torch_dataloader_utils.dataset.iceberg._resolve_files", patched_resolve):
        _, dataset = IcebergDataset.create_dataloader(
            table="mydb.mock_delete_table",
            catalog_config=catalog_config,
            num_workers=0,
        )

    assert dataset._has_deletes is True


# ---------------------------------------------------------------------------
# Scenario: Output formats
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_output_format_numpy(iceberg_table_3files):
    """numpy output: each column is an ndarray."""
    import numpy as np

    catalog_config, _, _ = iceberg_table_3files

    loader, _ = IcebergDataset.create_dataloader(
        table="mydb.test_table",
        catalog_config=catalog_config,
        num_workers=0,
        output_format="numpy",
    )

    batch = next(iter(loader))
    assert isinstance(batch["row_id"], np.ndarray)
    assert isinstance(batch["label"], np.ndarray)


@pytest.mark.integration
def test_output_format_arrow_with_collate(iceberg_table_3files):
    """arrow output with collate_fn: yields pa.RecordBatch directly."""
    catalog_config, _, _ = iceberg_table_3files

    loader, _ = IcebergDataset.create_dataloader(
        table="mydb.test_table",
        catalog_config=catalog_config,
        num_workers=0,
        output_format="arrow",
        collate_fn=lambda x: x,
    )

    batch = next(iter(loader))
    assert isinstance(batch, pa.RecordBatch)


@pytest.mark.integration
def test_output_format_dict_with_collate(iceberg_table_3files):
    """dict output with collate_fn: yields dict[str, list]."""
    catalog_config, _, _ = iceberg_table_3files

    loader, _ = IcebergDataset.create_dataloader(
        table="mydb.test_table",
        catalog_config=catalog_config,
        num_workers=0,
        output_format="dict",
        collate_fn=lambda x: x,
    )

    batch = next(iter(loader))
    assert isinstance(batch, dict)
    assert isinstance(batch["row_id"], list)


# ---------------------------------------------------------------------------
# Scenario: Validation errors
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_invalid_output_format_raises():
    """ValueError for unsupported output_format — raised before _resolve_files."""
    with pytest.raises(ValueError, match="xml"):
        IcebergDataset(table="db.table", catalog_config={}, output_format="xml")


@pytest.mark.integration
def test_arrow_without_collate_fn_raises():
    """ValueError when output_format='arrow' and no collate_fn provided."""
    with pytest.raises(ValueError, match="collate_fn"):
        IcebergDataset(table="db.table", catalog_config={}, output_format="arrow")


@pytest.mark.integration
def test_dict_without_collate_fn_raises():
    """ValueError when output_format='dict' and no collate_fn provided."""
    with pytest.raises(ValueError, match="collate_fn"):
        IcebergDataset(table="db.table", catalog_config={}, output_format="dict")


# ---------------------------------------------------------------------------
# Scenario: create_dataloader auto-detects num_workers
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_create_dataloader_num_workers_auto(iceberg_table_3files):
    """num_workers=None auto-detects CPU count - 1 (min 1)."""
    from torch.utils.data import DataLoader

    catalog_config, _, _ = iceberg_table_3files

    loader, _ = IcebergDataset.create_dataloader(
        table="mydb.test_table",
        catalog_config=catalog_config,
        num_workers=None,
    )

    expected = max(1, (os.cpu_count() or 1) - 1)
    assert isinstance(loader, DataLoader)
    assert loader.num_workers == expected


# ---------------------------------------------------------------------------
# Scenario: Shuffle produces different orders across epochs
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_shuffle_epoch_changes_split_order(iceberg_table_3files):
    """shuffle=True assigns files in different order across epochs."""
    catalog_config, _, _ = iceberg_table_3files

    _, dataset = IcebergDataset.create_dataloader(
        table="mydb.test_table",
        catalog_config=catalog_config,
        num_workers=1,
        shuffle=True,
        shuffle_seed=42,
    )

    orders = []
    for epoch in range(6):
        dataset.set_epoch(epoch)
        orders.append([sp.file.path for sp in dataset._splits[0].splits])

    assert not all(o == orders[0] for o in orders), "All epochs produced identical split order"


# ---------------------------------------------------------------------------
# Scenario: Multiple workers read disjoint data
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_multiple_workers_read_disjoint_data(iceberg_table_3files):
    """3 workers each read a different subset; together cover all 300 rows exactly once."""
    from unittest.mock import MagicMock, patch

    catalog_config, _, _ = iceberg_table_3files

    _, dataset = IcebergDataset.create_dataloader(
        table="mydb.test_table",
        catalog_config=catalog_config,
        num_workers=3,
    )

    all_row_ids = []
    for worker_id in range(3):
        mock_info = MagicMock()
        mock_info.id = worker_id
        with patch("torch.utils.data.get_worker_info", return_value=mock_info):
            for batch in dataset:
                all_row_ids.extend(batch["row_id"].tolist())

    assert len(all_row_ids) == 300
    assert sorted(all_row_ids) == list(range(300))


# ---------------------------------------------------------------------------
# Scenario: Delete path — __iter__ uses _read_task_with_deletes
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_iter_with_delete_flag_uses_slow_path(iceberg_catalog):
    """When _has_deletes=True, __iter__ routes through _read_task_with_deletes."""
    from unittest.mock import patch

    catalog, tmp_path = iceberg_catalog

    schema = Schema(NestedField(1, "row_id", IntegerType(), required=False))
    table = catalog.create_table("mydb.delete_iter_table", schema=schema)
    table.append(pa.table({"row_id": pa.array(list(range(10)), type=pa.int32())}))

    catalog_config = {
        "name": "test",
        "uri": f"sqlite:///{tmp_path}/catalog.db",
        "warehouse": f"file://{tmp_path}/warehouse",
    }

    # Force has_deletes=True so __iter__ takes the slow path
    from torch_dataloader_utils.dataset.iceberg import _resolve_files as _real_resolve

    def _patched_resolve(table_id, config, snap_id, scan_filter=None):
        files, _, delete_paths = _real_resolve(table_id, config, snap_id, scan_filter)
        return files, True, delete_paths

    test_batch = pa.record_batch({"row_id": pa.array(list(range(10)), pa.int32())})

    with patch("torch_dataloader_utils.dataset.iceberg._resolve_files", _patched_resolve):
        with patch(
            "torch_dataloader_utils.dataset.iceberg._read_task_with_deletes",
            side_effect=lambda *a, **kw: iter([test_batch]),
        ) as mock_fn:
            _, dataset = IcebergDataset.create_dataloader(
                table="mydb.delete_iter_table",
                catalog_config=catalog_config,
                num_workers=0,
            )
            rows = _collect(iter(dataset))

    assert mock_fn.called
    assert sorted(rows["row_id"]) == list(range(10))


# ---------------------------------------------------------------------------
# Scenario: Worker beyond split count yields nothing
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_worker_beyond_split_count_yields_nothing(iceberg_table_3files):
    """A worker with id >= num_splits returns without yielding."""
    from unittest.mock import MagicMock, patch

    catalog_config, _, _ = iceberg_table_3files

    _, dataset = IcebergDataset.create_dataloader(
        table="mydb.test_table",
        catalog_config=catalog_config,
        num_workers=2,
    )

    mock_info = MagicMock()
    mock_info.id = 99  # no such split
    with patch("torch.utils.data.get_worker_info", return_value=mock_info):
        batches = list(dataset)

    assert batches == []


# ---------------------------------------------------------------------------
# Fixture: unequal files with controlled row groups (for sub-file + rank tests)
# ---------------------------------------------------------------------------


@pytest.fixture()
def iceberg_table_unequal(iceberg_catalog):
    """Iceberg table with 4 unequal files registered via add_files.

    Files and row groups:
      part_0000.parquet — 100 rows, row_group_size= 50 → 2 row groups
      part_0001.parquet — 300 rows, row_group_size=100 → 3 row groups
      part_0002.parquet — 500 rows, row_group_size=100 → 5 row groups
      part_0003.parquet — 200 rows, row_group_size=100 → 2 row groups
    Total: 1100 rows, 12 row groups
    """
    catalog, tmp_path = iceberg_catalog

    schema = Schema(
        NestedField(1, "row_id", IntegerType(), required=False),
        NestedField(2, "feature_a", FloatType(), required=False),
        NestedField(3, "feature_b", IntegerType(), required=False),
        NestedField(4, "label", IntegerType(), required=False),
    )
    table = catalog.create_table("mydb.unequal_table", schema=schema)

    files_dir = tmp_path / "files"
    files_dir.mkdir()

    file_specs = [(100, 50), (300, 100), (500, 100), (200, 100)]
    file_paths = []
    offset = 0
    for i, (n_rows, rg_size) in enumerate(file_specs):
        row_ids = list(range(offset, offset + n_rows))
        arrow_table = pa.table(
            {
                "row_id": pa.array(row_ids, pa.int32()),
                "feature_a": pa.array([float(v % 10) for v in row_ids], pa.float32()),
                "feature_b": pa.array([v % 100 for v in row_ids], pa.int32()),
                "label": pa.array([v % 2 for v in row_ids], pa.int32()),
            }
        )
        path = files_dir / f"part_{i:04d}.parquet"
        pq.write_table(arrow_table, path, row_group_size=rg_size)
        file_paths.append(f"file://{path}")
        offset += n_rows

    table.add_files(file_paths)

    catalog_config = {
        "name": "test",
        "uri": f"sqlite:///{tmp_path}/catalog.db",
        "warehouse": f"file://{tmp_path}/warehouse",
    }
    total_rows = sum(n for n, _ in file_specs)
    return catalog_config, table, tmp_path, total_rows


# ---------------------------------------------------------------------------
# Scenario: Rank-aware sharding — disjoint, complete coverage
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_rank_aware_sharding_two_ranks(iceberg_table_3files):
    """Two ranks get disjoint file subsets that together cover all 300 rows."""
    from unittest.mock import MagicMock, patch

    catalog_config, _, _ = iceberg_table_3files

    all_ids = []
    for rank in range(2):
        _, dataset = IcebergDataset.create_dataloader(
            table="mydb.test_table",
            catalog_config=catalog_config,
            num_workers=1,
            num_ranks=2,
            rank=rank,
        )
        mock_info = MagicMock()
        mock_info.id = 0
        with patch("torch.utils.data.get_worker_info", return_value=mock_info):
            for batch in dataset:
                all_ids.extend(batch["row_id"].tolist())

    assert len(all_ids) == 300
    assert sorted(all_ids) == list(range(300))
    assert len(set(all_ids)) == 300  # no cross-rank duplicates


@pytest.mark.integration
def test_rank_aware_sharding_three_ranks(iceberg_table_3files):
    """Three ranks together cover all rows exactly once."""
    from unittest.mock import MagicMock, patch

    catalog_config, _, _ = iceberg_table_3files

    all_ids = []
    for rank in range(3):
        _, dataset = IcebergDataset.create_dataloader(
            table="mydb.test_table",
            catalog_config=catalog_config,
            num_workers=1,
            num_ranks=3,
            rank=rank,
        )
        mock_info = MagicMock()
        mock_info.id = 0
        with patch("torch.utils.data.get_worker_info", return_value=mock_info):
            for batch in dataset:
                all_ids.extend(batch["row_id"].tolist())

    assert sorted(all_ids) == list(range(300))
    assert len(set(all_ids)) == len(all_ids)


@pytest.mark.integration
def test_rank_aware_sharding_more_ranks_than_files(iceberg_table_3files):
    """With more ranks than files, some ranks get 0 splits — no rows missed."""
    from unittest.mock import MagicMock, patch

    catalog_config, _, _ = iceberg_table_3files

    all_ids = []
    for rank in range(5):
        _, dataset = IcebergDataset.create_dataloader(
            table="mydb.test_table",
            catalog_config=catalog_config,
            num_workers=1,
            num_ranks=5,
            rank=rank,
        )
        mock_info = MagicMock()
        mock_info.id = 0
        with patch("torch.utils.data.get_worker_info", return_value=mock_info):
            for batch in dataset:
                all_ids.extend(batch["row_id"].tolist())

    assert sorted(all_ids) == list(range(300))


# ---------------------------------------------------------------------------
# Scenario: Sub-file splitting with split_rows
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_sub_file_splitting_produces_row_group_chunks(iceberg_table_unequal):
    """split_rows forces Parquet row-group-level chunks; all rows still returned."""
    catalog_config, _, _, total_rows = iceberg_table_unequal

    loader, dataset = IcebergDataset.create_dataloader(
        table="mydb.unequal_table",
        catalog_config=catalog_config,
        num_workers=0,
        batch_size=128,
        split_rows=50,  # smaller than any row group → one chunk per row group
    )

    # 12 row groups total → 12 chunks
    total_splits = sum(len(s.splits) for s in dataset._splits)
    assert total_splits == 12

    rows = _collect(loader)
    assert sorted(rows["row_id"]) == list(range(total_rows))


@pytest.mark.integration
def test_sub_file_splitting_with_rank_sharding(iceberg_table_unequal):
    """split_rows + num_ranks together: each rank sees disjoint row-group chunks."""
    from unittest.mock import MagicMock, patch

    catalog_config, _, _, total_rows = iceberg_table_unequal

    all_ids = []
    for rank in range(3):
        _, dataset = IcebergDataset.create_dataloader(
            table="mydb.unequal_table",
            catalog_config=catalog_config,
            num_workers=1,
            num_ranks=3,
            rank=rank,
            split_rows=50,
        )
        mock_info = MagicMock()
        mock_info.id = 0
        with patch("torch.utils.data.get_worker_info", return_value=mock_info):
            for batch in dataset:
                all_ids.extend(batch["row_id"].tolist())

    assert sorted(all_ids) == list(range(total_rows))
    assert len(set(all_ids)) == len(all_ids)


@pytest.mark.integration
def test_split_bytes_string_form(iceberg_table_3files):
    """split_bytes accepts a human-readable string like '1MiB'."""
    catalog_config, _, _ = iceberg_table_3files

    loader, _ = IcebergDataset.create_dataloader(
        table="mydb.test_table",
        catalog_config=catalog_config,
        num_workers=0,
        split_bytes="1MiB",
    )
    rows = _collect(loader)
    assert sorted(rows["row_id"]) == list(range(300))


# ---------------------------------------------------------------------------
# Scenario: Overwrite snapshot — old data files not returned
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_overwrite_snapshot_returns_only_new_data(iceberg_catalog):
    """After overwriting a table, only new data files are returned — old snapshot
    files are marked DELETED in the manifest and excluded from scan."""
    catalog, tmp_path = iceberg_catalog

    schema = Schema(NestedField(1, "row_id", IntegerType(), required=False))
    table = catalog.create_table("mydb.overwrite_table", schema=schema)

    # Write snapshot 1: rows 0–99
    table.append(pa.table({"row_id": pa.array(list(range(100)), pa.int32())}))
    snap1_id = table.current_snapshot().snapshot_id

    # Overwrite (append-then-replace via overwrite): rows 1000–1099
    table.overwrite(pa.table({"row_id": pa.array(list(range(1000, 1100)), pa.int32())}))

    catalog_config = {
        "name": "test",
        "uri": f"sqlite:///{tmp_path}/catalog.db",
        "warehouse": f"file://{tmp_path}/warehouse",
    }

    # Current snapshot should return only new rows
    loader, _ = IcebergDataset.create_dataloader(
        table="mydb.overwrite_table",
        catalog_config=catalog_config,
        num_workers=0,
    )
    rows = _collect(loader)
    assert all(v >= 1000 for v in rows["row_id"]), "Old rows must not appear after overwrite"
    assert len(rows["row_id"]) == 100

    # Time travel to snapshot 1 returns only original rows
    loader_snap1, _ = IcebergDataset.create_dataloader(
        table="mydb.overwrite_table",
        catalog_config=catalog_config,
        num_workers=0,
        snapshot_id=snap1_id,
    )
    rows_snap1 = _collect(loader_snap1)
    assert all(v < 100 for v in rows_snap1["row_id"])
    assert len(rows_snap1["row_id"]) == 100


# ---------------------------------------------------------------------------
# Scenario: backward-compat V1 custom strategy (no num_ranks in signature)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_v1_custom_strategy_without_rank_params(iceberg_table_3files):
    """A custom strategy with V1 generate() signature works without num_ranks/rank."""
    from torch_dataloader_utils.splits.core import DataFileInfo, Shard, Split

    class V1Strategy:
        def generate(self, files: list[DataFileInfo], num_workers: int, epoch: int = 0):
            shards = [Shard(id=i) for i in range(num_workers)]
            for i, f in enumerate(files):
                shards[i % num_workers].splits.append(Split(file=f))
            return shards

    catalog_config, table, _ = iceberg_table_3files
    loader, _ = IcebergDataset.create_dataloader(
        table="mydb.test_table",
        catalog_config=catalog_config,
        num_workers=0,
        split_strategy=V1Strategy(),
        num_ranks=2,
        rank=0,
    )
    rows = _collect(loader)
    # Strategy ignores num_ranks — returns all 300 rows for rank 0
    assert len(rows["row_id"]) == 300
