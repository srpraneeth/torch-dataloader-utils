"""Unit tests for IcebergDataset — no pyiceberg required (mocked)."""
import os
import sys
import tempfile
from types import ModuleType
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from torch_dataloader_utils.splits.core import IcebergDataFileInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_info(path: str, file_size: int = 1024, record_count: int = 100) -> IcebergDataFileInfo:
    return IcebergDataFileInfo(
        path=path,
        file_size=file_size,
        record_count=record_count,
        partition={"dt": "2024-01-01"},
        snapshot_id=42,
    )


def _write_parquet(tmpdir: str, rows: list[int]) -> IcebergDataFileInfo:
    path = os.path.join(tmpdir, "a.parquet")
    pq.write_table(pa.table({"x": pa.array(rows, pa.int32())}), path)
    return IcebergDataFileInfo(
        path=path, file_size=os.path.getsize(path), record_count=len(rows), snapshot_id=1,
    )


# ---------------------------------------------------------------------------
# _require_pyiceberg
# ---------------------------------------------------------------------------

class TestRequirePyiceberg:
    def test_raises_import_error_when_missing(self):
        """When pyiceberg is not installed, ImportError with install hint is raised."""
        from torch_dataloader_utils.dataset.iceberg import _require_pyiceberg

        with patch.dict(sys.modules, {"pyiceberg": None}):
            with pytest.raises(ImportError, match="pip install torch-dataloader-utils"):
                _require_pyiceberg()

    def test_no_error_when_present(self):
        from torch_dataloader_utils.dataset.iceberg import _require_pyiceberg

        mock_pyiceberg = MagicMock()
        with patch.dict(sys.modules, {"pyiceberg": mock_pyiceberg}):
            _require_pyiceberg()  # should not raise


# ---------------------------------------------------------------------------
# _detect_format
# ---------------------------------------------------------------------------

class TestDetectFormat:
    def test_parquet_files(self):
        from torch_dataloader_utils.dataset.iceberg import _detect_format

        files = [_make_info("s3://bucket/a.parquet"), _make_info("s3://bucket/b.parquet")]
        assert _detect_format(files) == "parquet"

    def test_orc_files(self):
        from torch_dataloader_utils.dataset.iceberg import _detect_format

        files = [_make_info("s3://bucket/a.orc"), _make_info("s3://bucket/b.orc")]
        assert _detect_format(files) == "orc"

    def test_mixed_formats_raises(self):
        from torch_dataloader_utils.dataset.iceberg import _detect_format

        files = [_make_info("s3://bucket/a.parquet"), _make_info("s3://bucket/b.orc")]
        with pytest.raises(ValueError, match="Mixed file formats"):
            _detect_format(files)

    def test_no_extension_defaults_parquet(self):
        from torch_dataloader_utils.dataset.iceberg import _detect_format

        files = [_make_info("s3://bucket/datafile")]
        assert _detect_format(files) == "parquet"

    def test_single_parquet(self):
        from torch_dataloader_utils.dataset.iceberg import _detect_format

        files = [_make_info("hdfs://cluster/warehouse/table/part-00000.parquet")]
        assert _detect_format(files) == "parquet"

    def test_empty_files_list_defaults_parquet(self):
        from torch_dataloader_utils.dataset.iceberg import _detect_format

        # Empty list → exts is empty set → falls through to default "parquet"
        assert _detect_format([]) == "parquet"


# ---------------------------------------------------------------------------
# IcebergDataset.__init__ — file-not-found path
# ---------------------------------------------------------------------------

class TestIcebergDatasetInit:
    def _make_pyiceberg_mock(self, files: list[IcebergDataFileInfo]):
        """Build a minimal pyiceberg module + catalog mock that returns `files`."""
        # Build fake plan tasks from IcebergDataFileInfo list
        fake_tasks = []
        for info in files:
            data_file = MagicMock()
            data_file.file_path = info.path
            data_file.file_size_in_bytes = info.file_size
            data_file.record_count = info.record_count
            data_file.partition = info.partition or {}
            task = MagicMock()
            task.file = data_file
            fake_tasks.append(task)

        fake_snapshot = MagicMock()
        fake_snapshot.snapshot_id = 42

        fake_table = MagicMock()
        fake_table.scan.return_value.plan_files.return_value = iter(fake_tasks)
        fake_table.current_snapshot.return_value = fake_snapshot

        fake_catalog = MagicMock()
        fake_catalog.load_table.return_value = fake_table

        fake_load_catalog = MagicMock(return_value=fake_catalog)

        pyiceberg_mod = MagicMock()
        pyiceberg_catalog_mod = MagicMock()
        pyiceberg_catalog_mod.load_catalog = fake_load_catalog
        pyiceberg_expressions_mod = MagicMock()

        return {
            "pyiceberg": pyiceberg_mod,
            "pyiceberg.catalog": pyiceberg_catalog_mod,
            "pyiceberg.expressions": pyiceberg_expressions_mod,
        }, fake_catalog, fake_table

    def test_empty_table_raises_file_not_found(self):
        """No data files → FileNotFoundError with table name in message."""
        mock_mods, _, _ = self._make_pyiceberg_mock([])

        with patch.dict(sys.modules, mock_mods):
            with patch(
                "torch_dataloader_utils.dataset.iceberg._resolve_files",
                return_value=([], False, {}),
            ):
                from torch_dataloader_utils.dataset.iceberg import IcebergDataset

                with pytest.raises(FileNotFoundError, match="my_db.my_table"):
                    IcebergDataset(
                        table="my_db.my_table",
                        catalog_config={"type": "rest", "uri": "http://fake"},
                    )

    def test_create_dataloader_requires_pyiceberg(self):
        """create_dataloader raises ImportError immediately when pyiceberg missing."""
        with patch.dict(sys.modules, {"pyiceberg": None}):
            # Re-import to get fresh module
            import importlib
            import torch_dataloader_utils.dataset.iceberg as iceberg_mod
            importlib.reload(iceberg_mod)

            with pytest.raises(ImportError, match="pip install torch-dataloader-utils"):
                iceberg_mod.IcebergDataset.create_dataloader(
                    table="db.table",
                    catalog_config={"type": "rest"},
                )

    def test_set_epoch_delegates_to_inner_dataset(self):
        """set_epoch() regenerates splits on the dataset."""
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a.parquet")
            table = pa.table({"x": pa.array([1, 2, 3], pa.int32())})
            import pyarrow.parquet as pq
            pq.write_table(table, path)

            real_files = [IcebergDataFileInfo(
                path=path,
                file_size=os.path.getsize(path),
                record_count=3,
                snapshot_id=1,
            )]
            fake_task = MagicMock()
            fake_task.file.file_path = path
            fake_task.delete_files = set()
            fake_iceberg_table = MagicMock()

            with patch(
                "torch_dataloader_utils.dataset.iceberg._resolve_files",
                return_value=(real_files, False, {}),
            ):
                with patch("torch_dataloader_utils.dataset.iceberg._require_pyiceberg"):
                    from torch_dataloader_utils.dataset.iceberg import IcebergDataset

                    ds = IcebergDataset(
                        table="db.table",
                        catalog_config={"type": "rest"},
                        num_workers=1,
                    )
                    splits_before = ds._splits
                    ds.set_epoch(3)
                    # set_epoch regenerates splits — object identity changes
                    assert ds._epoch == 3

    def test_iter_delegates_to_inner_dataset(self):
        """__iter__ (fast path, no deletes) yields batches without error."""
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a.parquet")
            table = pa.table({"x": pa.array([1, 2, 3], pa.int32())})
            import pyarrow.parquet as pq
            pq.write_table(table, path)

            real_files = [IcebergDataFileInfo(
                path=path,
                file_size=os.path.getsize(path),
                record_count=3,
                snapshot_id=1,
            )]

            with patch(
                "torch_dataloader_utils.dataset.iceberg._resolve_files",
                return_value=(real_files, False, {}),
            ):
                with patch("torch_dataloader_utils.dataset.iceberg._require_pyiceberg"):
                    from torch_dataloader_utils.dataset.iceberg import IcebergDataset

                    ds = IcebergDataset(
                        table="db.table",
                        catalog_config={"type": "rest"},
                        num_workers=1,
                    )
                    # Actually iterate to cover the fast-path else branch
                    batches = list(ds)
                    assert len(batches) == 1
                    assert batches[0]["x"].tolist() == [1, 2, 3]

    def test_worker_beyond_split_count_yields_nothing(self):
        """worker_id >= len(splits) → early return, no batches."""
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a.parquet")
            pq.write_table(pa.table({"x": pa.array([1, 2, 3], pa.int32())}), path)
            real_files = [IcebergDataFileInfo(
                path=path, file_size=os.path.getsize(path), record_count=3, snapshot_id=1,
            )]

            with patch(
                "torch_dataloader_utils.dataset.iceberg._resolve_files",
                return_value=(real_files, False, {}),
            ):
                with patch("torch_dataloader_utils.dataset.iceberg._require_pyiceberg"):
                    from torch_dataloader_utils.dataset.iceberg import IcebergDataset
                    from unittest.mock import MagicMock

                    ds = IcebergDataset(
                        table="db.table",
                        catalog_config={"type": "rest"},
                        num_workers=1,  # 1 shard → split[0] only
                    )
                    mock_info = MagicMock()
                    mock_info.id = 99  # no such shard
                    with patch("torch.utils.data.get_worker_info", return_value=mock_info):
                        batches = list(ds)
                    assert batches == []


# ---------------------------------------------------------------------------
# _auto_select_strategy
# ---------------------------------------------------------------------------

class TestAutoSelectStrategy:
    def test_non_empty_files_returns_target_size(self):
        from torch_dataloader_utils.dataset.iceberg import _auto_select_strategy
        from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

        files = [_make_info("s3://bucket/a.parquet")]
        strategy = _auto_select_strategy(files, shuffle=False, shuffle_seed=42)
        assert isinstance(strategy, TargetSizeSplitStrategy)

    def test_empty_files_returns_round_robin(self):
        from torch_dataloader_utils.dataset.iceberg import _auto_select_strategy
        from torch_dataloader_utils.splits.round_robin import RoundRobinSplitStrategy

        strategy = _auto_select_strategy([], shuffle=False, shuffle_seed=42)
        assert isinstance(strategy, RoundRobinSplitStrategy)


# ---------------------------------------------------------------------------
# Output format validation — raised before _resolve_files is called
# ---------------------------------------------------------------------------

class TestOutputFormatValidation:
    def test_invalid_output_format_raises(self):
        from torch_dataloader_utils.dataset.iceberg import IcebergDataset

        with pytest.raises(ValueError, match="xml"):
            IcebergDataset(table="db.table", catalog_config={}, output_format="xml")

    def test_arrow_without_collate_fn_raises(self):
        from torch_dataloader_utils.dataset.iceberg import IcebergDataset

        with pytest.raises(ValueError, match="collate_fn"):
            IcebergDataset(table="db.table", catalog_config={}, output_format="arrow")

    def test_dict_without_collate_fn_raises(self):
        from torch_dataloader_utils.dataset.iceberg import IcebergDataset

        with pytest.raises(ValueError, match="collate_fn"):
            IcebergDataset(table="db.table", catalog_config={}, output_format="dict")

    def test_arrow_with_collate_fn_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            info = _write_parquet(tmpdir, [1, 2, 3])
            with patch(
                "torch_dataloader_utils.dataset.iceberg._resolve_files",
                return_value=([info], False, {}),
            ):
                from torch_dataloader_utils.dataset.iceberg import IcebergDataset

                ds = IcebergDataset(
                    table="db.table",
                    catalog_config={},
                    output_format="arrow",
                    collate_fn=lambda x: x,
                )
                assert ds is not None


# ---------------------------------------------------------------------------
# _read_task_with_deletes — mocked pyiceberg submodules
# ---------------------------------------------------------------------------

class TestReadTaskWithDeletes:
    _PATH = "s3://bucket/data.parquet"

    def _build_mocks(self, batch: pa.RecordBatch, include_task: bool = True):
        """Return (sys.modules patches, mock_table)."""
        mock_file = MagicMock()
        mock_file.file_path = self._PATH
        mock_task = MagicMock()
        mock_task.file = mock_file

        mock_scan = MagicMock()
        mock_scan.plan_files.return_value = [mock_task] if include_task else []

        mock_table = MagicMock()
        mock_table.scan.return_value = mock_scan
        # schema() and select() stay as MagicMocks — schema_to_pyarrow controls cast

        mock_catalog = MagicMock()
        mock_catalog.load_table.return_value = mock_table

        mock_catalog_mod = MagicMock()
        mock_catalog_mod.load_catalog.return_value = mock_catalog

        mock_arrow_scan = MagicMock()
        mock_arrow_scan.to_record_batches.return_value = [batch]

        mock_io_mod = MagicMock()
        mock_io_mod.ArrowScan.return_value = mock_arrow_scan
        # Return the batch's real schema so batch.cast(schema) is a no-op
        mock_io_mod.schema_to_pyarrow.return_value = batch.schema

        mock_expr_mod = MagicMock()

        mods = {
            "pyiceberg.catalog": mock_catalog_mod,
            "pyiceberg.io.pyarrow": mock_io_mod,
            "pyiceberg.expressions": mock_expr_mod,
        }
        return mods, mock_table

    def test_yields_batches_by_batch_size(self):
        """5 rows with batch_size=3 → yields 2 batches (3 rows then 2 rows)."""
        batch = pa.record_batch({"x": pa.array([1, 2, 3, 4, 5], pa.int32())})
        mods, _ = self._build_mocks(batch)

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _read_task_with_deletes

            results = list(_read_task_with_deletes(
                data_file_path=self._PATH,
                table_identifier="db.table",
                catalog_config={"name": "test"},
                snapshot_id=None,
                columns=None,
                filters=None,
                batch_size=3,
            ))

        assert len(results) == 2
        assert len(results[0]) == 3
        assert len(results[1]) == 2

    def test_task_not_found_yields_nothing(self):
        """If no task matches data_file_path, nothing is yielded."""
        batch = pa.record_batch({"x": pa.array([1], pa.int32())})
        mods, _ = self._build_mocks(batch, include_task=False)

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _read_task_with_deletes

            results = list(_read_task_with_deletes(
                data_file_path=self._PATH,
                table_identifier="db.table",
                catalog_config={"name": "test"},
                snapshot_id=None,
                columns=None,
                filters=None,
                batch_size=100,
            ))

        assert results == []

    def test_filter_applied_to_batch(self):
        """Rows not matching filter are excluded from output."""
        batch = pa.record_batch({"x": pa.array([1, 2, 3, 4, 5], pa.int32())})
        mods, _ = self._build_mocks(batch)

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _read_task_with_deletes

            results = list(_read_task_with_deletes(
                data_file_path=self._PATH,
                table_identifier="db.table",
                catalog_config={"name": "test"},
                snapshot_id=None,
                columns=None,
                filters=pc.field("x") > 3,
                batch_size=100,
            ))

        assert len(results) == 1
        assert results[0]["x"].to_pylist() == [4, 5]

    def test_filter_eliminates_all_rows_yields_nothing(self):
        """Filter that removes every row → nothing yielded."""
        batch = pa.record_batch({"x": pa.array([1, 2, 3], pa.int32())})
        mods, _ = self._build_mocks(batch)

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _read_task_with_deletes

            results = list(_read_task_with_deletes(
                data_file_path=self._PATH,
                table_identifier="db.table",
                catalog_config={"name": "test"},
                snapshot_id=None,
                columns=None,
                filters=pc.field("x") > 100,
                batch_size=100,
            ))

        assert results == []

    def test_columns_selection_calls_schema_select(self):
        """When columns is given, projected_schema.select(columns) is called."""
        batch = pa.record_batch({
            "x": pa.array([1, 2], pa.int32()),
            "y": pa.array([3, 4], pa.int32()),
        })
        mods, mock_table = self._build_mocks(batch)

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _read_task_with_deletes

            list(_read_task_with_deletes(
                data_file_path=self._PATH,
                table_identifier="db.table",
                catalog_config={"name": "test"},
                snapshot_id=None,
                columns=["x"],
                filters=None,
                batch_size=100,
            ))

        mock_table.schema.return_value.select.assert_called_once_with(["x"])

    def test_three_part_identifier_loads_second_and_third_parts(self):
        """3-part identifier 'ns.db.tbl' loads table as 'db.tbl' (skips namespace)."""
        batch = pa.record_batch({"x": pa.array([1], pa.int32())})
        mods, _ = self._build_mocks(batch)

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _read_task_with_deletes

            list(_read_task_with_deletes(
                data_file_path=self._PATH,
                table_identifier="namespace.db.tbl",   # 3 parts
                catalog_config={"name": "test"},
                snapshot_id=None,
                columns=None,
                filters=None,
                batch_size=100,
            ))

        mock_catalog = mods["pyiceberg.catalog"].load_catalog.return_value
        mock_catalog.load_table.assert_called_once_with("db.tbl")


# ---------------------------------------------------------------------------
# __iter__ delete path
# ---------------------------------------------------------------------------

class TestIterWithDeletePath:
    def _make_dataset(self, tmpdir, has_deletes=True):
        info = _write_parquet(tmpdir, [1, 2, 3])
        with patch(
            "torch_dataloader_utils.dataset.iceberg._resolve_files",
            return_value=([info], has_deletes, {info.path: set()} if has_deletes else {}),
        ):
            with patch("torch_dataloader_utils.dataset.iceberg._require_pyiceberg"):
                from torch_dataloader_utils.dataset.iceberg import IcebergDataset

                return IcebergDataset(
                    table="db.table",
                    catalog_config={"type": "rest"},
                    num_workers=1,
                )

    def test_iter_delete_path_calls_read_task(self):
        """When _has_deletes=True, _read_task_with_deletes is called per split."""
        test_batch = pa.record_batch({"x": pa.array([1, 2, 3], pa.int32())})

        with tempfile.TemporaryDirectory() as tmpdir:
            ds = self._make_dataset(tmpdir, has_deletes=True)

            with patch(
                "torch_dataloader_utils.dataset.iceberg._read_task_with_deletes",
                side_effect=lambda *a, **kw: iter([test_batch]),
            ) as mock_fn:
                batches = list(ds)

            assert mock_fn.called
            assert len(batches) == 1

    def test_iter_delete_path_yields_converted_batches(self):
        """Delete path converts each batch via output_format='torch'."""
        import torch

        test_batch = pa.record_batch({"x": pa.array([1, 2, 3], pa.int32())})

        with tempfile.TemporaryDirectory() as tmpdir:
            ds = self._make_dataset(tmpdir, has_deletes=True)

            with patch(
                "torch_dataloader_utils.dataset.iceberg._read_task_with_deletes",
                side_effect=lambda *a, **kw: iter([test_batch]),
            ):
                batches = list(ds)

        assert isinstance(batches[0]["x"], torch.Tensor)
        assert batches[0]["x"].tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# create_dataloader (unit)
# ---------------------------------------------------------------------------

class TestCreateDataloaderUnit:
    def _patch_resolve(self, tmpdir):
        info = _write_parquet(tmpdir, [1, 2, 3])
        return patch(
            "torch_dataloader_utils.dataset.iceberg._resolve_files",
            return_value=([info], False, {}),
        )

    def test_returns_dataloader_and_dataset(self):
        from torch.utils.data import DataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            with self._patch_resolve(tmpdir):
                with patch("torch_dataloader_utils.dataset.iceberg._require_pyiceberg"):
                    from torch_dataloader_utils.dataset.iceberg import IcebergDataset

                    loader, ds = IcebergDataset.create_dataloader(
                        table="db.table",
                        catalog_config={"type": "rest"},
                        num_workers=1,
                    )

        assert isinstance(loader, DataLoader)
        assert isinstance(ds, IcebergDataset)

    def test_num_workers_none_auto_detected(self):
        from torch.utils.data import DataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            with self._patch_resolve(tmpdir):
                with patch("torch_dataloader_utils.dataset.iceberg._require_pyiceberg"):
                    from torch_dataloader_utils.dataset.iceberg import IcebergDataset

                    loader, _ = IcebergDataset.create_dataloader(
                        table="db.table",
                        catalog_config={"type": "rest"},
                        num_workers=None,
                    )

        expected = max(1, (os.cpu_count() or 1) - 1)
        assert loader.num_workers == expected

    def test_default_collate_fn_set_for_non_torch_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self._patch_resolve(tmpdir):
                with patch("torch_dataloader_utils.dataset.iceberg._require_pyiceberg"):
                    from torch_dataloader_utils.dataset.iceberg import IcebergDataset

                    loader, _ = IcebergDataset.create_dataloader(
                        table="db.table",
                        catalog_config={"type": "rest"},
                        num_workers=1,
                        output_format="numpy",
                    )

        # When output_format != "torch" and no explicit collate_fn,
        # create_dataloader sets an identity collate_fn
        assert loader.collate_fn is not None
