"""Unit tests for IcebergDataset — no pyiceberg required (mocked)."""

import os
import sys
import tempfile
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
        path=path,
        file_size=os.path.getsize(path),
        record_count=len(rows),
        snapshot_id=1,
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
# _try_to_iceberg
# ---------------------------------------------------------------------------


class TestTryToIceberg:
    """Tests for the pc.Expression → pyiceberg BooleanExpression translator."""

    def _t(self, expr):
        from torch_dataloader_utils.dataset.iceberg import _try_to_iceberg

        return _try_to_iceberg(expr)

    # --- comparison operators with integer literals ---

    def test_gte_integer(self):
        from pyiceberg.expressions import GreaterThanOrEqual

        r = self._t(pc.field("row_id") >= 1875)
        assert isinstance(r, GreaterThanOrEqual)
        assert r.term.name == "row_id"
        assert r.literal.value == 1875

    def test_gt_integer(self):
        from pyiceberg.expressions import GreaterThan

        r = self._t(pc.field("score") > 0)
        assert isinstance(r, GreaterThan)
        assert r.term.name == "score"

    def test_lte_integer(self):
        from pyiceberg.expressions import LessThanOrEqual

        assert isinstance(self._t(pc.field("count") <= 100), LessThanOrEqual)

    def test_lt_integer(self):
        from pyiceberg.expressions import LessThan

        assert isinstance(self._t(pc.field("count") < 50), LessThan)

    def test_eq_integer(self):
        from pyiceberg.expressions import EqualTo

        r = self._t(pc.field("label") == 1)
        assert isinstance(r, EqualTo)
        assert r.literal.value == 1

    def test_ne_integer(self):
        from pyiceberg.expressions import NotEqualTo

        assert isinstance(self._t(pc.field("label") != 0), NotEqualTo)

    # --- other literal types ---

    def test_eq_string(self):
        from pyiceberg.expressions import EqualTo

        r = self._t(pc.field("region") == "us")
        assert isinstance(r, EqualTo)
        assert r.literal.value == "us"

    def test_gt_float(self):
        from pyiceberg.expressions import GreaterThan

        r = self._t(pc.field("score") > 0.5)
        assert isinstance(r, GreaterThan)
        assert abs(r.literal.value - 0.5) < 1e-9

    # --- compound expressions ---

    def test_and_combination(self):
        from pyiceberg.expressions import And

        r = self._t((pc.field("row_id") >= 100) & (pc.field("label") == 1))
        assert isinstance(r, And)

    def test_or_combination(self):
        from pyiceberg.expressions import Or

        r = self._t((pc.field("region") == "us") | (pc.field("region") == "eu"))
        assert isinstance(r, Or)

    # --- untranslatable expressions return None ---

    def test_returns_none_for_field_vs_field(self):
        """field >= field has no scalar literal — cannot translate."""
        r = self._t(pc.field("a") >= pc.field("b"))
        assert r is None

    def test_returns_none_when_pyiceberg_not_installed(self):
        """Returns None gracefully when pyiceberg.expressions is absent."""
        from torch_dataloader_utils.dataset.iceberg import _try_to_iceberg

        with patch.dict(sys.modules, {"pyiceberg.expressions": None}):
            r = _try_to_iceberg(pc.field("x") >= 1)
        assert r is None


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

        return (
            {
                "pyiceberg": pyiceberg_mod,
                "pyiceberg.catalog": pyiceberg_catalog_mod,
                "pyiceberg.expressions": pyiceberg_expressions_mod,
            },
            fake_catalog,
            fake_table,
        )

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
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a.parquet")
            table = pa.table({"x": pa.array([1, 2, 3], pa.int32())})
            import pyarrow.parquet as pq

            pq.write_table(table, path)

            real_files = [
                IcebergDataFileInfo(
                    path=path,
                    file_size=os.path.getsize(path),
                    record_count=3,
                    snapshot_id=1,
                )
            ]
            fake_task = MagicMock()
            fake_task.file.file_path = path
            fake_task.delete_files = set()

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
                    ds.set_epoch(3)
                    # set_epoch regenerates splits — object identity changes
                    assert ds._epoch == 3

    def test_iter_delegates_to_inner_dataset(self):
        """__iter__ (fast path, no deletes) yields batches without error."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a.parquet")
            table = pa.table({"x": pa.array([1, 2, 3], pa.int32())})
            import pyarrow.parquet as pq

            pq.write_table(table, path)

            real_files = [
                IcebergDataFileInfo(
                    path=path,
                    file_size=os.path.getsize(path),
                    record_count=3,
                    snapshot_id=1,
                )
            ]

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
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a.parquet")
            pq.write_table(pa.table({"x": pa.array([1, 2, 3], pa.int32())}), path)
            real_files = [
                IcebergDataFileInfo(
                    path=path,
                    file_size=os.path.getsize(path),
                    record_count=3,
                    snapshot_id=1,
                )
            ]

            with patch(
                "torch_dataloader_utils.dataset.iceberg._resolve_files",
                return_value=(real_files, False, {}),
            ):
                with patch("torch_dataloader_utils.dataset.iceberg._require_pyiceberg"):
                    from unittest.mock import MagicMock

                    from torch_dataloader_utils.dataset.iceberg import IcebergDataset

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

    def test_split_bytes_string_forwarded_to_strategy(self):
        """split_bytes='10MiB' is parsed and stored on the strategy."""
        from torch_dataloader_utils.dataset.iceberg import _auto_select_strategy
        from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

        files = [_make_info("s3://bucket/a.parquet")]
        strategy = _auto_select_strategy(files, shuffle=False, shuffle_seed=42, split_bytes="10MiB")
        assert isinstance(strategy, TargetSizeSplitStrategy)
        assert strategy.target_bytes == 10 * 1024 * 1024

    def test_split_rows_forwarded_to_strategy(self):
        """split_rows is forwarded as target_rows to TargetSizeSplitStrategy."""
        from torch_dataloader_utils.dataset.iceberg import _auto_select_strategy
        from torch_dataloader_utils.splits.target_size import TargetSizeSplitStrategy

        files = [_make_info("s3://bucket/a.parquet")]
        strategy = _auto_select_strategy(files, shuffle=False, shuffle_seed=42, split_rows=5000)
        assert isinstance(strategy, TargetSizeSplitStrategy)
        assert strategy.target_rows == 5000


# ---------------------------------------------------------------------------
# scan_filter auto-derivation
# ---------------------------------------------------------------------------


class TestAutoDeriveScanFilter:
    """Tests for the automatic scan_filter derivation from pc.Expression filters."""

    def _make_files(self, tmpdir):
        import os

        path = os.path.join(tmpdir, "a.parquet")
        pq.write_table(pa.table({"x": pa.array([1, 2, 3], pa.int32())}), path)
        return [
            IcebergDataFileInfo(
                path=path,
                file_size=os.path.getsize(path),
                record_count=3,
                snapshot_id=1,
            )
        ]

    def test_translatable_filters_derives_scan_filter(self):
        """When filters translates successfully, the derived iceberg expr reaches _resolve_files."""
        from pyiceberg.expressions import GreaterThanOrEqual

        with tempfile.TemporaryDirectory() as tmpdir:
            files = self._make_files(tmpdir)
            with patch(
                "torch_dataloader_utils.dataset.iceberg._resolve_files",
                return_value=(files, False, {}),
            ) as mock_resolve:
                with patch("torch_dataloader_utils.dataset.iceberg._require_pyiceberg"):
                    from torch_dataloader_utils.dataset.iceberg import IcebergDataset

                    IcebergDataset(
                        table="db.tbl",
                        catalog_config={},
                        num_workers=1,
                        filters=pc.field("row_id") >= 100,
                    )

                _, call_kwargs = mock_resolve.call_args
                derived = call_kwargs.get("scan_filter") or mock_resolve.call_args[0][3]
                assert isinstance(derived, GreaterThanOrEqual)
                assert derived.term.name == "row_id"

    def test_untranslatable_filters_leaves_scan_filter_none(self):
        """When filters cannot be translated, _resolve_files is called without scan_filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self._make_files(tmpdir)
            with patch(
                "torch_dataloader_utils.dataset.iceberg._resolve_files",
                return_value=(files, False, {}),
            ) as mock_resolve:
                with patch("torch_dataloader_utils.dataset.iceberg._require_pyiceberg"):
                    from torch_dataloader_utils.dataset.iceberg import IcebergDataset

                    IcebergDataset(
                        table="db.tbl",
                        catalog_config={},
                        num_workers=1,
                        filters=pc.field("a") >= pc.field("b"),  # field vs field — untranslatable
                    )

                _, call_kwargs = mock_resolve.call_args
                derived = call_kwargs.get("scan_filter") or mock_resolve.call_args[0][3]
                assert derived is None

    def test_explicit_scan_filter_not_overridden_by_derivation(self):
        """Explicit scan_filter takes precedence — _try_to_iceberg is not called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self._make_files(tmpdir)
            explicit = MagicMock(name="explicit_scan_filter")
            with patch(
                "torch_dataloader_utils.dataset.iceberg._resolve_files",
                return_value=(files, False, {}),
            ) as mock_resolve:
                with patch("torch_dataloader_utils.dataset.iceberg._require_pyiceberg"):
                    with patch(
                        "torch_dataloader_utils.dataset.iceberg._try_to_iceberg"
                    ) as mock_translate:
                        from torch_dataloader_utils.dataset.iceberg import IcebergDataset

                        IcebergDataset(
                            table="db.tbl",
                            catalog_config={},
                            num_workers=1,
                            filters=pc.field("row_id") >= 100,
                            scan_filter=explicit,
                        )

                mock_translate.assert_not_called()
                _, call_kwargs = mock_resolve.call_args
                passed = call_kwargs.get("scan_filter") or mock_resolve.call_args[0][3]
                assert passed is explicit

    def test_no_filters_no_scan_filter(self):
        """With neither filters nor scan_filter, _resolve_files gets scan_filter=None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self._make_files(tmpdir)
            with patch(
                "torch_dataloader_utils.dataset.iceberg._resolve_files",
                return_value=(files, False, {}),
            ) as mock_resolve:
                with patch("torch_dataloader_utils.dataset.iceberg._require_pyiceberg"):
                    from torch_dataloader_utils.dataset.iceberg import IcebergDataset

                    IcebergDataset(table="db.tbl", catalog_config={}, num_workers=1)

                _, call_kwargs = mock_resolve.call_args
                passed = call_kwargs.get("scan_filter") or mock_resolve.call_args[0][3]
                assert passed is None


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

            results = list(
                _read_task_with_deletes(
                    data_file_path=self._PATH,
                    table_identifier="db.table",
                    catalog_config={"name": "test"},
                    snapshot_id=None,
                    columns=None,
                    filters=None,
                    batch_size=3,
                )
            )

        assert len(results) == 2
        assert len(results[0]) == 3
        assert len(results[1]) == 2

    def test_task_not_found_yields_nothing(self):
        """If no task matches data_file_path, nothing is yielded."""
        batch = pa.record_batch({"x": pa.array([1], pa.int32())})
        mods, _ = self._build_mocks(batch, include_task=False)

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _read_task_with_deletes

            results = list(
                _read_task_with_deletes(
                    data_file_path=self._PATH,
                    table_identifier="db.table",
                    catalog_config={"name": "test"},
                    snapshot_id=None,
                    columns=None,
                    filters=None,
                    batch_size=100,
                )
            )

        assert results == []

    def test_filter_applied_to_batch(self):
        """Rows not matching filter are excluded from output."""
        batch = pa.record_batch({"x": pa.array([1, 2, 3, 4, 5], pa.int32())})
        mods, _ = self._build_mocks(batch)

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _read_task_with_deletes

            results = list(
                _read_task_with_deletes(
                    data_file_path=self._PATH,
                    table_identifier="db.table",
                    catalog_config={"name": "test"},
                    snapshot_id=None,
                    columns=None,
                    filters=pc.field("x") > 3,
                    batch_size=100,
                )
            )

        assert len(results) == 1
        assert results[0]["x"].to_pylist() == [4, 5]

    def test_filter_eliminates_all_rows_yields_nothing(self):
        """Filter that removes every row → nothing yielded."""
        batch = pa.record_batch({"x": pa.array([1, 2, 3], pa.int32())})
        mods, _ = self._build_mocks(batch)

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _read_task_with_deletes

            results = list(
                _read_task_with_deletes(
                    data_file_path=self._PATH,
                    table_identifier="db.table",
                    catalog_config={"name": "test"},
                    snapshot_id=None,
                    columns=None,
                    filters=pc.field("x") > 100,
                    batch_size=100,
                )
            )

        assert results == []

    def test_columns_selection_calls_schema_select(self):
        """When columns is given, projected_schema.select(columns) is called."""
        batch = pa.record_batch(
            {
                "x": pa.array([1, 2], pa.int32()),
                "y": pa.array([3, 4], pa.int32()),
            }
        )
        mods, mock_table = self._build_mocks(batch)

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _read_task_with_deletes

            list(
                _read_task_with_deletes(
                    data_file_path=self._PATH,
                    table_identifier="db.table",
                    catalog_config={"name": "test"},
                    snapshot_id=None,
                    columns=["x"],
                    filters=None,
                    batch_size=100,
                )
            )

        mock_table.schema.return_value.select.assert_called_once_with(["x"])

    def test_three_part_identifier_loads_second_and_third_parts(self):
        """3-part identifier 'ns.db.tbl' loads table as 'db.tbl' (skips namespace)."""
        batch = pa.record_batch({"x": pa.array([1], pa.int32())})
        mods, _ = self._build_mocks(batch)

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _read_task_with_deletes

            list(
                _read_task_with_deletes(
                    data_file_path=self._PATH,
                    table_identifier="namespace.db.tbl",  # 3 parts
                    catalog_config={"name": "test"},
                    snapshot_id=None,
                    columns=None,
                    filters=None,
                    batch_size=100,
                )
            )

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

    def test_scan_filter_forwarded_through_create_dataloader(self):
        """create_dataloader forwards scan_filter to _resolve_files."""
        mock_filter = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            info = _write_parquet(tmpdir, [1, 2, 3])
            with patch(
                "torch_dataloader_utils.dataset.iceberg._resolve_files",
                return_value=([info], False, {}),
            ) as mock_resolve:
                with patch("torch_dataloader_utils.dataset.iceberg._require_pyiceberg"):
                    from torch_dataloader_utils.dataset.iceberg import IcebergDataset

                    IcebergDataset.create_dataloader(
                        table="db.table",
                        catalog_config={"type": "rest"},
                        num_workers=1,
                        scan_filter=mock_filter,
                    )

        _, call_kwargs = mock_resolve.call_args
        assert (
            call_kwargs.get("scan_filter") is mock_filter
            or mock_resolve.call_args[0][3] is mock_filter
        )


# ---------------------------------------------------------------------------
# _resolve_files — exercises lines 65-124 via mocked pyiceberg.catalog
# ---------------------------------------------------------------------------


class TestResolveFiles:
    """Execute _resolve_files with a mocked pyiceberg.catalog to cover lines 65-124."""

    def _build_pyiceberg_catalog_mock(
        self,
        paths: list[str],
        has_deletes: bool = False,
        snapshot_id: int = 42,
        with_partition: bool = False,
    ):
        """Return (sys.modules patch dict, fake_catalog, fake_table)."""
        fake_tasks = []
        delete_stub = MagicMock()
        delete_stub.file_path = "s3://bucket/pos-deletes.avro"

        for path in paths:
            data_file = MagicMock()
            data_file.file_path = path
            data_file.file_size_in_bytes = 2048
            data_file.record_count = 100
            data_file.partition = {"dt": "2024-01-01"} if with_partition else None
            task = MagicMock()
            task.file = data_file
            # Non-empty list is truthy; empty list is falsy
            task.delete_files = [delete_stub] if has_deletes else []
            fake_tasks.append(task)

        fake_snap = MagicMock()
        fake_snap.snapshot_id = snapshot_id

        fake_table = MagicMock()
        fake_table.scan.return_value.plan_files.return_value = fake_tasks
        fake_table.current_snapshot.return_value = fake_snap

        fake_catalog = MagicMock()
        fake_catalog.load_table.return_value = fake_table

        fake_catalog_mod = MagicMock()
        fake_catalog_mod.load_catalog.return_value = fake_catalog

        return {"pyiceberg.catalog": fake_catalog_mod}, fake_catalog, fake_table

    # --- basic returns ---

    def test_returns_files_no_deletes(self):
        paths = ["s3://bucket/a.parquet", "s3://bucket/b.parquet"]
        mods, _, _ = self._build_pyiceberg_catalog_mock(paths)

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _resolve_files

            files, has_deletes, delete_paths = _resolve_files("db.tbl", {"name": "test"}, None)

        assert len(files) == 2
        assert has_deletes is False
        assert delete_paths == {}
        assert files[0].path == paths[0]
        assert files[1].path == paths[1]

    def test_file_size_and_record_count_populated(self):
        mods, _, _ = self._build_pyiceberg_catalog_mock(["s3://bucket/a.parquet"])

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _resolve_files

            files, _, _ = _resolve_files("db.tbl", {"name": "test"}, None)

        assert files[0].file_size == 2048
        assert files[0].record_count == 100

    # --- partition handling ---

    def test_partition_stored_when_present(self):
        mods, _, _ = self._build_pyiceberg_catalog_mock(
            ["s3://bucket/a.parquet"], with_partition=True
        )

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _resolve_files

            files, _, _ = _resolve_files("db.tbl", {"name": "test"}, None)

        assert files[0].partition == {"dt": "2024-01-01"}

    def test_no_partition_stored_as_none(self):
        mods, _, _ = self._build_pyiceberg_catalog_mock(
            ["s3://bucket/a.parquet"], with_partition=False
        )

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _resolve_files

            files, _, _ = _resolve_files("db.tbl", {"name": "test"}, None)

        assert files[0].partition is None

    # --- delete file detection ---

    def test_has_deletes_true_when_delete_files_present(self):
        mods, _, _ = self._build_pyiceberg_catalog_mock(["s3://bucket/a.parquet"], has_deletes=True)

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _resolve_files

            files, has_deletes, delete_paths = _resolve_files("db.tbl", {"name": "test"}, None)

        assert has_deletes is True
        assert "s3://bucket/a.parquet" in delete_paths
        assert "s3://bucket/pos-deletes.avro" in delete_paths["s3://bucket/a.parquet"]

    def test_has_deletes_false_no_delete_files(self):
        mods, _, _ = self._build_pyiceberg_catalog_mock(
            ["s3://bucket/a.parquet"], has_deletes=False
        )

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _resolve_files

            files, has_deletes, delete_paths = _resolve_files("db.tbl", {"name": "test"}, None)

        assert has_deletes is False
        assert delete_paths == {}

    # --- snapshot_id branches ---

    def test_current_snapshot_used_when_snapshot_id_is_none(self):
        mods, _, fake_table = self._build_pyiceberg_catalog_mock(
            ["s3://bucket/a.parquet"], snapshot_id=77
        )

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _resolve_files

            files, _, _ = _resolve_files("db.tbl", {"name": "test"}, snapshot_id=None)

        fake_table.current_snapshot.assert_called()
        assert files[0].snapshot_id == 77

    def test_explicit_snapshot_id_used_directly(self):
        mods, _, fake_table = self._build_pyiceberg_catalog_mock(
            ["s3://bucket/a.parquet"], snapshot_id=99
        )

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _resolve_files

            files, _, _ = _resolve_files("db.tbl", {"name": "test"}, snapshot_id=99)

        # When snapshot_id is provided it's passed directly to scan
        fake_table.scan.assert_called_with(snapshot_id=99)
        # snapshot_id stored on each file comes from the argument, not current_snapshot
        assert files[0].snapshot_id == 99

    # --- table identifier branches ---

    def test_two_part_identifier_loads_as_is(self):
        mods, fake_catalog, _ = self._build_pyiceberg_catalog_mock(["s3://bucket/a.parquet"])

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _resolve_files

            _resolve_files("db.tbl", {"name": "test"}, None)

        fake_catalog.load_table.assert_called_once_with("db.tbl")

    def test_three_part_identifier_loads_second_and_third_parts(self):
        mods, fake_catalog, _ = self._build_pyiceberg_catalog_mock(["s3://bucket/a.parquet"])

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _resolve_files

            _resolve_files("namespace.db.tbl", {"name": "test"}, None)

        fake_catalog.load_table.assert_called_once_with("db.tbl")

    # --- catalog_config forwarded to load_catalog ---

    def test_catalog_config_forwarded_to_load_catalog(self):
        mods, _, _ = self._build_pyiceberg_catalog_mock(["s3://bucket/a.parquet"])
        cfg = {"name": "my_catalog", "uri": "https://catalog.example.com"}

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _resolve_files

            _resolve_files("db.tbl", cfg, None)

        mods["pyiceberg.catalog"].load_catalog.assert_called_once_with(**cfg)

    # --- scan_filter forwarding ---

    def test_scan_filter_forwarded_to_table_scan(self):
        """scan_filter is passed as row_filter= to table.scan()."""
        mods, _, fake_table = self._build_pyiceberg_catalog_mock(["s3://bucket/a.parquet"])
        mock_filter = MagicMock()

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _resolve_files

            _resolve_files("db.tbl", {"name": "test"}, None, scan_filter=mock_filter)

        fake_table.scan.assert_called_with(snapshot_id=None, row_filter=mock_filter)

    def test_scan_filter_none_does_not_add_row_filter(self):
        """When scan_filter=None, table.scan is called without row_filter."""
        mods, _, fake_table = self._build_pyiceberg_catalog_mock(["s3://bucket/a.parquet"])

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _resolve_files

            _resolve_files("db.tbl", {"name": "test"}, None, scan_filter=None)

        fake_table.scan.assert_called_with(snapshot_id=None)

    def test_scan_filter_with_snapshot_id(self):
        """scan_filter and snapshot_id are both forwarded correctly."""
        mods, _, fake_table = self._build_pyiceberg_catalog_mock(
            ["s3://bucket/a.parquet"], snapshot_id=55
        )
        mock_filter = MagicMock()

        with patch.dict(sys.modules, mods):
            from torch_dataloader_utils.dataset.iceberg import _resolve_files

            _resolve_files("db.tbl", {"name": "test"}, snapshot_id=55, scan_filter=mock_filter)

        fake_table.scan.assert_called_with(snapshot_id=55, row_filter=mock_filter)
