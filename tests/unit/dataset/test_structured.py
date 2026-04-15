import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from torch_dataloader_utils.dataset.structured import StructuredDataset
from torch_dataloader_utils.splits.core import DataFileInfo

FIXTURES = __import__("pathlib").Path(__file__).parent.parent.parent / "fixtures"


def _make_files(*paths: str) -> list[DataFileInfo]:
    import os
    return [DataFileInfo(path=p, file_size=os.path.getsize(p)) for p in paths]


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

def test_invalid_format_raises():
    files = _make_files(str(FIXTURES / "sample.parquet"))
    with pytest.raises(ValueError, match="avro"):
        StructuredDataset(files=files, format="avro")


def test_invalid_output_format_raises():
    files = _make_files(str(FIXTURES / "sample.parquet"))
    with pytest.raises(ValueError, match="xml"):
        StructuredDataset(files=files, format="parquet", output_format="xml")


def test_arrow_without_collate_fn_raises():
    files = _make_files(str(FIXTURES / "sample.parquet"))
    with pytest.raises(ValueError, match="collate_fn"):
        StructuredDataset(files=files, format="parquet", output_format="arrow")


def test_dict_without_collate_fn_raises():
    files = _make_files(str(FIXTURES / "sample.parquet"))
    with pytest.raises(ValueError, match="collate_fn"):
        StructuredDataset(files=files, format="parquet", output_format="dict")


def test_arrow_with_collate_fn_does_not_raise():
    files = _make_files(str(FIXTURES / "sample.parquet"))
    ds = StructuredDataset(
        files=files, format="parquet", output_format="arrow", collate_fn=lambda x: x
    )
    assert ds is not None


# ---------------------------------------------------------------------------
# Single-process iteration (num_workers=0 → worker_id=0)
# ---------------------------------------------------------------------------

def test_single_process_returns_all_rows():
    import torch
    files = _make_files(str(FIXTURES / "sample.parquet"))
    ds = StructuredDataset(files=files, format="parquet", num_workers=1, batch_size=10)
    batches = list(ds)
    total = sum(b["label"].shape[0] for b in batches)
    assert total == 5


def test_single_process_output_is_torch():
    import torch
    files = _make_files(str(FIXTURES / "sample.parquet"))
    ds = StructuredDataset(files=files, format="parquet", num_workers=1)
    batch = next(iter(ds))
    assert isinstance(batch["feature_a"], torch.Tensor)
    assert isinstance(batch["label"], torch.Tensor)


def test_single_process_output_numpy():
    import numpy as np
    files = _make_files(str(FIXTURES / "sample.parquet"))
    ds = StructuredDataset(
        files=files, format="parquet", num_workers=1, output_format="numpy"
    )
    batch = next(iter(ds))
    assert isinstance(batch["feature_a"], np.ndarray)


def test_single_process_output_arrow_with_collate():
    files = _make_files(str(FIXTURES / "sample.parquet"))
    ds = StructuredDataset(
        files=files, format="parquet", num_workers=1,
        output_format="arrow", collate_fn=lambda x: x,
    )
    batch = next(iter(ds))
    assert isinstance(batch, pa.RecordBatch)


def test_single_process_output_dict_with_collate():
    files = _make_files(str(FIXTURES / "sample.parquet"))
    ds = StructuredDataset(
        files=files, format="parquet", num_workers=1,
        output_format="dict", collate_fn=lambda x: x,
    )
    batch = next(iter(ds))
    assert isinstance(batch, dict)
    assert isinstance(batch["label"], list)


# ---------------------------------------------------------------------------
# Column projection end-to-end
# ---------------------------------------------------------------------------

def test_column_projection():
    files = _make_files(str(FIXTURES / "sample.parquet"))
    ds = StructuredDataset(
        files=files, format="parquet", num_workers=1, columns=["feature_a", "label"]
    )
    batch = next(iter(ds))
    assert set(batch.keys()) == {"feature_a", "label"}
    assert "feature_b" not in batch


# ---------------------------------------------------------------------------
# Predicate pushdown end-to-end
# ---------------------------------------------------------------------------

def test_predicate_pushdown():
    files = _make_files(str(FIXTURES / "sample.parquet"))
    ds = StructuredDataset(
        files=files, format="parquet", num_workers=1,
        filters=pc.field("feature_b") > 30,
    )
    batches = list(ds)
    total = sum(b["label"].shape[0] for b in batches)
    assert total == 2  # feature_b values [10,20,30,40,50] → only 40,50 pass


# ---------------------------------------------------------------------------
# Shuffle
# ---------------------------------------------------------------------------

def test_no_shuffle_splits_cached(tmp_path):
    t = pa.table({"val": pa.array([1, 2], type=pa.int32())})
    for i in range(2):
        pq.write_table(t, tmp_path / f"f{i}.parquet")

    files = _make_files(*[str(tmp_path / f"f{i}.parquet") for i in range(2)])
    ds = StructuredDataset(files=files, format="parquet", num_workers=1, shuffle=False)

    splits_first = ds._splits
    list(ds)
    # splits should be the same object — not regenerated
    assert ds._splits is splits_first


def test_shuffle_regenerates_splits(tmp_path):
    t = pa.table({"val": pa.array([1, 2], type=pa.int32())})
    for i in range(8):
        pq.write_table(t, tmp_path / f"f{i}.parquet")

    files = _make_files(*[str(tmp_path / f"f{i}.parquet") for i in range(8)])
    ds = StructuredDataset(
        files=files, format="parquet", num_workers=1, shuffle=True, shuffle_seed=42
    )

    # set_epoch regenerates splits — collect orders across several epochs
    orders = []
    for epoch in range(5):
        ds.set_epoch(epoch)
        orders.append([fs.file.path for fs in ds._splits[0].file_splits])

    assert not all(o == orders[0] for o in orders), "All epochs produced identical order"


# ---------------------------------------------------------------------------
# Multi-worker assignment — mocked get_worker_info
# ---------------------------------------------------------------------------

def _make_multi_file_dataset(tmp_path, num_files: int, rows_per_file: int, num_workers: int):
    """Write num_files parquet files and return a StructuredDataset with num_workers splits."""
    for i in range(num_files):
        t = pa.table({
            "row_id": pa.array(
                list(range(i * rows_per_file, (i + 1) * rows_per_file)), type=pa.int32()
            )
        })
        pq.write_table(t, tmp_path / f"f{i}.parquet")

    files = _make_files(*[str(tmp_path / f"f{i}.parquet") for i in range(num_files)])
    return StructuredDataset(files=files, format="parquet", num_workers=num_workers)


def test_worker_0_reads_its_split(tmp_path):
    from unittest.mock import MagicMock, patch
    ds = _make_multi_file_dataset(tmp_path, num_files=4, rows_per_file=10, num_workers=4)

    mock_info = MagicMock()
    mock_info.id = 0
    with patch("torch.utils.data.get_worker_info", return_value=mock_info):
        rows = [b["row_id"].tolist() for b in ds]
    # worker 0 gets split 0 — exactly 1 file, 10 rows
    assert sum(len(r) for r in rows) == 10


def test_worker_1_reads_its_split(tmp_path):
    from unittest.mock import MagicMock, patch
    ds = _make_multi_file_dataset(tmp_path, num_files=4, rows_per_file=10, num_workers=4)

    mock_info = MagicMock()
    mock_info.id = 1
    with patch("torch.utils.data.get_worker_info", return_value=mock_info):
        rows = [b["row_id"].tolist() for b in ds]
    assert sum(len(r) for r in rows) == 10


def test_workers_read_disjoint_splits(tmp_path):
    from unittest.mock import MagicMock, patch
    ds = _make_multi_file_dataset(tmp_path, num_files=4, rows_per_file=10, num_workers=4)

    all_row_ids = []
    for worker_id in range(4):
        mock_info = MagicMock()
        mock_info.id = worker_id
        with patch("torch.utils.data.get_worker_info", return_value=mock_info):
            for batch in ds:
                all_row_ids.extend(batch["row_id"].tolist())

    # All 40 rows returned with no duplicates
    assert len(all_row_ids) == 40
    assert sorted(all_row_ids) == list(range(40))


def test_worker_beyond_split_count_yields_nothing(tmp_path):
    from unittest.mock import MagicMock, patch
    # 2 files, 2 workers — worker_id=5 has no split
    ds = _make_multi_file_dataset(tmp_path, num_files=2, rows_per_file=10, num_workers=2)

    mock_info = MagicMock()
    mock_info.id = 5
    with patch("torch.utils.data.get_worker_info", return_value=mock_info):
        batches = list(ds)
    assert batches == []


def test_more_files_than_workers_all_rows_covered(tmp_path):
    from unittest.mock import MagicMock, patch
    # 6 files, 2 workers — each worker reads 3 files
    ds = _make_multi_file_dataset(tmp_path, num_files=6, rows_per_file=10, num_workers=2)

    all_row_ids = []
    for worker_id in range(2):
        mock_info = MagicMock()
        mock_info.id = worker_id
        with patch("torch.utils.data.get_worker_info", return_value=mock_info):
            for batch in ds:
                all_row_ids.extend(batch["row_id"].tolist())

    assert len(all_row_ids) == 60
    assert sorted(all_row_ids) == list(range(60))


def test_imbalanced_files_no_rows_dropped(tmp_path):
    from unittest.mock import MagicMock, patch
    # 4 files with different row counts — SizeBalancedSplitStrategy kicks in
    row_counts = [40, 30, 20, 10]
    offset = 0
    paths = []
    for i, n in enumerate(row_counts):
        t = pa.table({
            "row_id": pa.array(list(range(offset, offset + n)), type=pa.int32())
        })
        p = tmp_path / f"f{i}.parquet"
        pq.write_table(t, p)
        paths.append(str(p))
        offset += n

    import os
    files = [DataFileInfo(path=p, file_size=os.path.getsize(p)) for p in paths]
    ds = StructuredDataset(files=files, format="parquet", num_workers=2)

    all_row_ids = []
    for worker_id in range(2):
        mock_info = MagicMock()
        mock_info.id = worker_id
        with patch("torch.utils.data.get_worker_info", return_value=mock_info):
            for batch in ds:
                all_row_ids.extend(batch["row_id"].tolist())

    assert len(all_row_ids) == 100
    assert sorted(all_row_ids) == list(range(100))


# ---------------------------------------------------------------------------
# num_workers=None auto-detect via create_dataloader
# ---------------------------------------------------------------------------

def test_create_dataloader_returns_dataloader():
    from torch.utils.data import DataLoader
    loader, _ = StructuredDataset.create_dataloader(
        path=str(FIXTURES),
        format="parquet",
        num_workers=0,
        batch_size=10,
    )
    assert isinstance(loader, DataLoader)


def test_create_dataloader_num_workers_none():
    import os
    from torch.utils.data import DataLoader
    loader, _ = StructuredDataset.create_dataloader(
        path=str(FIXTURES),
        format="parquet",
        num_workers=None,
        batch_size=10,
    )
    assert isinstance(loader, DataLoader)
    expected = max(1, (os.cpu_count() or 1) - 1)
    assert loader.num_workers == expected
