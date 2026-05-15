"""Unit tests for the record-level shuffle buffer."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from torch_dataloader_utils.dataset.base import _shuffle_buffer_iter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source(n_rows: int, batch_size: int = 100) -> list[pa.RecordBatch]:
    """Build a list of RecordBatches with a single 'id' column 0..n_rows-1."""
    batches = []
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        batches.append(pa.record_batch({"id": list(range(start, end))}))
    return batches


def _collect_ids(batches: list[pa.RecordBatch]) -> list[int]:
    return [v for b in batches for v in b.column("id").to_pylist()]


def _run_buffer(source_batches, buffer_size, batch_size=32, seed=0) -> list[pa.RecordBatch]:
    rng = np.random.default_rng(seed)
    return list(_shuffle_buffer_iter(iter(source_batches), buffer_size, batch_size, rng))


# ---------------------------------------------------------------------------
# 1. shuffle_buffer_size=None → bypass (not the buffer itself, handled by __iter__)
# ---------------------------------------------------------------------------


def test_buffer_preserves_all_rows_small():
    """All rows present, no duplicates."""
    source = _make_source(500, batch_size=50)
    out = _run_buffer(source, buffer_size=200, batch_size=64)
    ids = _collect_ids(out)
    assert sorted(ids) == list(range(500))


def test_buffer_preserves_all_rows_large():
    """Larger dataset — still no rows lost or duplicated."""
    source = _make_source(10_000, batch_size=1000)
    out = _run_buffer(source, buffer_size=3000, batch_size=256)
    ids = _collect_ids(out)
    assert sorted(ids) == list(range(10_000))


# ---------------------------------------------------------------------------
# 2. Shuffle actually reorders rows
# ---------------------------------------------------------------------------


def test_buffer_reorders_rows():
    """Output row order differs from input."""
    source = _make_source(1000, batch_size=100)
    in_order = list(range(1000))
    out = _run_buffer(source, buffer_size=500, batch_size=128, seed=7)
    out_order = _collect_ids(out)
    assert sorted(out_order) == in_order  # all present
    assert out_order != in_order  # shuffled


# ---------------------------------------------------------------------------
# 3. Determinism: same seed + epoch formula → same order
# ---------------------------------------------------------------------------


def test_buffer_is_deterministic():
    source = _make_source(500, batch_size=50)
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    out_a = _collect_ids(list(_shuffle_buffer_iter(iter(source), 200, 64, rng_a)))
    out_b = _collect_ids(list(_shuffle_buffer_iter(iter(source), 200, 64, rng_b)))
    assert out_a == out_b


def test_different_seeds_produce_different_order():
    source = _make_source(500, batch_size=50)
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)
    out_a = _collect_ids(list(_shuffle_buffer_iter(iter(source), 200, 64, rng_a)))
    out_b = _collect_ids(list(_shuffle_buffer_iter(iter(source), 200, 64, rng_b)))
    assert sorted(out_a) == sorted(out_b)  # same rows
    assert out_a != out_b  # different order


# ---------------------------------------------------------------------------
# 4. Buffer smaller than dataset
# ---------------------------------------------------------------------------


def test_buffer_smaller_than_dataset():
    """buffer_size << dataset_size still produces all rows."""
    source = _make_source(2000, batch_size=200)
    out = _run_buffer(source, buffer_size=100, batch_size=32)
    ids = _collect_ids(out)
    assert sorted(ids) == list(range(2000))


# ---------------------------------------------------------------------------
# 5. Buffer larger than dataset — drains entirely in final step
# ---------------------------------------------------------------------------


def test_buffer_larger_than_dataset():
    """When buffer_size > dataset rows, all rows land in the final drain."""
    source = _make_source(50, batch_size=10)
    out = _run_buffer(source, buffer_size=10_000, batch_size=32)
    ids = _collect_ids(out)
    assert sorted(ids) == list(range(50))


# ---------------------------------------------------------------------------
# 6. Last batch smaller than batch_size (tail handling)
# ---------------------------------------------------------------------------


def test_last_batch_can_be_smaller():
    """Last output batch may have fewer than batch_size rows."""
    source = _make_source(105, batch_size=50)
    out = _run_buffer(source, buffer_size=50, batch_size=32)
    ids = _collect_ids(out)
    assert sorted(ids) == list(range(105))
    # At least one batch should be smaller than 32 (the tail)
    assert any(len(b) < 32 for b in out)


# ---------------------------------------------------------------------------
# 7. Empty source → yields nothing
# ---------------------------------------------------------------------------


def test_empty_source_yields_nothing():
    out = _run_buffer([], buffer_size=1000, batch_size=32)
    assert out == []


# ---------------------------------------------------------------------------
# 8. Single-row source
# ---------------------------------------------------------------------------


def test_single_row_source():
    source = [pa.record_batch({"id": [99]})]
    out = _run_buffer(source, buffer_size=1000, batch_size=32)
    ids = _collect_ids(out)
    assert ids == [99]


# ---------------------------------------------------------------------------
# 9. batch_size larger than buffer_size
# ---------------------------------------------------------------------------


def test_batch_size_larger_than_buffer():
    """Works correctly when batch_size > buffer_size."""
    source = _make_source(300, batch_size=30)
    out = _run_buffer(source, buffer_size=10, batch_size=64)
    ids = _collect_ids(out)
    assert sorted(ids) == list(range(300))


# ---------------------------------------------------------------------------
# 10. Integration with BaseDataset.__iter__ (via StructuredDataset)
# ---------------------------------------------------------------------------


def test_structured_dataset_with_shuffle_buffer(tmp_path):
    """End-to-end: StructuredDataset with shuffle_buffer_size yields all rows shuffled."""
    import pyarrow.parquet as pq

    from torch_dataloader_utils.dataset.structured import StructuredDataset
    from torch_dataloader_utils.splits.core import DataFileInfo

    n_rows = 1000
    f = tmp_path / "data.parquet"
    pq.write_table(pa.table({"x": list(range(n_rows))}), f)

    files = [DataFileInfo(path=str(f), file_size=f.stat().st_size, record_count=n_rows)]
    ds = StructuredDataset(
        files=files,
        format="parquet",
        batch_size=128,
        num_workers=0,
        output_format="arrow",
        shuffle_buffer_size=200,
        collate_fn=lambda x: x,
    )

    out_batches = list(ds)
    all_x = [v for b in out_batches for v in b.column("x").to_pylist()]
    assert sorted(all_x) == list(range(n_rows))
    assert all_x != list(range(n_rows)), "rows should be shuffled"


def test_structured_dataset_no_shuffle_buffer_default(tmp_path):
    """Without shuffle_buffer_size, output is in file order (existing behaviour)."""
    import pyarrow.parquet as pq

    from torch_dataloader_utils.dataset.structured import StructuredDataset
    from torch_dataloader_utils.splits.core import DataFileInfo

    n_rows = 500
    f = tmp_path / "data.parquet"
    pq.write_table(pa.table({"x": list(range(n_rows))}), f)

    files = [DataFileInfo(path=str(f), file_size=f.stat().st_size, record_count=n_rows)]
    ds = StructuredDataset(
        files=files,
        format="parquet",
        batch_size=128,
        num_workers=0,
        output_format="arrow",
        collate_fn=lambda x: x,
    )

    out_batches = list(ds)
    all_x = [v for b in out_batches for v in b.column("x").to_pylist()]
    assert all_x == list(range(n_rows))
