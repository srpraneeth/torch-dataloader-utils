import pyarrow as pa
import numpy as np
import pytest

from torch_dataloader_utils.dataset.output import convert_batch


def _batch(include_string: bool = False) -> pa.RecordBatch:
    data = {
        "feature_a": pa.array([1.0, 2.0, 3.0], type=pa.float32()),
        "label": pa.array([0, 1, 0], type=pa.int32()),
    }
    if include_string:
        data["category"] = pa.array(["cat", "dog", "cat"])
    return pa.record_batch(data)


# ---------------------------------------------------------------------------
# output_format="arrow"
# ---------------------------------------------------------------------------

def test_arrow_returns_record_batch():
    batch = _batch()
    result = convert_batch(batch, "arrow")
    assert isinstance(result, pa.RecordBatch)
    assert result is batch


# ---------------------------------------------------------------------------
# output_format="dict"
# ---------------------------------------------------------------------------

def test_dict_returns_lists():
    batch = _batch()
    result = convert_batch(batch, "dict")
    assert isinstance(result, dict)
    assert isinstance(result["feature_a"], list)
    assert result["feature_a"] == [1.0, 2.0, 3.0]
    assert result["label"] == [0, 1, 0]


def test_dict_handles_string_columns():
    batch = _batch(include_string=True)
    result = convert_batch(batch, "dict")
    assert result["category"] == ["cat", "dog", "cat"]


# ---------------------------------------------------------------------------
# output_format="numpy"
# ---------------------------------------------------------------------------

def test_numpy_numeric_columns_are_ndarray():
    batch = _batch()
    result = convert_batch(batch, "numpy")
    assert isinstance(result["feature_a"], np.ndarray)
    assert isinstance(result["label"], np.ndarray)


def test_numpy_string_columns_pass_through_as_list():
    batch = _batch(include_string=True)
    result = convert_batch(batch, "numpy")
    assert isinstance(result["category"], list)
    assert result["category"] == ["cat", "dog", "cat"]


def test_numpy_values_correct():
    batch = _batch()
    result = convert_batch(batch, "numpy")
    np.testing.assert_array_equal(result["label"], [0, 1, 0])


# ---------------------------------------------------------------------------
# output_format="torch"
# ---------------------------------------------------------------------------

def test_torch_numeric_columns_are_tensors():
    import torch
    batch = _batch()
    result = convert_batch(batch, "torch")
    assert isinstance(result["feature_a"], torch.Tensor)
    assert isinstance(result["label"], torch.Tensor)


def test_torch_string_columns_pass_through_as_list():
    batch = _batch(include_string=True)
    result = convert_batch(batch, "torch")
    assert isinstance(result["category"], list)
    assert result["category"] == ["cat", "dog", "cat"]


def test_torch_values_correct():
    import torch
    batch = _batch()
    result = convert_batch(batch, "torch")
    assert result["label"].tolist() == [0, 1, 0]


def test_torch_dtype_preserved():
    import torch
    batch = _batch()
    result = convert_batch(batch, "torch")
    assert result["feature_a"].dtype == torch.float32
    assert result["label"].dtype == torch.int32


# ---------------------------------------------------------------------------
# Unsupported format
# ---------------------------------------------------------------------------

def test_unsupported_output_format_raises():
    batch = _batch()
    with pytest.raises(ValueError, match="xml"):
        convert_batch(batch, "xml")
