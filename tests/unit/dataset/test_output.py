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


# ---------------------------------------------------------------------------
# All numeric types produce tensors / ndarrays
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arrow_type,torch_dtype", [
    (pa.int8(),    "torch.int8"),
    (pa.int16(),   "torch.int16"),
    (pa.int32(),   "torch.int32"),
    (pa.int64(),   "torch.int64"),
    (pa.uint8(),   "torch.uint8"),
    (pa.float16(), "torch.float16"),
    (pa.float32(), "torch.float32"),
    (pa.float64(), "torch.float64"),
    (pa.bool_(),   "torch.bool"),
])
def test_torch_numeric_types(arrow_type, torch_dtype):
    import torch
    values = [True, False, True] if arrow_type == pa.bool_() else [1, 0, 1]
    arr = pa.array(values, type=arrow_type)
    batch = pa.record_batch({"col": arr})
    result = convert_batch(batch, "torch")
    assert isinstance(result["col"], torch.Tensor)
    assert str(result["col"].dtype) == torch_dtype


@pytest.mark.parametrize("arrow_type", [
    pa.int8(), pa.int16(), pa.int32(), pa.int64(),
    pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64(),
    pa.float16(), pa.float32(), pa.float64(),
    pa.bool_(),
])
def test_numpy_numeric_types(arrow_type):
    values = [True, False, True] if arrow_type == pa.bool_() else [1, 0, 1]
    arr = pa.array(values, type=arrow_type)
    batch = pa.record_batch({"col": arr})
    result = convert_batch(batch, "numpy")
    assert isinstance(result["col"], np.ndarray)
