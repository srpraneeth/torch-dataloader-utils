import pyarrow as pa
import numpy as np


_NUMERIC_TYPES = (
    pa.int8(), pa.int16(), pa.int32(), pa.int64(),
    pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64(),
    pa.float16(), pa.float32(), pa.float64(),
    pa.bool_(),
)


def _is_numeric(field: pa.Field) -> bool:
    return field.type in _NUMERIC_TYPES


def convert_batch(batch: pa.RecordBatch, output_format: str) -> object:
    """Convert a pyarrow RecordBatch to the requested output format.

    Args:
        batch: The RecordBatch to convert.
        output_format: One of "torch", "numpy", "arrow", "dict".

    Returns:
        Converted batch in the requested format.
    """
    if output_format == "arrow":
        return batch

    if output_format == "dict":
        return {name: batch.column(name).to_pylist() for name in batch.schema.names}

    if output_format == "numpy":
        result = {}
        for name in batch.schema.names:
            col = batch.column(name)
            field = batch.schema.field(name)
            if _is_numeric(field):
                result[name] = col.to_numpy(zero_copy_only=False)
            else:
                result[name] = col.to_pylist()
        return result

    if output_format == "torch":
        import torch
        result = {}
        for name in batch.schema.names:
            col = batch.column(name)
            field = batch.schema.field(name)
            if _is_numeric(field):
                arr = col.to_numpy(zero_copy_only=False)
                result[name] = torch.from_numpy(arr.copy())
            else:
                result[name] = col.to_pylist()
        return result

    raise ValueError(
        f"Unsupported output_format {output_format!r}. "
        "Supported: torch, numpy, arrow, dict"
    )
