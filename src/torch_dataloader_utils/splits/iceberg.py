"""
Iceberg-aware split strategy.

Not yet implemented — placeholder for V1 Iceberg support.
Will use IcebergDataFileInfo.record_count for equi-row splits
and IcebergDataFileInfo.partition for partition-aware assignment.
"""
from torch_dataloader_utils.splits.core import DataFileInfo, IcebergDataFileInfo, Split  # noqa: F401
