from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class DataFileInfo:
    path: str
    file_size: int | None = None
    record_count: int | None = None


@dataclass
class IcebergDataFileInfo(DataFileInfo):
    partition: dict[str, str] | None = None
    snapshot_id: int | None = None


@dataclass
class RowRange:
    """Defines a row slice within a file. Used for sub-file splitting."""

    offset: int  # start row (inclusive)
    length: int  # number of rows to read


@dataclass
class Split:
    """A single unit of work: one file paired with an optional row range.

    row_range=None means read the entire file.
    row_range=RowRange(offset, length) reads a sub-file slice.
    """

    file: DataFileInfo
    row_range: RowRange | None = None


@dataclass
class Shard:
    """A worker's assignment: a collection of splits to process sequentially."""

    id: int
    splits: list[Split] = field(default_factory=list)
    row_count: int | None = None
    size_bytes: int | None = None


class SplitStrategy(Protocol):
    def generate(
        self,
        files: list[DataFileInfo],
        num_workers: int,
        epoch: int = 0,
    ) -> list[Shard]: ...
