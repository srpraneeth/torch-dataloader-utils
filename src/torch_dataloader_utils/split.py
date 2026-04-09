from dataclasses import dataclass, field


@dataclass
class Split:
    id: int
    files: list[str]
    row_count: int | None = None
    size_bytes: int | None = None
