"""
Script to generate test fixtures.
Run once: uv run python tests/fixtures/generate.py
"""

import json
import pathlib

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.orc as orc
import pyarrow.parquet as pq

FIXTURES = pathlib.Path(__file__).parent

# Shared sample data
TABLE = pa.table(
    {
        "feature_a": pa.array([1.0, 2.0, 3.0, 4.0, 5.0], type=pa.float32()),
        "feature_b": pa.array([10, 20, 30, 40, 50], type=pa.int64()),
        "label": pa.array([0, 1, 0, 1, 0], type=pa.int32()),
    }
)

# Parquet
pq.write_table(TABLE, FIXTURES / "sample.parquet")

# ORC
orc.write_table(TABLE, str(FIXTURES / "sample.orc"))

# CSV
pa_csv.write_csv(TABLE, FIXTURES / "sample.csv")

# JSONL
with open(FIXTURES / "sample.jsonl", "w") as f:
    for row in TABLE.to_pylist():
        f.write(json.dumps(row) + "\n")

print(f"Fixtures written to {FIXTURES}")
for p in sorted(FIXTURES.glob("sample.*")):
    print(f"  {p.name} ({p.stat().st_size} bytes)")
