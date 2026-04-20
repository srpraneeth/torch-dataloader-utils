"""
Generate synthetic Parquet test data for e2e pipeline testing.

Usage:
    uv run python e2e/gen_data.py --out-dir /tmp/e2e_data --files 20 --rows 100000
    uv run python e2e/gen_data.py --out-dir /tmp/e2e_data --files 5 --rows 10000 --unequal
"""
import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq


def make_table(n_rows: int, row_id_offset: int, row_group_size: int) -> pa.Table:
    row_ids = list(range(row_id_offset, row_id_offset + n_rows))
    # 64 float features — realistic embedding input size
    columns = {"row_id": pa.array(row_ids, type=pa.int32())}
    for i in range(64):
        columns[f"feat_{i:02d}"] = pa.array(
            [float((r + i) % 100) / 100.0 for r in row_ids], type=pa.float32()
        )
    columns["label"] = pa.array([r % 10 for r in row_ids], type=pa.int32())
    return pa.table(columns, schema=_schema())


def _schema() -> pa.Schema:
    fields = [pa.field("row_id", pa.int32())]
    for i in range(64):
        fields.append(pa.field(f"feat_{i:02d}", pa.float32()))
    fields.append(pa.field("label", pa.int32()))
    return pa.schema(fields)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/tmp/e2e_data")
    parser.add_argument("--files", type=int, default=20, help="Number of Parquet files")
    parser.add_argument("--rows", type=int, default=100_000, help="Rows per file (base)")
    parser.add_argument(
        "--unequal", action="store_true",
        help="Make files very unequal in size (tests worker balance)"
    )
    parser.add_argument("--row-group-size", type=int, default=10_000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    total = 0
    for i in range(args.files):
        if args.unequal:
            # Geometric distribution: file 0 has 1x rows, file N-1 has ~10x rows
            multiplier = 2 ** (i % 4)
            n_rows = max(args.rows // 8, args.rows * multiplier // 8)
        else:
            n_rows = args.rows

        path = os.path.join(args.out_dir, f"part_{i:04d}.parquet")
        table = make_table(n_rows, row_id_offset=total, row_group_size=args.row_group_size)
        pq.write_table(table, path, row_group_size=args.row_group_size)
        print(f"  wrote {path}  rows={n_rows:,}  size={os.path.getsize(path):,} bytes")
        total += n_rows

    print(f"\nTotal rows: {total:,}  across {args.files} files  → {args.out_dir}")


if __name__ == "__main__":
    main()
