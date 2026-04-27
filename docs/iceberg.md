# IcebergDataset

`IcebergDataset` reads Apache Iceberg tables via `pyiceberg`, resolves them to data files, and streams batches using the same worker split mechanism as `StructuredDataset`.

## Install

```bash
pip install torch-dataloader-utils[iceberg]
```

## create_dataloader()

```python
from torch_dataloader_utils import IcebergDataset
import pyarrow.compute as pc

loader, dataset = IcebergDataset.create_dataloader(
    table="my_db.my_table",               # or "namespace.db.table"
    catalog_config={
        "type": "rest",                   # rest | glue | hive | jdbc
        "uri": "https://catalog.example.com",
        "credential": "token:abc123",
    },
    num_workers=4,
    batch_size=1024,
    columns=["feature_a", "feature_b", "label"],
    filters=pc.field("region_id") >= 5,   # auto-prunes files AND filters rows
    snapshot_id=None,                     # None = current; set for time travel
    shuffle=True,
    split_bytes="64MiB",
    output_format="torch",
)
```

## Scan-Filter Auto-Derivation

Passing `filters` is all you need. The library auto-translates common pyarrow expressions into a native pyiceberg expression and pushes it into `table.scan(row_filter=...)` at `plan_files()` time — pruning entire partitions and files **before** splits are generated.

Supported auto-translation: `>=`, `>`, `<=`, `<`, `==`, `!=` with integer, float, or string literals; `&` (AND), `|` (OR), arbitrarily nested.

```
INFO  Auto-derived scan_filter:  pc.Expression (region_id >= 5)
                               → pyiceberg GreaterThanOrEqual(...)
INFO  Iceberg scan complete: files=4  (down from 6 without filter)
```

### Explicit Two-Layer Control

For cases where file-pruning and row-filtering need different predicates:

```python
from pyiceberg.expressions import GreaterThan

loader, dataset = IcebergDataset.create_dataloader(
    ...
    scan_filter=GreaterThan("partition_dt", 20240101),  # file/partition pruning
    filters=pc.field("score") > 0.9,                   # row-level filter in workers
)
```

When `scan_filter` is provided explicitly, auto-derivation is skipped.

## Parameters

Most `StructuredDataset` parameters apply — except `path`, `format`, `partitioning`, and `storage_options`. Storage credentials for Iceberg are handled via `catalog_config`, not `storage_options`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `table` | `str` | — | Fully qualified table identifier (`"db.table"` or `"ns.db.table"`) |
| `catalog_config` | `dict` | — | Forwarded to `pyiceberg.catalog.load_catalog()`. Pass credentials here, not in `storage_options` |
| `snapshot_id` | `int \| None` | `None` | Pin a snapshot for time travel. `None` = current snapshot |
| `scan_filter` | `BooleanExpression \| None` | `None` | Native pyiceberg expression for file/partition pruning. Auto-derived from `filters` when not set |

## Catalog Config Examples

=== "REST"
    ```python
    catalog_config = {
        "type": "rest",
        "uri": "https://catalog.example.com",
        "credential": "token:abc123",
    }
    ```

=== "AWS Glue"
    ```python
    catalog_config = {
        "type": "glue",
        "aws_access_key_id": "...",
        "aws_secret_access_key": "...",
        "region_name": "us-east-1",
    }
    ```

=== "Local SQLite (testing)"
    ```python
    catalog_config = {
        "name": "local",
        "uri": "sqlite:////tmp/catalog.db",
        "warehouse": "file:///tmp/warehouse",
    }
    ```

## Delete Files

When the Iceberg table contains **position delete files** (written by `DELETE` or `MERGE INTO`), `IcebergDataset` automatically switches to `pyiceberg.io.pyarrow.ArrowScan` per file — deleted rows are never returned.

!!! warning "Limitations with delete files"
    - **Equality deletes** are not supported by pyiceberg and will raise `NotImplementedError`. Compact the table first.
    - **Sub-file splitting is disabled** when delete files are present — position delete offsets reference absolute row positions in the original file.

## Time Travel

```python
loader, _ = IcebergDataset.create_dataloader(
    table="my_db.my_table",
    catalog_config=catalog_config,
    snapshot_id=8271638172635,    # pin to a historical snapshot
)
```
