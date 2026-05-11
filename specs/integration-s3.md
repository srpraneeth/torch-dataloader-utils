# Spec: S3 Integration Tests (moto)

## Core Principle

S3 integration tests validate the full pipeline against a mocked S3 backend.
They exercise the remote filesystem path in `discovery.py` and `reader.py` that
cannot be covered by local tests — specifically the `PyFileSystem` wrapping of
fsspec for remote backends.

```
moto S3 (in-process mock)
        ↓
discover_files("s3://bucket/...")
        ↓
read_split(split, storage_options={"endpoint_url": ...})
        ↓
StructuredDataset.create_dataloader(path="s3://...", storage_options=...)
        ↓
batches in requested output format
```

## Backend Coverage Note

GCS and Azure use the **same code path** as S3 — `fsspec.url_to_fs()` + `PyFileSystem(FSSpecHandler(fs))`. Testing S3 with moto covers this shared abstraction layer.

However, per-backend differences exist that S3 moto tests cannot catch:
- Auth flow differences (gcsfs, adlfs have their own credential chains)
- Path format edge cases (`gs://`, `az://`, `abfs://`)
- Subtle `stat()` response shape differences between backends

**V1:** S3 moto tests cover the shared code path. README documents this limitation.
**V2:** Real GCS (fake-gcs-server) and Azure (Azurite) CI tests via Docker Compose.

---

## Requirements

### Test Infrastructure `[v1]`
Tests SHALL use `moto` to mock the S3 backend — no real AWS credentials needed.
Tests SHALL create a fresh S3 bucket per test via a pytest fixture.
Tests SHALL upload test Parquet files to the mocked bucket before each test.
Tests SHALL pass `storage_options` with the moto endpoint to fsspec.
All tests SHALL be marked `@pytest.mark.integration`.

### Coverage Goals `[v1]`
Tests SHALL exercise `discover_files()` against a mocked S3 path.
Tests SHALL exercise `read_split()` reading Parquet from mocked S3.
Tests SHALL exercise `StructuredDataset.create_dataloader()` end-to-end against mocked S3.
Tests SHALL cover directory discovery, glob pattern, and single file paths on S3.
Tests SHALL cover the `PyFileSystem` wrapping path in `format/reader.py`.

---

## Scenarios

| Scenario | Input | Expected |
|----------|-------|----------|
| Directory discovery | 3 Parquet files under `s3://bucket/data/` | 3 `DataFileInfo` with `file_size` populated |
| Glob pattern | 3 `.parquet` + 2 `.csv`, glob `*.parquet` | Exactly 3 `DataFileInfo` returned |
| Single file | `s3://bucket/data/f1.parquet` | Exactly 1 `DataFileInfo` returned |
| End-to-end read | 2 Parquet × 100 rows, `num_workers=0` | 200 rows, each batch is `dict[str, torch.Tensor]` |
| Column projection | File with `[feature_a, feature_b, label]`, `columns=["feature_a", "label"]` | Each batch contains only those two keys |
| Predicate pushdown | `feature_b` values 0–99, `filters=pc.field("feature_b") >= 50` | Only rows ≥ 50 returned |
| Path not found | `s3://bucket/does-not-exist/` | `FileNotFoundError` raised |
| No rows dropped or duplicated | 3 Parquet × 100 rows, row_id 0–299 | Set of all row_ids equals exactly {0..299} |
