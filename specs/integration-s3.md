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

#### Scenario: S3 directory discovery
- GIVEN a mocked S3 bucket with 3 Parquet files under `s3://bucket/data/`
- WHEN `discover_files("s3://bucket/data/", storage_options=moto_opts)` is called
- THEN 3 DataFileInfo objects are returned with `file_size` populated

#### Scenario: S3 glob pattern
- GIVEN a mocked S3 bucket with 3 Parquet and 2 CSV files under `s3://bucket/data/`
- WHEN `discover_files("s3://bucket/data/*.parquet", storage_options=moto_opts)` is called
- THEN exactly 3 DataFileInfo objects are returned

#### Scenario: S3 single file
- GIVEN a single Parquet file at `s3://bucket/data/f1.parquet`
- WHEN `discover_files("s3://bucket/data/f1.parquet", storage_options=moto_opts)` is called
- THEN exactly 1 DataFileInfo is returned

#### Scenario: S3 end-to-end Parquet read
- GIVEN a mocked S3 bucket with 2 Parquet files, 100 rows each (200 total)
- WHEN `create_dataloader(path="s3://bucket/data/", format="parquet", storage_options=moto_opts, num_workers=0)` is called
- AND the DataLoader is fully iterated
- THEN total rows collected equals 200
- AND each batch is a `dict[str, torch.Tensor]`

#### Scenario: S3 column projection
- GIVEN a mocked S3 Parquet file with columns [feature_a, feature_b, label]
- WHEN `create_dataloader(..., columns=["feature_a", "label"])` is called
- THEN each batch contains only keys "feature_a" and "label"

#### Scenario: S3 predicate pushdown
- GIVEN a mocked S3 Parquet file with feature_b values 0-99
- WHEN `create_dataloader(..., filters=pc.field("feature_b") >= 50)` is called
- THEN only rows where feature_b >= 50 are returned

#### Scenario: S3 path does not exist
- GIVEN a path `s3://bucket/does-not-exist/` that does not exist in the bucket
- WHEN `discover_files("s3://bucket/does-not-exist/")` is called
- THEN a `FileNotFoundError` is raised

#### Scenario: S3 no rows dropped or duplicated
- GIVEN a mocked S3 bucket with 3 Parquet files, 100 rows each (300 total)
- WHEN `create_dataloader` is fully iterated
- THEN the set of all row_ids equals exactly {0..299} with no duplicates
