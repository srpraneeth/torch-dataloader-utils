# Spec: GCS and Azure Integration Tests

## Core Principle

GCS and Azure integration tests validate the full pipeline against mocked remote
backends. They cover per-backend differences that S3/moto tests cannot:
- Path format (`gs://`, `az://`, `abfs://`)
- Non-local fsspec filesystem → `PyFileSystem(FSSpecHandler(fs))` wrapping path
- Install-hint errors for missing `gcsfs` / `adlfs`

```
MemoryFileSystem (fsspec, in-process)
    patched via fsspec.url_to_fs for gs:// / az:// URIs
        ↓
discover_files("gs://bucket/..." or "az://container/...")
        ↓
read_split(split, storage_options=...)
        ↓
StructuredDataset.create_dataloader(path="gs://...", storage_options=...)
        ↓
batches in requested output format
```

## Mocking Strategy

Both backends are mocked **in-process** using `fsspec`'s built-in
`MemoryFileSystem` — no real credentials, no Docker, no network.

`MemoryFileSystem` has protocol `"memory"` (non-local), so it exercises the
`PyFileSystem(FSSpecHandler(fs))` wrapping path in `reader.py` — the same code
path that real `gcsfs` / `adlfs` use. `fsspec.url_to_fs` is patched via
`unittest.mock.patch` to return the `MemoryFileSystem` instance for `gs://` and
`az://` URIs, stripping the scheme so paths resolve correctly inside the fs.

Why not a real mock library (`gcsfs` anonymous mode, `fake-gcs-server`, Azurite)?
- Docker adds CI complexity and flakiness
- `gcsfs`/`adlfs` anonymous modes still attempt network resolution
- The goal is validating *this library's* handling of the wrapping and path
  restoration logic — not validating `gcsfs`/`adlfs` internals

`gcsfs` and `adlfs` are installed in the dev environment so that the install-hint
path (ImportError when missing) can also be tested.

---

## Requirements

### Test Infrastructure `[v2]`
Tests SHALL mock `fsspec.url_to_fs` to return a `MemoryFileSystem` for `gs://` and
`az://`/`abfs://` URIs — no real GCP or Azure credentials needed.
Tests SHALL use a `gcs_dataset` / `azure_dataset` pytest fixture that:
  1. Creates a `MemoryFileSystem` and populates it with Parquet fixture files
  2. Patches `fsspec.url_to_fs` to serve those files for the target scheme
  3. Exposes a scheme URI (`gs://test-bucket/data/`) for use in test assertions
`gcsfs` and `adlfs` SHALL be added to the `dev` extras in `pyproject.toml`.
All tests SHALL be marked `@pytest.mark.integration`.
CI SHALL install `gcsfs` and `adlfs` alongside `s3` in the integration job.

### Coverage Goals `[v2]`
Tests SHALL exercise `discover_files()` against a mocked `gs://` / `az://` path.
Tests SHALL exercise `read_split()` reading Parquet from the mocked filesystem.
Tests SHALL exercise `StructuredDataset.create_dataloader()` end-to-end.
Tests SHALL cover directory, glob, and single-file path forms per backend.
Tests SHALL confirm the `PyFileSystem(FSSpecHandler(fs))` wrapping path is taken
  (i.e., that a non-local fsspec fs is returned and wrapped for pyarrow).
Tests SHALL cover the install-hint `ImportError` when the backend is not installed.
Tests SHALL verify no rows are dropped or duplicated across multiple files.

---

## Scenarios (identical for GCS and Azure)

| Scenario | Input | Expected |
|----------|-------|----------|
| Directory discovery | 3 Parquet files under `gs://bucket/data/` | 3 `DataFileInfo` with `file_size > 0` |
| Glob pattern | 3 `.parquet` + 2 injected `.csv` names, glob `*.parquet` | Exactly 3 `DataFileInfo` |
| Single file | `gs://bucket/data/f1.parquet` | Exactly 1 `DataFileInfo` |
| End-to-end read | 2 Parquet × 100 rows, `num_workers=0` | 200 rows, `dict[str, torch.Tensor]` |
| Column projection | 4-col file, `columns=["feature_a", "label"]` | Only those two keys in each batch |
| Predicate pushdown | `feature_b` 0–99, `filters=pc.field("feature_b") >= 50` | 50 rows, all `feature_b >= 50` |
| No rows dropped/duplicated | 3 Parquet × 100 rows, row_id 0–299 | `sorted(row_ids) == list(range(300))` |
| Path not found | `gs://bucket/does-not-exist/` | `FileNotFoundError` |
| Missing backend | `gcsfs` / `adlfs` not importable | `ImportError` with install hint |

---

## Fixture Design

```python
@pytest.fixture
def gcs_dataset():
    """Populate a MemoryFileSystem and patch fsspec.url_to_fs for gs:// URIs."""
    from fsspec.implementations.memory import MemoryFileSystem

    mem = MemoryFileSystem()
    for i in range(3):
        buf = io.BytesIO()
        pq.write_table(_make_table(100, row_id_offset=i * 100), buf)
        mem.pipe(f"test-bucket/data/f{i}.parquet", buf.getvalue())

    real_url_to_fs = fsspec.url_to_fs

    def _patched(path, **kwargs):
        if path.startswith("gs://"):
            return mem, path[len("gs://"):]
        return real_url_to_fs(path, **kwargs)

    with mock.patch("fsspec.url_to_fs", side_effect=_patched):
        yield "gs://test-bucket/data"
```

Azure fixture is identical with `"az://"` and `"abfs://"` scheme handling.

---

## CI Changes

`pyproject.toml` dev extras:
```toml
dev = [
    ...
    "gcsfs>=2024.2",
    "adlfs>=2024.2",
]
```

`.github/workflows/ci.yml` integration job — extend the install step:
```yaml
- name: Install dependencies
  run: |
    ...
    uv pip install --system -e ".[dev,s3,gcs,azure,iceberg]"
```

No new CI job needed — GCS and Azure tests run in the existing integration job.
