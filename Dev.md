# torch-dataloader-utils — Dev Plan

## V1 Tasks

### 1. Project Setup
- [x] Initialize `pyproject.toml` with core and optional dependencies
- [x] Set up package structure under `torch_dataloader_utils/`
- [x] Configure `ruff` for linting and formatting
- [x] Set up `pre-commit` hooks
- [x] Add `.gitignore`
- [x] Add GitHub Actions CI workflow

### 2. Testing Setup
- [x] Install and configure `pytest`
- [x] Set up `pytest-cov` for coverage
- [x] Create local test fixtures — small Parquet, ORC, CSV, JSON files
- [x] Configure CI matrix: Python 3.11/3.12 × PyTorch 2.2/latest
- [x] Add `moto` for mocked S3 tests (no real cloud credentials needed in CI)

### 3. Core Abstractions
- [x] Define `Split` dataclass
- [x] Implement split generation algorithm (round-robin, file-count balanced)
- [x] Implement epoch reshuffle logic with seed + epoch offset
- [x] Write tests for split generation and shuffle reproducibility

### 4. Filesystem Layer
- [x] Implement file discovery via `fsspec` (directory, glob, single file)
- [x] Validate `storage_options` passthrough
- [x] Write tests for local filesystem
- [x] Write tests for mocked S3 via `moto`

### 5. Format Layer
- [x] Implement `pyarrow.dataset` reader (Parquet, ORC, CSV, JSON, JSONL)
- [x] Wire up column projection (`columns` parameter)
- [x] Wire up predicate pushdown (`filters` parameter)
- [x] Wire up Hive partitioning (`partitioning` parameter)
- [x] Write tests for each format
- [x] Write tests for column projection and filter pushdown

### 6. Dataset Layer
- [x] Implement `StructuredDataset(IterableDataset)`
- [x] Implement `__iter__` with `get_worker_info()` based split assignment
- [x] Implement output format conversion (`torch`, `numpy`, `arrow`, `dict`)
- [x] Implement `collate_fn` passthrough with early validation
- [x] Implement `create_dataloader()` static method
- [x] Implement `set_epoch()` for shuffle reproducibility across epochs
- [x] Write tests for single-process iteration
- [x] Write tests for multi-worker assignment (mocked `get_worker_info`)
- [ ] Write tests for real multi-worker iteration (runs on Linux CI via GitHub Actions — macOS spawn mode causes deadlocks with pyarrow generators)
- [x] Write tests for output format conversion
- [x] Write tests for collate_fn validation

### 7. Iceberg Layer
- [x] Implement `IcebergDataset(IterableDataset)`
- [x] Connect to catalog via `pyiceberg` using `catalog_config`
- [x] Resolve table to data file URIs
- [x] Support `snapshot_id` for time travel
- [x] Apply partition filters before file resolution
- [x] Delegate to `StructuredDataset` after file resolution
- [x] Implement `create_dataloader()` static method
- [x] Write tests with a local Iceberg table fixture

### 7b. Target-Size Sub-File Splitting
- [x] Implement `TargetSizeSplitStrategy` — packs row groups into `target_bytes` chunks
- [x] Read Parquet row group metadata in main process (no data scan)
- [x] Sub-file `RowRange` support in `reader.py` via `pq.ParquetFile.read_row_groups()`
- [x] Non-Parquet files (ORC, CSV, JSONL) treated as single unsplittable chunks
- [x] Make `TargetSizeSplitStrategy` the default auto-selected strategy
- [x] Write unit tests for strategy and reader
- [x] Write integration tests for sub-file splitting and predicate pushdown

### 8. Error Handling
- [x] Missing optional dependency — clear `ImportError` with install command
- [x] Invalid `format` value — clear `ValueError`
- [x] `output_format` is `arrow`/`dict` with no `collate_fn` — clear error at construction
- [x] Path not found or inaccessible — surface fsspec error cleanly (bad S3 bucket / permission errors)

### 9. Publishing
- [x] Write `README.md` install and quickstart section
- [x] Configure `pyproject.toml` for PyPI metadata
- [ ] Set up GitHub Actions release workflow (tag → publish to PyPI)
- [ ] Publish `0.1.0` to PyPI
