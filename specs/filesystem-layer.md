# Spec: Filesystem Layer

## Core Principle

The filesystem layer has one responsibility — discover files and return `list[DataFileInfo]`.
It knows nothing about splits, formats, or reading data.

```
path (str)  +  storage_options (dict)
        ↓
   fsspec filesystem
        ↓
list[DataFileInfo(path, file_size)]
```

---

## Requirements

### File Discovery `[v1]`
The system SHALL accept a path that is one of:
- A directory URI — discover all files directly inside it
- A glob pattern — discover all files matching the pattern
- A single file URI — return that file only

The system SHALL use `fsspec.url_to_fs()` to resolve the filesystem from the path scheme.
The system SHALL pass `storage_options` directly to fsspec without interpretation.
The system SHALL return a `list[DataFileInfo]` with `path` and `file_size` populated.
The system SHALL populate `file_size` from fsspec `stat()` where available.
The system SHALL return an empty list when no files match — not raise an error.
The system SHALL raise a clear `FileNotFoundError` when the path itself does not exist.

### Backend Support `[v1]`
The system SHALL support any fsspec-compatible backend via path scheme auto-detection:
- `s3://` → s3fs
- `gs://` → gcsfs
- `az://` or `abfs://` → adlfs
- No scheme or `file://` → local filesystem

The system SHALL raise an `ImportError` with the install command when an optional
backend package (s3fs, gcsfs, adlfs) is not installed.

### Format Filtering `[v1]`
The system SHALL accept an optional `extensions` parameter (e.g. `[".parquet", ".orc"]`).
When `extensions` is provided the system SHALL exclude files that do not match.
When `extensions` is None the system SHALL return all files regardless of extension.

### Logging `[v1]`
The system SHALL log at `INFO` level: path being scanned and number of files found.
The system SHALL log at `DEBUG` level: each discovered file path and size.

---

## Scenarios

#### Scenario: Directory discovery
- GIVEN a directory containing 5 Parquet files
- WHEN `discover_files(path)` is called
- THEN all 5 files are returned as DataFileInfo with file_size populated

#### Scenario: Glob pattern
- GIVEN a glob `s3://bucket/data/*.parquet` matching 3 files
- WHEN `discover_files(path)` is called
- THEN exactly 3 DataFileInfo objects are returned

#### Scenario: Single file
- GIVEN a direct file path `s3://bucket/data/f1.parquet`
- WHEN `discover_files(path)` is called
- THEN exactly 1 DataFileInfo is returned

#### Scenario: Empty directory
- GIVEN a directory containing no files
- WHEN `discover_files(path)` is called
- THEN an empty list is returned — no error raised

#### Scenario: Path does not exist
- GIVEN a path that does not exist on the filesystem
- WHEN `discover_files(path)` is called
- THEN a FileNotFoundError is raised with the path in the message

#### Scenario: Extension filtering
- GIVEN a directory with 3 .parquet and 2 .csv files
- WHEN `discover_files(path, extensions=[".parquet"])` is called
- THEN only the 3 Parquet files are returned

#### Scenario: Missing optional backend
- GIVEN an `s3://` path and `s3fs` is not installed
- WHEN `discover_files(path)` is called
- THEN an ImportError is raised with: `pip install torch-dataloader-utils[s3]`
