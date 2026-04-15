"""
S3 integration tests using moto (in-process mock).

Exercises the remote filesystem path: PyFileSystem(FSSpecHandler(fs))
in discovery.py and reader.py — the path that local tests cannot cover.

Run with:
    uv run pytest tests/integration/test_s3.py -m integration -v
"""
import io

import boto3
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import torch
from moto import mock_aws

from torch_dataloader_utils.dataset.structured import StructuredDataset
from torch_dataloader_utils.filesystem.discovery import discover_files

BUCKET = "test-bucket"
REGION = "us-east-1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_table(n_rows: int, row_id_offset: int = 0) -> pa.Table:
    row_ids = list(range(row_id_offset, row_id_offset + n_rows))
    return pa.table({
        "row_id":    pa.array(row_ids, type=pa.int32()),
        "feature_a": pa.array([float(i % 10) for i in row_ids], type=pa.float32()),
        "feature_b": pa.array([i % 100 for i in row_ids], type=pa.int32()),
        "label":     pa.array([i % 2 for i in row_ids], type=pa.int32()),
    })


def _table_to_bytes(table: pa.Table) -> bytes:
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def _upload(s3_client, key: str, table: pa.Table) -> None:
    s3_client.put_object(Bucket=BUCKET, Key=key, Body=_table_to_bytes(table))


def _collect(loader) -> dict[str, list]:
    result: dict[str, list] = {}
    for batch in loader:
        for key, val in batch.items():
            if key not in result:
                result[key] = []
            if isinstance(val, torch.Tensor):
                result[key].extend(val.tolist())
            else:
                result[key].extend(val)
    return result


def _storage_options() -> dict:
    """storage_options for s3fs pointing at the active moto mock.
    skip_instance_cache and use_listings_cache=False prevent s3fs from
    reusing a cached filesystem instance with stale ETag metadata across tests.
    """
    return {
        "key": "test",
        "secret": "test",
        "client_kwargs": {"region_name": REGION},
        "skip_instance_cache": True,
        "use_listings_cache": False,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def aws_credentials(monkeypatch):
    """Ensure boto3/s3fs never hit real AWS."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "test")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)


# ---------------------------------------------------------------------------
# Scenario: S3 directory discovery
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_s3_directory_discovery():
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET)
        for i in range(3):
            _upload(client, f"data/f{i}.parquet", _make_table(10))

        files = discover_files(
            f"s3://{BUCKET}/data/",
            storage_options=_storage_options(),
        )

    assert len(files) == 3
    assert all(f.file_size > 0 for f in files)


# ---------------------------------------------------------------------------
# Scenario: S3 glob pattern
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_s3_glob_pattern():
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET)
        for i in range(3):
            _upload(client, f"data/f{i}.parquet", _make_table(10))
        # Upload 2 CSV-named objects that should be excluded by the glob
        for i in range(2):
            client.put_object(Bucket=BUCKET, Key=f"data/f{i}.csv", Body=b"a,b\n1,2\n")

        files = discover_files(
            f"s3://{BUCKET}/data/*.parquet",
            storage_options=_storage_options(),
        )

    assert len(files) == 3


# ---------------------------------------------------------------------------
# Scenario: S3 single file
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_s3_single_file():
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET)
        _upload(client, "data/f1.parquet", _make_table(10))

        files = discover_files(
            f"s3://{BUCKET}/data/f1.parquet",
            storage_options=_storage_options(),
        )

    assert len(files) == 1


# ---------------------------------------------------------------------------
# Scenario: S3 path does not exist
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_s3_path_not_found():
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET)

        with pytest.raises(FileNotFoundError):
            discover_files(
                f"s3://{BUCKET}/does-not-exist/",
                storage_options=_storage_options(),
            )


# ---------------------------------------------------------------------------
# Scenario: S3 end-to-end Parquet read
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_s3_end_to_end_read():
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET)
        for i in range(2):
            _upload(client, f"data/f{i}.parquet", _make_table(100, row_id_offset=i * 100))

        loader, _ = StructuredDataset.create_dataloader(
            path=f"s3://{BUCKET}/data/",
            format="parquet",
            num_workers=0,
            batch_size=50,
            storage_options=_storage_options(),
        )
        rows = _collect(loader)

    assert len(rows["row_id"]) == 200
    assert isinstance(rows["row_id"][0], int)


# ---------------------------------------------------------------------------
# Scenario: S3 column projection
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_s3_column_projection():
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET)
        _upload(client, "data/f1.parquet", _make_table(50))

        loader, _ = StructuredDataset.create_dataloader(
            path=f"s3://{BUCKET}/data/",
            format="parquet",
            num_workers=0,
            columns=["feature_a", "label"],
            storage_options=_storage_options(),
        )
        batch = next(iter(loader))

    assert set(batch.keys()) == {"feature_a", "label"}
    assert "feature_b" not in batch


# ---------------------------------------------------------------------------
# Scenario: S3 predicate pushdown
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_s3_predicate_pushdown():
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET)
        _upload(client, "data/f1.parquet", _make_table(100))

        loader, _ = StructuredDataset.create_dataloader(
            path=f"s3://{BUCKET}/data/",
            format="parquet",
            num_workers=0,
            filters=pc.field("feature_b") >= 50,
            storage_options=_storage_options(),
        )
        rows = _collect(loader)

    # feature_b = row_id % 100; rows 0-99 → feature_b 0-99 → 50 rows pass
    assert len(rows["row_id"]) == 50
    assert all(v >= 50 for v in rows["feature_b"])


# ---------------------------------------------------------------------------
# Scenario: S3 no rows dropped or duplicated
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_s3_no_rows_dropped_or_duplicated():
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET)
        for i in range(3):
            _upload(client, f"data/f{i}.parquet", _make_table(100, row_id_offset=i * 100))

        loader, _ = StructuredDataset.create_dataloader(
            path=f"s3://{BUCKET}/data/",
            format="parquet",
            num_workers=0,
            batch_size=32,
            storage_options=_storage_options(),
        )
        rows = _collect(loader)

    row_ids = rows["row_id"]
    assert len(row_ids) == 300
    assert sorted(row_ids) == list(range(300))
