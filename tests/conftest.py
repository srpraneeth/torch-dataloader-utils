import pathlib
import pytest

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture
def sample_parquet(fixtures_dir):
    return str(fixtures_dir / "sample.parquet")


@pytest.fixture
def sample_orc(fixtures_dir):
    return str(fixtures_dir / "sample.orc")


@pytest.fixture
def sample_csv(fixtures_dir):
    return str(fixtures_dir / "sample.csv")


@pytest.fixture
def sample_jsonl(fixtures_dir):
    return str(fixtures_dir / "sample.jsonl")
