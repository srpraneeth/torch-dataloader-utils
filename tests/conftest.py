import pathlib

import pytest

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# moto 5.x + aiobotocore 3.x / s3fs 2024.10+ compatibility shim
#
# s3fs 2024.10+ uses aiobotocore 3.x, which expects aiohttp-style responses
# (async .content property, .content.read() coroutine). moto 5.x mocks at
# the botocore HTTP layer and returns MockRawResponse — a BytesIO subclass
# with no async interface.
#
# This shim patches two spots:
#   1. aiobotocore.endpoint.convert_to_response_dict — makes .content awaitable
#   2. aiobotocore.response.StreamingBody.read — falls back to sync .read()
#      when the wrapped object has no async .content attribute
# ---------------------------------------------------------------------------
try:
    import inspect as _inspect

    # ---- patch 1: endpoint.convert_to_response_dict ----
    import aiobotocore.endpoint as _aioep

    async def _compat_convert(http_response, operation_model):
        response_dict = {
            "headers": http_response.headers,
            "status_code": http_response.status_code,
            "context": {"operation_name": operation_model.name},
        }

        async def _read_body(obj):
            val = obj.content
            if _inspect.isawaitable(val):
                return await val
            # moto MockRawResponse: .content is bytes, not a coroutine
            return val

        if response_dict["status_code"] >= 300:
            response_dict["body"] = await _read_body(http_response)
        elif operation_model.has_event_stream_output:
            response_dict["body"] = http_response.raw
        elif operation_model.has_streaming_output:
            try:
                import httpx as _httpx
                if isinstance(http_response.raw, _httpx.Response):
                    from aiobotocore.endpoint import HttpxStreamingBody
                    response_dict["body"] = HttpxStreamingBody(http_response.raw)
                    return response_dict
            except ImportError:
                pass
            from botocore.response import StreamingBody as _SyncBody
            length = response_dict["headers"].get("content-length")
            response_dict["body"] = _SyncBody(http_response.raw, length)
        else:
            response_dict["body"] = await _read_body(http_response)

        return response_dict

    _aioep.convert_to_response_dict = _compat_convert

    # ---- patch 2: response.StreamingBody.read ----
    # aiobotocore wraps the raw stream and calls self.__wrapped__.content.read(n)
    # moto's MockRawResponse has no .content — it IS the readable bytes stream.
    import aiobotocore.response as _aioresp

    _orig_sb_read = _aioresp.StreamingBody.read

    async def _compat_sb_read(self, amt=None):
        raw = self.__wrapped__
        if not hasattr(raw, "content"):
            # moto MockRawResponse: read directly
            data = raw.read() if amt is None else raw.read(amt)
            self._self_amount_read += len(data)
            if amt is None or (not data and amt is not None and amt > 0):
                self._verify_content_length()
            return data
        return await _orig_sb_read(self, amt)

    _aioresp.StreamingBody.read = _compat_sb_read

except ImportError:
    pass  # aiobotocore not installed — no patch needed


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
