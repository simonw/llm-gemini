import pytest


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["x-goog-api-key"],
        "decode_compressed_response": True,
    }
