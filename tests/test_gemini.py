import llm
import os
import pytest

GEMINI_API_KEY = os.environ.get("PYTEST_GEMINI_API_KEY", None) or "gm-..."


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_prompt():
    model = llm.get_model("gemini-1.5-flash-latest")
    model.key = model.key or GEMINI_API_KEY
    response = model.prompt("Name for a pet pelican, just the name")
    assert str(response) == "Percy"
    assert response.response_json == [
        {
            "candidates": [
                {"content": {"parts": [{"text": "Percy"}], "role": "model"}}
            ],
            "modelVersion": "gemini-1.5-flash-002",
        }
    ]
    assert response.token_details is None
    assert response.input_tokens == 10
    # Not sure why our pytest-recording setup doesn't report output tokens
    # https://github.com/simonw/llm-gemini/issues/25#issuecomment-2487464339
    assert response.output_tokens is None

    # And try it async too
    async_model = llm.get_async_model("gemini-1.5-flash-latest")
    async_model.key = async_model.key or GEMINI_API_KEY
    response = await async_model.prompt("Name for a pet pelican, just the name")
    text = await response.text()
    assert text == "Percy"
