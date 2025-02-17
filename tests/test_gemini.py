import llm
import nest_asyncio
import os
import pytest

nest_asyncio.apply()

GEMINI_API_KEY = os.environ.get("PYTEST_GEMINI_API_KEY", None) or "gm-..."


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_prompt():
    model = llm.get_model("gemini-1.5-flash-latest")
    response = model.prompt("Name for a pet pelican, just the name", key=GEMINI_API_KEY)
    assert str(response) == "Percy"
    assert response.response_json == [
        {
            "candidates": [
                {"content": {"parts": [{"text": "Percy"}], "role": "model"}}
            ],
            "modelVersion": "gemini-1.5-flash-latest",
        }
    ]
    assert response.token_details is None
    assert response.input_tokens == 10
    # Not sure why our pytest-recording setup doesn't report output tokens
    # https://github.com/simonw/llm-gemini/issues/25#issuecomment-2487464339
    assert response.output_tokens is None

    # And try it async too
    async_model = llm.get_async_model("gemini-1.5-flash-latest")
    response = await async_model.prompt(
        "Name for a pet pelican, just the name", key=GEMINI_API_KEY
    )
    text = await response.text()
    assert text == "Percy"
