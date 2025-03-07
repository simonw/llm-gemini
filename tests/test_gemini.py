import llm
import nest_asyncio
import json
import os
import pytest
import pydantic

nest_asyncio.apply()

GEMINI_API_KEY = os.environ.get("PYTEST_GEMINI_API_KEY", None) or "gm-..."


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_prompt():
    model = llm.get_model("gemini-1.5-flash-latest")
    response = model.prompt("Name for a pet pelican, just the name", key=GEMINI_API_KEY)
    assert str(response) == "Percy\n"
    assert response.response_json == {
        "candidates": [
            {
                "finishReason": "STOP",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE",
                    },
                ],
            }
        ],
        "modelVersion": "gemini-1.5-flash-latest",
    }
    assert response.token_details == {
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 9}],
        "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 2}],
    }
    assert response.input_tokens == 9
    assert response.output_tokens == 2

    # And try it async too
    async_model = llm.get_async_model("gemini-1.5-flash-latest")
    response = await async_model.prompt(
        "Name for a pet pelican, just the name", key=GEMINI_API_KEY
    )
    text = await response.text()
    assert text == "Percy\n"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_prompt_with_pydantic_schema():
    class Dog(pydantic.BaseModel):
        name: str
        age: int
        bio: str

    model = llm.get_model("gemini-1.5-flash-latest")
    response = model.prompt(
        "Invent a cool dog", key=GEMINI_API_KEY, schema=Dog, stream=False
    )
    assert json.loads(response.text()) == {
        "age": 3,
        "bio": "A fluffy Samoyed with exceptional intelligence and a love for belly rubs. He's mastered several tricks, including fetching the newspaper and opening doors.",
        "name": "Cloud",
    }
    assert response.response_json == {
        "candidates": [
            {
                "finishReason": "STOP",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE",
                    },
                ],
            }
        ],
        "modelVersion": "gemini-1.5-flash-latest",
    }
    assert response.input_tokens == 10


@pytest.mark.vcr
@pytest.mark.parametrize(
    "model_id",
    (
        "gemini-embedding-exp-03-07",
        "gemini-embedding-exp-03-07-128",
        "gemini-embedding-exp-03-07-512",
    ),
)
def test_embedding(model_id, monkeypatch):
    monkeypatch.setenv("LLM_GEMINI_KEY", GEMINI_API_KEY)
    model = llm.get_embedding_model(model_id)
    response = model.embed("Some text goes here")
    expected_length = 3072
    if model_id.endswith("-128"):
        expected_length = 128
    elif model_id.endswith("-512"):
        expected_length = 512
    assert len(response) == expected_length
