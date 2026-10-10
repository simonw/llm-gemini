from click.testing import CliRunner
import httpx
import json
from llm.cli import cli
import pytest


@pytest.mark.parametrize(
    "methods, expected_names",
    [
        ([], ["models/first", "models/embedding", "models/last"]),
        (["embedContent"], ["models/embedding"]),
        (
            ["embedContent", "generateContent"],
            ["models/first", "models/embedding", "models/last"],
        ),
    ],
)
def test_cli_gemini_models_pagination(monkeypatch, methods, expected_names):
    pages = {
        None: {
            "models": [
                {
                    "name": "models/first",
                    "supportedGenerationMethods": ["generateContent"],
                }
            ],
            "nextPageToken": "second/+?=",
        },
        "second/+?=": {
            "models": [
                {
                    "name": "models/embedding",
                    "supportedGenerationMethods": ["embedContent"],
                }
            ],
            "nextPageToken": "third/+?=",
        },
        "third/+?=": {
            "models": [
                {
                    "name": "models/last",
                    "supportedGenerationMethods": ["generateContent"],
                }
            ]
        },
    }
    page_tokens = []

    def handle(request):
        assert request.url.host == "generativelanguage.googleapis.com"
        assert request.url.path == "/v1beta/models"
        assert request.headers["x-goog-api-key"] == "test-key"
        page_token = request.url.params.get("pageToken")
        page_tokens.append(page_token)
        return httpx.Response(200, json=pages[page_token])

    args = ["gemini", "models", "--key", "test-key"]
    for method in methods:
        args.extend(["--method", method])
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr("llm_gemini.httpx.get", client.get)
        result = CliRunner().invoke(cli, args)

    assert result.exit_code == 0, result.exception
    assert [model["name"] for model in json.loads(result.output)] == expected_names
    assert page_tokens == [None, "second/+?=", "third/+?="]


def test_cli_gemini_models_empty_page(monkeypatch):
    def handle(request):
        return httpx.Response(200, json={})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr("llm_gemini.httpx.get", client.get)
        result = CliRunner().invoke(cli, ["gemini", "models", "--key", "test-key"])

    assert result.exit_code == 0, result.exception
    assert json.loads(result.output) == []


def test_cli_gemini_models_later_page_error(monkeypatch):
    page_tokens = []

    def handle(request):
        page_token = request.url.params.get("pageToken")
        page_tokens.append(page_token)
        if page_token is None:
            return httpx.Response(
                200,
                json={"models": [{"name": "models/first"}], "nextPageToken": "second"},
            )
        return httpx.Response(503, json={"error": {"message": "Unavailable"}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr("llm_gemini.httpx.get", client.get)
        result = CliRunner().invoke(cli, ["gemini", "models", "--key", "test-key"])

    assert result.exit_code == 1
    assert isinstance(result.exception, httpx.HTTPStatusError)
    assert result.exception.response.status_code == 503
    assert result.output == ""
    assert page_tokens == [None, "second"]
