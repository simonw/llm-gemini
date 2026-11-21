import asyncio
import json

import httpx
import httpx._api
import llm
import pytest
from click.testing import CliRunner
from llm.cli import cli

MODEL = "gemini-2.5-flash"
KEY = "test-key-not-real"


class TrackedStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.chunks_read = 0
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            self.chunks_read += 1
            yield chunk

    async def __aiter__(self):
        for chunk in self:
            yield chunk

    def close(self):
        self.closed = True

    async def aclose(self):
        self.closed = True


@pytest.fixture(autouse=True)
def isolated_user_path(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path / "llm"))
    monkeypatch.setenv("LLM_GEMINI_KEY", KEY)


@pytest.fixture
def mock_http(monkeypatch):
    def install(status, chunks):
        stream = TrackedStream(chunks)
        clients = []
        requests = []

        def handler(request):
            requests.append(request)
            assert request.method == "POST"
            assert request.url.path.endswith(f"/{MODEL}:streamGenerateContent")
            assert request.headers["x-goog-api-key"] == KEY
            return httpx.Response(
                status,
                stream=stream,
                headers={"x-private-debug": "do-not-display"},
            )

        transport = httpx.MockTransport(handler)
        sync_client = httpx.Client
        async_client = httpx.AsyncClient

        def make_sync_client(*args, **kwargs):
            client = sync_client(*args, **kwargs, transport=transport)
            clients.append(client)
            return client

        def make_async_client(*args, **kwargs):
            client = async_client(*args, **kwargs, transport=transport)
            clients.append(client)
            return client

        # Keep httpx.stream itself real, including its client context manager.
        monkeypatch.setattr(httpx._api, "Client", make_sync_client)
        monkeypatch.setattr(httpx, "AsyncClient", make_async_client)

        def assert_closed():
            assert len(requests) == 1
            assert stream.closed
            assert len(clients) == 1
            assert clients[0].is_closed

        return stream, assert_closed

    return install


def prompt_text(async_):
    if async_:

        async def run():
            response = await llm.get_async_model(MODEL).prompt("Hello", key=KEY)
            return await response.text()

        return asyncio.run(run())
    return llm.get_model(MODEL).prompt("Hello", key=KEY).text()


def encoded(value):
    return json.dumps(value).encode()


@pytest.mark.parametrize("async_", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "status,body,message",
    [
        (400, encoded({"error": {"message": "Invalid request"}}), "Invalid request"),
        (
            403,
            encoded({"error": {"message": "Permission denied"}}),
            "Permission denied",
        ),
        (429, encoded({"error": {"message": "Quota exhausted"}}), "Quota exhausted"),
        (500, encoded([{"error": {"message": "Legacy error"}}]), "Legacy error"),
        (503, b"<html>private upstream error</html>", "Gemini API error: HTTP 503"),
        (500, b'{"error":', "Gemini API error: HTTP 500"),
        (500, encoded({"error": []}), "Gemini API error: HTTP 500"),
        (500, encoded({"error": {}}), "Gemini API error: HTTP 500"),
        (500, encoded({"error": {"message": 123}}), "Gemini API error: HTTP 500"),
        (500, encoded({"error": {"message": "  "}}), "Gemini API error: HTTP 500"),
    ],
    ids=[
        "bad-request",
        "forbidden",
        "quota",
        "legacy-array",
        "html",
        "invalid-json",
        "invalid-error-shape",
        "missing-message",
        "non-string-message",
        "empty-message",
    ],
)
def test_http_errors_are_model_errors(mock_http, async_, status, body, message):
    # Split the body to exercise read()/aread() across transport chunks.
    _, assert_closed = mock_http(status, [body[:5], body[5:]])
    try:
        with pytest.raises(llm.ModelError) as exc:
            prompt_text(async_)
        assert str(exc.value) == message
    finally:
        assert_closed()


@pytest.mark.parametrize("async_", [False, True], ids=["sync", "async"])
def test_http_200_array_error_keeps_provider_message(mock_http, async_):
    message = "Existing in-band error"
    _, assert_closed = mock_http(200, [encoded([{"error": {"message": message}}])])
    try:
        with pytest.raises(llm.ModelError) as exc:
            prompt_text(async_)
        assert str(exc.value) == message
    finally:
        assert_closed()


@pytest.mark.parametrize("async_", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "status,body,message",
    [
        (429, encoded({"error": {"message": "Quota exhausted"}}), "Quota exhausted"),
        (503, b"<html>private upstream error</html>", "Gemini API error: HTTP 503"),
    ],
    ids=["provider-message", "fallback"],
)
def test_cli_reports_http_error(mock_http, async_, status, body, message):
    _, assert_closed = mock_http(status, [body])
    args = ["-m", MODEL, "--key", KEY, "--no-log", "Hello"]
    if async_:
        args.append("--async")
    try:
        result = CliRunner().invoke(cli, args)
        assert result.exit_code == 1
        assert result.output == f"Error: {message}\n"
        assert isinstance(result.exception, SystemExit)
    finally:
        assert_closed()


@pytest.mark.parametrize("async_", [False, True], ids=["sync", "async"])
def test_success_is_incremental(mock_http, async_):
    def event(text):
        return {
            "candidates": [{"content": {"parts": [{"text": text}]}}],
            "modelVersion": MODEL,
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2},
        }

    stream, assert_closed = mock_http(
        200, [b"[" + encoded(event("Hello")) + b",", encoded(event(" world")) + b"]"]
    )

    def assert_first_chunk():
        assert stream.chunks_read == 1
        assert not stream.closed

    if async_:

        async def run():
            response = llm.get_async_model(MODEL).prompt("Hello", key=KEY)
            chunks = response.__aiter__()
            assert await chunks.__anext__() == "Hello"
            assert_first_chunk()
            assert [chunk async for chunk in chunks] == [" world"]
            assert await response.text() == "Hello world"
            return response

        response = asyncio.run(run())
    else:
        response = llm.get_model(MODEL).prompt("Hello", key=KEY)
        chunks = iter(response)
        assert next(chunks) == "Hello"
        assert_first_chunk()
        assert list(chunks) == [" world"]
        assert response.text() == "Hello world"

    assert response.resolved_model == MODEL
    assert (response.input_tokens, response.output_tokens) == (3, 2)
    assert_closed()
