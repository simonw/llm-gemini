from contextlib import contextmanager
import copy
import httpx
import json
import llm
from llm.cli import cli
from llm.parts import ReasoningPart, TextPart, ToolCallPart
from click.testing import CliRunner
import pytest


class SyncChunks(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self):
        self.closed = True


class AsyncChunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.fixture
def transport(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))
    client_class = httpx.Client
    async_client_class = httpx.AsyncClient

    def install(chunks, content_type="text/event-stream", status=200):
        state = {"requests": [], "streams": []}

        def response(request, stream_class):
            assert request.method == "POST"
            assert request.url.host == "generativelanguage.googleapis.com"
            assert request.url.path.endswith(":streamGenerateContent")
            assert not request.url.query  # v1 must not force alt=sse
            assert request.headers["x-goog-api-key"] == "test-key"
            state["requests"].append(json.loads(request.content))
            stream = stream_class(chunks)
            state["streams"].append(stream)
            return httpx.Response(
                status, headers={"Content-Type": content_type}, stream=stream
            )

        @contextmanager
        def stream(method, url, **kwargs):
            with client_class(
                transport=httpx.MockTransport(lambda r: response(r, SyncChunks))
            ) as client:
                with client.stream(method, url, **kwargs) as result:
                    yield result

        monkeypatch.setattr("llm_gemini.httpx.stream", stream)
        monkeypatch.setattr(
            "llm_gemini.httpx.AsyncClient",
            lambda: async_client_class(
                transport=httpx.MockTransport(lambda r: response(r, AsyncChunks))
            ),
        )
        return state

    return install


def event(parts=None, **extra):
    return {
        "candidates": [{"content": {"parts": parts or []}, "finishReason": "STOP"}],
        "modelVersion": "gemini-test-resolved",
        **extra,
    }


def wire(events, ending="\n"):
    return "".join(
        "data: " + json.dumps(e, ensure_ascii=False) + ending * 2 for e in events
    ).encode()


async def consume(async_):
    model = (
        llm.get_async_model("gemini-flash-latest")
        if async_
        else llm.get_model("gemini-flash-latest")
    )
    response = model.prompt("hello", key="test-key")
    text = await response.text() if async_ else response.text()
    return response, text


@pytest.mark.asyncio
@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("chunk_size", [1, 7, 65536])
async def test_sse_chunk_boundaries(transport, async_, ending, chunk_size):
    # U+2028 is JSON text, not an SSE line delimiter.
    raw = (
        b"\xef\xbb\xbf"
        + (
            ": keepalive"
            + ending * 2
            + "event: message"
            + ending
            + "id: 12"
            + ending
            + "retry: 50"
            + ending
        ).encode()
    )
    raw += wire(
        [event([{"text": "안녕🙂\u2028"}]), event([{"text": " world"}])], ending
    )
    state = transport(
        [raw[i : i + chunk_size] for i in range(0, len(raw), chunk_size)],
        "Text/Event-Stream; charset=iso-8859-1",
    )
    response, text = await consume(async_)
    assert text == "안녕🙂\u2028 world"
    assert response.resolved_model == "gemini-test-resolved"
    assert len(state["requests"]) == 1
    assert all(s.closed for s in state["streams"])


@pytest.mark.asyncio
@pytest.mark.parametrize("async_", [False, True])
async def test_sse_multiline_and_empty_events(transport, async_):
    data = json.dumps(event([{"text": "hello"}]), indent=2)
    raw = (
        ": ping\n\ndata:\n\nunknown: ignored\n\n"
        + "\n".join("data:" + line for line in data.splitlines())
        + "\n\n"
    )
    raw += 'data: {"candidates": '  # unfinished event must not be dispatched at EOF
    state = transport([raw.encode()])
    _, text = await consume(async_)
    assert text == "hello"
    assert all(s.closed for s in state["streams"])


@pytest.mark.asyncio
@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.parametrize("content_type", ["text/event-stream", "application/json", ""])
async def test_stream_metadata_preserved(transport, async_, content_type):
    grounding = {"groundingSupports": [{"confidenceScores": [0.98]}]}
    parts = [
        {"thought": True, "text": "thinking", "thoughtSignature": "reason-signature"},
        {"text": "answer", "thoughtSignature": "text-signature"},
        {
            "functionCall": {
                "name": "lookup",
                "args": {"city": "서울"},
                "id": "call-1",
            },
            "thoughtSignature": "tool-signature",
        },
    ]
    events = [
        event(parts),
        event(
            usageMetadata={
                "promptTokenCount": 4,
                "candidatesTokenCount": 3,
                "thoughtsTokenCount": 2,
                "totalTokenCount": 9,
            }
        ),
    ]
    events[0]["candidates"][0]["groundingMetadata"] = grounding
    original = copy.deepcopy(events)
    raw = (
        wire(events)
        if content_type == "text/event-stream"
        else json.dumps(events).encode()
    )
    state = transport([raw[i : i + 3] for i in range(0, len(raw), 3)], content_type)
    response, text = await consume(async_)
    assert text == "answer"
    assert response.resolved_model == "gemini-test-resolved"
    assert response.input_tokens == 4
    assert response.output_tokens == 5
    messages = await response.messages() if async_ else response.messages()
    all_parts = messages[-1].parts
    reasoning = next(p for p in all_parts if isinstance(p, ReasoningPart))
    assert reasoning.text == "thinking"
    assert (
        reasoning.provider_metadata["gemini"]["thoughtSignature"] == "reason-signature"
    )
    text_part = next(p for p in all_parts if isinstance(p, TextPart))
    assert text_part.provider_metadata["gemini"]["thoughtSignature"] == "text-signature"
    assert any(
        isinstance(p, TextPart)
        and (p.provider_metadata or {}).get("gemini", {}).get("groundingMetadata")
        == grounding
        for p in all_parts
    )
    tool_calls = await response.tool_calls() if async_ else response.tool_calls()
    assert [(t.name, t.arguments, t.tool_call_id) for t in tool_calls] == [
        ("lookup", {"city": "서울"}, "call-1")
    ]
    assert (
        next(p for p in all_parts if isinstance(p, ToolCallPart)).provider_metadata[
            "gemini"
        ]["thoughtSignature"]
        == "tool-signature"
    )
    json.dumps(response.response_json)
    assert events == original
    assert all(s.closed for s in state["streams"])


@pytest.mark.asyncio
@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.parametrize("content_type", ["text/event-stream", "application/json"])
async def test_stream_provider_error(transport, async_, content_type):
    events = [{"error": {"message": "Provider unavailable"}}]
    raw = (
        wire(events)
        if content_type == "text/event-stream"
        else json.dumps(events).encode()
    )
    state = transport([raw], content_type, 503)
    with pytest.raises(llm.ModelError, match="Provider unavailable"):
        await consume(async_)
    assert all(s.closed for s in state["streams"])


@pytest.mark.asyncio
@pytest.mark.parametrize("async_", [False, True])
@pytest.mark.parametrize(
    "raw,message",
    [
        (b"data: invalid\n\n", "Invalid JSON in Gemini SSE data"),
        (b"data: []\n\n", "Expected a JSON object in Gemini SSE data"),
        (b": keepalive\n\n", "Gemini SSE response contained no JSON data"),
        (b"data: {}", "Gemini SSE response contained no JSON data"),
    ],
)
async def test_sse_invalid_or_empty_response(transport, async_, raw, message):
    state = transport([raw])
    with pytest.raises(llm.ModelError, match=message):
        await consume(async_)
    assert all(s.closed for s in state["streams"])


def test_cli_sse(transport):
    state = transport([wire([event([{"text": "안녕 SSE"}])])])
    result = CliRunner().invoke(
        cli, ["-m", "gemini-flash-latest", "--key", "test-key", "--no-log", "hello"]
    )
    assert result.exit_code == 0, result.exception
    assert result.stdout == "안녕 SSE\n"
    assert all(s.closed for s in state["streams"])


@pytest.mark.asyncio
@pytest.mark.parametrize("async_", [False, True])
async def test_sse_server_tool_metadata(transport, async_):
    raw_call = {
        "toolCall": {
            "toolType": "GOOGLE_SEARCH_WEB",
            "args": {"queries": ["test"]},
            "id": "search-1",
        },
        "thoughtSignature": "call-sig",
    }
    raw_result = {
        "toolResponse": {
            "toolType": "GOOGLE_SEARCH_WEB",
            "response": {"result": "found"},
            "id": "search-1",
        },
        "thoughtSignature": "result-sig",
    }
    state = transport([wire([event([raw_call, raw_result, {"text": "found"}])])])
    response, text = await consume(async_)
    assert text == "found"
    messages = await response.messages() if async_ else response.messages()
    parts = messages[-1].parts
    call = next(p for p in parts if isinstance(p, ToolCallPart))
    assert call.server_executed and call.tool_call_id == "search-1"
    assert call.provider_metadata["gemini"]["part"] == raw_call
    assert any(
        (p.provider_metadata or {}).get("gemini", {}).get("part") == raw_result
        for p in parts
    )
    assert all(s.closed for s in state["streams"])


@pytest.mark.asyncio
@pytest.mark.parametrize("async_", [False, True])
async def test_stream_is_incremental_and_closes_on_read_error(transport, async_):
    state = transport(
        [wire([event([{"text": "first"}])]), httpx.ReadError("broken stream")]
    )
    model = (
        llm.get_async_model("gemini-flash-latest")
        if async_
        else llm.get_model("gemini-flash-latest")
    )
    response = model.prompt("hello", key="test-key")
    iterator = response.__aiter__() if async_ else iter(response)
    first = await iterator.__anext__() if async_ else next(iterator)
    assert first == "first"
    assert not state["streams"][0].closed
    with pytest.raises(httpx.ReadError, match="broken stream"):
        if async_:
            await iterator.__anext__()
        else:
            next(iterator)
    assert state["streams"][0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("async_", [False, True])
async def test_sse_nonstreaming_consumer(transport, async_):
    state = transport([wire([event([{"text": "first"}]), event([{"text": " last"}])])])
    model = (
        llm.get_async_model("gemini-flash-latest")
        if async_
        else llm.get_model("gemini-flash-latest")
    )
    response = model.prompt("hello", key="test-key", stream=False)
    text = await response.text() if async_ else response.text()
    assert text == "first last"
    assert all(s.closed for s in state["streams"])
