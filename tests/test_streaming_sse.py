import contextlib
import json
import sys

import httpx
import llm
from llm.parts import ReasoningPart, TextPart, ToolCallPart
import pytest

import llm_gemini

SSE_URL_PREFIX = "https://generativelanguage.googleapis.com/v1beta/models/"


class SyncChunks(httpx.SyncByteStream):
    def __init__(self, chunks):
        self._chunks = chunks

    def __iter__(self):
        yield from self._chunks


class AsyncChunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


def sse(events, line_ending="\n", bom=False):
    "Encode Gemini streaming events the way ?alt=sse does."
    text = "".join(
        "data: " + json.dumps(event, ensure_ascii=False) + line_ending * 2
        for event in events
    )
    payload = text.encode("utf-8")
    return (b"\xef\xbb\xbf" + payload) if bom else payload


def install(monkeypatch, chunks, content_type="text/event-stream"):
    "Route sync and async HTTPX through a MockTransport serving these chunks."
    requests = []

    def sync_handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": content_type},
            stream=SyncChunks(chunks),
        )

    def async_handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": content_type},
            stream=AsyncChunks(chunks),
        )

    sync_transport = httpx.MockTransport(sync_handler)
    async_transport = httpx.MockTransport(async_handler)
    real_client = httpx.Client
    real_async_client = httpx.AsyncClient

    @contextlib.contextmanager
    def fake_stream(method, url, **kwargs):
        with real_client(transport=sync_transport) as client:
            with client.stream(method, url, **kwargs) as response:
                yield response

    def fake_async_client():
        return real_async_client(transport=async_transport)

    monkeypatch.setattr(llm_gemini.httpx, "stream", fake_stream)
    monkeypatch.setattr(llm_gemini.httpx, "AsyncClient", fake_async_client)
    return requests


def event(text="", **extra):
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "modelVersion": "gemini-test-resolved",
        **extra,
    }


def prompt_sync():
    model = llm.get_model("gemini-flash-latest")
    response = model.prompt("Say hi", key="test-key")
    return response


async def prompt_async():
    model = llm.get_async_model("gemini-flash-latest")
    return await model.prompt("Say hi", key="test-key")


def assert_stream_request(requests):
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url).startswith(SSE_URL_PREFIX)
    assert request.url.path.endswith(":streamGenerateContent")
    # The plugin must not change its default request to force alt=sse
    assert request.url.query == b""
    assert request.headers["x-goog-api-key"] == "test-key"


def test_sync_sse_text_and_usage(monkeypatch):
    chunks = [
        sse(
            [
                event("Hello"),
                event(
                    " world",
                    usageMetadata={
                        "promptTokenCount": 5,
                        "candidatesTokenCount": 7,
                        "thoughtsTokenCount": 2,
                        "totalTokenCount": 14,
                    },
                ),
            ]
        )
    ]
    requests = install(monkeypatch, chunks)
    response = prompt_sync()

    assert response.text() == "Hello world"
    assert_stream_request(requests)
    assert response.response_json["modelVersion"] == "gemini-test-resolved"
    assert response.resolved_model == "gemini-test-resolved"
    assert response.input_tokens == 5
    assert response.output_tokens == 9
    assert response.token_details["candidatesTokenCount"] == 7


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="httpcore async cleanup issue on 3.14"
)
@pytest.mark.asyncio
async def test_async_sse_text_and_usage(monkeypatch):
    chunks = [
        sse(
            [
                event("Hello"),
                event(
                    " world",
                    usageMetadata={
                        "promptTokenCount": 5,
                        "candidatesTokenCount": 7,
                        "thoughtsTokenCount": 2,
                    },
                ),
            ]
        )
    ]
    requests = install(monkeypatch, chunks)
    response = await prompt_async()

    assert await response.text() == "Hello world"
    assert_stream_request(requests)
    assert response.resolved_model == "gemini-test-resolved"
    assert response.input_tokens == 5
    assert response.output_tokens == 9


def test_content_type_detection_is_case_and_parameter_insensitive(monkeypatch):
    chunks = [sse([event("hi")])]
    install(monkeypatch, chunks, content_type="Text/Event-Stream; charset=utf-8")
    response = prompt_sync()
    assert response.text() == "hi"


def test_json_array_response_still_parses(monkeypatch):
    body = json.dumps([event("array "), event("response")]).encode("utf-8")
    requests = install(monkeypatch, [body], content_type="application/json")
    response = prompt_sync()
    assert response.text() == "array response"
    assert_stream_request(requests)


@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"])
def test_sse_line_endings(monkeypatch, line_ending):
    chunks = [sse([event("one "), event("two")], line_ending=line_ending)]
    install(monkeypatch, chunks)
    response = prompt_sync()
    assert response.text() == "one two"


def test_sse_line_endings_split_across_chunks(monkeypatch):
    # A CRLF pair split between two HTTP chunks must count as one boundary
    payload = sse([event("split")], line_ending="\r\n")
    chunks = [payload[i : i + 1] for i in range(len(payload))]
    install(monkeypatch, chunks)
    response = prompt_sync()
    assert response.text() == "split"


def test_sse_crlf_split_with_empty_chunk_between(monkeypatch):
    payload = sse([event("one "), event("two")], line_ending="\r\n")
    chunks = []
    for byte in payload:
        chunks.append(bytes([byte]))
        chunks.append(b"")
    install(monkeypatch, chunks)
    assert prompt_sync().text() == "one two"


def test_sse_bom(monkeypatch):
    chunks = [sse([event("bom")], bom=True)]
    install(monkeypatch, chunks)
    assert prompt_sync().text() == "bom"


def test_sse_bom_split_across_chunks(monkeypatch):
    payload = sse([event("bom")], bom=True)
    chunks = [payload[:1], payload[1:2], payload[2:]]
    install(monkeypatch, chunks)
    assert prompt_sync().text() == "bom"


def test_sse_comments_and_keepalives_ignored(monkeypatch):
    payload = (
        b": keepalive\n\n"
        b"event: message\n"
        b"id: 1\n\n"
        b"data: " + json.dumps(event("kept")).encode() + b"\n\n"
        b":another comment\n\n"
    )
    chunks = [payload]
    install(monkeypatch, chunks)
    assert prompt_sync().text() == "kept"


def test_sse_multiline_data_joined(monkeypatch):
    payload = json.dumps(event("joined"))
    split_at = payload.index(",") + 1
    wire = (
        "data: " + payload[:split_at] + "\n" "data: " + payload[split_at:] + "\n\n"
    ).encode("utf-8")
    chunks = [wire]
    install(monkeypatch, chunks)
    assert prompt_sync().text() == "joined"


def test_sse_incomplete_event_at_eof_discarded(monkeypatch):
    complete = sse([event("complete")])
    incomplete = b'data: {"candidates": [{"content": {"parts": [{"text": "dropped'
    chunks = [complete + incomplete]
    install(monkeypatch, chunks)
    response = prompt_sync()
    assert response.text() == "complete"


def test_sse_malformed_json_raises_model_error(monkeypatch):
    chunks = [b"data: {not json}\n\n"]
    install(monkeypatch, chunks)
    response = prompt_sync()
    with pytest.raises(llm.ModelError):
        response.text()


def test_sse_non_object_payload_raises_model_error(monkeypatch):
    chunks = [b'data: ["not", "an", "object"]\n\n']
    install(monkeypatch, chunks)
    response = prompt_sync()
    with pytest.raises(llm.ModelError):
        response.text()


def test_sse_without_data_raises_model_error(monkeypatch):
    chunks = [b": keepalive\n\n"]
    install(monkeypatch, chunks)
    response = prompt_sync()
    with pytest.raises(llm.ModelError):
        response.text()


def test_sse_error_event_raises_model_error(monkeypatch):
    chunks = [sse([{"error": {"message": "quota exhausted"}}])]
    install(monkeypatch, chunks)
    response = prompt_sync()
    with pytest.raises(llm.ModelError, match="quota exhausted"):
        response.text()


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="httpcore async cleanup issue on 3.14"
)
@pytest.mark.asyncio
async def test_async_sse_malformed_json_raises_model_error(monkeypatch):
    chunks = [b"data: {not json}\n\n"]
    install(monkeypatch, chunks)
    with pytest.raises(llm.ModelError):
        await (await prompt_async()).text()


def test_sse_unicode_and_grounding_floats(monkeypatch):
    grounding = {
        "groundingChunks": [{"web": {"uri": "https://example.com/"}}],
        "groundingSupports": [{"confidenceScores": [0.98]}],
    }
    payload = event("안녕 🐦")
    payload["candidates"][0]["groundingMetadata"] = grounding
    chunks = [sse([payload])]
    install(monkeypatch, chunks)
    response = prompt_sync()

    assert response.text() == "안녕 🐦"
    scores = (
        response.messages()[0]
        .parts[0]
        .provider_metadata["gemini"]["groundingMetadata"]["groundingSupports"][0][
            "confidenceScores"
        ]
    )
    assert isinstance(scores[0], float)
    assert scores[0] == pytest.approx(0.98)


def test_sse_reasoning_and_tool_call(monkeypatch):
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "thinking it over", "thought": True},
                        {"text": "The answer is 6."},
                        {
                            "functionCall": {
                                "name": "multiply",
                                "args": {"a": 2, "b": 3},
                                "id": "call-1",
                            },
                            "thoughtSignature": "signature-1",
                        },
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "modelVersion": "gemini-test-resolved",
    }
    chunks = [sse([payload])]
    install(monkeypatch, chunks)
    response = prompt_sync()

    assert response.text() == "The answer is 6."
    parts = response.messages()[0].parts
    assert isinstance(parts[0], ReasoningPart)
    assert isinstance(parts[1], TextPart)
    assert isinstance(parts[2], ToolCallPart)
    tool_calls = response.tool_calls()
    assert [call.name for call in tool_calls] == ["multiply"]
    assert tool_calls[0].arguments == {"a": 2, "b": 3}
    assert tool_calls[0].thought_signature == "signature-1"


@pytest.mark.parametrize("chunk_size", [1, 7])
def test_sse_byte_chunked_delivery(monkeypatch, chunk_size):
    payload = sse([event("chunked 안녕 🐦")])
    chunks = [payload[i : i + chunk_size] for i in range(0, len(payload), chunk_size)]
    install(monkeypatch, chunks)
    assert prompt_sync().text() == "chunked 안녕 🐦"


def test_sse_stream_false(monkeypatch):
    chunks = [sse([event("no "), event("stream")])]
    install(monkeypatch, chunks)
    model = llm.get_model("gemini-flash-latest")
    response = model.prompt("Say hi", key="test-key", stream=False)
    assert response.text() == "no stream"


def test_sse_decoder_dispatches_incrementally():
    events = []
    decoder = llm_gemini._SSEDecoder(events)
    decoder.send(b'data: {"candidates": []}\n\n')
    assert events == [{"candidates": []}]
    decoder.send(b'data: {"candidates": [')
    assert len(events) == 1
    decoder.send(b'{"content": {"parts": []}}]}\n\n')
    assert len(events) == 2
