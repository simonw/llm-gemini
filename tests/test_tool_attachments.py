import base64
import copy
import json
from contextlib import contextmanager

import httpx
import llm
import pytest
from llm.parts import AttachmentPart, Message, TextPart, ToolCallPart, ToolResultPart

MODEL = "gemini-2.5-flash"
KEY = "not-real"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8A"
    "AwMCAO+aVZkAAAAASUVORK5CYII="
)
CALL = {
    "function_call": {"name": "report", "args": {}, "id": "report-1"},
    "thoughtSignature": "call-signature",
}


def inline(content, mime_type):
    return {
        "inlineData": {
            "data": base64.b64encode(content).decode("ascii"),
            "mimeType": mime_type,
        }
    }


def function_response(output="Report ready", name="report", call_id="report-1"):
    result = {"name": name, "response": {"output": output}}
    if call_id:
        result["id"] = call_id
    return {"function_response": result}


@pytest.fixture
def image():
    return llm.Attachment(type="image/png", content=PNG)


@pytest.fixture
def wire(monkeypatch):
    requests = []

    def handler(request):
        assert request.method == "POST"
        assert request.url.path.endswith(":streamGenerateContent")
        body = json.loads(request.content)
        requests.append(body)
        has_result = any(
            "function_response" in part
            for content in body["contents"]
            for part in content["parts"]
        )
        part = {"text": "Reviewed"}
        if body.get("tools") and not has_result:
            part = {
                "functionCall": CALL["function_call"],
                "thoughtSignature": CALL["thoughtSignature"],
            }
        return httpx.Response(
            200,
            json=[
                {
                    "candidates": [{"content": {"role": "model", "parts": [part]}}],
                    "modelVersion": MODEL,
                }
            ],
        )

    transport = httpx.MockTransport(handler)
    client_class = httpx.Client
    async_client_class = httpx.AsyncClient

    @contextmanager
    def stream(method, url, **kwargs):
        with (
            client_class(transport=transport) as client,
            client.stream(method, url, **kwargs) as response,
        ):
            yield response

    def async_client(*args, **kwargs):
        return async_client_class(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "stream", stream)
    monkeypatch.setattr(httpx, "AsyncClient", async_client)
    return requests


def assert_manual_request(body):
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "Read the report"}]},
        {"role": "model", "parts": [CALL]},
        {
            "role": "user",
            "parts": [function_response(), inline(PNG, "image/png")],
        },
    ]


def test_manual_conversation_tool_attachment_and_replay(wire, image):
    def report():
        return llm.ToolOutput(output="Report ready", attachments=[image])

    conversation = llm.get_model(MODEL).conversation()
    first = conversation.prompt("Read the report", tools=[report], key=KEY)
    results = first.execute_tool_calls()
    assert results[0].attachments == [image]
    before = copy.deepcopy(results)
    second = conversation.prompt(tool_results=results, tools=[report], key=KEY)
    assert second.text() == "Reviewed"
    assert results == before
    assert_manual_request(wire[-1])
    previous = copy.deepcopy(wire[-1]["contents"])
    assert conversation.prompt("Explain", key=KEY).text() == "Reviewed"
    assert wire[-1]["contents"] == previous + [
        {"role": "model", "parts": [{"text": "Reviewed"}]},
        {"role": "user", "parts": [{"text": "Explain"}]},
    ]
    assert results == before


@pytest.mark.asyncio
async def test_async_manual_conversation_tool_attachment_and_replay(wire, image):
    async def report():
        return llm.ToolOutput(output="Report ready", attachments=[image])

    conversation = llm.get_async_model(MODEL).conversation()
    first = await conversation.prompt("Read the report", tools=[report], key=KEY)
    results = await first.execute_tool_calls()
    assert results[0].attachments == [image]
    before = copy.deepcopy(results)
    second = await conversation.prompt(tool_results=results, tools=[report], key=KEY)
    assert await second.text() == "Reviewed"
    assert results == before
    assert_manual_request(wire[-1])
    previous = copy.deepcopy(wire[-1]["contents"])
    third = await conversation.prompt("Explain", key=KEY)
    assert await third.text() == "Reviewed"
    assert wire[-1]["contents"] == previous + [
        {"role": "model", "parts": [{"text": "Reviewed"}]},
        {"role": "user", "parts": [{"text": "Explain"}]},
    ]
    assert results == before


def test_explicit_history_preserves_attachment_order_and_caller_data(wire, tmp_path):
    path = tmp_path / "chart.png"
    path.write_bytes(PNG)
    messages = [
        Message(role="system", parts=[TextPart(text="Review carefully")]),
        Message(
            role="assistant",
            parts=[
                TextPart(
                    text="Fetching",
                    provider_metadata={
                        "gemini": {"thoughtSignature": "text-signature"}
                    },
                ),
                ToolCallPart(
                    name="report",
                    arguments={"page": 2},
                    tool_call_id="report-1",
                    provider_metadata={
                        "gemini": {"thoughtSignature": "call-signature"}
                    },
                ),
                ToolCallPart(name="audio", tool_call_id="audio-2"),
            ],
        ),
        Message(
            role="tool",
            parts=[
                TextPart(text="Before"),
                ToolResultPart(
                    name="report",
                    output="",
                    tool_call_id="report-1",
                    attachments=[
                        llm.Attachment(path=str(path)),
                        llm.Attachment(type="application/pdf", content=b"%PDF-1.4\n"),
                    ],
                ),
                TextPart(text="Between"),
                ToolResultPart(
                    name="audio",
                    output="Sound",
                    tool_call_id="audio-2",
                    attachments=[llm.Attachment(type="audio/mpeg", content=b"audio")],
                ),
                AttachmentPart(
                    attachment=llm.Attachment(type="image/png", content=PNG)
                ),
                TextPart(text="After"),
            ],
        ),
    ]
    before = copy.deepcopy(messages)
    expected = [
        {
            "role": "model",
            "parts": [
                {"text": "Fetching", "thoughtSignature": "text-signature"},
                {
                    "function_call": {
                        "name": "report",
                        "args": {"page": 2},
                        "id": "report-1",
                    },
                    "thoughtSignature": "call-signature",
                },
                {"function_call": {"name": "audio", "args": {}, "id": "audio-2"}},
            ],
        },
        {
            "role": "user",
            "parts": [
                {"text": "Before"},
                function_response(""),
                inline(PNG, "image/png"),
                inline(b"%PDF-1.4\n", "application/pdf"),
                {"text": "Between"},
                function_response("Sound", "audio", "audio-2"),
                inline(b"audio", "audio/mp3"),
                inline(PNG, "image/png"),
                {"text": "After"},
            ],
        },
    ]
    for _ in range(2):
        assert (
            llm.get_model(MODEL).prompt(messages=messages, key=KEY).text() == "Reviewed"
        )
        assert messages == before
        assert wire[-1]["contents"] == expected


def assert_automatic_request(body):
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "Read the report"}]},
        {"role": "model", "parts": [CALL]},
        {"role": "user", "parts": [function_response()]},
        {"role": "user", "parts": [inline(PNG, "image/png")]},
    ]


@pytest.mark.parametrize("method", ["chain", "reply"])
def test_automatic_tool_attachment_sent_once(wire, image, method):
    calls = []

    def report():
        calls.append(True)
        return llm.ToolOutput(output="Report ready", attachments=[image])

    model = llm.get_model(MODEL)
    if method == "chain":
        response = model.chain("Read the report", tools=[report], key=KEY)
    else:
        first = model.prompt("Read the report", tools=[report], key=KEY)
        response = first.reply(key=KEY)
    assert response.text() == "Reviewed"
    assert calls == [True]
    assert len(wire) == 2
    assert_automatic_request(wire[-1])


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["chain", "reply"])
async def test_async_automatic_tool_attachment_sent_once(wire, image, method):
    calls = []

    async def report():
        calls.append(True)
        return llm.ToolOutput(output="Report ready", attachments=[image])

    model = llm.get_async_model(MODEL)
    if method == "chain":
        response = model.chain("Read the report", tools=[report], key=KEY)
    else:
        first = await model.prompt("Read the report", tools=[report], key=KEY)
        response = await first.reply(key=KEY)
    assert await response.text() == "Reviewed"
    assert calls == [True]
    assert len(wire) == 2
    assert_automatic_request(wire[-1])


@pytest.mark.parametrize("output,call_id", [("Plain result", "report-1"), ("", None)])
def test_plain_tool_result_encoding_unchanged(wire, output, call_id):
    result = llm.ToolResult(name="report", output=output, tool_call_id=call_id)
    response = llm.get_model(MODEL).prompt(tool_results=[result], key=KEY)
    assert response.text() == "Reviewed"
    assert wire[-1]["contents"] == [
        {"role": "user", "parts": [function_response(output, call_id=call_id)]}
    ]


@pytest.mark.parametrize("native", [True, False])
def test_server_tool_results_keep_native_replay_only(wire, native):
    raw = {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "done"}}
    result = ToolResultPart(
        name="code_execution",
        output="ignored",
        server_executed=True,
        attachments=[llm.Attachment(path="/nonexistent/server-attachment")],
        provider_metadata={"gemini": {"part": raw}} if native else None,
    )
    messages = [Message(role="assistant", parts=[result, TextPart(text="Done")])]
    before = copy.deepcopy(messages)
    response = llm.get_model(MODEL).prompt(messages=messages, key=KEY)
    assert response.text() == "Reviewed"
    assert messages == before
    assert wire[-1]["contents"] == [
        {"role": "model", "parts": ([raw] if native else []) + [{"text": "Done"}]}
    ]
    if native:
        wire[-1]["contents"][0]["parts"][0]["codeExecutionResult"]["output"] = "changed"
        assert messages == before


@pytest.mark.parametrize("missing_file", [False, True])
def test_invalid_tool_attachment_fails_before_post(wire, tmp_path, missing_file):
    if missing_file:
        attachment = llm.Attachment(
            type="image/png", path=str(tmp_path / "missing.png")
        )
        error = FileNotFoundError
        match = "missing.png"
    else:
        attachment = llm.Attachment()
        error = ValueError
        match = "Attachment has no type"
    result = llm.ToolResult(
        name="report", output="Report ready", attachments=[attachment]
    )
    before = copy.deepcopy(result)
    response = llm.get_model(MODEL).prompt(tool_results=[result], key=KEY)
    with pytest.raises(error, match=match):
        response.text()
    assert wire == []
    assert result == before
