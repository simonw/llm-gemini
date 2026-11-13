import copy

import llm
import pytest

from llm_gemini import AsyncGeminiPro, GeminiPro, cleanup_schema

REFERENCE_NAMES = [
    ("ordinary", "ordinary"),
    ("slash/name", "slash~1name"),
    ("tilde~name", "tilde~0name"),
    ("literal~1name", "literal~01name"),
]


def _schema(name, token):
    definition = {"type": "string", "enum": ["first", "second"]}
    definitions = {name: definition}
    if token != name:
        # The encoded spelling is a different literal name, not the target.
        definitions[token] = {"type": "integer"}
    if name == "literal~1name":
        definitions["literal/name"] = {"type": "integer"}
    return {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/" + token}},
        "required": ["value"],
        "$defs": definitions,
    }


@pytest.mark.parametrize("model_class", [GeminiPro, AsyncGeminiPro])
@pytest.mark.parametrize("surface", ["response", "tool"])
@pytest.mark.parametrize("name,token", REFERENCE_NAMES)
def test_request_schema_resolves_escaped_definition_names(
    model_class, surface, name, token
):
    schema = _schema(name, token)
    original = copy.deepcopy(schema)
    model = model_class("gemini-flash-latest", can_schema=True)
    prompt = llm.Prompt(
        "Use the supplied schema",
        model,
        options=model.Options(),
        schema=schema if surface == "response" else None,
        tools=(
            [llm.Tool(name="lookup", input_schema=schema)]
            if surface == "tool"
            else None
        ),
    )

    body = model.build_request_body(prompt, conversation=None)
    if surface == "response":
        actual = body["generationConfig"]["response_schema"]
    else:
        actual = body["tools"][0]["functionDeclarations"][0]["parameters"]

    assert actual == {
        "type": "object",
        "properties": {"value": {"type": "string", "enum": ["first", "second"]}},
        "required": ["value"],
    }
    assert schema == original


def test_repeated_escaped_reference_is_not_treated_as_recursion():
    schema = {
        "type": "object",
        "properties": {
            "first": {"$ref": "#/$defs/a~1b"},
            "second": {"type": "array", "items": {"$ref": "#/$defs/a~1b"}},
        },
        "$defs": {"a/b": {"type": "integer"}},
    }

    assert cleanup_schema(schema) == {
        "type": "object",
        "properties": {
            "first": {"type": "integer"},
            "second": {"type": "array", "items": {"type": "integer"}},
        },
    }


@pytest.mark.parametrize("name,token", REFERENCE_NAMES)
def test_recursive_escaped_definition_is_rejected(name, token):
    schema = {
        "$ref": "#/$defs/" + token,
        "$defs": {
            name: {
                "type": "object",
                "properties": {"next": {"$ref": "#/$defs/" + token}},
            }
        },
    }

    with pytest.raises(ValueError, match="Recursive schema detected") as exc:
        cleanup_schema(schema)
    assert name in str(exc.value)
