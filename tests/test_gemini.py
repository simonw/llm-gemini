from click.testing import CliRunner
import llm
from llm.cli import cli
from llm.parts import StreamEvent
import json
import os
import pytest
import pydantic
from pydantic import BaseModel
import sys
from typing import List, Optional
from llm_gemini import cleanup_schema, is_youtube_url

GEMINI_API_KEY = os.environ.get("PYTEST_GEMINI_API_KEY", None) or "gm-..."


@pytest.mark.vcr
def test_prompt():
    model = llm.get_model("gemini-flash-latest")
    response = model.prompt("Name for a pet pelican, just the name", key=GEMINI_API_KEY)
    assert str(response).strip()
    assert response.response_json["candidates"][0]["finishReason"] == "STOP"
    assert response.response_json["modelVersion"] == "gemini-3.6-flash"
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.token_details["candidatesTokenCount"] > 0


# Skip async test on Python 3.14 due to httpcore cleanup incompatibility
# https://github.com/encode/httpcore/issues - AsyncLibraryNotFoundError during __aexit__
@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="httpcore async cleanup issue on 3.14"
)
@pytest.mark.vcr
@pytest.mark.asyncio
async def test_prompt_async():
    async_model = llm.get_async_model("gemini-flash-latest")
    response = await async_model.prompt(
        "Name for a pet pelican, just the name", key=GEMINI_API_KEY
    )
    text = await response.text()
    assert text.strip()
    assert response.resolved_model == response.response_json["modelVersion"]


@pytest.mark.vcr
def test_prompt_with_pydantic_schema():
    class Dog(pydantic.BaseModel):
        name: str
        age: int
        bio: str

    model = llm.get_model("gemini-flash-latest")
    response = model.prompt(
        "Invent a cool dog", key=GEMINI_API_KEY, schema=Dog, stream=False
    )
    dog = json.loads(response.text())
    assert set(dog) == {"name", "age", "bio"}
    assert isinstance(dog["name"], str)
    assert isinstance(dog["age"], int)
    assert isinstance(dog["bio"], str)
    assert response.response_json["candidates"][0]["finishReason"] == "STOP"
    assert response.response_json["modelVersion"] == "gemini-3.6-flash"
    assert response.input_tokens > 0


@pytest.mark.vcr
def test_prompt_with_multiple_dogs():
    class Dog(pydantic.BaseModel):
        name: str
        age: int
        bio: str

    class Dogs(BaseModel):
        dogs: List[Dog]

    model = llm.get_model("gemini-flash-latest")
    response = model.prompt(
        "Invent 3 cool dogs", key=GEMINI_API_KEY, schema=Dogs, stream=False
    )
    result = json.loads(response.text())

    # Verify we got 3 dogs
    assert "dogs" in result
    assert len(result["dogs"]) == 3

    # Verify each dog has the required fields
    for dog in result["dogs"]:
        assert "name" in dog
        assert "age" in dog
        assert "bio" in dog
        assert isinstance(dog["name"], str)
        assert isinstance(dog["age"], int)
        assert isinstance(dog["bio"], str)


@pytest.mark.vcr
@pytest.mark.parametrize(
    "model_id",
    (
        "gemini-embedding-001",
        "gemini-embedding-001-768",
        "gemini-embedding-2",
        "gemini-embedding-2-768",
    ),
)
def test_embedding(model_id, monkeypatch):
    monkeypatch.setenv("LLM_GEMINI_KEY", GEMINI_API_KEY)
    model = llm.get_embedding_model(model_id)
    response = model.embed("Some text goes here")
    expected_length = 768 if model_id.endswith("-768") else 3072
    assert len(response) == expected_length
    magnitude = sum(component**2 for component in response) ** 0.5
    assert magnitude == pytest.approx(1.0, abs=1e-6)


@pytest.mark.vcr
def test_embedding_batch(monkeypatch):
    monkeypatch.setenv("LLM_GEMINI_KEY", GEMINI_API_KEY)
    model = llm.get_embedding_model("gemini-embedding-2-768")
    responses = model.embed_batch(["First text", "Second text"])
    assert len(responses) == 2
    assert all(len(response) == 768 for response in responses)


def test_embedding_models():
    dimensions = (768, 1536)
    for gemini_model_id in ("gemini-embedding-2", "gemini-embedding-001"):
        model = llm.get_embedding_model(gemini_model_id)
        assert model.gemini_model_id == gemini_model_id
        assert model.output_dimensionality is None
        for dimension in dimensions:
            model = llm.get_embedding_model(f"{gemini_model_id}-{dimension}")
            assert model.gemini_model_id == gemini_model_id
            assert model.output_dimensionality == dimension


@pytest.mark.parametrize(
    "schema,expected",
    [
        # Test 1: Top-level keys removal
        (
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "title": "Example Schema",
                "additionalProperties": False,
                "type": "object",
            },
            {"type": "object"},
        ),
        # Test 2: Preserve keys within a "properties" block
        (
            {
                "type": "object",
                "properties": {
                    "authors": {"type": "string"},
                    "title": {"type": "string"},
                    "reference": {"type": "string"},
                    "year": {"type": "string"},
                },
                "title": "This should be removed from the top-level",
            },
            {
                "type": "object",
                "properties": {
                    "authors": {"type": "string"},
                    "title": {"type": "string"},
                    "reference": {"type": "string"},
                    "year": {"type": "string"},
                },
            },
        ),
        # Test 3: Nested keys outside and inside properties block
        (
            {
                "definitions": {
                    "info": {
                        "title": "Info title",  # should be removed because it's not inside a "properties" block
                        "description": "A description",
                        "properties": {
                            "name": {
                                "title": "Name Title",
                                "type": "string",
                            },  # title here should be preserved
                            "$schema": {
                                "type": "string"
                            },  # should be preserved as it's within properties
                        },
                    }
                },
                "$schema": "http://example.com/schema",
            },
            {
                "definitions": {
                    "info": {
                        "description": "A description",
                        "properties": {
                            "name": {"title": "Name Title", "type": "string"},
                            "$schema": {"type": "string"},
                        },
                    }
                }
            },
        ),
        # Test 4: List of schemas
        (
            [
                {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                },
                {"title": "Should be removed", "type": "array"},
            ],
            [{"type": "object"}, {"type": "array"}],
        ),
    ],
)
def test_cleanup_schema(schema, expected):
    # Use a deep copy so the original test data remains unchanged.
    result = cleanup_schema(schema)
    assert result == expected


# Tests for $ref resolution - patterns that now work with nested models
@pytest.mark.parametrize(
    "schema,expected",
    [
        # Test 1: Direct model reference (Person with Address)
        (
            {
                "properties": {
                    "name": {"type": "string"},
                    "address": {"$ref": "#/$defs/Address"},
                },
                "required": ["name", "address"],
                "type": "object",
                "$defs": {
                    "Address": {
                        "properties": {
                            "street": {"type": "string"},
                            "city": {"type": "string"},
                        },
                        "required": ["street", "city"],
                        "type": "object",
                    }
                },
            },
            {
                "properties": {
                    "name": {"type": "string"},
                    "address": {
                        "properties": {
                            "street": {"type": "string"},
                            "city": {"type": "string"},
                        },
                        "required": ["street", "city"],
                        "type": "object",
                    },
                },
                "required": ["name", "address"],
                "type": "object",
            },
        ),
        # Test 2: List of models (Dogs with List[Dog])
        (
            {
                "properties": {
                    "dogs": {"items": {"$ref": "#/$defs/Dog"}, "type": "array"}
                },
                "required": ["dogs"],
                "type": "object",
                "$defs": {
                    "Dog": {
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                        "required": ["name", "age"],
                        "type": "object",
                    }
                },
            },
            {
                "properties": {
                    "dogs": {
                        "items": {
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer"},
                            },
                            "required": ["name", "age"],
                            "type": "object",
                        },
                        "type": "array",
                    }
                },
                "required": ["dogs"],
                "type": "object",
            },
        ),
        # Test 3: Optional model field
        (
            {
                "properties": {
                    "name": {"type": "string"},
                    "employer": {
                        "anyOf": [{"$ref": "#/$defs/Company"}, {"type": "null"}]
                    },
                },
                "required": ["name"],
                "type": "object",
                "$defs": {
                    "Company": {
                        "properties": {"company_name": {"type": "string"}},
                        "required": ["company_name"],
                        "type": "object",
                    }
                },
            },
            {
                "properties": {
                    "name": {"type": "string"},
                    "employer": {
                        "anyOf": [
                            {
                                "properties": {"company_name": {"type": "string"}},
                                "required": ["company_name"],
                                "type": "object",
                            },
                            {"type": "null"},
                        ]
                    },
                },
                "required": ["name"],
                "type": "object",
            },
        ),
        # Test 4: Nested composition (Customer -> List[Order] -> List[Item])
        (
            {
                "properties": {
                    "name": {"type": "string"},
                    "orders": {"items": {"$ref": "#/$defs/Order"}, "type": "array"},
                },
                "required": ["name", "orders"],
                "type": "object",
                "$defs": {
                    "Order": {
                        "properties": {
                            "items": {
                                "items": {"$ref": "#/$defs/Item"},
                                "type": "array",
                            }
                        },
                        "required": ["items"],
                        "type": "object",
                    },
                    "Item": {
                        "properties": {
                            "product_name": {"type": "string"},
                            "quantity": {"type": "integer"},
                        },
                        "required": ["product_name", "quantity"],
                        "type": "object",
                    },
                },
            },
            {
                "properties": {
                    "name": {"type": "string"},
                    "orders": {
                        "items": {
                            "properties": {
                                "items": {
                                    "items": {
                                        "properties": {
                                            "product_name": {"type": "string"},
                                            "quantity": {"type": "integer"},
                                        },
                                        "required": ["product_name", "quantity"],
                                        "type": "object",
                                    },
                                    "type": "array",
                                }
                            },
                            "required": ["items"],
                            "type": "object",
                        },
                        "type": "array",
                    },
                },
                "required": ["name", "orders"],
                "type": "object",
            },
        ),
    ],
)
def test_cleanup_schema_with_refs(schema, expected):
    """Test that $ref resolution works for various nested model patterns."""
    import copy

    result = cleanup_schema(copy.deepcopy(schema))
    assert result == expected


def test_recursive_schema_raises_error():
    """Test that recursive self-referential schemas raise a descriptive error."""
    # Simulate a recursive Node class: Node has a field 'next' that references Node
    recursive_schema = {
        "properties": {
            "value": {"type": "string"},
            "next": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]},
        },
        "required": ["value"],
        "type": "object",
        "$defs": {
            "Node": {
                "properties": {
                    "value": {"type": "string"},
                    "next": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]},
                },
                "required": ["value"],
                "type": "object",
            }
        },
    }

    import copy

    with pytest.raises(ValueError) as exc_info:
        cleanup_schema(copy.deepcopy(recursive_schema))

    error_message = str(exc_info.value)
    assert "Recursive schema detected" in error_message
    assert "Node" in error_message
    assert "directly references itself" in error_message
    assert "Gemini API does not support recursive Pydantic models" in error_message


def test_indirect_recursive_schema_raises_error():
    """Test that indirect recursion (A -> B -> A) is detected and raises an error."""
    # Simulate class A with a field of type B, and class B with a field of type A
    indirect_recursive_schema = {
        "properties": {"name": {"type": "string"}, "b_field": {"$ref": "#/$defs/B"}},
        "required": ["name"],
        "type": "object",
        "$defs": {
            "A": {
                "properties": {
                    "name": {"type": "string"},
                    "b_field": {"$ref": "#/$defs/B"},
                },
                "required": ["name"],
                "type": "object",
            },
            "B": {
                "properties": {
                    "id": {"type": "integer"},
                    "a_field": {"$ref": "#/$defs/A"},
                },
                "required": ["id"],
                "type": "object",
            },
        },
    }

    import copy

    with pytest.raises(ValueError) as exc_info:
        cleanup_schema(copy.deepcopy(indirect_recursive_schema))

    error_message = str(exc_info.value)
    assert "Recursive schema detected" in error_message
    assert "indirectly references itself through" in error_message
    # Should mention both types involved in the cycle
    assert "A" in error_message and "B" in error_message
    assert "Gemini API does not support recursive Pydantic models" in error_message


def test_recursive_pydantic_model_raises_error():
    """Test that recursive Pydantic models raise an error when schema is generated at test time.

    This test complements test_recursive_schema_raises_error by using actual Pydantic
    classes rather than hand-crafted JSON schemas. If Pydantic changes how it generates
    schemas in a future version, this test will catch any incompatibility.
    """

    class Node(BaseModel):
        value: str
        next: Optional["Node"] = None

    # Generate schema from Pydantic model at test time
    pydantic_schema = Node.model_json_schema()

    import copy

    with pytest.raises(ValueError) as exc_info:
        cleanup_schema(copy.deepcopy(pydantic_schema))

    error_message = str(exc_info.value)
    assert "Recursive schema detected" in error_message
    assert "directly references itself" in error_message
    assert "Gemini API does not support recursive Pydantic models" in error_message


def test_indirect_recursive_pydantic_models_raise_error():
    """Test that indirectly recursive Pydantic models are detected when schema is generated at test time.

    This test complements test_indirect_recursive_schema_raises_error by using actual
    Pydantic classes rather than hand-crafted JSON schemas. If Pydantic changes how it
    generates schemas in a future version, this test will catch any incompatibility.
    """

    class B(BaseModel):
        id: int
        a_field: "A"

    class A(BaseModel):
        name: str
        b_field: B

    # Generate schema from Pydantic model at test time
    pydantic_schema = A.model_json_schema()

    import copy

    with pytest.raises(ValueError) as exc_info:
        cleanup_schema(copy.deepcopy(pydantic_schema))

    error_message = str(exc_info.value)
    assert "Recursive schema detected" in error_message
    assert "indirectly references itself through" in error_message
    # Should mention both types involved in the cycle
    assert "A" in error_message and "B" in error_message
    assert "Gemini API does not support recursive Pydantic models" in error_message


# Integration tests with real Pydantic models
@pytest.mark.vcr
def test_nested_model_direct_reference():
    """Test Pattern 1: Direct model reference (Person with Address)"""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    model = llm.get_model("gemini-flash-latest")
    response = model.prompt(
        "Create a person named Alice living in San Francisco",
        key=GEMINI_API_KEY,
        schema=Person,
        stream=False,
    )
    result = json.loads(response.text())
    assert "name" in result
    assert "address" in result
    assert "street" in result["address"]
    assert "city" in result["address"]


@pytest.mark.vcr
def test_nested_model_list():
    """Test Pattern 2: List of models (already covered by test_prompt_with_multiple_dogs)"""
    pass  # Covered by test_prompt_with_multiple_dogs


@pytest.mark.vcr
def test_nested_model_optional():
    """Test Pattern 3: Optional model field"""

    class Company(BaseModel):
        company_name: str

    class Person(BaseModel):
        name: str
        employer: Optional[Company]

    model = llm.get_model("gemini-flash-latest")
    response = model.prompt(
        "Create a person named Bob who works at TechCorp",
        key=GEMINI_API_KEY,
        schema=Person,
        stream=False,
    )
    result = json.loads(response.text())
    assert "name" in result
    assert "employer" in result
    if result["employer"] is not None:
        assert "company_name" in result["employer"]


@pytest.mark.vcr
def test_nested_model_deep_composition():
    """Test Pattern 4: Nested composition (Customer -> Orders -> Items)"""

    class Item(BaseModel):
        product_name: str
        quantity: int

    class Order(BaseModel):
        items: List[Item]

    class Customer(BaseModel):
        name: str
        orders: List[Order]

    model = llm.get_model("gemini-flash-latest")
    response = model.prompt(
        "Create a customer named Carol with 2 orders, each containing 2 items",
        key=GEMINI_API_KEY,
        schema=Customer,
        stream=False,
    )
    result = json.loads(response.text())
    assert "name" in result
    assert "orders" in result
    assert len(result["orders"]) > 0
    for order in result["orders"]:
        assert "items" in order
        assert len(order["items"]) > 0
        for item in order["items"]:
            assert "product_name" in item
            assert "quantity" in item


@pytest.mark.vcr
def test_cli_gemini_models(tmpdir, monkeypatch):
    user_dir = tmpdir / "llm.datasette.io"
    user_dir.mkdir()
    monkeypatch.setenv("LLM_USER_PATH", str(user_dir))
    # With no key set should error nicely
    runner = CliRunner()
    result = runner.invoke(cli, ["gemini", "models"])
    assert result.exit_code == 1
    assert (
        "Error: You must set the LLM_GEMINI_KEY environment variable or use --key\n"
        == result.output
    )
    # Try again with --key
    result2 = runner.invoke(cli, ["gemini", "models", "--key", GEMINI_API_KEY])
    assert result2.exit_code == 0
    assert "gemini-3.6-flash" in result2.output
    # And with --method
    result3 = runner.invoke(
        cli, ["gemini", "models", "--key", GEMINI_API_KEY, "--method", "embedContent"]
    )
    assert result3.exit_code == 0
    models = json.loads(result3.output)
    for model in models:
        assert "embedContent" in model["supportedGenerationMethods"]


@pytest.mark.vcr
def test_resolved_model():
    model = llm.get_model("gemini-flash-latest")
    response = model.prompt("hi", key=GEMINI_API_KEY)
    response.text()
    assert response.resolved_model == "gemini-3.6-flash"


@pytest.mark.vcr
def test_tools():
    model = llm.get_model("gemini-2.5-flash")
    names = ["Charles", "Sammy"]
    chain_response = model.chain(
        "Two names for a pet pelican",
        tools=[
            llm.Tool.function(lambda: names.pop(0), name="pelican_name_generator"),
        ],
        key=GEMINI_API_KEY,
    )
    text = chain_response.text()
    assert "Charles" in text
    assert "Sammy" in text
    # This one did three
    assert len(chain_response._responses) == 3
    first, second, third = chain_response._responses
    assert len(first.tool_calls()) == 1
    assert first.tool_calls()[0].name == "pelican_name_generator"
    assert len(second.tool_calls()) == 1
    assert second.tool_calls()[0].name == "pelican_name_generator"
    assert second.prompt.tool_results[0].output == "Charles"
    assert third.prompt.tool_results[0].output == "Sammy"


@pytest.mark.vcr
def test_tools_with_nested_pydantic_models():
    """Test that tools with nested Pydantic model schemas work correctly.

    This verifies that the fix from PR #107 is applied to tool schemas,
    ensuring $ref references in tool input schemas are properly resolved.
    """

    class Address(BaseModel):
        """Address information"""

        street: str
        city: str
        zipcode: str

    class PersonInput(BaseModel):
        """Input for creating a person with address"""

        name: str
        age: int
        address: Address

    # Mock database of people
    people_db = []

    def add_person(name: str, age: int, address: dict) -> str:
        """Add a person with their address to the database"""
        people_db.append({"name": name, "age": age, "address": address})
        return (
            f"Added {name} (age {age}) living at {address['street']}, {address['city']}"
        )

    model = llm.get_model("gemini-flash-latest")

    # Create a tool with nested Pydantic model schema
    # Convert Pydantic model to JSON schema
    input_schema = PersonInput.model_json_schema()

    add_person_tool = llm.Tool(
        name="add_person",
        description="Add a person with their address to the database",
        input_schema=input_schema,
        implementation=add_person,
    )

    chain_response = model.chain(
        "Add Alice who is 30 years old and lives at 123 Main St, San Francisco, CA 94102 to the database",
        tools=[add_person_tool],
        key=GEMINI_API_KEY,
    )

    text = chain_response.text()

    # Verify the tool was called
    assert len(chain_response._responses) >= 2
    first_response = chain_response._responses[0]
    assert len(first_response.tool_calls()) == 1

    tool_call = first_response.tool_calls()[0]
    assert tool_call.name == "add_person"

    # Verify the nested address structure was properly parsed
    assert "name" in tool_call.arguments
    assert "age" in tool_call.arguments
    assert "address" in tool_call.arguments
    assert isinstance(tool_call.arguments["address"], dict)
    assert "street" in tool_call.arguments["address"]
    assert "city" in tool_call.arguments["address"]
    assert "zipcode" in tool_call.arguments["address"]

    # Verify the person was added to the database
    assert len(people_db) == 1
    assert people_db[0]["name"] == tool_call.arguments["name"]
    assert people_db[0]["age"] == tool_call.arguments["age"]


def test_youtube_url_detection():
    assert is_youtube_url("https://www.youtube.com/watch?v=abc123")
    assert is_youtube_url("https://youtu.be/abc123")
    assert is_youtube_url("https://www.youtube.com/embed/abc123")
    assert is_youtube_url("http://www.youtube.com/watch?v=abc123")
    assert is_youtube_url("http://youtu.be/abc123")
    assert is_youtube_url("https://www.youtube.com/watch?v=abc123&feature=share")
    assert not is_youtube_url("https://example.com/video.mp4")
    assert not is_youtube_url("https://vimeo.com/123456")
    assert not is_youtube_url("https://www.youtube.com/user/username")
    assert not is_youtube_url("https://www.youtube.com/")
    assert not is_youtube_url(None)


def test_gemini_3_flash_has_all_thinking_levels():
    """Gemini 3 Flash should support minimal, low, medium, high thinking levels."""
    import typing

    model = llm.get_model("gemini-3-flash-preview")
    options_class = model.Options

    # Check that thinking_level field exists
    assert "thinking_level" in options_class.model_fields

    # Get the allowed values from the field's annotation
    field_info = options_class.model_fields["thinking_level"]
    # The annotation should be Optional[SomeEnum]
    annotation = field_info.annotation
    # Extract the enum from Optional[enum]
    args = typing.get_args(annotation)
    thinking_enum = args[0] if args else annotation

    # Check all 4 levels are available
    level_values = {e.value for e in thinking_enum}
    assert level_values == {"minimal", "low", "medium", "high"}


def test_gemini_31_pro_has_thinking_levels():
    """Gemini 3.1 Pro should support low, medium and high thinking levels."""
    import typing

    model = llm.get_model("gemini-3.1-pro-preview")
    options_class = model.Options

    # Check that thinking_level field exists
    assert "thinking_level" in options_class.model_fields

    # Get the allowed values from the field's annotation
    field_info = options_class.model_fields["thinking_level"]
    annotation = field_info.annotation
    args = typing.get_args(annotation)
    thinking_enum = args[0] if args else annotation

    # Check all 3 supported levels are available
    level_values = {e.value for e in thinking_enum}
    assert level_values == {"low", "medium", "high"}


def test_gemini_25_flash_has_thinking_budget_not_level():
    """Gemini 2.5 Flash should have thinking_budget but not thinking_level."""
    model = llm.get_model("gemini-2.5-flash")
    options_class = model.Options

    assert "thinking_budget" in options_class.model_fields
    assert "thinking_level" not in options_class.model_fields


def test_gemma_4_has_neither_thinking_option():
    """Gemma 4 should have neither thinking_budget nor thinking_level."""
    model = llm.get_model("gemma-4-26b-a4b-it")
    options_class = model.Options

    assert "thinking_budget" not in options_class.model_fields
    assert "thinking_level" not in options_class.model_fields


def test_thinking_level_in_request_body():
    """Thinking level should be included in the request body when set."""
    model = llm.get_model("gemini-3-flash-preview")

    # Create a mock prompt with thinking_level option
    class MockPrompt:
        prompt = "test"
        system = None
        attachments = []
        tools = None
        schema = None
        tool_results = None

    mock_prompt = MockPrompt()
    mock_prompt.options = model.Options(thinking_level="high")

    body = model.build_request_body(mock_prompt, None)

    assert "generationConfig" in body
    assert "thinkingConfig" in body["generationConfig"]
    assert body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "high"


def test_thought_summaries_in_request_by_default():
    """Thinking models should request visible thought summaries by default."""
    model = llm.get_model("gemini-3-flash-preview")

    class MockPrompt:
        prompt = "test"
        system = None
        attachments = []
        tools = None
        schema = None
        tool_results = None

    mock_prompt = MockPrompt()
    mock_prompt.options = model.Options()

    body = model.build_request_body(mock_prompt, None)

    assert body["generationConfig"]["thinkingConfig"] == {"includeThoughts": True}


def test_thought_summaries_not_in_request_when_reasoning_hidden():
    """hide_reasoning should disable visible Gemini thought summaries."""
    model = llm.get_model("gemini-3-flash-preview")

    class MockPrompt:
        prompt = "test"
        system = None
        attachments = []
        tools = None
        schema = None
        tool_results = None
        hide_reasoning = True

    mock_prompt = MockPrompt()
    mock_prompt.options = model.Options()

    body = model.build_request_body(mock_prompt, None)

    if "generationConfig" in body:
        assert "thinkingConfig" not in body["generationConfig"]


def test_thought_part_yields_reasoning_stream_event():
    """Gemini thought text parts should be surfaced as LLM reasoning events."""
    model = llm.get_model("gemini-3-flash-preview")

    class MockResponse:
        def add_tool_call(self, tool_call):
            raise AssertionError("No tool call expected")

    events = list(
        model.process_part(
            {"thought": True, "text": "I should inspect the table first."},
            MockResponse(),
        )
    )

    assert len(events) == 1
    assert events[0].type == "reasoning"
    assert events[0].chunk == "I should inspect the table first."


def test_empty_thought_part_yields_no_stream_event():
    """Empty Gemini thought text should not open an empty reasoning block."""
    model = llm.get_model("gemini-3-flash-preview")

    class MockResponse:
        def add_tool_call(self, tool_call):
            raise AssertionError("No tool call expected")

    events = list(model.process_part({"thought": True, "text": ""}, MockResponse()))

    assert events == []


@pytest.mark.vcr
def test_tools_with_gemini_3_thought_signatures():
    """Test that tools work with Gemini 3 models which require thought signatures.

    Gemini 3 models return thoughtSignature with function calls, and these must be
    included when sending function responses back to the model.
    """
    model = llm.get_model("gemini-3-flash-preview")

    def multiply(x: int, y: int) -> int:
        """Multiply two numbers."""
        return x * y

    chain_response = model.chain(
        "What is 5 times 3?",
        tools=[
            llm.Tool.function(multiply, name="multiply"),
        ],
        key=GEMINI_API_KEY,
    )
    text = chain_response.text()

    # Verify the tool was called and the response mentions the result
    assert len(chain_response._responses) >= 2
    first_response = chain_response._responses[0]
    assert len(first_response.tool_calls()) == 1
    assert first_response.tool_calls()[0].name == "multiply"
    # The result should be 15
    assert "15" in text


def test_build_messages_replays_stateless_history():
    """Explicit messages should retain prior turns without a Conversation."""
    from llm.parts import Message, TextPart

    model = llm.get_model("gemini-2.5-flash")
    chain = [
        Message(role="user", parts=[TextPart(text="How many products in Germany?")]),
        Message(role="assistant", parts=[TextPart(text="2,100 products.")]),
        Message(role="user", parts=[TextPart(text="What was my first question?")]),
    ]

    class MockPrompt:
        prompt = "What was my first question?"
        system = None
        attachments = []
        tools = None
        schema = None
        tool_results = None
        messages = chain

    contents = model.build_messages(MockPrompt(), None)

    assert contents[0] == {
        "role": "user",
        "parts": [{"text": "How many products in Germany?"}],
    }
    assert contents[1] == {
        "role": "model",
        "parts": [{"text": "2,100 products."}],
    }
    assert contents[-1] == {
        "role": "user",
        "parts": [{"text": "What was my first question?"}],
    }


def test_hosted_tool_options_are_removed():
    model = llm.get_model("gemini-3.6-flash")

    assert {
        "google_search",
        "url_context",
        "code_execution",
        "grounding_links",
        "format_links",
    }.isdisjoint(model.Options.model_fields)


@pytest.mark.parametrize(
    "model_id,expected_tools",
    (
        (
            "gemini-3.6-flash",
            {"GoogleSearch", "URLContext", "CodeExecution"},
        ),
        (
            "gemini-2.5-flash",
            {"GoogleSearch", "URLContext", "CodeExecution"},
        ),
        (
            "gemini-flash-latest",
            {"GoogleSearch", "URLContext", "CodeExecution"},
        ),
        ("gemma-4-26b-a4b-it", set()),
    ),
)
def test_supported_server_side_tools(model_id, expected_tools):
    model = llm.get_model(model_id)

    assert {
        tool.__name__ for tool in model.supported_server_side_tools
    } == expected_tools


def test_server_side_tool_request_specs_and_config():
    from llm_gemini import CodeExecution, GoogleSearch, URLContext

    model = llm.get_model("gemini-3.6-flash")
    prompt = llm.Prompt(
        "Research this and check the result with code",
        model,
        options=model.Options(),
        tools=[GoogleSearch(), URLContext(), CodeExecution()],
    )

    body = model.build_request_body(prompt, None)

    assert body["tools"] == [
        {"googleSearch": {}},
        {"urlContext": {}},
        {"codeExecution": {}},
    ]
    assert body["toolConfig"] == {
        "includeServerSideToolInvocations": True,
        "functionCallingConfig": {"mode": "VALIDATED"},
    }


@pytest.mark.parametrize(
    "model_id", ("gemini-flash-latest", "gemini-flash-lite-latest")
)
def test_latest_aliases_enable_server_tool_context(model_id):
    from llm_gemini import GoogleSearch

    model = llm.get_model(model_id)
    local_tool = llm.Tool(name="local_tool", input_schema={"type": "object"})
    prompt = llm.Prompt(
        "Search for pelicans",
        model,
        options=model.Options(),
        tools=[GoogleSearch(), local_tool],
    )

    body = model.build_request_body(prompt, None)

    assert body["toolConfig"]["includeServerSideToolInvocations"] is True
    assert body["tools"][1]["functionDeclarations"][0]["name"] == "local_tool"


def test_gemini_3_combines_function_and_server_side_tools():
    from llm_gemini import GoogleSearch

    model = llm.get_model("gemini-3.6-flash")
    weather = llm.Tool(
        name="weather",
        description="Look up the weather for a city",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    prompt = llm.Prompt(
        "Compare sources with the weather tool",
        model,
        options=model.Options(),
        tools=[weather, GoogleSearch()],
    )

    body = model.build_request_body(prompt, None)

    assert body["tools"] == [
        {"googleSearch": {}},
        {
            "functionDeclarations": [
                {
                    "name": "weather",
                    "description": "Look up the weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ]
        },
    ]
    assert body["toolConfig"] == {
        "includeServerSideToolInvocations": True,
        "functionCallingConfig": {"mode": "VALIDATED"},
    }


def test_pre_gemini_3_rejects_mixed_function_and_server_side_tools():
    from llm_gemini import GoogleSearch

    model = llm.get_model("gemini-2.5-flash")
    local_tool = llm.Tool(name="local_tool", input_schema={"type": "object"})
    prompt = llm.Prompt(
        "Use both tools",
        model,
        options=model.Options(),
        tools=[local_tool, GoogleSearch()],
    )

    with pytest.raises(ValueError, match="Gemini 3"):
        model.build_request_body(prompt, None)


def test_unsupported_server_side_tool_raises():
    from llm_gemini import GoogleSearch

    model = llm.get_model("gemma-4-26b-a4b-it")
    prompt = llm.Prompt(
        "Search for pelicans",
        model,
        options=model.Options(),
        tools=[GoogleSearch()],
    )

    with pytest.raises(ValueError, match="does not support server-side tool"):
        model.build_request_body(prompt, None)


def test_server_side_tools_are_listed_by_cli():
    runner = CliRunner()

    result = runner.invoke(cli, ["tools", "list", "-m", "gemini-3.6-flash"])

    assert result.exit_code == 0
    assert "GoogleSearch()" in result.output
    assert "URLContext()" in result.output
    assert "CodeExecution()" in result.output


class _NoLocalToolCallsResponse:
    def add_tool_call(self, tool_call):
        raise AssertionError("Server-side tools must not be registered as local calls")


def test_native_server_tool_call_and_response_events():
    model = llm.get_model("gemini-3.6-flash")
    raw_call = {
        "thoughtSignature": "signature-123",
        "toolCall": {
            "toolType": "GOOGLE_SEARCH_WEB",
            "args": {"queries": ["latest pelican research"]},
            "id": "search-123",
        },
    }
    raw_response = {
        "thoughtSignature": "signature-456",
        "toolResponse": {
            "toolType": "GOOGLE_SEARCH_WEB",
            "response": {
                "search_suggestions": [
                    {"title": "Pelican research", "url": "https://example.com/"}
                ]
            },
            "id": "search-123",
        },
    }

    events = list(
        model.process_candidates(
            [{"content": {"parts": [raw_call, raw_response]}}],
            _NoLocalToolCallsResponse(),
        )
    )

    assert [event.type for event in events] == [
        "tool_call_name",
        "tool_call_args",
        "tool_result",
    ]
    assert [event.tool_call_id for event in events] == [
        "search-123",
        "search-123",
        "search-123",
    ]
    assert all(event.server_executed for event in events)
    assert events[0].chunk == "google_search"
    assert json.loads(events[1].chunk) == {"queries": ["latest pelican research"]}
    assert events[2].tool_name == "google_search"
    assert json.loads(events[2].chunk) == raw_response["toolResponse"]["response"]
    assert events[0].provider_metadata == {"gemini": {"part": raw_call}}
    assert events[2].provider_metadata == {"gemini": {"part": raw_response}}


def test_empty_text_part_retains_thought_signature(monkeypatch):
    model = llm.get_model("gemini-3.6-flash")

    def execute(prompt, stream, response, conversation, key):
        yield StreamEvent(type="text", chunk="answer")
        yield from model.process_part(
            {"text": "", "thoughtSignature": "signature-123"},
            response,
        )

    monkeypatch.setattr(model, "execute", execute)
    response = model.prompt("test", key="test")

    assert response.text() == "answer"
    text_part = response.messages()[0].parts[0]
    assert text_part.provider_metadata == {
        "gemini": {"thoughtSignature": "signature-123"}
    }


def test_native_code_execution_events_share_generated_id():
    model = llm.get_model("gemini-3.6-flash")
    raw_code = {
        "executableCode": {
            "language": "PYTHON",
            "code": "print(6 * 7)",
        }
    }
    raw_result = {
        "codeExecutionResult": {
            "outcome": "OUTCOME_OK",
            "output": "42\n",
        }
    }

    events = list(
        model.process_candidates(
            [{"content": {"parts": [raw_code, raw_result]}}],
            _NoLocalToolCallsResponse(),
        )
    )

    assert [event.type for event in events] == [
        "tool_call_name",
        "tool_call_args",
        "tool_result",
    ]
    assert events[0].tool_call_id
    assert len({event.tool_call_id for event in events}) == 1
    assert all(event.server_executed for event in events)
    assert events[0].chunk == "code_execution"
    assert json.loads(events[1].chunk) == raw_code["executableCode"]
    assert events[2].tool_name == "code_execution"
    assert json.loads(events[2].chunk) == raw_result["codeExecutionResult"]
    assert events[0].provider_metadata == {"gemini": {"part": raw_code}}
    assert events[2].provider_metadata == {"gemini": {"part": raw_result}}


def test_native_server_tool_parts_replay_verbatim():
    from llm.parts import Message, ToolCallPart, ToolResultPart

    model = llm.get_model("gemini-3.6-flash")
    raw_call = {
        "thoughtSignature": "signature-123",
        "toolCall": {
            "toolType": "URL_CONTEXT",
            "args": {"urls": ["https://example.com/"]},
            "id": "url-123",
        },
    }
    raw_response = {
        "toolResponse": {
            "toolType": "URL_CONTEXT",
            "response": {"url": "https://example.com/", "status": "OK"},
            "id": "url-123",
        }
    }
    messages = [
        Message(
            role="assistant",
            parts=[
                ToolCallPart(
                    name="url_context",
                    arguments=raw_call["toolCall"]["args"],
                    tool_call_id="url-123",
                    server_executed=True,
                    provider_metadata={"gemini": {"part": raw_call}},
                ),
                ToolResultPart(
                    name="url_context",
                    output=json.dumps(raw_response["toolResponse"]["response"]),
                    tool_call_id="url-123",
                    server_executed=True,
                    provider_metadata={"gemini": {"part": raw_response}},
                ),
            ],
        )
    ]

    class MockPrompt:
        prompt = None
        system = None
        attachments = []
        tools = None
        schema = None
        tool_results = None

    prompt = MockPrompt()
    prompt.messages = messages
    prompt.options = model.Options()

    assert model.build_messages(prompt, None) == [
        {"role": "model", "parts": [raw_call, raw_response]}
    ]


def test_google_search_grounding_metadata_is_raw_and_text_is_unchanged():
    model = llm.get_model("gemini-3.6-flash")
    grounding = {
        "groundingChunks": [
            {
                "web": {
                    "title": "Example source",
                    "uri": "https://example.com/source",
                }
            },
        ],
        "groundingSupports": [
            {
                "segment": {
                    "startIndex": 0,
                    "endIndex": 24,
                    "text": "Pelicans live worldwide.",
                },
                "groundingChunkIndices": [0],
                "confidenceScores": [0.98],
            }
        ],
        "webSearchQueries": ["where pelicans live"],
        "searchEntryPoint": {
            "renderedContent": (
                '<a href="https://example.com/search">'
                "<span>Latest pelican research</span></a>"
            )
        },
    }

    events = list(
        model.process_candidates(
            [
                {
                    "content": {"parts": [{"text": "Pelicans live worldwide."}]},
                    "groundingMetadata": grounding,
                }
            ],
            response=_NoLocalToolCallsResponse(),
        )
    )

    assert len(events) == 1
    assert events[0].type == "text"
    assert events[0].chunk == "Pelicans live worldwide."
    assert events[0].provider_metadata == {"gemini": {"groundingMetadata": grounding}}
