from click.testing import CliRunner
import llm
from llm.cli import cli
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
        "candidatesTokenCount": 2,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 9}],
        "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 2}],
    }
    assert response.input_tokens == 9
    assert response.output_tokens == 2

    # Skip async test on Python 3.14 due to httpcore cleanup incompatibility
    # https://github.com/encode/httpcore/issues - AsyncLibraryNotFoundError during __aexit__
    if sys.version_info < (3, 14):
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

    class Dogs(BaseModel):
        dogs: List[Dog]

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
@pytest.mark.asyncio
async def test_prompt_with_multiple_dogs():
    class Dog(pydantic.BaseModel):
        name: str
        age: int
        bio: str

    class Dogs(BaseModel):
        dogs: List[Dog]

    model = llm.get_model("gemini-2.0-flash")
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

    model = llm.get_model("gemini-2.0-flash")
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

    model = llm.get_model("gemini-2.0-flash")
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

    model = llm.get_model("gemini-2.0-flash")
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
    assert "gemini-1.5-flash-latest" in result2.output
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
    assert response.resolved_model == "gemini-2.5-flash-preview-09-2025"


@pytest.mark.vcr
def test_tools():
    model = llm.get_model("gemini-2.0-flash")
    names = ["Charles", "Sammy"]
    chain_response = model.chain(
        "Two names for a pet pelican",
        tools=[
            llm.Tool.function(lambda: names.pop(0), name="pelican_name_generator"),
        ],
        key=GEMINI_API_KEY,
    )
    text = chain_response.text()
    assert text == "Okay, here are two names for a pet pelican: Charles and Sammy.\n"
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
