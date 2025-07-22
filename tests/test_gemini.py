import base64
from click.testing import CliRunner
import llm
from llm.cli import cli
import nest_asyncio
import json
import os
import pytest
import pydantic
import tempfile
from llm_gemini import cleanup_schema

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
        "candidatesTokenCount": 2,
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


@pytest.mark.vcr
def test_cli_gemini_models(tmpdir, monkeypatch):
    user_dir = tmpdir / "llm.datasette.io"
    user_dir.mkdir()
    monkeypatch.setenv("LLM_USER_PATH", str(user_dir))
    monkeypatch.setenv("LLM_GEMINI_KEY", "")
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
def test_cli_gemini_generate_with_output_file(tmpdir, monkeypatch):
    """Test image generation with --output option"""
    user_dir = tmpdir / "llm.datasette.io"
    user_dir.mkdir()
    monkeypatch.setenv("LLM_USER_PATH", str(user_dir))
    
    # Create a temporary output file
    output_file = tmpdir / "test_image.jpg"
    
    runner = CliRunner()
    result = runner.invoke(cli, [
        "gemini", "generate", "A pelican riding a bicycle",
        "--output", str(output_file),
        "--key", GEMINI_API_KEY,
        "--model", "gemini-2.0-flash-exp-image-generation"
    ])
    
    assert result.exit_code == 0
    assert f"Image saved to {output_file}" in result.output
    assert output_file.exists()
    
    # Check that the file contains some data (should be binary image data)
    image_data = output_file.read_binary()
    assert len(image_data) > 0
    # Basic check that it might be image data (common image file headers)
    assert image_data[:4] in [b'\xff\xd8\xff\xe0', b'\x89PNG', b'RIFF'] or image_data[8:12] == b'WEBP'


@pytest.mark.vcr  
def test_cli_gemini_generate_to_stdout(tmpdir, monkeypatch):
    """Test image generation output to stdout for piping"""
    user_dir = tmpdir / "llm.datasette.io"
    user_dir.mkdir()
    monkeypatch.setenv("LLM_USER_PATH", str(user_dir))
    
    runner = CliRunner()
    result = runner.invoke(cli, [
        "gemini", "generate", "A pelican riding a bicycle with a pink helmet",
        "--key", GEMINI_API_KEY,
        "--model", "gemini-2.0-flash-exp-image-generation"
    ])
    
    assert result.exit_code == 0
    # stdout_bytes should contain the binary image data
    assert len(result.stdout_bytes) > 0
    # Basic check that it might be image data
    image_data = result.stdout_bytes
    assert image_data[:4] in [b'\xff\xd8\xff\xe0', b'\x89PNG', b'RIFF'] or image_data[8:12] == b'WEBP'


def test_cli_gemini_generate_no_key_error(tmpdir, monkeypatch):
    """Test that generate command fails gracefully without API key"""
    user_dir = tmpdir / "llm.datasette.io"
    user_dir.mkdir()
    monkeypatch.setenv("LLM_USER_PATH", str(user_dir))
    monkeypatch.setenv("LLM_GEMINI_KEY", "")
    
    runner = CliRunner()
    result = runner.invoke(cli, [
        "gemini", "generate", "A pelican riding a bicycle past a No-Entry sign"
    ])
    
    assert result.exit_code == 1
    assert "You must set the LLM_GEMINI_KEY environment variable or use --key" in result.output


def test_cli_gemini_generate_invalid_model_error(tmpdir, monkeypatch):
    """Test that generate command fails with invalid model"""
    user_dir = tmpdir / "llm.datasette.io"  
    user_dir.mkdir()
    monkeypatch.setenv("LLM_USER_PATH", str(user_dir))
    
    runner = CliRunner()
    result = runner.invoke(cli, [
        "gemini", "generate", "A pelican riding a bicycle past a question mark sign",
        "--key", GEMINI_API_KEY,
        "--model", "gemini-1.5-flash-latest"  # Not an image generation model
    ])
    
    assert result.exit_code == 1
    assert "does not support image generation" in result.output


def test_image_generation_models_registered():
    """Test that image generation models are properly registered"""
    # Test that the models exist
    model1 = llm.get_model("gemini-2.0-flash-exp-image-generation")
    model2 = llm.get_model("gemini-2.0-flash-preview-image-generation")
    
    assert model1 is not None
    assert model2 is not None
    
    # Check that they have the image generation capability
    assert hasattr(model1, 'can_image_generation')
    assert hasattr(model2, 'can_image_generation')
    assert model1.can_image_generation == True
    assert model2.can_image_generation == True


@pytest.mark.vcr
def test_cli_gemini_generate_with_single_attachment(tmpdir, monkeypatch):
    """Test image generation with a single attached image"""
    user_dir = tmpdir / "llm.datasette.io"
    user_dir.mkdir()
    monkeypatch.setenv("LLM_USER_PATH", str(user_dir))
    
    # Create a simple test image file (small but valid PNG)
    # This is a minimal 3x3 red pixel PNG
    test_image_data = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAYAAABWKLW/AAAABmJLR0QA/wD/AP+gvaeTAAAAHklEQVQImWP8z8DAwMjIyMDAwMDwn4EBDRgZGNAARAAAJgADABGl')
    test_image_path = tmpdir / "input.png"
    test_image_path.write_binary(test_image_data)
    
    output_file = tmpdir / "output.jpg"
    
    runner = CliRunner()
    result = runner.invoke(cli, [
        "gemini", "generate", "Make this image larger",
        "--output", str(output_file),
        "--attach", str(test_image_path),
        "--key", GEMINI_API_KEY,
        "--model", "gemini-2.0-flash-exp-image-generation"
    ])
    
    assert result.exit_code == 0
    assert f"Image saved to {output_file}" in result.output
    assert output_file.exists()


@pytest.mark.vcr
def test_cli_gemini_generate_with_multiple_attachments(tmpdir, monkeypatch):
    """Test image generation with multiple attached images"""
    user_dir = tmpdir / "llm.datasette.io"
    user_dir.mkdir()
    monkeypatch.setenv("LLM_USER_PATH", str(user_dir))
    
    # Create two simple test image files (4x4 pixel PNGs with different colors)
    # Red 4x4 image
    red_image_data = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAQAAAAEAQMAAACTPww9AAAAAXNSR0IB2cksfwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAANQTFRF/wAAGeIJNwAAAAxJREFUeJxjZGAEQwAAHAAFSJZPvAAAAABJRU5ErkJggg==')
    red_image_path = tmpdir / "red.png"
    red_image_path.write_binary(red_image_data)
    
    # Blue 4x4 image  
    blue_image_data = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAQAAAAEAQMAAACTPww9AAAAAXNSR0IB2cksfwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAANQTFRFAAD/injSVwAAAAxJREFUeJxjZGAEQwAAHAAFSJZPvAAAAABJRU5ErkJggg==')
    blue_image_path = tmpdir / "blue.png"
    blue_image_path.write_binary(blue_image_data)
    
    output_file = tmpdir / "combined.jpg"
    
    runner = CliRunner()
    result = runner.invoke(cli, [
        "gemini", "generate", "Combine these images side by side",
        "--output", str(output_file),
        "--attach", str(red_image_path),
        "--attach", str(blue_image_path),
        "--key", GEMINI_API_KEY,
        "--model", "gemini-2.0-flash-exp-image-generation"
    ])
    
    assert result.exit_code == 0
    assert f"Image saved to {output_file}" in result.output
    assert output_file.exists()


def test_cli_gemini_generate_with_nonexistent_attachment(tmpdir, monkeypatch):
    """Test that generate command fails gracefully with nonexistent attachment"""
    user_dir = tmpdir / "llm.datasette.io"
    user_dir.mkdir()
    monkeypatch.setenv("LLM_USER_PATH", str(user_dir))
    
    nonexistent_file = tmpdir / "does_not_exist.jpg"
    
    runner = CliRunner()
    result = runner.invoke(cli, [
        "gemini", "generate", "Edit this image",
        "--attach", str(nonexistent_file),
        "--key", GEMINI_API_KEY,
        "--model", "gemini-2.0-flash-exp-image-generation"
    ])
    
    assert result.exit_code == 1
    assert "Failed to read attachment" in result.output
    assert str(nonexistent_file) in result.output


