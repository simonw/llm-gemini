import httpx
import ijson
import llm
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, create_model, Field


def pydantic_to_gemini_schema(schema: dict) -> dict:
    """Convert a Pydantic JSON schema to Gemini's schema format."""
    type_mapping = {
        "string": "STRING",
        "integer": "INTEGER", 
        "number": "NUMBER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
        "object": "OBJECT",
    }

    def convert_property(prop, definitions=None):
        if "$ref" in prop and definitions:
            ref_path = prop["$ref"].split("/")[-1]
            if ref_path in definitions:
                return convert_property(definitions[ref_path], definitions)
            
        if "type" not in prop:
            return {"type": "STRING"}  # default to string for complex types
        
        result = {"type": type_mapping.get(prop["type"], "STRING")}
        
        if prop["type"] == "array" and "items" in prop:
            result["items"] = convert_property(prop["items"], definitions)
        elif prop["type"] == "object" and "properties" in prop:
            result["properties"] = {
                k: convert_property(v, definitions) for k, v in prop["properties"].items()
            }
        
        return result

    schema = schema.copy()
    definitions = schema.get("$defs") or schema.get("definitions")

    if "type" in schema:
        return convert_property(schema, definitions)
    elif "properties" in schema:
        # Handle root level schema
        return {
            "type": "OBJECT",
            "properties": {
                k: convert_property(v, definitions) for k, v in schema["properties"].items()
            }
        }
    else:
        raise ValueError("Invalid schema format")


def resolve_ref(ref: str, schema_dict: Dict) -> Dict:
    """Resolve a $ref reference in the schema."""
    if not ref.startswith("#/"):
        raise ValueError(f"Only local references are supported: {ref}")

    path = ref[2:].split("/")
    current = schema_dict
    for part in path:
        if part not in current:
            raise ValueError(f"Invalid reference: {ref}")
        current = current[part]
    return current


def create_models_from_schema(schema_dict: Dict) -> Dict[str, Type[BaseModel]]:
    """Create all models defined in the schema."""
    models = {}

    def parse_schema_to_pydantic_model(
        schema_dict: Dict, model_name: str = None
    ) -> Type[BaseModel]:
        """Parse a schema into a Pydantic model, supporting nested objects and references."""
        # If it's a reference, resolve it
        if isinstance(schema_dict, dict) and "$ref" in schema_dict:
            ref_path = schema_dict["$ref"][2:].split("/")
            if ref_path[0] in models:
                return models[ref_path[0]]
            referenced_schema = {
                ref_path[0]: resolve_ref(schema_dict["$ref"], root_schema)
            }
            return parse_schema_to_pydantic_model(referenced_schema)

        if len(schema_dict) == 1 and isinstance(next(iter(schema_dict.values())), dict):
            model_name = next(iter(schema_dict.keys()))
            fields_dict = next(iter(schema_dict.values()))

            # Handle array type with items
            if (
                isinstance(fields_dict, dict)
                and fields_dict.get("type") == "array"
                and "items" in fields_dict
            ):
                item_model = parse_schema_to_pydantic_model(fields_dict["items"])
                # Create a wrapper model for arrays to satisfy OpenAI's object requirement
                model = create_model(model_name, items=(List[item_model], ...))
                models[model_name] = model
                return model

            # Map string type names to actual types
            type_mapping = {
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": List,
                "dict": Dict,
            }

            # Convert fields
            fields: Dict[str, Any] = {}
            for field_name, field_type in fields_dict.items():
                if isinstance(field_type, dict):
                    # Nested object
                    nested_model = parse_schema_to_pydantic_model(
                        field_type, field_name
                    )
                    fields[field_name] = (nested_model, ...)
                elif isinstance(field_type, str):
                    python_type = type_mapping.get(field_type, Any)
                    fields[field_name] = (python_type, ...)

            model = create_model(model_name, **fields)
            models[model_name] = model
            return model

        return create_model(model_name or "DynamicModel", **{})

    root_schema = schema_dict
    # First pass: create all top-level models
    for model_name, model_schema in schema_dict.items():
        if model_name not in models:
            parse_schema_to_pydantic_model({model_name: model_schema})

    return models

SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
]

# https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-gemini#supported_models_2
GOOGLE_SEARCH_MODELS = {
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-001",
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-002",
    "gemini-1.5-flash-002",
    "gemini-2.0-flash-exp",
}


@llm.hookimpl
def register_models(register):
    # Register both sync and async versions of each model
    for model_id in [
        "gemini-pro",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-001",
        "gemini-1.5-flash-001",
        "gemini-1.5-pro-002",
        "gemini-1.5-flash-002",
        "gemini-1.5-flash-8b-latest",
        "gemini-1.5-flash-8b-001",
        "gemini-exp-1114",
        "gemini-exp-1121",
        "gemini-exp-1206",
        "gemini-2.0-flash-exp",
        "learnlm-1.5-pro-experimental",
        "gemini-2.0-flash-thinking-exp-1219",
        "gemini-2.0-flash-thinking-exp-01-21",
    ]:
        can_google_search = model_id in GOOGLE_SEARCH_MODELS
        register(
            GeminiPro(model_id, can_google_search=can_google_search),
            AsyncGeminiPro(model_id, can_google_search=can_google_search),
        )


def resolve_type(attachment):
    mime_type = attachment.resolve_type()
    # https://github.com/simonw/llm/issues/587#issuecomment-2439785140
    if mime_type == "audio/mpeg":
        mime_type = "audio/mp3"
    if mime_type == "application/ogg":
        mime_type = "audio/ogg"
    return mime_type


class _SharedGemini:
    needs_key = "gemini"
    key_env_var = "LLM_GEMINI_KEY"
    can_stream = True

    attachment_types = (
        # Text
        "text/plain",
        "text/csv",
        # PDF
        "application/pdf",
        # Images
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/heic",
        "image/heif",
        # Audio
        "audio/wav",
        "audio/mp3",
        "audio/aiff",
        "audio/aac",
        "audio/ogg",
        "application/ogg",
        "audio/flac",
        "audio/mpeg",  # Treated as audio/mp3
        # Video
        "video/mp4",
        "video/mpeg",
        "video/mov",
        "video/avi",
        "video/x-flv",
        "video/mpg",
        "video/webm",
        "video/wmv",
        "video/3gpp",
        "video/quicktime",
    )

    class Options(llm.Options):
        code_execution: Optional[bool] = Field(
            description="Enables the model to generate and run Python code",
            default=None,
        )
        temperature: Optional[float] = Field(
            description=(
                "Controls the randomness of the output. Use higher values for "
                "more creative responses, and lower values for more "
                "deterministic responses."
            ),
            default=None,
            ge=0.0,
            le=2.0,
        )
        max_output_tokens: Optional[int] = Field(
            description="Sets the maximum number of tokens to include in a candidate.",
            default=None,
        )
        top_p: Optional[float] = Field(
            description=(
                "Changes how the model selects tokens for output. Tokens are "
                "selected from the most to least probable until the sum of "
                "their probabilities equals the topP value."
            ),
            default=None,
            ge=0.0,
            le=1.0,
        )
        top_k: Optional[int] = Field(
            description=(
                "Changes how the model selects tokens for output. A topK of 1 "
                "means the selected token is the most probable among all the "
                "tokens in the model's vocabulary, while a topK of 3 means "
                "that the next token is selected from among the 3 most "
                "probable using the temperature."
            ),
            default=None,
            ge=1,
        )
        json_object: Optional[bool] = Field(
            description="Output a valid JSON object {...}",
            default=None,
        )
        response_mime_type: Optional[str] = Field(
            description="Structured response MIME type",
            default=None,
        )
        response_schema: Optional[Union[dict, str]] = Field(
            description="Path to YAML schema file or schema dict",
            default=None,
        )

    class OptionsWithGoogleSearch(Options):
        google_search: Optional[bool] = Field(
            description="Enables the model to use Google Search to improve the accuracy and recency of responses from the model",
            default=None,
        )

    def __init__(self, model_id, can_google_search=False):
        self.model_id = model_id
        self.can_google_search = can_google_search
        if can_google_search:
            self.Options = self.OptionsWithGoogleSearch

    def build_messages(self, prompt, conversation):
        messages = []
        if conversation:
            for response in conversation.responses:
                parts = []
                for attachment in response.attachments:
                    mime_type = resolve_type(attachment)
                    parts.append(
                        {
                            "inlineData": {
                                "data": attachment.base64_content(),
                                "mimeType": mime_type,
                            }
                        }
                    )
                if response.prompt.prompt:
                    parts.append({"text": response.prompt.prompt})
                messages.append({"role": "user", "parts": parts})
                messages.append(
                    {"role": "model", "parts": [{"text": response.text_or_raise()}]}
                )

        parts = []
        if prompt.prompt:
            parts.append({"text": prompt.prompt})
        for attachment in prompt.attachments:
            mime_type = resolve_type(attachment)
            parts.append(
                {
                    "inlineData": {
                        "data": attachment.base64_content(),
                        "mimeType": mime_type,
                    }
                }
            )

        messages.append({"role": "user", "parts": parts})
        return messages

    def build_request_body(self, prompt, conversation):
        body = {
            "contents": self.build_messages(prompt, conversation),
            "safetySettings": SAFETY_SETTINGS,
        }
        if prompt.options and prompt.options.code_execution:
            body["tools"] = [{"codeExecution": {}}]
        if prompt.options and self.can_google_search and prompt.options.google_search:
            body["tools"] = [{"google_search_retrieval": {}}]
        if prompt.system:
            body["systemInstruction"] = {"parts": [{"text": prompt.system}]}

        generation_config = {}
        config_map = {
            "temperature": "temperature",
            "max_output_tokens": "maxOutputTokens",
            "top_p": "topP",
            "top_k": "topK",
        }
        if prompt.options:
            if prompt.options.response_schema:
                if isinstance(prompt.options.response_schema, str):
                    schema_path = Path(prompt.options.response_schema)
                    if not schema_path.exists():
                        raise ValueError(f"Schema file not found: {schema_path}")
                    try:
                        with open(schema_path, "r") as f:
                            schema = yaml.full_load(f)
                            models = create_models_from_schema(schema)
                            # For array types, we want the container model
                            model_name = next(iter(schema.keys()))
                            DynamicModel = models[model_name]
                            pydantic_schema = DynamicModel.model_json_schema()
                            # Add referenced models to definitions
                            pydantic_schema["$defs"] = {
                                k: v.model_json_schema() 
                                for k, v in models.items() 
                                if k != model_name
                            }
                            gemini_schema = pydantic_to_gemini_schema(pydantic_schema)
                    except yaml.YAMLError as e:
                        raise ValueError(f"Invalid YAML in schema file: {e}") from e
                    generation_config["response_schema"] = gemini_schema
                    generation_config["response_mime_type"] = "application/json"
                else:
                    generation_config["response_schema"] = (
                        prompt.options.response_schema
                    )

            if prompt.options.response_mime_type:
                generation_config["response_mime_type"] = (
                    prompt.options.response_mime_type
                )
            elif prompt.options.json_object:
                generation_config["response_mime_type"] = "application/json"

            if generation_config:
                body["generationConfig"] = generation_config

        if any(
            getattr(prompt.options, key, None) is not None for key in config_map.keys()
        ):
            generation_config = {}
            for key, other_key in config_map.items():
                config_value = getattr(prompt.options, key, None)
                if config_value is not None:
                    generation_config[other_key] = config_value
            body["generationConfig"] = generation_config

        return body

    def process_part(self, part):
        if "text" in part:
            return part["text"]
        elif "executableCode" in part:
            return f'```{part["executableCode"]["language"].lower()}\n{part["executableCode"]["code"].strip()}\n```\n'
        elif "codeExecutionResult" in part:
            return f'```\n{part["codeExecutionResult"]["output"].strip()}\n```\n'
        return ""

    def set_usage(self, response):
        try:
            usage = response.response_json[-1].pop("usageMetadata")
            input_tokens = usage.pop("promptTokenCount", None)
            output_tokens = usage.pop("candidatesTokenCount", None)
            usage.pop("totalTokenCount", None)
            if input_tokens is not None:
                response.set_usage(
                    input=input_tokens, output=output_tokens, details=usage or None
                )
        except (IndexError, KeyError):
            pass


class GeminiPro(_SharedGemini, llm.Model):
    def execute(self, prompt, stream, response, conversation):
        key = self.get_key()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:streamGenerateContent"
        gathered = []
        body = self.build_request_body(prompt, conversation)

        with httpx.stream(
            "POST",
            url,
            timeout=None,
            headers={"x-goog-api-key": key},
            json=body,
        ) as http_response:
            events = ijson.sendable_list()
            coro = ijson.items_coro(events, "item")
            for chunk in http_response.iter_bytes():
                coro.send(chunk)
                if events:
                    event = events[0]
                    if isinstance(event, dict) and "error" in event:
                        raise llm.ModelError(event["error"]["message"])
                    try:
                        part = event["candidates"][0]["content"]["parts"][0]
                        yield self.process_part(part)
                    except KeyError:
                        yield ""
                    gathered.append(event)
                    events.clear()
        response.response_json = gathered
        self.set_usage(response)


class AsyncGeminiPro(_SharedGemini, llm.AsyncModel):
    async def execute(self, prompt, stream, response, conversation):
        key = self.get_key()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:streamGenerateContent"
        gathered = []
        body = self.build_request_body(prompt, conversation)

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                url,
                timeout=None,
                headers={"x-goog-api-key": key},
                json=body,
            ) as http_response:
                events = ijson.sendable_list()
                coro = ijson.items_coro(events, "item")
                async for chunk in http_response.aiter_bytes():
                    coro.send(chunk)
                    if events:
                        event = events[0]
                        if isinstance(event, dict) and "error" in event:
                            raise llm.ModelError(event["error"]["message"])
                        try:
                            part = event["candidates"][0]["content"]["parts"][0]
                            yield self.process_part(part)
                        except KeyError:
                            yield ""
                        gathered.append(event)
                        events.clear()
        response.response_json = gathered
        self.set_usage(response)


@llm.hookimpl
def register_embedding_models(register):
    register(
        GeminiEmbeddingModel("text-embedding-004", "text-embedding-004"),
    )


class GeminiEmbeddingModel(llm.EmbeddingModel):
    needs_key = "gemini"
    key_env_var = "LLM_GEMINI_KEY"
    batch_size = 20

    def __init__(self, model_id, gemini_model_id):
        self.model_id = model_id
        self.gemini_model_id = gemini_model_id

    def embed_batch(self, items):
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.get_key(),
        }
        data = {
            "requests": [
                {
                    "model": "models/" + self.gemini_model_id,
                    "content": {"parts": [{"text": item}]},
                }
                for item in items
            ]
        }

        with httpx.Client() as client:
            response = client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model_id}:batchEmbedContents",
                headers=headers,
                json=data,
                timeout=None,
            )

        response.raise_for_status()
        return [item["values"] for item in response.json()["embeddings"]]
