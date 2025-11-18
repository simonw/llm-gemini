import click
import copy
from enum import Enum
import httpx
import ijson
import json
import llm
import re
from pydantic import Field
from typing import Optional

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
    "gemini-2.0-flash",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-2.5-flash-lite-preview-09-2025",
    "gemini-3-pro-preview",
}

# Older Google models used google_search_retrieval instead of google_search
GOOGLE_SEARCH_MODELS_USING_SEARCH_RETRIEVAL = {
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-001",
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-002",
    "gemini-1.5-flash-002",
    "gemini-2.0-flash-exp",
}

THINKING_BUDGET_MODELS = {
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.0-flash-thinking-exp-1219",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-2.5-flash-lite-preview-09-2025",
}

THINKING_LEVEL_MODELS = {
    "gemini-3-pro-preview",
}

NO_VISION_MODELS = {"gemma-3-1b-it", "gemma-3n-e4b-it"}


class ThinkingLevel(str, Enum):
    """Allowed thinking levels for Gemini models."""

    LOW = "low"
    HIGH = "high"


class MediaResolution(str, Enum):
    """Allowed media resolution values for Gemini models."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNSPECIFIED = "unspecified"


ATTACHMENT_TYPES = {
    # Text
    "text/plain",
    "text/csv",
    "text/html; charset=utf-8",
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
    "video/youtube",
}


@llm.hookimpl
def register_models(register):
    # Register both sync and async versions of each model
    for model_id in (
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
        # Gemma 3 models:
        "gemma-3-1b-it",
        "gemma-3-4b-it",
        "gemma-3-12b-it",  # 12th March 2025
        "gemma-3-27b-it",
        "gemma-3n-e4b-it",  # 20th May 2025
        "gemini-2.0-flash-thinking-exp-1219",
        "gemini-2.0-flash-thinking-exp-01-21",
        # Released 5th Feb 2025:
        "gemini-2.0-flash",
        "gemini-2.0-pro-exp-02-05",
        # Released 25th Feb 2025:
        "gemini-2.0-flash-lite",
        # 25th March 2025:
        "gemini-2.5-pro-exp-03-25",
        # 4th April 2025 (paid):
        "gemini-2.5-pro-preview-03-25",
        # 17th April 2025:
        "gemini-2.5-flash-preview-04-17",
        # 6th May 2025:
        "gemini-2.5-pro-preview-05-06",
        # 20th May 2025:
        "gemini-2.5-flash-preview-05-20",
        # 5th June 2025:
        "gemini-2.5-pro-preview-06-05",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        # 22nd July 2025:
        "gemini-2.5-flash-lite",
        # 25th Spetember 2025:
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash-preview-09-2025",
        "gemini-2.5-flash-lite-preview-09-2025",
        # 18th November 2025:
        "gemini-3-pro-preview",
    ):
        can_google_search = model_id in GOOGLE_SEARCH_MODELS
        can_thinking_budget = model_id in THINKING_BUDGET_MODELS
        can_thinking_level = model_id in THINKING_LEVEL_MODELS
        can_vision = model_id not in NO_VISION_MODELS
        can_schema = "flash-thinking" not in model_id and "gemma-3" not in model_id
        register(
            GeminiPro(
                model_id,
                can_vision=can_vision,
                can_google_search=can_google_search,
                can_thinking_budget=can_thinking_budget,
                can_thinking_level=can_thinking_level,
                can_schema=can_schema,
            ),
            AsyncGeminiPro(
                model_id,
                can_vision=can_vision,
                can_google_search=can_google_search,
                can_thinking_budget=can_thinking_budget,
                can_thinking_level=can_thinking_level,
                can_schema=can_schema,
            ),
            aliases=(model_id,),
        )


def resolve_type(attachment):
    mime_type = attachment.resolve_type()
    # https://github.com/simonw/llm/issues/587#issuecomment-2439785140
    if mime_type == "audio/mpeg":
        mime_type = "audio/mp3"
    if mime_type == "application/ogg":
        mime_type = "audio/ogg"
    # Check if this is a YouTube URL
    if attachment.url and is_youtube_url(attachment.url):
        return "video/youtube"
    return mime_type


def is_youtube_url(url):
    """Check if a URL is a YouTube video URL"""
    if not url:
        return False
    youtube_patterns = [
        r"^https?://(www\.)?youtube\.com/watch\?v=",
        r"^https?://youtu\.be/",
        r"^https?://(www\.)?youtube\.com/embed/",
    ]
    return any(re.match(pattern, url) for pattern in youtube_patterns)


def cleanup_schema(schema, in_properties=False):
    "Gemini supports only a subset of JSON schema"
    keys_to_remove = ("$schema", "additionalProperties", "title")

    # First pass: resolve $ref references using $defs
    if isinstance(schema, dict) and "$defs" in schema:
        defs = schema.pop("$defs")
        _resolve_refs(schema, defs)

    if isinstance(schema, dict):
        # Only remove keys if we're not inside a 'properties' block.
        if not in_properties:
            for key in keys_to_remove:
                schema.pop(key, None)
        for key, value in list(schema.items()):
            # If the key is 'properties', set the flag for its value.
            if key == "properties" and isinstance(value, dict):
                cleanup_schema(value, in_properties=True)
            else:
                cleanup_schema(value, in_properties=in_properties)
    elif isinstance(schema, list):
        for item in schema:
            cleanup_schema(item, in_properties=in_properties)
    return schema


def _resolve_refs(schema, defs, expansion_stack=None):
    """Recursively resolve $ref references in schema using definitions.

    Args:
        schema: The schema to resolve references in
        defs: Dictionary of definitions from $defs
        expansion_stack: List tracking the order of definition expansions (for cycle detection)

    Raises:
        ValueError: If a recursive self-reference is detected
    """
    if expansion_stack is None:
        expansion_stack = []

    if isinstance(schema, dict):
        if "$ref" in schema:
            # Extract the reference path (e.g., "#/$defs/Dog" -> "Dog")
            ref_path = schema.pop("$ref")
            if ref_path.startswith("#/$defs/"):
                def_name = ref_path.split("/")[-1]
                if def_name in defs:
                    # Check for recursive reference
                    if def_name in expansion_stack:
                        # Determine if this is direct or indirect recursion
                        if expansion_stack[-1] == def_name:
                            # Direct recursion: the definition references itself
                            raise ValueError(
                                f"Recursive schema detected: '{def_name}' directly references itself. "
                                f"The Gemini API does not support recursive Pydantic models. "
                                f"Please use a non-recursive schema structure."
                            )
                        else:
                            # Indirect recursion: the class being referenced is earlier in the stack
                            # The last item in expansion_stack is the immediate class that references def_name
                            intermediate = expansion_stack[-1]
                            raise ValueError(
                                f"Recursive schema detected: '{def_name}' indirectly references itself through '{intermediate}'. "
                                f"The Gemini API does not support recursive Pydantic models. "
                                f"Please use a non-recursive schema structure."
                            )

                    # Add this definition to the expansion stack
                    expansion_stack.append(def_name)
                    # Replace the $ref with the actual definition
                    resolved = copy.deepcopy(defs[def_name])
                    schema.update(resolved)
                    # Resolve any nested refs in the newly added definition
                    _resolve_refs(schema, defs, expansion_stack)
                    # Remove from stack after processing
                    expansion_stack.pop()
                    return

        # Recursively resolve refs in nested structures
        for value in schema.values():
            _resolve_refs(value, defs, expansion_stack)
    elif isinstance(schema, list):
        for item in schema:
            _resolve_refs(item, defs, expansion_stack)


class _SharedGemini:
    needs_key = "gemini"
    key_env_var = "LLM_GEMINI_KEY"
    can_stream = True
    supports_schema = True
    supports_tools = True

    attachment_types = set()

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
        timeout: Optional[float] = Field(
            description=(
                "The maximum time in seconds to wait for a response. "
                "If the model does not respond within this time, "
                "the request will be aborted."
            ),
            default=None,
        )
        url_context: Optional[bool] = Field(
            description=(
                "Enable the URL context tool so the model can fetch content "
                "from URLs mentioned in the prompt"
            ),
            default=None,
        )
        media_resolution: Optional[MediaResolution] = Field(
            description=(
                "Media resolution for the input media (esp. YouTube) "
                "- default is low, other values are medium, high, or unspecified"
            ),
            default=MediaResolution.LOW,
        )

    class OptionsWithGoogleSearch(Options):
        google_search: Optional[bool] = Field(
            description="Enables the model to use Google Search to improve the accuracy and recency of responses from the model",
            default=None,
        )

    class OptionsWithThinkingBudget(OptionsWithGoogleSearch):
        thinking_budget: Optional[int] = Field(
            description="Indicates the thinking budget in tokens. Set to 0 to disable.",
            default=None,
        )

    class OptionsWithThinkingLevel(OptionsWithGoogleSearch):
        thinking_level: Optional[ThinkingLevel] = Field(
            description="Indicates the thinking level. Can be 'low' or 'high'.",
            default=None,
        )

    def __init__(
        self,
        gemini_model_id,
        can_vision=True,
        can_google_search=False,
        can_thinking_budget=False,
        can_thinking_level=False,
        can_schema=False,
    ):
        self.model_id = "gemini/{}".format(gemini_model_id)
        self.gemini_model_id = gemini_model_id
        self.can_google_search = can_google_search
        self.supports_schema = can_schema
        if can_google_search:
            self.Options = self.OptionsWithGoogleSearch
        self.can_thinking_budget = can_thinking_budget
        if can_thinking_budget:
            self.Options = self.OptionsWithThinkingBudget
        self.can_thinking_level = can_thinking_level
        if can_thinking_level:
            self.Options = self.OptionsWithThinkingLevel
        if can_vision:
            self.attachment_types = ATTACHMENT_TYPES

    def build_messages(self, prompt, conversation):
        messages = []
        if conversation:
            for response in conversation.responses:
                parts = []
                for attachment in response.attachments:
                    mime_type = resolve_type(attachment)
                    parts.append(self._build_attachment_part(attachment, mime_type))
                if response.prompt.prompt:
                    parts.append({"text": response.prompt.prompt})
                if response.prompt.tool_results:
                    parts.extend(
                        [
                            {
                                "function_response": {
                                    "name": tool_result.name,
                                    "response": {
                                        "output": tool_result.output,
                                    },
                                }
                            }
                            for tool_result in response.prompt.tool_results
                        ]
                    )
                messages.append({"role": "user", "parts": parts})
                model_parts = []
                response_text = response.text_or_raise()
                model_parts.append({"text": response_text})
                tool_calls = response.tool_calls_or_raise()
                if tool_calls:
                    model_parts.extend(
                        [
                            {
                                "function_call": {
                                    "name": tool_call.name,
                                    "args": tool_call.arguments,
                                }
                            }
                            for tool_call in tool_calls
                        ]
                    )
                messages.append({"role": "model", "parts": model_parts})

        parts = []
        if prompt.prompt:
            parts.append({"text": prompt.prompt})
        if prompt.tool_results:
            parts.extend(
                [
                    {
                        "function_response": {
                            "name": tool_result.name,
                            "response": {
                                "output": tool_result.output,
                            },
                        }
                    }
                    for tool_result in prompt.tool_results
                ]
            )
        for attachment in prompt.attachments:
            mime_type = resolve_type(attachment)
            parts.append(self._build_attachment_part(attachment, mime_type))

        messages.append({"role": "user", "parts": parts})
        return messages

    def _build_attachment_part(self, attachment, mime_type):
        """Build the appropriate part for an attachment based on its type"""
        if mime_type == "video/youtube":
            return {"fileData": {"mimeType": mime_type, "fileUri": attachment.url}}
        else:
            return {
                "inlineData": {
                    "data": attachment.base64_content(),
                    "mimeType": mime_type,
                }
            }

    def build_request_body(self, prompt, conversation):
        body = {
            "contents": self.build_messages(prompt, conversation),
            "safetySettings": SAFETY_SETTINGS,
        }
        if prompt.system:
            body["systemInstruction"] = {"parts": [{"text": prompt.system}]}

        tools = []
        if prompt.options and prompt.options.code_execution:
            tools.append({"codeExecution": {}})
        if prompt.options and self.can_google_search and prompt.options.google_search:
            tool_name = (
                "google_search_retrieval"
                if self.model_id in GOOGLE_SEARCH_MODELS_USING_SEARCH_RETRIEVAL
                else "google_search"
            )
            tools.append({tool_name: {}})
        if prompt.options and prompt.options.url_context:
            tools.append({"url_context": {}})
        if prompt.tools:
            tools.append(
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": cleanup_schema(copy.deepcopy(tool.input_schema)),
                        }
                        for tool in prompt.tools
                    ]
                }
            )
        if tools:
            body["tools"] = tools

        generation_config = {}

        if prompt.schema:
            generation_config.update(
                {
                    "response_mime_type": "application/json",
                    "response_schema": cleanup_schema(copy.deepcopy(prompt.schema)),
                }
            )

        if self.can_thinking_budget and prompt.options.thinking_budget is not None:
            generation_config["thinking_config"] = {
                "thinking_budget": prompt.options.thinking_budget
            }

        if self.can_thinking_level and prompt.options.thinking_level is not None:
            generation_config["thinkingConfig"] = {
                "thinkingLevel": prompt.options.thinking_level
            }

        config_map = {
            "temperature": "temperature",
            "max_output_tokens": "maxOutputTokens",
            "top_p": "topP",
            "top_k": "topK",
        }
        if prompt.options and prompt.options.json_object:
            generation_config["response_mime_type"] = "application/json"

        if any(
            getattr(prompt.options, key, None) is not None for key in config_map.keys()
        ):
            for key, other_key in config_map.items():
                config_value = getattr(prompt.options, key, None)
                if config_value is not None:
                    generation_config[other_key] = config_value

        has_youtube = any(
            attachment.url and is_youtube_url(attachment.url)
            for attachment in prompt.attachments
        ) or (
            conversation
            and any(
                attachment.url and is_youtube_url(attachment.url)
                for response in conversation.responses
                for attachment in response.attachments
            )
        )

        # See https://ai.google.dev/api/generate-content#MediaResolution for mediaResolution token counts
        if prompt.options and prompt.options.media_resolution:
            generation_config["mediaResolution"] = (
                f"MEDIA_RESOLUTION_{prompt.options.media_resolution.value.upper()}"
            )
        elif has_youtube:  # support longer videos even if no option set
            generation_config["mediaResolution"] = "MEDIA_RESOLUTION_LOW"

        if generation_config:
            body["generationConfig"] = generation_config

        return body

    def process_part(self, part, response):
        if "functionCall" in part:
            response.add_tool_call(
                llm.ToolCall(
                    name=part["functionCall"]["name"],
                    arguments=part["functionCall"]["args"],
                )
            )
        if "text" in part:
            return part["text"]
        elif "executableCode" in part:
            return f'```{part["executableCode"]["language"].lower()}\n{part["executableCode"]["code"].strip()}\n```\n'
        elif "codeExecutionResult" in part:
            return f'```\n{part["codeExecutionResult"]["output"].strip()}\n```\n'
        return ""

    def process_candidates(self, candidates, response):
        # We only use the first candidate
        for part in candidates[0]["content"]["parts"]:
            yield self.process_part(part, response)

    def set_usage(self, response):
        try:
            # Don't record the "content" key from that last candidate
            for candidate in response.response_json["candidates"]:
                candidate.pop("content", None)
            usage = response.response_json.pop("usageMetadata")
            input_tokens = usage.pop("promptTokenCount", None)
            # See https://github.com/simonw/llm-gemini/issues/75#issuecomment-2861827509
            candidates_token_count = usage.get("candidatesTokenCount") or 0
            thoughts_token_count = usage.get("thoughtsTokenCount") or 0
            output_tokens = candidates_token_count + thoughts_token_count
            tool_token_count = usage.get("toolUsePromptTokenCount") or 0
            if tool_token_count:
                if input_tokens is None:
                    input_tokens = tool_token_count
                else:
                    input_tokens += tool_token_count
            usage.pop("totalTokenCount", None)
            if input_tokens is not None:
                response.set_usage(
                    input=input_tokens, output=output_tokens, details=usage or None
                )
        except (IndexError, KeyError):
            pass


class GeminiPro(_SharedGemini, llm.KeyModel):
    def execute(self, prompt, stream, response, conversation, key):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model_id}:streamGenerateContent"
        gathered = []
        body = self.build_request_body(prompt, conversation)

        with httpx.stream(
            "POST",
            url,
            timeout=prompt.options.timeout,
            headers={"x-goog-api-key": self.get_key(key)},
            json=body,
        ) as http_response:
            events = ijson.sendable_list()
            coro = ijson.items_coro(events, "item")
            for chunk in http_response.iter_bytes():
                coro.send(chunk)
                if events:
                    for event in events:
                        if isinstance(event, dict) and "error" in event:
                            raise llm.ModelError(event["error"]["message"])
                        try:
                            yield from self.process_candidates(
                                event["candidates"], response
                            )
                        except KeyError:
                            yield ""
                        gathered.append(event)
                    events.clear()
        response.response_json = gathered[-1]
        resolved_model = gathered[-1]["modelVersion"]
        response.set_resolved_model(resolved_model)
        self.set_usage(response)


class AsyncGeminiPro(_SharedGemini, llm.AsyncKeyModel):
    async def execute(self, prompt, stream, response, conversation, key):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model_id}:streamGenerateContent"
        gathered = []
        body = self.build_request_body(prompt, conversation)

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                url,
                timeout=prompt.options.timeout,
                headers={"x-goog-api-key": self.get_key(key)},
                json=body,
            ) as http_response:
                events = ijson.sendable_list()
                coro = ijson.items_coro(events, "item")
                async for chunk in http_response.aiter_bytes():
                    coro.send(chunk)
                    if events:
                        for event in events:
                            if isinstance(event, dict) and "error" in event:
                                raise llm.ModelError(event["error"]["message"])
                            try:
                                for chunk in self.process_candidates(
                                    event["candidates"], response
                                ):
                                    yield chunk
                            except KeyError:
                                yield ""
                            gathered.append(event)
                        events.clear()
        response.response_json = gathered[-1]
        self.set_usage(response)


@llm.hookimpl
def register_embedding_models(register):
    register(GeminiEmbeddingModel("text-embedding-004", "text-embedding-004"))
    # gemini-embedding-exp-03-07 in different truncation sizes
    register(
        GeminiEmbeddingModel(
            "gemini-embedding-exp-03-07", "gemini-embedding-exp-03-07"
        ),
    )
    for i in (128, 256, 512, 1024, 2048):
        register(
            GeminiEmbeddingModel(
                f"gemini-embedding-exp-03-07-{i}", f"gemini-embedding-exp-03-07", i
            ),
        )


class GeminiEmbeddingModel(llm.EmbeddingModel):
    needs_key = "gemini"
    key_env_var = "LLM_GEMINI_KEY"
    batch_size = 20

    def __init__(self, model_id, gemini_model_id, truncate=None):
        self.model_id = model_id
        self.gemini_model_id = gemini_model_id
        self.truncate = truncate

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
        values = [item["values"] for item in response.json()["embeddings"]]
        if self.truncate:
            values = [value[: self.truncate] for value in values]
        return values


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def gemini():
        "Commands relating to the llm-gemini plugin"

    @gemini.command()
    @click.option("--key", help="API key to use")
    @click.option(
        "methods",
        "--method",
        multiple=True,
        help="Filter by supported generation methods",
    )
    def models(key, methods):
        """
        List of Gemini models pulled from their API

        Use --method to filter by supported generation methods for example:

        llm gemini models --method generateContent --method embedContent
        """
        key = llm.get_key(key, "gemini", "LLM_GEMINI_KEY")
        if not key:
            raise click.ClickException(
                "You must set the LLM_GEMINI_KEY environment variable or use --key"
            )
        url = f"https://generativelanguage.googleapis.com/v1beta/models"
        response = httpx.get(url, headers={"x-goog-api-key": key})
        response.raise_for_status()
        models = response.json()["models"]
        if methods:
            models = [
                model
                for model in models
                if any(
                    method in model["supportedGenerationMethods"] for method in methods
                )
            ]
        click.echo(json.dumps(models, indent=2))

    @gemini.command()
    @click.option("--key", help="API key to use")
    def files(key):
        "List of files uploaded to the Gemini API"
        key = llm.get_key(key, "gemini", "LLM_GEMINI_KEY")
        response = httpx.get(
            f"https://generativelanguage.googleapis.com/v1beta/files?key={key}",
        )
        response.raise_for_status()
        if "files" in response.json():
            click.echo(json.dumps(response.json()["files"], indent=2))
        else:
            click.echo("No files uploaded to the Gemini API.", err=True)
