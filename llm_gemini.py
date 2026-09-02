import click
import copy
from enum import Enum
import httpx
import ijson
import json
import llm
import math
from llm.models import _partition_tools
from llm.parts import (
    AttachmentPart,
    ReasoningPart,
    StreamEvent,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
import re
from pydantic import Field, create_model
from typing import Any, Dict, Optional
from uuid import uuid4

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
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
}


def _supports_url_context(model_id):
    return model_id.startswith(("gemini-2.5", "gemini-3")) or model_id in {
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
    }


def _supports_code_execution(model_id):
    return model_id.startswith(("gemini-2.5", "gemini-3", "gemini-flash"))


def _supports_server_tool_context(model_id):
    return model_id.startswith("gemini-3") or model_id in {
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
    }


def _prepare_server_tool_request(model, body):
    if not model.supports_server_tool_context:
        return
    tool_config = body.setdefault("toolConfig", {})
    tool_config["includeServerSideToolInvocations"] = True
    tool_config.setdefault("functionCallingConfig", {})["mode"] = "VALIDATED"


class GoogleSearch(llm.ServerSideTool):
    """Ground responses using up-to-date information from Google Search."""

    name = "google_search"
    plugin = "llm-gemini"

    def __init__(self):
        super().__init__()

    def tool_spec(self, model):
        return {"googleSearch": {}}

    def prepare_request(self, model, body):
        _prepare_server_tool_request(model, body)


class URLContext(llm.ServerSideTool):
    """Fetch and use content from URLs included in the prompt."""

    name = "url_context"
    plugin = "llm-gemini"

    def __init__(self):
        super().__init__()

    def tool_spec(self, model):
        return {"urlContext": {}}

    def prepare_request(self, model, body):
        _prepare_server_tool_request(model, body)


class CodeExecution(llm.ServerSideTool):
    """Let Gemini write and execute Python code while answering."""

    name = "code_execution"
    plugin = "llm-gemini"

    def __init__(self):
        super().__init__()

    def tool_spec(self, model):
        return {"codeExecution": {}}

    def prepare_request(self, model, body):
        _prepare_server_tool_request(model, body)


SERVER_TOOL_NAMES = {
    "GOOGLE_SEARCH": "google_search",
    "GOOGLE_SEARCH_WEB": "google_search",
    "URL_CONTEXT": "url_context",
    "CODE_EXECUTION": "code_execution",
}


def _server_tool_name(tool_type):
    if not tool_type:
        return "server_side_tool"
    return SERVER_TOOL_NAMES.get(tool_type, tool_type.lower())


def _native_part_metadata(part):
    return {"gemini": {"part": copy.deepcopy(part)}}


THINKING_BUDGET_MODELS = {
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
}

# Model-specific thinking levels - models not listed here don't support thinking_level
MODEL_THINKING_LEVELS = {
    "gemini-3-flash-preview": ["minimal", "low", "medium", "high"],
    "gemini-3.1-pro-preview": ["low", "medium", "high"],
    "gemini-3.1-pro-preview-customtools": ["low", "medium", "high"],
    "gemini-3.1-flash-lite-preview": ["minimal", "low", "medium", "high"],
    "gemini-3.1-flash-lite": ["minimal", "low", "medium", "high"],
    "gemini-3.5-flash": ["minimal", "low", "medium", "high"],
    "gemini-3.6-flash": ["minimal", "low", "medium", "high"],
    "gemini-3.7-flash": ["low", "medium", "high"],
    "gemini-3.8-flash": ["low", "medium", "high"],
    "gemini-3.5-flash-lite": ["minimal", "low", "medium", "high"],
}


class MediaResolution(str, Enum):
    """Allowed media resolution values for Gemini models."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA_HIGH = "ultra_high"
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
        "gemini-2.5-flash",
        # 25th September 2025:
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        # 17th December 2025:
        "gemini-3-flash-preview",
        # 19th February 2026
        "gemini-3.1-pro-preview",
        "gemini-3.1-pro-preview-customtools",
        # 3rd March 2026
        "gemini-3.1-flash-lite-preview",
        # 2nd April 2026
        "gemma-4-26b-a4b-it",
        "gemma-4-31b-it",
        # 7th May 2026
        "gemini-3.1-flash-lite",
        # 19th May 2026
        "gemini-3.5-flash",
        # 21st July 2026
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        # 13th August 2026
        "gemini-3.7-flash",
        # 2nd September 2026
        "gemini-3.8-flash",
    ):
        can_google_search = model_id in GOOGLE_SEARCH_MODELS
        can_url_context = _supports_url_context(model_id)
        can_code_execution = _supports_code_execution(model_id)
        can_thinking_budget = model_id in THINKING_BUDGET_MODELS
        thinking_levels = MODEL_THINKING_LEVELS.get(model_id)
        register(
            GeminiPro(
                model_id,
                can_google_search=can_google_search,
                can_url_context=can_url_context,
                can_code_execution=can_code_execution,
                can_thinking_budget=can_thinking_budget,
                thinking_levels=thinking_levels,
                can_schema=True,
            ),
            AsyncGeminiPro(
                model_id,
                can_google_search=can_google_search,
                can_url_context=can_url_context,
                can_code_execution=can_code_execution,
                can_thinking_budget=can_thinking_budget,
                thinking_levels=thinking_levels,
                can_schema=True,
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

    def __init__(
        self,
        gemini_model_id,
        can_vision=True,
        can_google_search=False,
        can_url_context=False,
        can_code_execution=False,
        can_thinking_budget=False,
        thinking_levels=None,
        can_schema=False,
        can_media_resolution=True,
    ):
        self.model_id = "gemini/{}".format(gemini_model_id)
        self.gemini_model_id = gemini_model_id
        self.can_google_search = can_google_search
        self.can_url_context = can_url_context
        self.can_code_execution = can_code_execution
        self.supports_server_tool_context = _supports_server_tool_context(
            gemini_model_id
        )
        self.supports_schema = can_schema
        self.can_thinking_budget = can_thinking_budget
        self.thinking_levels = thinking_levels
        self.can_media_resolution = can_media_resolution

        # Build Options class dynamically based on capabilities
        extra_fields = {}

        if can_thinking_budget:
            extra_fields["thinking_budget"] = (
                Optional[int],
                Field(
                    description="Indicates the thinking budget in tokens. Set to 0 to disable.",
                    default=None,
                ),
            )

        if thinking_levels:
            # Create a dynamic enum with only the supported levels
            ThinkingLevelEnum = Enum(
                "ThinkingLevel",
                {level.upper(): level for level in thinking_levels},
                type=str,
            )
            level_choices = ", ".join(f"'{level}'" for level in thinking_levels)
            extra_fields["thinking_level"] = (
                Optional[ThinkingLevelEnum],
                Field(
                    description=f"Indicates the thinking level. Can be {level_choices}.",
                    default=None,
                ),
            )

        if can_media_resolution:
            extra_fields["media_resolution"] = (
                Optional[MediaResolution],
                Field(
                    description=(
                        "Media resolution for the input media (esp. YouTube) "
                        "- default is low, other values are medium, high, or unspecified"
                    ),
                    default=None,
                ),
            )

        if extra_fields:
            self.Options = create_model(
                "Options",
                __base__=self.Options,
                **extra_fields,
            )

        if can_vision:
            self.attachment_types = ATTACHMENT_TYPES

    @property
    def supported_server_side_tools(self):
        tools = []
        if self.can_google_search:
            tools.append(GoogleSearch)
        if self.can_url_context:
            tools.append(URLContext)
        if self.can_code_execution:
            tools.append(CodeExecution)
        return tuple(tools)

    def build_messages(self, prompt, conversation):
        if hasattr(prompt, "messages"):
            messages = []
            for msg in prompt.messages:
                if msg.role == "system":
                    continue
                gemini_role = "model" if msg.role == "assistant" else "user"
                parts = []
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        if part.text:
                            text_part = {"text": part.text}
                            sig = (
                                (part.provider_metadata or {})
                                .get("gemini", {})
                                .get("thoughtSignature")
                            )
                            if sig:
                                text_part["thoughtSignature"] = sig
                            parts.append(text_part)
                    elif isinstance(part, ToolCallPart):
                        if part.server_executed:
                            native_part = (
                                (part.provider_metadata or {})
                                .get("gemini", {})
                                .get("part")
                            )
                            if native_part:
                                parts.append(copy.deepcopy(native_part))
                            continue
                        fc_part = {
                            "function_call": {
                                "name": part.name,
                                "args": part.arguments,
                            }
                        }
                        if part.tool_call_id:
                            fc_part["function_call"]["id"] = part.tool_call_id
                        sig = (
                            (part.provider_metadata or {})
                            .get("gemini", {})
                            .get("thoughtSignature")
                        )
                        if sig:
                            fc_part["thoughtSignature"] = sig
                        parts.append(fc_part)
                    elif isinstance(part, ToolResultPart):
                        if part.server_executed:
                            native_part = (
                                (part.provider_metadata or {})
                                .get("gemini", {})
                                .get("part")
                            )
                            if native_part:
                                parts.append(copy.deepcopy(native_part))
                            continue
                        function_response = {
                            "name": part.name,
                            "response": {"output": part.output},
                        }
                        if part.tool_call_id:
                            function_response["id"] = part.tool_call_id
                        parts.append({"function_response": function_response})
                    elif isinstance(part, AttachmentPart) and part.attachment:
                        mime_type = resolve_type(part.attachment)
                        parts.append(
                            self._build_attachment_part(part.attachment, mime_type)
                        )
                    elif isinstance(part, ReasoningPart):
                        # Gemini does not accept visible reasoning text back in
                        # input. Tool-call state is preserved using
                        # thoughtSignature metadata on the function_call part.
                        pass
                if parts:
                    messages.append({"role": gemini_role, "parts": parts})
            return messages

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
                    for tool_call in tool_calls:
                        fc_part = {
                            "function_call": {
                                "name": tool_call.name,
                                "args": tool_call.arguments,
                            }
                        }
                        # Include thought signature if present (required for Gemini 3 models)
                        if hasattr(tool_call, "thought_signature"):
                            fc_part["thoughtSignature"] = tool_call.thought_signature
                        model_parts.append(fc_part)
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
        server_side_tools = []
        if prompt.tools:
            function_tools, server_side_tools = _partition_tools(self, prompt.tools)
            if (
                function_tools
                and server_side_tools
                and not self.supports_server_tool_context
            ):
                raise ValueError(
                    "Combining server-side tools with function tools is only "
                    "supported by Gemini 3 models"
                )
            tools.extend(tool.tool_spec(self) for tool in server_side_tools)
            if function_tools:
                tools.append(
                    {
                        "functionDeclarations": [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": cleanup_schema(
                                    copy.deepcopy(tool.input_schema)
                                ),
                            }
                            for tool in function_tools
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

        thinking_config: Dict[str, Any] = {}
        if self.can_thinking_budget and prompt.options.thinking_budget is not None:
            thinking_config["thinkingBudget"] = prompt.options.thinking_budget

        if (
            self.thinking_levels
            and getattr(prompt.options, "thinking_level", None) is not None
        ):
            # Get the string value from the enum
            thinking_level = prompt.options.thinking_level
            if hasattr(thinking_level, "value"):
                thinking_level = thinking_level.value
            thinking_config["thinkingLevel"] = thinking_level

        if (self.can_thinking_budget or self.thinking_levels) and not getattr(
            prompt, "hide_reasoning", False
        ):
            thinking_config["includeThoughts"] = True

        if thinking_config:
            generation_config["thinkingConfig"] = thinking_config

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
        if self.can_media_resolution:
            media_resolution = getattr(prompt.options, "media_resolution", None)
            if media_resolution is not None:
                generation_config["mediaResolution"] = (
                    f"MEDIA_RESOLUTION_{media_resolution.value.upper()}"
                )

        if generation_config:
            body["generationConfig"] = generation_config

        for tool in server_side_tools:
            tool.prepare_request(self, body)

        return body

    def process_part(self, part, response):
        if "functionCall" in part:
            function_call = part["functionCall"]
            tool_call_id = function_call.get("id") or f"call_{uuid4().hex}"
            tool_call = llm.ToolCall(
                name=function_call["name"],
                arguments=function_call.get("args") or {},
                tool_call_id=tool_call_id,
            )
            # Store thought signature if present (required for Gemini 3 models)
            if "thoughtSignature" in part:
                tool_call.thought_signature = part["thoughtSignature"]
            response.add_tool_call(tool_call)
            pm = (
                {"gemini": {"thoughtSignature": part["thoughtSignature"]}}
                if "thoughtSignature" in part
                else None
            )
            yield StreamEvent(
                type="tool_call_name",
                chunk=function_call["name"],
                tool_call_id=tool_call_id,
                provider_metadata=pm,
            )
            yield StreamEvent(
                type="tool_call_args",
                chunk=json.dumps(function_call.get("args") or {}),
                tool_call_id=tool_call_id,
            )
            return

        if "toolCall" in part:
            tool_call = part["toolCall"]
            tool_type = tool_call.get("toolType")
            tool_name = _server_tool_name(tool_type)
            tool_call_id = tool_call.get("id") or f"{tool_name}_{uuid4().hex}"
            if response is not None:
                pending_ids = getattr(response, "_gemini_server_tool_ids", {})
                pending_ids[tool_type] = tool_call_id
                response._gemini_server_tool_ids = pending_ids
            yield StreamEvent(
                type="tool_call_name",
                chunk=tool_name,
                tool_call_id=tool_call_id,
                server_executed=True,
                provider_metadata=_native_part_metadata(part),
            )
            yield StreamEvent(
                type="tool_call_args",
                chunk=json.dumps(tool_call.get("args") or {}),
                tool_call_id=tool_call_id,
                server_executed=True,
            )
            return

        if "toolResponse" in part:
            tool_response = part["toolResponse"]
            tool_type = tool_response.get("toolType")
            tool_name = _server_tool_name(tool_type)
            pending_ids = getattr(response, "_gemini_server_tool_ids", {})
            tool_call_id = (
                tool_response.get("id")
                or pending_ids.get(tool_type)
                or f"{tool_name}_{uuid4().hex}"
            )
            yield StreamEvent(
                type="tool_result",
                chunk=json.dumps(tool_response.get("response", {})),
                tool_call_id=tool_call_id,
                server_executed=True,
                tool_name=tool_name,
                provider_metadata=_native_part_metadata(part),
            )
            return

        if "text" in part:
            text = part["text"]
            if not text and "thoughtSignature" not in part:
                return
            provider_metadata = None
            if "thoughtSignature" in part:
                provider_metadata = {
                    "gemini": {"thoughtSignature": part["thoughtSignature"]}
                }
            if part.get("thought"):
                yield StreamEvent(
                    type="reasoning",
                    chunk=text,
                    provider_metadata=provider_metadata,
                )
            else:
                yield StreamEvent(
                    type="text",
                    chunk=text,
                    provider_metadata=provider_metadata,
                )
            return

        if "executableCode" in part:
            executable_code = part["executableCode"]
            tool_call_id = executable_code.get("id") or f"code_{uuid4().hex}"
            if response is not None:
                response._gemini_code_execution_id = tool_call_id
            yield StreamEvent(
                type="tool_call_name",
                chunk="code_execution",
                tool_call_id=tool_call_id,
                server_executed=True,
                provider_metadata=_native_part_metadata(part),
            )
            yield StreamEvent(
                type="tool_call_args",
                chunk=json.dumps(executable_code),
                tool_call_id=tool_call_id,
                server_executed=True,
            )
            return

        if "codeExecutionResult" in part:
            code_execution_result = part["codeExecutionResult"]
            tool_call_id = (
                code_execution_result.get("id")
                or getattr(response, "_gemini_code_execution_id", None)
                or f"code_{uuid4().hex}"
            )
            yield StreamEvent(
                type="tool_result",
                chunk=json.dumps(code_execution_result),
                tool_call_id=tool_call_id,
                server_executed=True,
                tool_name="code_execution",
                provider_metadata=_native_part_metadata(part),
            )

    def process_candidates(self, candidates, response):
        # We only use the first candidate
        candidate = candidates[0]
        events = []
        for part in candidate.get("content", {}).get("parts", []):
            events.extend(self.process_part(part, response))

        grounding = candidate.get("groundingMetadata")
        if grounding:
            grounding_metadata = {"gemini": {"groundingMetadata": grounding}}
            for event in reversed(events):
                if event.type == "text":
                    provider_metadata = dict(event.provider_metadata or {})
                    gemini_metadata = dict(provider_metadata.get("gemini", {}))
                    gemini_metadata["groundingMetadata"] = grounding
                    provider_metadata["gemini"] = gemini_metadata
                    event.provider_metadata = provider_metadata
                    break
            else:
                # If grounding arrives in a metadata-only streaming event, this
                # empty text event will group with the preceding text part.
                events.append(
                    StreamEvent(
                        type="text",
                        chunk="",
                        provider_metadata=grounding_metadata,
                    )
                )

        yield from events

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
            # Keep provider metadata JSON-serializable: ijson otherwise parses
            # fractional values such as grounding confidence scores as Decimal.
            coro = ijson.items_coro(events, "item", use_float=True)
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
                            yield StreamEvent(type="text", chunk="")
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
                coro = ijson.items_coro(events, "item", use_float=True)
                async for chunk in http_response.aiter_bytes():
                    coro.send(chunk)
                    if events:
                        for event in events:
                            if isinstance(event, dict) and "error" in event:
                                raise llm.ModelError(event["error"]["message"])
                            try:
                                for stream_event in self.process_candidates(
                                    event["candidates"], response
                                ):
                                    yield stream_event
                            except KeyError:
                                yield StreamEvent(type="text", chunk="")
                            gathered.append(event)
                        events.clear()
        response.response_json = gathered[-1]
        resolved_model = gathered[-1]["modelVersion"]
        response.set_resolved_model(resolved_model)
        self.set_usage(response)


@llm.hookimpl
def register_embedding_models(register):
    for model_id in ("gemini-embedding-2", "gemini-embedding-001"):
        register(GeminiEmbeddingModel(model_id, model_id))
        for output_dimensionality in (768, 1536):
            register(
                GeminiEmbeddingModel(
                    f"{model_id}-{output_dimensionality}",
                    model_id,
                    output_dimensionality,
                )
            )


class GeminiEmbeddingModel(llm.EmbeddingModel):
    needs_key = "gemini"
    key_env_var = "LLM_GEMINI_KEY"
    batch_size = 20

    def __init__(self, model_id, gemini_model_id, output_dimensionality=None):
        self.model_id = model_id
        self.gemini_model_id = gemini_model_id
        self.output_dimensionality = output_dimensionality

    def embed_batch(self, items):
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.get_key(),
        }
        requests = []
        for item in items:
            request = {
                "model": "models/" + self.gemini_model_id,
                "content": {"parts": [{"text": item}]},
            }
            if self.output_dimensionality:
                request["outputDimensionality"] = self.output_dimensionality
            requests.append(request)
        data = {"requests": requests}

        with httpx.Client() as client:
            response = client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model_id}:batchEmbedContents",
                headers=headers,
                json=data,
                timeout=None,
            )

        response.raise_for_status()
        values = [item["values"] for item in response.json()["embeddings"]]
        # gemini-embedding-2 normalizes reduced vectors automatically, but
        # gemini-embedding-001 requires callers to normalize them.
        if (
            self.gemini_model_id == "gemini-embedding-001"
            and self.output_dimensionality
        ):
            normalized = []
            for value in values:
                magnitude = math.sqrt(sum(component**2 for component in value))
                normalized.append(
                    [component / magnitude for component in value]
                    if magnitude
                    else value
                )
            values = normalized
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
