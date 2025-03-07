import copy
import httpx
import ijson
import llm
import time
import click
from pathlib import Path
from pydantic import Field
from typing import Optional, Any

KEY_NAME = "gemini"
ENV_NAME = "LLM_GEMINI_KEY"

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
}

# Size threshold for using the Files API (5MB in bytes)
FILE_SIZE_THRESHOLD = 5 * 1024 * 1024


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
        # Released 5th Feb 2025:
        "gemini-2.0-flash",
        "gemini-2.0-pro-exp-02-05",
        # Released 25th Feb 2025:
        "gemini-2.0-flash-lite",
    ]:
        can_google_search = model_id in GOOGLE_SEARCH_MODELS
        register(
            GeminiPro(
                model_id,
                can_google_search=can_google_search,
                can_schema="flash-thinking" not in model_id,
            ),
            AsyncGeminiPro(
                model_id,
                can_google_search=can_google_search,
                can_schema="flash-thinking" not in model_id,
            ),
        )


@llm.hookimpl
def register_commands(cli):
    @cli.group(name="gemini")
    def gemini_group():
        """Commands for managing Gemini Files API"""
        pass

    @gemini_group.command(name="list-files")
    @click.option("--key", help="Gemini API key")
    @click.option("json_", "--json", is_flag=True, help="Output as JSON")
    def gemini_list_files(key, json_):
        """List files that have been uploaded to the Gemini Files API"""
        gemini_key = llm.get_key(key, KEY_NAME, ENV_NAME)

        try:
            response = httpx.get(
                "https://generativelanguage.googleapis.com/v1beta/files",
                headers={"x-goog-api-key": gemini_key},
                timeout=10,
            )

            if response.status_code != 200:
                click.echo(
                    f"Error listing files: {response.status_code} - {response.text}"
                )
                return

            data = response.json()
            files = data.get("files", [])
            if json_:
                click.echo(json.dumps(files, indent=2))
                return

            if not files:
                click.echo("No files found.", err=True)
                return

            click.echo(f"Found {len(files)} uploaded files:")
            for file in files:
                file_name = file['name']
                click.echo(
                    f"File name: {file_name}\n"
                    f"  MIME Type: {file.get('mimeType', 'N/A')}\n"
                    f"  Create Time: {file.get('createTime', 'N/A')}\n"
                    f"  Expiration Time: {file.get('expirationTime', 'N/A')}\n"
                    f"  Size: {file.get('sizeBytes', 'N/A')} bytes\n"
                )
        except Exception as e:
            click.echo(f"Error: {str(e)}")

    @gemini_group.command(name="delete")
    @click.argument("file_id")
    def gemini_delete(file_id):
        """Delete a file from Gemini Files API"""
        gemini_key = llm.get_key(None, KEY_NAME, ENV_NAME)

        try:
            response = httpx.delete(
                f"https://generativelanguage.googleapis.com/v1beta/files/{file_id}",
                headers={"x-goog-api-key": gemini_key},
                timeout=10,
            )

            if response.status_code in (200, 204):
                click.echo(f"Successfully deleted file {file_id}")
            else:
                click.echo(
                    f"Failed to delete file {file_id}: {response.status_code} - {response.text}"
                )
        except Exception as e:
            click.echo(f"Error: {str(e)}")

    @gemini_group.command(name="upload")
    @click.argument("file", type=click.Path(exists=True))
    def gemini_upload(file):
        """Upload a file to Gemini Files API for reuse"""
        gemini_key = llm.get_key(None, KEY_NAME, ENV_NAME)

        # Create an attachment from the file
        path = Path(file)
        attachment = llm.Attachment(path=str(path.resolve()))

        try:
            # Generate a valid filename (alphanumeric and dashes only, max 40 chars)
            attachment_id = attachment.id()
            response = httpx.get(
                f"https://generativelanguage.googleapis.com/upload/v1beta/files",
                headers={"x-goog-api-key": gemini_key},
                timeout=10,
            )

            if response.status_code == 200:
                click.echo(f"File already exists with ID: {safe_filename}")
                return

            # Upload the file
            files = {
                "file": (
                    attachment_id,
                    attachment.content_bytes(),
                    attachment.resolve_type(),
                )
            }

            response = httpx.post(
                "https://generativelanguage.googleapis.com/v1beta/files",
                headers={"x-goog-api-key": gemini_key},
                files=files,
                timeout=None,  # No timeout for large file uploads
            )
            print(files)

            if response.status_code == 200:
                file_data = response.json()
                print(file_data)
                file_id = file_data.get("name", "").split("/")[-1]
                click.echo(f"Successfully uploaded file. File ID: {file_id}")
            else:
                click.echo(
                    f"Failed to upload file: {response.status_code} - {response.text}"
                )
        except Exception as e:
            click.echo(f"Error: {str(e)}")


def resolve_type(attachment):
    mime_type = attachment.resolve_type()
    # https://github.com/simonw/llm/issues/587#issuecomment-2439785140
    if mime_type == "audio/mpeg":
        mime_type = "audio/mp3"
    if mime_type == "application/ogg":
        mime_type = "audio/ogg"
    return mime_type


def cleanup_schema(schema):
    "Gemini supports only a subset of JSON schema"
    keys_to_remove = ("$schema", "additionalProperties", "title")
    # Recursively remove them
    if isinstance(schema, dict):
        for key in keys_to_remove:
            schema.pop(key, None)
        for value in schema.values():
            cleanup_schema(value)
    elif isinstance(schema, list):
        for value in schema:
            cleanup_schema(value)
    return schema


def should_use_files_api(attachment):
    """Determine if an individual attachment should use the Files API based on its size"""
    try:
        size = len(attachment.content_bytes())
        return size > FILE_SIZE_THRESHOLD
    except Exception:
        # If we can't determine the size, assume it's large
        return True


def check_file_exists(file_id, api_key):
    """Check if a file exists in Gemini's Files API"""
    try:
        response = httpx.get(
            f"https://generativelanguage.googleapis.com/v1beta/files/{file_id}",
            headers={"x-goog-api-key": api_key},
            timeout=10,
        )
        return response.status_code == 200
    except Exception:
        return False


class UploadError(Exception):
    pass


def upload_file(attachment, api_key, wait_until_active=True):
    """Upload a file to Gemini's Files API"""
    print("Uploading", attachment)
    try:
        # Truncate to 32 because it's max of 40 and needs room for files/ prefix
        filename = attachment.id()[:32]
        print("filename will be", filename)
        if check_file_exists(filename, api_key):
            return filename
        response = httpx.post(
            "https://generativelanguage.googleapis.com/upload/v1beta/files",
            headers={"x-goog-api-key": api_key},
            files={
                "file": (
                    filename,
                    attachment.content_bytes(),
                    attachment.resolve_type(),
                )
            },
            timeout=None,
        )
        if response.status_code == 200:
            print('response.json()', response.json())
            if not wait_until_active:
                return response.json()
            else:
                attempts = 0
                while response.json()['file']['state'] == 'PROCESSING':
                    attempts += 1
                    if attempts > 10:
                        raise Exception("File upload took too long to process")
                    time.sleep(2)
                    response = httpx.get(
                        f"https://generativelanguage.googleapis.com/v1beta/files/{filename}",
                        headers={"x-goog-api-key": api_key},
                        timeout=10,
                    )
                return response.json()
        else:
            raise Exception(
                f"Failed to upload file: {response.status_code} - {response.text}"
            )
    except Exception as ex:
        raise UploadError(str(ex))


class _SharedGemini:
    needs_key = KEY_NAME
    key_env_var = ENV_NAME
    can_stream = True
    supports_schema = True

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
        force_files_api: Optional[bool] = Field(
            description="Force using Gemini Files API for all attachments",
            default=None,
        )

    class OptionsWithGoogleSearch(Options):
        google_search: Optional[bool] = Field(
            description="Enables the model to use Google Search to improve the accuracy and recency of responses from the model",
            default=None,
        )

    def __init__(self, model_id, can_google_search=False, can_schema=False):
        self.model_id = model_id
        self.can_google_search = can_google_search
        self.supports_schema = can_schema
        if can_google_search:
            self.Options = self.OptionsWithGoogleSearch

    def process_attachment(self, attachment, key, force_files_api=False):
        """Process a single attachment, deciding whether to use Files API or inline data"""
        if force_files_api or should_use_files_api(attachment):
            # Use Files API
            file_info = upload_file(attachment, key)
            uri = file_info["file"]["uri"]
            return {
                "file_data": {
                    "file_uri": uri,
                    "mime_type": attachment.resolve_type(),
                }
            }

        # Default to inline data
        return {
            "inlineData": {
                "data": attachment.base64_content(),
                "mimeType": resolve_type(attachment),
            }
        }

    def build_messages(self, prompt, conversation, key):
        messages = []
        force_files_api = prompt.options and getattr(
            prompt.options, "force_files_api", False
        )

        if conversation:
            for response in conversation.responses:
                parts = []
                for attachment in response.attachments:
                    parts.append(
                        self.process_attachment(attachment, key, force_files_api)
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
            parts.append(self.process_attachment(attachment, key, force_files_api))

        messages.append({"role": "user", "parts": parts})
        return messages

    def build_request_body(self, prompt, conversation, key):
        body = {
            "contents": self.build_messages(prompt, conversation, key),
            "safetySettings": SAFETY_SETTINGS,
        }
        if prompt.options and prompt.options.code_execution:
            body["tools"] = [{"codeExecution": {}}]
        if prompt.options and self.can_google_search and prompt.options.google_search:
            body["tools"] = [{"google_search": {}}]
        if prompt.system:
            body["systemInstruction"] = {"parts": [{"text": prompt.system}]}

        if prompt.schema:
            body["generationConfig"] = {
                "response_mime_type": "application/json",
                "response_schema": cleanup_schema(copy.deepcopy(prompt.schema)),
            }

        config_map = {
            "temperature": "temperature",
            "max_output_tokens": "maxOutputTokens",
            "top_p": "topP",
            "top_k": "topK",
        }
        if prompt.options and prompt.options.json_object:
            body["generationConfig"] = {"response_mime_type": "application/json"}

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

    def process_candidates(self, candidates):
        # We only use the first candidate
        for part in candidates[0]["content"]["parts"]:
            yield self.process_part(part)

    def set_usage(self, response):
        try:
            # Don't record the "content" key from that last candidate
            for candidate in response.response_json["candidates"]:
                candidate.pop("content", None)
            usage = response.response_json.pop("usageMetadata")
            input_tokens = usage.pop("promptTokenCount", None)
            output_tokens = usage.pop("candidatesTokenCount", None)
            usage.pop("totalTokenCount", None)
            if input_tokens is not None:
                response.set_usage(
                    input=input_tokens, output=output_tokens, details=usage or None
                )
        except (IndexError, KeyError):
            pass


class GeminiPro(_SharedGemini, llm.KeyModel):
    def execute(self, prompt, stream, response, conversation, key):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:streamGenerateContent"
        gathered = []
        body = self.build_request_body(prompt, conversation, key)

        with httpx.stream(
            "POST",
            url,
            timeout=None,
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
                            yield from self.process_candidates(event["candidates"])
                        except KeyError:
                            yield ""
                        gathered.append(event)
                    events.clear()
        response.response_json = gathered[-1]
        self.set_usage(response)


class AsyncGeminiPro(_SharedGemini, llm.AsyncKeyModel):
    async def execute(self, prompt, stream, response, conversation, key):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:streamGenerateContent"
        gathered = []
        body = self.build_request_body(prompt, conversation, key)

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                url,
                timeout=None,
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
                                    event["candidates"]
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
    register(
        GeminiEmbeddingModel("text-embedding-004", "text-embedding-004"),
    )


class GeminiEmbeddingModel(llm.EmbeddingModel):
    needs_key = KEY_NAME
    key_env_var = ENV_NAME
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


import httpx
import json
import asyncio
from typing import Optional, Dict, Any, Union


async def async_upload_file_resumable(
    attachment, api_key: str, display_name: Optional[str] = None
) -> Optional[str]:
    """
    Upload a file to Gemini Files API using the resumable upload protocol.

    Args:
        attachment: The attachment object with content_bytes() and resolve_type() methods
        api_key: The Gemini API key
        display_name: Optional display name for the file (defaults to attachment ID)

    Returns:
        The file_id if successful, None otherwise
    """
    try:
        # Get file content and metadata
        content = attachment.content_bytes()
        mime_type = attachment.resolve_type()
        num_bytes = len(content)

        # Use attachment ID as display name if not provided
        if not display_name:
            display_name = attachment.id()[:40]

        # Base URL for the upload
        base_url = "https://generativelanguage.googleapis.com/upload/v1beta/files"

        # Step 1: Initialize the resumable upload and get the upload URL
        async with httpx.AsyncClient() as client:
            # Make the initial request to start the resumable upload
            headers = {
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(num_bytes),
                "X-Goog-Upload-Header-Content-Type": mime_type,
                "Content-Type": "application/json",
            }

            # Create the request with file metadata
            metadata = json.dumps({"file": {"display_name": display_name}})

            # Get the upload URL from the response headers
            init_response = await client.post(
                f"{base_url}?key={api_key}",
                headers=headers,
                content=metadata,
                timeout=30,
            )

            if init_response.status_code != 200:
                print(
                    f"Failed to initialize upload: {init_response.status_code} - {init_response.text}"
                )
                return None

            # Extract the upload URL from headers
            upload_url = init_response.headers.get("x-goog-upload-url")
            if not upload_url:
                print("Failed to get upload URL from response headers")
                return None

            # Step 2: Upload the actual file data
            upload_headers = {
                "Content-Length": str(num_bytes),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            }

            upload_response = await client.post(
                upload_url,
                headers=upload_headers,
                content=content,
                timeout=None,  # No timeout for large uploads
            )

            if upload_response.status_code != 200:
                print(
                    f"Failed to upload file: {upload_response.status_code} - {upload_response.text}"
                )
                return None

            # Parse the response to get the file ID
            try:
                response_data = upload_response.json()
                print(response_data)
                # Extract the file ID from the response
                file_name = response_data.get("file", {}).get("name", "")
                if file_name:
                    # Extract the ID from the full resource name
                    file_id = file_name.split("/")[-1]
                    return file_id
                else:
                    print("File ID not found in response")
                    return None
            except Exception as e:
                print(f"Failed to parse response: {str(e)}")
                return None

    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        return None
