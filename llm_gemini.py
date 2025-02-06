from bs4 import BeautifulSoup
import httpx
import ijson
import llm
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
        # Released 5th Feb 2025:
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite-preview-02-05",
        "gemini-2.0-pro-exp-02-05",
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


def format_link(osc8format, title, link):
    if osc8format:
        return f"\x1b]8;;{link}\a{title}\x1b]8;;\a"
    else:
        return f"{title}: {link}"


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

    class OptionsWithGoogleSearch(Options):
        google_search: Optional[bool] = Field(
            description="Enables the model to use Google Search to improve the accuracy and recency of responses from the model",
            default=None,
        )
        grounding_links: Optional[bool] = Field(
            description="Whether to show the grounding links from the response",
            default=True
        )
        format_links: Optional[bool] = Field(
            description="Whether to format the grounding links and search recommendation with OSC 8 escape sequences",
            default=True
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

    def process_grounding(self, event, options):
        if not options or not self.can_google_search or not options.google_search:
            return ""

        try:
            text = ""
            grounding = event["candidates"][0]["groundingMetadata"]
            if options.grounding_links and grounding:
                chunks = grounding["groundingChunks"]
                if chunks:
                    text += "\n\nGrounding Sources:\n"
                    for index, chunk in enumerate(chunks):
                        if "web" in chunk:
                            title = chunk['web']['title']
                            href = chunk['web']['uri']
                            text += f"{index + 1}. {format_link(options.format_links, title, href)}\n"
                rendered_content = grounding["searchEntryPoint"]["renderedContent"]
                if rendered_content:
                    soup = BeautifulSoup(rendered_content, 'html.parser')
                    links = soup.find_all('a')
                    if links:
                        text += f"\nGoogle Search Suggestions:\n"
                        for link in links:
                            title = link.string
                            href = link['href']
                            text += f"{format_link(options.format_links, title, href)}\n"
            return text
        except KeyError:
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
                        text = self.process_part(part)
                        text += self.process_grounding(event, prompt.options)
                        yield text
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
                            text = self.process_part(part)
                            text + self.process_grounding(event, prompt.options)
                            yield text
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
