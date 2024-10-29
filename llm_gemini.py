import httpx
import ijson
import llm
import urllib.parse

# We disable all of these to avoid random unexpected errors
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


@llm.hookimpl
def register_models(register):
    register(GeminiPro("gemini-pro"))
    register(GeminiPro("gemini-1.5-pro-latest"))
    register(GeminiPro("gemini-1.5-flash-latest"))
    register(GeminiPro("gemini-1.5-pro-001"))
    register(GeminiPro("gemini-1.5-flash-001"))
    register(GeminiPro("gemini-1.5-pro-002"))
    register(GeminiPro("gemini-1.5-flash-002"))
    register(GeminiPro("gemini-1.5-flash-8b-latest"))
    register(GeminiPro("gemini-1.5-flash-8b-001"))


def resolve_type(attachment):
    mime_type = attachment.resolve_type()
    # https://github.com/simonw/llm/issues/587#issuecomment-2439785140
    if mime_type == "audio/mpeg":
        mime_type = "audio/mp3"
    return mime_type


class GeminiPro(llm.Model):
    needs_key = "gemini"
    key_env_var = "LLM_GEMINI_KEY"
    can_stream = True

    attachment_types = (
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
    )

    def __init__(self, model_id):
        self.model_id = model_id

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
                parts.append({"text": response.prompt.prompt})
                messages.append({"role": "user", "parts": parts})
                messages.append({"role": "model", "parts": [{"text": response.text()}]})

        parts = [{"text": prompt.prompt}]
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

    def execute(self, prompt, stream, response, conversation):
        key = self.get_key()
        url = "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?".format(
            self.model_id
        ) + urllib.parse.urlencode(
            {"key": key}
        )
        gathered = []
        body = {
            "contents": self.build_messages(prompt, conversation),
            "safetySettings": SAFETY_SETTINGS,
        }
        if prompt.system:
            body["systemInstruction"] = {"parts": [{"text": prompt.system}]}
        with httpx.stream(
            "POST",
            url,
            timeout=None,
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
                        yield event["candidates"][0]["content"]["parts"][0]["text"]
                    except KeyError:
                        yield ""
                    gathered.append(event)
                    events.clear()
        response.response_json = gathered


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
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model_id}:batchEmbedContents?key={self.get_key()}",
                headers=headers,
                json=data,
                timeout=None,
            )

        response.raise_for_status()
        return [item["values"] for item in response.json()["embeddings"]]
