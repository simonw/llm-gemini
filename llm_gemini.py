import httpx
import ijson
import llm
import urllib.parse
from pydantic import BaseModel

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


class GeminiPro(llm.Model):
    can_stream = True

    class Options(BaseModel):
        """
            some generation_config parameters 
            complete list https://ai.google.dev/api/python/google/generativeai/types/GenerationConfig
        """
        max_output_tokens: int = 1024
        temperature: float = 1.0 
        top_p: int = 1
        top_k: int = 32

        model_config = {
                "extra": 'forbid'
        }


    def __init__(self, model_id):
        self.model_id = model_id

    def build_messages(self, prompt, conversation):
        if not conversation:
            return [{"role": "user", "parts": [{"text": prompt.prompt}]}]
        messages = []
        for response in conversation.responses:
            messages.append(
                {"role": "user", "parts": [{"text": response.prompt.prompt}]}
            )
            messages.append({"role": "model", "parts": [{"text": response.text()}]})
        messages.append({"role": "user", "parts": [{"text": prompt.prompt}]})
        return messages

    def execute(self, prompt, stream, response, conversation):
        key = llm.get_key("", "gemini", "LLM_GEMINI_KEY")
        url = "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?".format(
            self.model_id
        ) + urllib.parse.urlencode(
            {"key": key}
        )
        gathered = []
        body = {
            "contents": self.build_messages(prompt, conversation),
            "safetySettings": SAFETY_SETTINGS,
            "generation_config": dict(prompt.options)
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
