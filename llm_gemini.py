import httpx
import ijson
import llm
from typing import Optional, List
from pydantic import Field
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

class GeminiOptions(llm.Options):
    max_tokens: Optional[int] = Field(
        description="The maximum number of tokens to generate before stopping",
        default=8000,
    )

    temperature: Optional[float] = Field(
        description="Amount of randomness injected into the response. Defaults to 1.0. Ranges from 0.0 to 1.0. Use temperature closer to 0.0 for analytical / multiple choice, and closer to 1.0 for creative and generative tasks. Note that even with temperature of 0.0, the results will not be fully deterministic.",
        default=1.0,
    )

    top_p: Optional[float] = Field(
        description="For Nucleus sampling. Nucleus sampling considers the smallest set of tokens whose probability sum is at least topP.",
        default=None,
    )

    top_k: Optional[int] = Field(
        description="For Top-k sampling. Top-k sampling considers the set of topK most probable tokens. If empty, indicates the model doesn't use top-k sampling, and topK isn't allowed as a generation parameter.",
        default=None,
    )


def fetch_available_models(api_key: str) -> List[str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    with httpx.Client() as client:
        response = client.get(url)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model["name"].split("/")[-1] for model in models if "generateContent" in model.get("supportedGenerationMethods", [])]

@llm.hookimpl
def register_models(register):
    api_key = llm.get_key("", "gemini", "LLM_GEMINI_KEY")
    available_models = fetch_available_models(api_key)
    for model_id in available_models:
        register(GeminiPro(model_id))


class GeminiPro(llm.Model):
    can_stream = True
    
    class Options(GeminiOptions): ...
    
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
    api_key = llm.get_key("", "gemini", "LLM_GEMINI_KEY")
    available_models = fetch_available_models(api_key)
    embedding_models = [model for model in available_models if "embedContent" in model]
    for model_id in embedding_models:
        register(GeminiEmbeddingModel(model_id, model_id))


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
