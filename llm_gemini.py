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
    register(GeminiPro())


class GeminiPro(llm.Model):
    model_id = "gemini-pro"
    can_stream = True

    def build_messages(self, prompt, conversation):
        if not conversation:
            return [{"role": "user", "parts": [{"text": prompt.prompt}]}]
        messages = []
        for response in conversation.responses:
            messages.append({"role": "user", "parts": [{"text": response.prompt.prompt}]})
            messages.append({"role": "model", "parts": [{"text": response.text()}]})
        messages.append({"role": "user", "parts": [{"text": prompt.prompt}]})
        return messages

    def execute(self, prompt, stream, response, conversation):
        key = llm.get_key("", "gemini", "LLM_GEMINI_KEY")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:streamGenerateContent?"
            + urllib.parse.urlencode({"key": key})
        )
        gathered = []
        with httpx.stream(
            "POST",
            url,
            timeout=None,
            json={
                "contents": self.build_messages(prompt, conversation),
                "safetySettings": SAFETY_SETTINGS,
            },
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
