# llm-gemini

[![PyPI](https://img.shields.io/pypi/v/llm-gemini.svg)](https://pypi.org/project/llm-gemini/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-gemini?include_prereleases&label=changelog)](https://github.com/simonw/llm-gemini/releases)
[![Tests](https://github.com/simonw/llm-gemini/workflows/Test/badge.svg)](https://github.com/simonw/llm-gemini/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-gemini/blob/main/LICENSE)

API access to Google's Gemini models

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-gemini
```
## Usage

Configure the model by setting a key called "gemini" to your [API key](https://aistudio.google.com/app/apikey):
```bash
llm keys set gemini
```
```
<paste key here>
```
You can also set the API key by assigning it to the environment variable `LLM_GEMINI_KEY`.

Now run the model using `-m gemini-2.0-flash`, for example:

```bash
llm -m gemini-2.0-flash "A short joke about a pelican and a walrus"
```

> A pelican and a walrus are sitting at a bar. The pelican orders a fishbowl cocktail, and the walrus orders a plate of clams. The bartender asks, "So, what brings you two together?"
>
> The walrus sighs and says, "It's a long story. Let's just say we met through a mutual friend... of the fin."

You can set the [default model](https://llm.datasette.io/en/stable/setup.html#setting-a-custom-default-model) to avoid the extra `-m` option:

```bash
llm models default gemini-2.0-flash
llm "A joke about a pelican and a walrus"
```

## Available models

<!-- [[[cog
import cog
from llm import cli
from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(cli.cli, ["models", "-q", "gemini/"])
lines = reversed(result.output.strip().split("\n"))
to_output = []
NOTES = {
    "gemini/gemini-3-pro-preview": "Gemini 3 Pro Preview",
    "gemini/gemini-flash-latest": "Latest Gemini Flash",
    "gemini/gemini-flash-lite-latest": "Latest Gemini Flash Lite",
    "gemini/gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini/gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini/gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "gemini/gemini-2.5-flash-preview-05-20": "Gemini 2.5 Flash preview (priced differently from 2.5 Flash)",
    "gemini/gemini-2.0-flash-thinking-exp-01-21": "Experimental \"thinking\" model from January 2025",
    "gemini/gemini-1.5-flash-8b-latest": "The least expensive model",
}
for line in lines:
    model_id, rest = line.split(None, 2)[1:]
    note = NOTES.get(model_id, "")
    to_output.append(
        "- `{}`{}".format(
            model_id,
            ': {}'.format(note) if note else ""
        )
    )
cog.out("\n".join(to_output))
]]] -->
- `gemini/gemini-3-pro-preview`: Gemini 3 Pro Preview
- `gemini/gemini-2.5-flash-lite-preview-09-2025`
- `gemini/gemini-2.5-flash-preview-09-2025`
- `gemini/gemini-flash-lite-latest`: Latest Gemini Flash Lite
- `gemini/gemini-flash-latest`: Latest Gemini Flash
- `gemini/gemini-2.5-flash-lite`: Gemini 2.5 Flash Lite
- `gemini/gemini-2.5-pro`: Gemini 2.5 Pro
- `gemini/gemini-2.5-flash`: Gemini 2.5 Flash
- `gemini/gemini-2.5-pro-preview-06-05`
- `gemini/gemini-2.5-flash-preview-05-20`: Gemini 2.5 Flash preview (priced differently from 2.5 Flash)
- `gemini/gemini-2.5-pro-preview-05-06`
- `gemini/gemini-2.5-flash-preview-04-17`
- `gemini/gemini-2.5-pro-preview-03-25`
- `gemini/gemini-2.5-pro-exp-03-25`
- `gemini/gemini-2.0-flash-lite`
- `gemini/gemini-2.0-pro-exp-02-05`
- `gemini/gemini-2.0-flash`
- `gemini/gemini-2.0-flash-thinking-exp-01-21`: Experimental "thinking" model from January 2025
- `gemini/gemini-2.0-flash-thinking-exp-1219`
- `gemini/gemma-3n-e4b-it`
- `gemini/gemma-3-27b-it`
- `gemini/gemma-3-12b-it`
- `gemini/gemma-3-4b-it`
- `gemini/gemma-3-1b-it`
- `gemini/learnlm-1.5-pro-experimental`
- `gemini/gemini-2.0-flash-exp`
- `gemini/gemini-exp-1206`
- `gemini/gemini-exp-1121`
- `gemini/gemini-exp-1114`
- `gemini/gemini-1.5-flash-8b-001`
- `gemini/gemini-1.5-flash-8b-latest`: The least expensive model
- `gemini/gemini-1.5-flash-002`
- `gemini/gemini-1.5-pro-002`
- `gemini/gemini-1.5-flash-001`
- `gemini/gemini-1.5-pro-001`
- `gemini/gemini-1.5-flash-latest`
- `gemini/gemini-1.5-pro-latest`
- `gemini/gemini-pro`
<!-- [[[end]]] -->

All of these models have aliases that omit the `gemini/` prefix, for example:

```bash
llm -m gemini-1.5-flash-8b-latest --schema 'name,age int,bio' 'invent a dog'
```

### Images, audio and video

Gemini models are multi-modal. You can provide images, audio or video files as input like this:

```bash
llm -m gemini-2.0-flash 'extract text' -a image.jpg
```
Or with a URL:
```bash
llm -m gemini-2.0-flash-lite 'describe image' \
  -a https://static.simonwillison.net/static/2024/pelicans.jpg
```
Audio works too:

```bash
llm -m gemini-2.0-flash 'transcribe audio' -a audio.mp3
```

And video:

```bash
llm -m gemini-2.0-flash 'describe what happens' -a video.mp4
```
The Gemini prompting guide includes [extensive advice](https://ai.google.dev/gemini-api/docs/file-prompting-strategies) on multi-modal prompting.

### YouTube videos

You can provide YouTube video URLs as attachments as well:

```bash
llm -m gemini-3-pro-preview -a 'https://www.youtube.com/watch?v=9o1_DL9uNlM' \
  'Produce a summary with relevant URLs and code example snippets, then an accurate transcript with timestamps.'
```
[Example output here](https://gist.github.com/simonw/1b07aafb2bfc112b180ab68c864511cb).

These will be processed with media resolution `low` by default. You can use the `-o media_resolution X` option to set that to `medium`, `high`, or `unspecified`.

### JSON output

Use `-o json_object 1` to force the output to be JSON:

```bash
llm -m gemini-2.0-flash -o json_object 1 \
  '3 largest cities in California, list of {"name": "..."}'
```
Outputs:
```json
{"cities": [{"name": "Los Angeles"}, {"name": "San Diego"}, {"name": "San Jose"}]}
```

### Code execution

Gemini models can [write and execute code](https://ai.google.dev/gemini-api/docs/code-execution) - they can decide to write Python code, execute it in a secure sandbox and use the result as part of their response.

To enable this feature, use `-o code_execution 1`:

```bash
llm -m gemini-2.0-flash -o code_execution 1 \
'use python to calculate (factorial of 13) * 3'
```
### Google search

Some Gemini models support [Grounding with Google Search](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-gemini#web-ground-gemini), where the model can run a Google search and use the results as part of answering a prompt.

Using this feature may incur additional requirements in terms of how you use the results. Consult [Google's documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-gemini#web-ground-gemini) for more details.

To run a prompt with Google search enabled, use `-o google_search 1`:

```bash
llm -m gemini-2.0-flash -o google_search 1 \
  'What happened in Ireland today?'
```

Use `llm logs -c --json` after running a prompt to see the full JSON response, which includes [additional information](https://github.com/simonw/llm-gemini/pull/29#issuecomment-2606201877) about grounded results.

### URL context

Gemini models support a [URL context](https://ai.google.dev/gemini-api/docs/url-context) tool which, when enabled, allows the models to fetch additional content from URLs as part of their execution.

You can enable that with the `-o url_context 1` option - for example:

```bash
llm -m gemini-2.5-flash -o url_context 1 'Latest headline on simonwillison.net'
```
Extra tokens introduced by this tool will be charged as input tokens. Use `--usage` to see details of those:
```bash
llm -m gemini-2.5-flash -o url_context 1 --usage \
  'Latest headline on simonwillison.net'
```
Outputs:
```
The latest headline on simonwillison.net as of August 17, 2025, is "TIL: Running a gpt-oss eval suite against LM Studio on a Mac.".
Token usage: 9,613 input, 87 output, {"candidatesTokenCount": 57, "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 10}], "toolUsePromptTokenCount": 9603, "toolUsePromptTokensDetails": [{"modality": "TEXT", "tokenCount": 9603}], "thoughtsTokenCount": 30}
```
The `"toolUsePromptTokenCount"` key shows how many tokens were used for that URL context.

### Chat

To chat interactively with the model, run `llm chat`:

```bash
llm chat -m gemini-2.0-flash
```

### Timeouts

By default there is no `timeout` against the Gemini API. You can use the `timeout` option to protect against API requests that hang indefinitely.

With the CLI tool that looks like this, to set a 1.5 second timeout:

```bash
llm -m gemini-2.5-flash-preview-05-20 'epic saga about mice' -o timeout 1.5
```
In the Python library timeouts are used like this:
```python
import httpx, llm

model = llm.get_model("gemini/gemini-2.5-flash-preview-05-20")

try:
    response = model.prompt(
        "epic saga about mice", timeout=1.5
    )
    print(response.text())
except httpx.TimeoutException:
    print("Timeout exceeded")
```
An `httpx.TimeoutException` subclass will be raised if the timeout is exceeded.

## Embeddings

The plugin also adds support for the `gemini-embedding-exp-03-07` and `text-embedding-004` embedding models.

Run that against a single string like this:
```bash
llm embed -m text-embedding-004 -c 'hello world'
```
This returns a JSON array of 768 numbers.

The `gemini-embedding-exp-03-07` model is larger, returning 3072 numbers. You can also use variants of it that are truncated down to smaller sizes:

- `gemini-embedding-exp-03-07` - 3072 numbers
- `gemini-embedding-exp-03-07-2048` - 2048 numbers
- `gemini-embedding-exp-03-07-1024` - 1024 numbers
- `gemini-embedding-exp-03-07-512` - 512 numbers
- `gemini-embedding-exp-03-07-256` - 256 numbers
- `gemini-embedding-exp-03-07-128` - 128 numbers

This command will embed every `README.md` file in child directories of the current directory and store the results in a SQLite database called `embed.db` in a collection called `readmes`:

```bash
llm embed-multi readmes -d embed.db -m gemini-embedding-exp-03-07-128 \
  --files . '*/README.md'
```
You can then run similarity searches against that collection like this:
```bash
llm similar readmes -c 'upload csvs to stuff' -d embed.db
```

See the [LLM embeddings documentation](https://llm.datasette.io/en/stable/embeddings/cli.html) for further details.

## Listing all Gemini API models

The `llm gemini models` command lists all of the models that are exposed by the Gemini API, some of which may not be available through this plugin.

```bash
llm gemini models
```
You can add a `--key X` option to use a different API key.

To filter models by their supported generation methods use `--method` one or more times:
```bash
llm gemini models --method embedContent
```
If you provide multiple methods you will see models that support any of them.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-gemini
python3 -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
llm install -e '.[test]'
```
To run the tests:
```bash
pytest
```

This project uses [pytest-recording](https://github.com/kiwicom/pytest-recording) to record Gemini API responses for the tests.

If you add a new test that calls the API you can capture the API response like this:
```bash
PYTEST_GEMINI_API_KEY="$(llm keys get gemini)" pytest --record-mode once
```
You will need to have stored a valid Gemini API key using this command first:
```bash
llm keys set gemini
# Paste key here
```
