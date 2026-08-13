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

Now run the model using `-m gemini-flash-latest`, for example:

```bash
llm -m gemini-flash-latest "A short joke about a pelican and a walrus"
```

> A pelican and a walrus are sitting at a bar. The pelican orders a fishbowl cocktail, and the walrus orders a plate of clams. The bartender asks, "So, what brings you two together?"
>
> The walrus sighs and says, "It's a long story. Let's just say we met through a mutual friend... of the fin."

You can set the [default model](https://llm.datasette.io/en/stable/setup.html#setting-a-custom-default-model) to avoid the extra `-m` option:

```bash
llm models default gemini-flash-latest
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
    "gemini/gemini-3.7-flash": "Gemini 3.7 Flash",
    "gemini/gemini-3.6-flash": "Gemini 3.6 Flash",
    "gemini/gemini-3.5-flash": "Gemini 3.5 Flash",
    "gemini/gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    "gemini/gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
    "gemini/gemma-4-31b-it": "Gemma 4 31B Instruct",
    "gemini/gemma-4-26b-a4b-it": "Gemma 4 26B-A4B Instruct",
    "gemini/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite Preview",
    "gemini/gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
    "gemini/gemini-3-flash-preview": "Gemini 3 Flash Preview",
    "gemini/gemini-flash-latest": "Latest Gemini Flash",
    "gemini/gemini-flash-lite-latest": "Latest Gemini Flash Lite",
    "gemini/gemini-2.5-flash": "Gemini 2.5 Flash",
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
- `gemini/gemini-3.7-flash`: Gemini 3.7 Flash
- `gemini/gemini-3.5-flash-lite`: Gemini 3.5 Flash Lite
- `gemini/gemini-3.6-flash`: Gemini 3.6 Flash
- `gemini/gemini-3.5-flash`: Gemini 3.5 Flash
- `gemini/gemini-3.1-flash-lite`: Gemini 3.1 Flash Lite
- `gemini/gemma-4-31b-it`: Gemma 4 31B Instruct
- `gemini/gemma-4-26b-a4b-it`: Gemma 4 26B-A4B Instruct
- `gemini/gemini-3.1-flash-lite-preview`: Gemini 3.1 Flash Lite Preview
- `gemini/gemini-3.1-pro-preview-customtools`
- `gemini/gemini-3.1-pro-preview`: Gemini 3.1 Pro Preview
- `gemini/gemini-3-flash-preview`: Gemini 3 Flash Preview
- `gemini/gemini-flash-lite-latest`: Latest Gemini Flash Lite
- `gemini/gemini-flash-latest`: Latest Gemini Flash
- `gemini/gemini-2.5-flash`: Gemini 2.5 Flash
<!-- [[[end]]] -->

All of these models have aliases that omit the `gemini/` prefix, for example:

```bash
llm -m gemini-flash-latest --schema 'name,age int,bio' 'invent a dog'
```

### Images, audio and video

Gemini models are multi-modal. You can provide images, audio or video files as input like this:

```bash
llm -m gemini-flash-latest 'extract text' -a image.jpg
```
Or with a URL:
```bash
llm -m gemini-flash-latest 'describe image' \
  -a https://static.simonwillison.net/static/2024/pelicans.jpg
```
Audio works too:

```bash
llm -m gemini-flash-latest 'transcribe audio' -a audio.mp3
```

And video:

```bash
llm -m gemini-flash-latest 'describe what happens' -a video.mp4
```
The Gemini prompting guide includes [extensive advice](https://ai.google.dev/gemini-api/docs/file-prompting-strategies) on multi-modal prompting.

### YouTube videos

You can provide YouTube video URLs as attachments as well:

```bash
llm -m gemini-flash-latest -a 'https://www.youtube.com/watch?v=9o1_DL9uNlM' \
  'Produce a summary with relevant URLs and code example snippets, then an accurate transcript with timestamps.'
```
[Example output here](https://gist.github.com/simonw/1b07aafb2bfc112b180ab68c864511cb).

These will be processed with media resolution `low` by default. You can use the `-o media_resolution X` option to set that to `medium`, `high`, or `unspecified`.

### JSON output

Use `-o json_object 1` to force the output to be JSON:

```bash
llm -m gemini-flash-latest -o json_object 1 \
  '3 largest cities in California, list of {"name": "..."}'
```
Outputs:
```json
{"cities": [{"name": "Los Angeles"}, {"name": "San Diego"}, {"name": "San Jose"}]}
```

### Code execution

Gemini models can [write and execute code](https://ai.google.dev/gemini-api/docs/code-execution) - they can decide to write Python code, execute it in a secure sandbox and use the result as part of their response.

Enable this server-side tool with `-T CodeExecution`:

```bash
llm -m gemini-3.6-flash -T CodeExecution \
'use python to calculate (factorial of 13) * 3'
```
### Google search

Some Gemini models support [Grounding with Google Search](https://ai.google.dev/gemini-api/docs/google-search), where the model can run a Google search and use the results as part of answering a prompt.

Using this feature may incur additional requirements in terms of how you use the results. Consult [Google's documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-gemini#web-ground-gemini) for more details.

Enable this server-side tool with `-T GoogleSearch`:

```bash
llm -m gemini-3.6-flash -T GoogleSearch \
  'What happened in Ireland today?'
```

The plugin leaves the model's response text unchanged and retains Gemini's raw
`groundingMetadata` on the response part. Use `llm logs -c --json` after running
a prompt to inspect that metadata, which includes [additional information](https://github.com/simonw/llm-gemini/pull/29#issuecomment-2606201877) about grounded results.

When Gemini returns native server-side tool invocation parts, the plugin exposes
those as structured server-side tool call and result events as well.

### URL context

Gemini models support a [URL context](https://ai.google.dev/gemini-api/docs/url-context) tool which, when enabled, allows the models to fetch additional content from URLs as part of their execution.

Enable this server-side tool with `-T URLContext` - for example:

```bash
llm -m gemini-2.5-flash -T URLContext 'Latest headline on simonwillison.net'
```
Extra tokens introduced by this tool will be charged as input tokens. Use `--usage` to see details of those:
```bash
llm -m gemini-2.5-flash -T URLContext --usage \
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
llm chat -m gemini-flash-latest
```

### Timeouts

By default there is no `timeout` against the Gemini API. You can use the `timeout` option to protect against API requests that hang indefinitely.

With the CLI tool that looks like this, to set a 1.5 second timeout:

```bash
llm -m gemini-flash-latest 'epic saga about mice' -o timeout 1.5
```
In the Python library timeouts are used like this:
```python
import httpx, llm

model = llm.get_model("gemini/gemini-flash-latest")

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

The plugin supports Google's current [Gemini embedding models](https://ai.google.dev/gemini-api/docs/embeddings):

- `gemini-embedding-2` is the latest model.
- `gemini-embedding-001` remains available for text-only use cases.

Run that against a single string like this:
```bash
llm embed -m gemini-embedding-2 -c 'hello world'
```
This returns a JSON array of 3072 numbers.

Both models have variants that ask Gemini to return its recommended smaller
vector sizes of 768 or 1536 dimensions, specified as a suffix on the model ID:

- `gemini-embedding-2` - 3072 numbers
- `gemini-embedding-2-1536` - 1536 numbers
- `gemini-embedding-2-768` - 768 numbers
- `gemini-embedding-001` - 3072 numbers
- `gemini-embedding-001-1536` - 1536 numbers
- `gemini-embedding-001-768` - 768 numbers

The embedding spaces used by the two models are incompatible. If you switch an
existing collection from `gemini-embedding-001` to `gemini-embedding-2`, you
must re-embed all of its content.

This command will embed every `README.md` file in child directories of the current directory and store the results in a SQLite database called `embed.db` in a collection called `readmes`:

```bash
llm embed-multi readmes -d embed.db -m gemini-embedding-2-768 \
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

To set up this plugin locally, first checkout the code, then run the tests with `uv`:
```bash
cd llm-gemini
uv run pytest
```
Run `llm` with the plugin like this:
```bash
uv run llm models -q gemini
```

This project uses [pytest-recording](https://github.com/kiwicom/pytest-recording) to record Gemini API responses for the tests.

If you add a new test that calls the API you can capture the API response like this:
```bash
PYTEST_GEMINI_API_KEY="$(llm keys get gemini)" uv run pytest --record-mode once
```
You will need to have stored a valid Gemini API key using this command first:
```bash
llm keys set gemini
# Paste key here
```
