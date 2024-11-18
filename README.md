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

Now run the model using `-m gemini-1.5-pro-latest`, for example:

```bash
llm -m gemini-1.5-pro-latest "A joke about a pelican and a walrus"
```

> A pelican walks into a seafood restaurant with a huge fish hanging out of its beak.  The walrus, sitting at the bar, eyes it enviously.
>
> "Hey," the walrus says, "That looks delicious! What kind of fish is that?"
>
> The pelican taps its beak thoughtfully. "I believe," it says, "it's a billfish."

Other models are:

- `gemini-1.5-flash-latest`
- `gemini-1.5-flash-8b-latest` - the least expensive
- `gemini-exp-1114` - recent experimental

### Images, audio and video

Gemini models are multi-modal. You can provide images, audio or video files as input like this:

```bash
llm -m gemini-1.5-flash-latest 'extract text' -a image.jpg
```
Or with a URL:
```bash
llm -m gemini-1.5-flash-8b-latest 'describe image' \
  -a https://static.simonwillison.net/static/2024/pelicans.jpg
```
Audio works too:

```bash
llm -m gemini-1.5-pro-latest 'transcribe audio' -a audio.mp3
```

And video:

```bash
llm -m gemini-1.5-pro-latest 'describe what happens' -a video.mp4
```
The Gemini prompting guide includes [extensive advice](https://ai.google.dev/gemini-api/docs/file-prompting-strategies) on multi-modal prompting.

### JSON output

Use `-o json_object 1` to force the output to be JSON:

```bash
llm -m gemini-1.5-flash-latest -o json_object 1 \
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
llm -m gemini-1.5-pro-latest -o code_execution 1 \
'use python to calculate (factorial of 13) * 3'
```

### Chat

To chat interactively with the model, run `llm chat`:

```bash
llm chat -m gemini-1.5-pro-latest
```

## Embeddings

The plugin also adds support for the `text-embedding-004` embedding model.

Run that against a single string like this:
```bash
llm embed -m text-embedding-004 -c 'hello world'
```
This returns a JSON array of 768 numbers.

This command will embed every `README.md` file in child directories of the current directory and store the results in a SQLite database called `embed.db` in a collection called `readmes`:

```bash
llm embed-multi readmes --files . '*/README.md' -d embed.db -m text-embedding-004
```
You can then run similarity searches against that collection like this:
```bash
llm similar readmes -c 'upload csvs to stuff' -d embed.db
```

See the [LLM embeddings documentation](https://llm.datasette.io/en/stable/embeddings/cli.html) for further details.

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
