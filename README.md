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

Configure the model by setting a key called "gemini" to your [API key](https://ai.google.dev/):

```bash
llm keys set gemini
```
```
<paste key here>
```

Now run the model using `-m gemini-pro`, for example:

```bash
llm -m gemini-pro "A joke about a pelican and a walrus"
```

> Why did the pelican get mad at the walrus?
>
> Because he called him a hippo-crit.

To chat interactively with the model, run `llm chat`:

```bash
llm chat -m gemini-pro
```

If you have access to the Gemini 1.5 Pro preview you can use `-m gemini-1.5-pro-latest` to work with that model.

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
