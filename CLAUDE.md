# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running tests
```bash
pytest
```

To run tests with API recording for new tests:
```bash
PYTEST_GEMINI_API_KEY="$(llm keys get gemini)" pytest --record-mode once
```

### Updating README documentation
The README uses cog for dynamic content generation:
```bash
python -m cogapp --check README.md
```

### Installing development environment
```bash
python3 -m venv venv
source venv/bin/activate
llm install -e '.[test]'
```

## Architecture Overview

This is an LLM plugin that provides access to Google's Gemini API models. Key components:

### Core Model Implementation
- **llm_gemini.py**: Main plugin module containing all model implementations
  - `GeminiModel` class handles text generation for 45+ Gemini models
  - `GeminiEmbeddingModel` handles embedding models (text-embedding-004, gemini-embedding-exp-03-07 with size variants)
  - Model registration via `register_models()` hookimpl with both sync and async versions

### Model Configuration & Capabilities
- **Model Sets**: Models are categorized by capabilities:
  - `GOOGLE_SEARCH_MODELS`: 44 models supporting Google search grounding
  - `THINKING_BUDGET_MODELS`: 8 models supporting "thinking" mode for reasoning
  - `NO_VISION_MODELS`: 2 models without multi-modal support (gemma-3-1b-it, gemma-3n-e4b-it)
  - `ATTACHMENT_TYPES`: 21 supported MIME types for multi-modal inputs

- **Model Capability Flags**:
  - `can_vision`: Multi-modal support for images/audio/video
  - `can_google_search`: Web grounding capability
  - `can_thinking_budget`: Reasoning with thinking tokens
  - `can_schema`: JSON schema support (excluded for flash-thinking and gemma-3 models)

### Options System
- **Base Options** (via `Options` class):
  - `temperature`, `max_output_tokens`, `top_p`, `top_k`
  - `json_object`: Force JSON output
  - `timeout`: Request timeout (httpx)
  - `code_execution`: Enable Python sandbox execution

- **Extended Options** (via inheritance):
  - `OptionsWithGoogleSearch`: Adds `google_search` flag
  - `OptionsWithThinkingBudget`: Adds `thinking_budget` parameter

### Advanced Features
- **Schema Cleanup**: `cleanup_schema()` removes unsupported JSON schema properties
- **Google Search API Compatibility**: Handles both `google_search_retrieval` (older) and `google_search` (newer) tool formats
- **Token Usage Tracking**: Separates candidate vs thinking tokens for accurate billing
- **Safety Settings**: All categories set to BLOCK_NONE by default
- **Response Processing**: Handles executableCode, codeExecutionResult, functionCall parts

### CLI Commands
- **`llm gemini models`**: Lists all Gemini API models
  - `--method` flag for filtering by supported methods
  - `--key` for API key override
- **`llm gemini files`**: Lists uploaded files in Gemini API

### Testing Infrastructure
- **pytest-recording**: VCR-style test recording with API key filtering
- **Test Coverage**:
  - Prompt generation with various options
  - Embedding model variants
  - Tool/function calling
  - CLI commands
  - JSON schema cleanup
  - Async model support

### Implementation Details
- **Model ID System**: Internal `gemini_model_id` vs external `gemini/` prefixed `model_id`
- **Message Building**: Complex conversation history with tool results and attachments
- **Streaming**: Uses ijson for efficient JSON streaming from API responses
- **MIME Type Resolution**: Custom handling (e.g., audio/mpeg → audio/mp3)

The plugin integrates with the LLM ecosystem via entry points defined in pyproject.toml.