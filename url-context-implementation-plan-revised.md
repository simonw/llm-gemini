# URL Context Feature Implementation Plan (Revised)

## Overview
This plan outlines how to add URL Context support to the llm-gemini plugin, following the same pattern as the existing `google_search` feature. This revision corrects inaccuracies based on the actual codebase structure.

## Implementation Steps

### 1. Define Supported Models
Create a new constant in `llm_gemini.py`:
```python
URL_CONTEXT_MODELS = {
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-flash-preview-05-20", 
    "gemini-2.0-flash",
    "gemini-2.0-flash-live-001",
}
```

### 2. Add Model Capability Flag
Update the model registration in `register_models()`:
```python
can_url_context = model_id in URL_CONTEXT_MODELS
```

Add to `_SharedGemini.__init__()`:
```python
def __init__(self, gemini_model_id, can_vision=True, can_google_search=False, 
             can_thinking_budget=False, can_schema=False, can_url_context=False):
    self.can_url_context = can_url_context
```

### 3. Handle Options Class Complexity
The current codebase uses a simple sequential assignment pattern for Options classes. Adding URL context requires careful consideration because:
- Some models support both google_search and url_context
- OptionsWithThinkingBudget currently extends OptionsWithGoogleSearch
- We need to handle all possible combinations

#### Option A: Simple Sequential (Current Pattern)
```python
# In _SharedGemini.__init__()
if can_google_search:
    self.Options = self.OptionsWithGoogleSearch
if can_url_context:
    self.Options = self.OptionsWithUrlContext
if can_thinking_budget:
    self.Options = self.OptionsWithThinkingBudget
```

**Limitation**: Models can only have one feature's options exposed.

#### Option B: Combined Options Classes (Recommended)
Create new combined options classes:
```python
class OptionsWithUrlContext(Options):
    url_context: Optional[bool] = Field(
        description="Enables the model to retrieve and analyze content from specified URLs",
        default=None,
    )

class OptionsWithGoogleSearchAndUrlContext(OptionsWithGoogleSearch):
    url_context: Optional[bool] = Field(
        description="Enables the model to retrieve and analyze content from specified URLs",  
        default=None,
    )

# Update OptionsWithThinkingBudget to support all features
class OptionsWithThinkingBudget(OptionsWithGoogleSearchAndUrlContext):
    thinking_budget: Optional[int] = Field(
        description="Maximum number of thinking tokens to use (min: 2000, max: 100000, multiple of 1000)",
        default=None,
        ge=2000,
        le=100000,
    )
```

Update Options assignment logic:
```python
# In _SharedGemini.__init__()
if self.can_thinking_budget:
    self.Options = self.OptionsWithThinkingBudget
elif self.can_google_search and self.can_url_context:
    self.Options = self.OptionsWithGoogleSearchAndUrlContext
elif self.can_url_context:
    self.Options = self.OptionsWithUrlContext
elif self.can_google_search:
    self.Options = self.OptionsWithGoogleSearch
```

### 4. Add Tool to Request
Update `build_request_body()` to handle URL context:
```python
# After existing google_search handling
if prompt.options and self.can_url_context and prompt.options.url_context:
    tools.append({"url_context": {}})
```

### 5. Update Model Registration
Pass the capability flag during model registration in both `GeminiPro` and `AsyncGeminiPro` creation:
```python
models.append(
    GeminiPro(
        f"gemini/{model_id}",
        can_google_search=can_google_search,
        can_thinking_budget=can_thinking_budget,
        can_vision=can_vision,
        can_schema=can_schema,
        can_url_context=can_url_context,
    )
)
models.append(
    AsyncGeminiPro(
        f"gemini/{model_id}",
        can_google_search=can_google_search,
        can_thinking_budget=can_thinking_budget,
        can_vision=can_vision,
        can_schema=can_schema,
        can_url_context=can_url_context,
    )
)
```

### 6. Handle URL Input
URLs can be provided in two ways:
1. **As attachments**: `-a https://example.com`
2. **In prompt text**: Direct URL references

The implementation should handle both patterns. When URLs are provided as attachments, they should be processed appropriately in `build_request_body()`.

### 7. Documentation Updates
Update README.md with:
- New section explaining URL Context feature
- Usage examples: `llm -m gemini/gemini-2.0-flash -o url_context 1 'Compare these articles' -a https://url1.com -a https://url2.com`
- List of supported models
- Limitations (20 URLs max, experimental status)
- Quota information (1500/day via API)

### 8. Testing Strategy
Create tests in `test_gemini.py`:
- Test that only supported models expose the option
- Test request body generation with url_context enabled
- Mock API responses to avoid actual URL fetching
- Test error handling for unsupported models
- Test combinations with google_search for models that support both

### 9. Handle Response Metadata
The API returns `url_context_metadata` in responses. Consider adding logic to:
- Extract and display URL retrieval status
- Show which URLs were successfully fetched
- Handle error cases gracefully

## Implementation Order
1. Add URL_CONTEXT_MODELS constant
2. Update _SharedGemini.__init__ to accept can_url_context
3. Create new Options classes (including combined ones)
4. Update Options assignment logic
5. Add url_context tool to request body
6. Update model registration
7. Write tests
8. Update documentation
9. Test with real API calls

## Key Differences from Original Plan
1. **Class Names**: Uses `_SharedGemini` instead of `GeminiModel`
2. **Options Hierarchy**: More complex due to feature combinations
3. **Model Classes**: `GeminiPro` and `AsyncGeminiPro` instead of single `GeminiModel`
4. **Tool API**: Confirmed that tools are added as `{"tool_name": {}}` format

## Considerations
- Models that support url_context also support google_search (based on research doc)
- Need to handle cases where both features are enabled simultaneously
- The experimental nature means the API might change
- Consider adding deprecation warnings if needed

## Future Enhancements
- URL validation before sending to API
- Better error messages for URL fetch failures
- Caching of URL content to reduce API calls
- Progress indicators for multiple URL fetches
- Support for specifying which URLs to use for which parts of the prompt