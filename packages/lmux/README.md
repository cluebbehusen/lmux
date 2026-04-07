# lmux

Core types, protocols, and utilities for the lmux ecosystem.

You don't need to install this directly; provider packages (e.g., `lmux-openai`) include it as a dependency. Install it only if you're building a custom provider.

## Types

### Messages

- `SystemMessage`: system/instruction message
- `DeveloperMessage`: developer message (for o-series models)
- `UserMessage`: user message, supports text and multimodal content (`TextContent`, `ImageContent`)
- `AssistantMessage`: assistant message with optional tool calls
- `ToolMessage`: tool result

### Responses

- `ChatResponse`: chat completion result with `content`, `usage`, `cost`, `model`, `provider`, `finish_reason`
- `ChatChunk`: streaming chunk with `delta`, `tool_call_deltas`, `usage`, `cost`
- `EmbeddingResponse`: embedding result with `embeddings`, `usage`, `cost`
- `ResponseResponse`: Responses API result with `output_text`, `usage`, `cost`

### Cost

- `Usage`: token counts (`input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_creation_tokens`)
- `Cost`: cost breakdown (`input_cost`, `output_cost`, `total_cost`, plus cache costs)
- `ModelPricing` / `PricingTier`: tiered pricing configuration
- `per_million_tokens()`: converts per-million price to per-token price
- `calculate_cost()`: calculates cost from usage and pricing

### Tools

- `Tool`: function tool definition
- `ToolChoice` / `ToolChoiceFunction`: control whether and which tool the model calls
- `ToolCall` / `ToolCallDelta`: tool call in responses and streaming
- `FunctionDefinition` / `FunctionCallResult` / `FunctionCallDelta`

### Response Format

- `TextResponseFormat` / `JsonObjectResponseFormat` / `JsonSchemaResponseFormat`

## Protocols

```python
from lmux import CompletionProvider, EmbeddingProvider, ResponsesProvider, PricingProvider, AsyncCloseable
```

- `CompletionProvider[ParamsT]`: `chat`, `achat`, `chat_stream`, `achat_stream`
- `EmbeddingProvider[ParamsT]`: `embed`, `aembed`
- `ResponsesProvider[ParamsT]`: `create_response`, `acreate_response`
- `PricingProvider`: `register_pricing`
- `AuthProvider[AuthT]`: `get_credentials`, `aget_credentials`
- `AsyncCloseable`: `aclose`

All are `@runtime_checkable`, so you can use `isinstance()` to check support.

## Registry

Route `"prefix/model"` strings to provider instances:

```python
from lmux import Registry

registry = Registry()
registry.register("openai", openai_provider)
registry.register("anthropic", anthropic_provider)

response = registry.chat("openai/gpt-4o", messages)
response = registry.chat("anthropic/claude-sonnet-4-20250514", messages)

# Close all providers that implement AsyncCloseable
await registry.aclose()
```

## Exceptions

All exceptions inherit from `LmuxError` and carry optional `provider` and `status_code` fields:

- `AuthenticationError`
- `RateLimitError` (with `retry_after`)
- `InvalidRequestError`
- `NotFoundError`
- `ProviderError`
- `TimeoutError`
- `UnsupportedFeatureError`

## MockProvider

Built-in mock for testing. Implements all protocols with configurable responses and call tracking.

```python
from lmux import MockProvider, ChatResponse

mock = MockProvider(chat_responses=[ChatResponse(...)])
response = mock.chat("any-model", messages)
assert len(mock.calls) == 1
```
