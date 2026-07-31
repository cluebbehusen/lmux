# lmux

Modular Python library for unified LLM provider access with cost reporting.

Separate packages per provider. Lazy dependency loading. No global state. Install only what you use.

## Install

Provider packages include `lmux` as a dependency, so there's no need to install it separately.

```bash
uv add lmux-openai         # OpenAI
uv add lmux-anthropic      # Anthropic ([vertex]: Claude on Vertex AI; [bedrock]: Claude on Bedrock)
uv add lmux-azure-foundry  # Azure AI Foundry
uv add lmux-aws-bedrock    # AWS Bedrock
uv add lmux-google         # Google (Gemini)
uv add lmux-groq           # Groq
```

Or with pip: `pip install lmux-openai`, etc.

## Usage

Use providers directly:

```python
from lmux import UserMessage
from lmux_openai import OpenAIProvider

provider = OpenAIProvider()  # reads OPENAI_API_KEY from env by default
response = provider.chat("gpt-4o", [UserMessage(content="Hello")])

print(response.content)
print(response.cost)  # Cost(input_cost=..., output_cost=..., total_cost=...)
```

Or route through a registry:

```python
from lmux import Registry, UserMessage
from lmux_openai import OpenAIProvider, OpenAIParams
from lmux_anthropic import AnthropicProvider

registry = Registry()
registry.register("openai", OpenAIProvider(), default_params=OpenAIParams(reasoning_effort="high"))
registry.register("anthropic", AnthropicProvider())

response = registry.chat("openai/gpt-4o", [UserMessage(content="Hello")])
response = registry.chat("anthropic/claude-sonnet-4-20250514", [UserMessage(content="Hello")])
```

### Streaming

```python
for chunk in provider.chat_stream("gpt-4o", [UserMessage(content="Hello")]):
    if chunk.delta:
        print(chunk.delta, end="")
```

### Async

Every method has an async counterpart: `achat`, `achat_stream`, `aembed`, `acreate_response`.

```python
response = await provider.achat("gpt-4o", [UserMessage(content="Hello")])
```

### Resource Cleanup

Providers that cache async HTTP clients implement `AsyncCloseable`. In long-running or serverless environments, close these when done:

```python
# Close a single provider
await provider.aclose()

# Close all providers via the registry
await registry.aclose()
```

Providers also detect event loop changes (e.g. in AWS Lambda) and automatically recreate their async clients when needed.

### Embeddings

```python
response = provider.embed("text-embedding-3-small", "Hello")
print(response.embeddings)
print(response.cost)
```

### Reasoning

Request reasoning with a unified `reasoning_effort` parameter that works across all providers:

```python
response = provider.chat("o3", messages, reasoning_effort="high")
print(response.reasoning)  # normalized reasoning text (or None)
print(response.usage.reasoning_tokens)  # reasoning token count (or None)
```

Streaming returns reasoning in `chunk.reasoning_delta`. Each provider maps the effort level to its native API:

| Provider               | Mapping                                                                                                                                  |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| OpenAI / Azure Foundry | `reasoning_effort`                                                                                                                       |
| Anthropic              | adaptive `thinking` + `output_config.effort` on Claude 4.6+; `thinking` with budget tokens (capped to `max_tokens - 1`) on 4.5 and older; summarized display in both modes |
| Google                 | `thinking_config` with `thinking_budget` and `include_thoughts`                                                                          |
| AWS Bedrock            | `additionalModelRequestFields.thinking` — adaptive + `output_config.effort` on Claude 4.6+, capped budget tokens on 4.5 and older, with summarized display in both modes |
| Groq                   | `reasoning_effort` + `include_reasoning`                                                                                                 |

For fine-grained control, use provider-specific params instead (e.g., `AnthropicParams(thinking=...)`). Provider params always take precedence over the top-level `reasoning_effort`.

### Provider continuations

Some provider protocols return opaque signed state that must be preserved across a tool-use turn. Providers that support this attach a `ProviderContinuation` to the response. Convert the response directly into the next assistant message to carry it forward:

```python
messages.append(response.to_assistant_message())
messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call_id))
```

Continuation data belongs to the provider package. Preserve it unchanged or drop it; do not construct or edit it. It may contain full reasoning text and signatures and is included in normal Pydantic serialization.

### Cost

Every response includes a `.cost` field when the model's pricing is known. Unknown models return `None`, not an error.

Register custom pricing for models not in the built-in tables:

```python
from lmux import ModelPricing, PricingTier, per_million_tokens

provider.register_pricing("my-fine-tune", ModelPricing(tiers=[
    PricingTier(
        input_cost_per_token=per_million_tokens(3.00),
        output_cost_per_token=per_million_tokens(15.00),
    ),
]))
```

## Providers

| Package                                              | Protocols                        | Auth                                                 |
| ---------------------------------------------------- | -------------------------------- | ---------------------------------------------------- |
| [lmux-openai](packages/lmux-openai)                  | Completion, Embedding, Responses | `OPENAI_API_KEY`                                     |
| [lmux-anthropic](packages/lmux-anthropic)            | Completion                       | `ANTHROPIC_API_KEY`                                  |
| [lmux-anthropic\[vertex\]](packages/lmux-anthropic)  | Completion                       | ADC, service account                                 |
| [lmux-anthropic](packages/lmux-anthropic) (Foundry)  | Completion                       | `ANTHROPIC_FOUNDRY_API_KEY`, Entra ID token provider |
| [lmux-anthropic\[bedrock\]](packages/lmux-anthropic) | Completion                       | boto3 credential chain, `AWS_BEARER_TOKEN_BEDROCK`   |
| [lmux-azure-foundry](packages/lmux-azure-foundry)    | Completion, Embedding, Responses | `AZURE_FOUNDRY_API_KEY`, Azure AD, token provider    |
| [lmux-aws-bedrock](packages/lmux-aws-bedrock)        | Completion, Embedding            | boto3 credential chain                               |
| [lmux-google](packages/lmux-google)                  | Completion, Embedding            | ADC, service account, `GOOGLE_API_KEY`               |
| [lmux-groq](packages/lmux-groq)                      | Completion                       | `GROQ_API_KEY`                                       |

## Load balancing

[lmux-load-balancer](packages/lmux-load-balancer) adds a meta-provider that distributes requests across a group of registered providers, weighted and sticky per routing key, with failover. It keeps one conversation on one endpoint (preserving its prompt cache) while spreading different conversations across endpoints.

```python
from lmux import Registry, UserMessage
from lmux_anthropic import AnthropicProvider
from lmux_aws_bedrock import BedrockProvider
from lmux_load_balancer import LoadBalancerProvider, LoadBalancerParams

registry = Registry()
registry.register("anthropic", AnthropicProvider())
registry.register("bedrock", BedrockProvider())
registry.register(
    "balanced",
    LoadBalancerProvider(
        registry,
        groups={
            "sonnet": {
                "bedrock/anthropic.claude-sonnet-4": 0.7,
                "anthropic/claude-sonnet-4-20250514": 0.3,
            },
        },
    ),
)

# Sticky per conversation; falls over to another endpoint on a retryable error.
response = registry.chat(
    "balanced/sonnet",
    [UserMessage(content="Hello")],
    provider_params=LoadBalancerParams(sticky_key="conversation-123"),
)
```

## Custom Providers

Implement the protocols you need. Only `lmux` is required as a dependency.

```python
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Literal

from lmux import (
    ChatChunk,
    ChatResponse,
    CompletionProvider,
    Message,
    ResponseFormat,
    Tool,
    ToolChoice,
)


class MyProvider(CompletionProvider[None]):
    def chat(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: None = None,
    ) -> ChatResponse: ...

    async def achat(self, model, messages, **kwargs) -> ChatResponse: ...
    def chat_stream(self, model, messages, **kwargs) -> Iterator[ChatChunk]: ...
    async def achat_stream(self, model, messages, **kwargs) -> AsyncIterator[ChatChunk]: ...
```

Works with the registry like any built-in provider:

```python
registry.register("my-provider", MyProvider())
response = registry.chat("my-provider/my-model", messages)
```

## Design

- **Lazy loading**: Each provider imports `httpx` (and any provider-specific dependencies, e.g. `google-auth` or `boto3`) on first API call, not on `import`. To pay the import cost at startup instead, each package exports a `preload()` function:

  ```python
  import lmux_openai
  lmux_openai.preload()  # eagerly imports httpx and the provider's dependencies
  ```

- **Protocols**: Providers implement `CompletionProvider`, `EmbeddingProvider`, `ResponsesProvider`, and/or `AsyncCloseable`. Check support at runtime with `isinstance()`.
- **Standardized inputs and outputs**: Same message types and response shapes across all providers.
- **Cost ownership**: Each provider owns its pricing data and calculation. Core provides utilities, not a database.
- **Serverless-friendly**: Lazy dependency loading, no global state, and automatic event loop detection keep cold starts fast and avoid stale client issues.

## Development

Checked with ruff, ty, and pytest with 100% branch coverage.

### Skills

- `/update-pricing`: Claude Code skill that validates and updates pricing data across all providers

### Scripts

- `scripts/update_bedrock_pricing.py`: generates Bedrock pricing from the AWS Pricing API — the Anthropic-on-Bedrock subset into `lmux-bedrock-shared`, the rest into `lmux-aws-bedrock`
- `scripts/validate_pricing.py`: cross-references provider pricing against external databases (LiteLLM, OpenRouter, genai-prices)

## Requirements

Python 3.13+
