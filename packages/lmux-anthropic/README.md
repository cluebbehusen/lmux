# lmux-anthropic

Anthropic provider for [lmux](https://github.com/cluebbehusen/lmux). Wraps the [anthropic](https://pypi.org/project/anthropic/) SDK.

Supports chat completions and streaming. Standardized interface, cost tracking on every response, and registry-based routing across providers.

## Auth

Set `ANTHROPIC_API_KEY` in your environment. The default `AnthropicEnvAuthProvider` reads it automatically.

```python
from lmux_anthropic import AnthropicProvider

provider = AnthropicProvider()
```

## Usage

### Chat

```python
from lmux import UserMessage

response = provider.chat("claude-sonnet-4-20250514", [UserMessage(content="Hello")])
print(response.content)
print(response.cost)
```

### Streaming

```python
for chunk in provider.chat_stream("claude-sonnet-4-20250514", [UserMessage(content="Hello")]):
    if chunk.delta:
        print(chunk.delta, end="")
```

### Async

All methods have async variants: `achat`, `achat_stream`.

### Registry

Use with the lmux registry to route across multiple providers:

```python
from lmux import Registry

registry = Registry()
registry.register("anthropic", provider)
response = registry.chat("anthropic/claude-sonnet-4-20250514", messages)
```

## Provider Params

```python
from lmux_anthropic import AnthropicParams

response = provider.chat(
    "claude-sonnet-4-20250514",
    messages,
    provider_params=AnthropicParams(inference_geo="us"),
)
```

| Parameter       | Type                        | Description                        |
| --------------- | --------------------------- | ---------------------------------- |
| `thinking`      | `dict`                      | Extended thinking configuration    |
| `metadata`      | `dict[str, str]`            | Request metadata                   |
| `top_k`         | `int`                       | Top-k sampling                     |
| `service_tier`  | `"auto" \| "standard_only"` | Service tier selection             |
| `inference_geo` | `"us"`                      | Inference geography (affects cost) |

## Constructor Options

```python
AnthropicProvider(
    auth=...,               # AuthProvider[str], default: AnthropicEnvAuthProvider()
    base_url=...,           # Optional base URL override
    timeout=...,            # Request timeout in seconds
    max_retries=...,        # Max retry attempts
    default_max_tokens=..., # Default max tokens (default: 4096)
)
```
