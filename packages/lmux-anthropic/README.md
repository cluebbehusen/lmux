# lmux-anthropic

Anthropic provider for [lmux](https://github.com/cluebbehusen/lmux). Wraps the [anthropic](https://pypi.org/project/anthropic/) SDK.

Supports chat completions and streaming.

Part of the [lmux](https://github.com/cluebbehusen/lmux) ecosystem: standardized interface, cost tracking on every response, and registry-based routing across providers.

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
| `cache_control` | `dict`                      | Top-level prompt-cache control — auto-places a breakpoint on the last cacheable block (e.g. `{"type": "ephemeral"}`) |

## Prompt Caching

Two ways to opt in:

- **Top-level (auto-placement):** pass `cache_control` via `AnthropicParams` (above) to cache the full rendered prefix.
- **Explicit breakpoints:** place `CachePointContent` parts in `UserMessage` content. A cache point marks the end of the stable prefix; it attaches `cache_control` to the preceding content block. A cache point with no preceding block in its message applies to whatever came before it: the prior message's last block, or the system text seen so far (system text after the marker stays outside the cached prefix). A marker with nothing cacheable before it is dropped, and when two markers resolve to the same block the first one wins.

```python
from lmux import CachePointContent, TextContent, UserMessage

messages = [
    UserMessage(content=[TextContent(text=big_stable_context), CachePointContent(ttl="1h")]),
    UserMessage(content="What changed since yesterday?"),
]
```

Cache reads/writes are reported on `response.usage` (`cache_read_tokens`, `cache_creation_tokens`, and the per-TTL `cache_creation_tokens_by_ttl` breakdown) and priced into `response.cost`, including the 2x write rate for `ttl="1h"`.

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
