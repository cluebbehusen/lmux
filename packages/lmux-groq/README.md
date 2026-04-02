# lmux-groq

Groq provider for [lmux](https://github.com/cluebbehusen/lmux). Wraps the [groq](https://pypi.org/project/groq/) SDK.

Supports chat completions and streaming.

Part of the [lmux](https://github.com/cluebbehusen/lmux) ecosystem: standardized interface, cost tracking on every response, and registry-based routing across providers.

## Auth

Set `GROQ_API_KEY` in your environment. The default `GroqEnvAuthProvider` reads it automatically.

```python
from lmux_groq import GroqProvider

provider = GroqProvider()
```

## Usage

### Chat

```python
from lmux import UserMessage

response = provider.chat("llama-3.3-70b-versatile", [UserMessage(content="Hello")])
print(response.content)
print(response.cost)
```

### Streaming

```python
for chunk in provider.chat_stream("llama-3.3-70b-versatile", [UserMessage(content="Hello")]):
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
registry.register("groq", provider)
response = registry.chat("groq/llama-3.3-70b-versatile", messages)
```

## Provider Params

```python
from lmux_groq import GroqParams

response = provider.chat(
    "llama-3.3-70b-versatile",
    messages,
    provider_params=GroqParams(service_tier="flex"),
)
```

| Parameter      | Type                                               | Description                 |
| -------------- | -------------------------------------------------- | --------------------------- |
| `service_tier` | `"auto" \| "on_demand" \| "flex" \| "performance"` | Service tier selection      |
| `seed`         | `int`                                              | Deterministic sampling seed |
| `user`         | `str`                                              | End-user identifier         |

## Constructor Options

```python
GroqProvider(
    auth=...,          # AuthProvider[str], default: GroqEnvAuthProvider()
    base_url=...,      # Optional base URL override
    timeout=...,       # Request timeout in seconds
    max_retries=...,   # Max retry attempts
)
```
