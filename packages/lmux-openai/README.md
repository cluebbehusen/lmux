# lmux-openai

OpenAI provider for [lmux](https://github.com/cluebbehusen/lmux). Wraps the [openai](https://pypi.org/project/openai/) SDK.

Supports chat completions, streaming, embeddings, and the Responses API.

Part of the [lmux](https://github.com/cluebbehusen/lmux) ecosystem: standardized interface, cost tracking on every response, and registry-based routing across providers.

## Auth

Set `OPENAI_API_KEY` in your environment. The default `OpenAIEnvAuthProvider` reads it automatically.

```python
from lmux_openai import OpenAIProvider

provider = OpenAIProvider()
```

Or pass a custom auth provider:

```python
provider = OpenAIProvider(auth=my_auth_provider)
```

## Usage

### Chat

```python
from lmux import UserMessage

response = provider.chat("gpt-4o", [UserMessage(content="Hello")])
print(response.content)
print(response.cost)
```

### Streaming

```python
for chunk in provider.chat_stream("gpt-4o", [UserMessage(content="Hello")]):
    if chunk.delta:
        print(chunk.delta, end="")
```

### Embeddings

```python
response = provider.embed("text-embedding-3-small", "Hello")
print(response.embeddings)
```

### Responses API

```python
response = provider.create_response("gpt-4o", "Hello")
print(response.output_text)
```

### Async

All methods have async variants: `achat`, `achat_stream`, `aembed`, `acreate_response`.

### Registry

Use with the lmux registry to route across multiple providers:

```python
from lmux import Registry

registry = Registry()
registry.register("openai", provider)
response = registry.chat("openai/gpt-4o", messages)
```

## Provider Params

Pass OpenAI-specific parameters via `provider_params`:

```python
from lmux_openai import OpenAIParams

response = provider.chat(
    "o3",
    messages,
    provider_params=OpenAIParams(reasoning_effort="high", service_tier="flex"),
)
```

| Parameter          | Type                            | Description                          |
| ------------------ | ------------------------------- | ------------------------------------ |
| `service_tier`     | `"auto" \| "default" \| "flex"` | Service tier selection               |
| `reasoning_effort` | `"low" \| "medium" \| "high"`   | Reasoning effort for o-series models |
| `seed`             | `int`                           | Deterministic sampling seed          |
| `user`             | `str`                           | End-user identifier                  |

## Constructor Options

```python
OpenAIProvider(
    auth=...,            # AuthProvider[str], default: OpenAIEnvAuthProvider()
    base_url=...,        # Optional base URL override
    timeout=...,         # Request timeout in seconds
    max_retries=...,     # Max retry attempts
    data_residency=...,  # bool, default: False — apply 10% uplift for regional endpoints
)
```

### Data Residency

OpenAI charges a 10% uplift on the `gpt-5.4` family (`gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.4-pro`) when requests go through a [regional processing (data residency) endpoint](https://developers.openai.com/api/docs/guides/your-data).

Data residency is selected at the _transport_ layer (regional hostname like `eu.api.openai.com`), not via a per-request parameter. Set `data_residency=True` on the provider so lmux applies the uplift to the reported cost.

```python
provider = OpenAIProvider(
    base_url="https://eu.api.openai.com/v1",
    data_residency=True,
)
```

The uplift is only applied to eligible models (checked via `regional_uplift_applies`); other models (e.g. `gpt-4o`, embeddings) return their standard cost even when `data_residency=True`.
