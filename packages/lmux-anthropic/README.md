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
| `pricing_as_of` | `datetime.date`             | Override the date used for dated pricing (e.g. a model's introductory-rate window); defaults to the current date |

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

## Claude on Vertex AI

Requires the `vertex` extra, which pulls in `google-auth` via `anthropic[vertex]`:

```bash
uv add "lmux-anthropic[vertex]"
```

`AnthropicVertexProvider` serves Claude through GCP Vertex AI with the same chat/streaming interface:

```python
from lmux_anthropic import AnthropicVertexProvider

provider = AnthropicVertexProvider(project_id="my-project", region="global")
response = provider.chat("claude-sonnet-4-5@20250929", [UserMessage(content="Hello")])
print(response.provider)  # "anthropic-vertex"
print(response.cost)
```

`project_id` falls back to the `ANTHROPIC_VERTEX_PROJECT_ID` environment variable, then to the project resolved by the auth provider (e.g. the `gcloud` default project under ADC, or the service account key file's project). `region` falls back to `CLOUD_ML_REGION`; a request without a region raises at first call. `region` accepts `"global"`, a multi-region (`"us"`, `"eu"`), or a specific region (`"us-east5"`, ...). Model IDs use Vertex's `@`-versioned format (`claude-sonnet-4-5@20250929`) or plain names for newer models (`claude-opus-4-6`).

### Vertex Auth

Application Default Credentials by default; a service account file is also supported:

```python
from lmux_anthropic import AnthropicVertexServiceAccountAuthProvider

provider = AnthropicVertexProvider(
    project_id="my-project",
    region="global",
    auth=AnthropicVertexServiceAccountAuthProvider(service_account_file="/path/to/key.json"),
)
```

Any `AuthProvider` that returns `google.auth` `Credentials` works — either bare, or as a `(credentials, project_id)` tuple so the provider can infer the project.

### Vertex Params Caveat

`AnthropicParams.service_tier` and `AnthropicParams.inference_geo` are Anthropic-API-only: the Vertex provider drops them from outgoing requests, and the `inference_geo` US cost multiplier never applies.

## Claude in Microsoft Foundry

No extra needed — `AnthropicFoundryProvider` ships with the base package and serves Claude through a Foundry resource with the same chat/streaming interface:

```python
from lmux_anthropic import AnthropicFoundryProvider

provider = AnthropicFoundryProvider(resource="example-resource")
response = provider.chat("claude-sonnet-4-6", [UserMessage(content="Hello")])
print(response.provider)  # "anthropic-foundry"
print(response.cost)
```

`resource` and the mutually exclusive `base_url` fall back to the `ANTHROPIC_FOUNDRY_RESOURCE` and `ANTHROPIC_FOUNDRY_BASE_URL` environment variables. Model IDs are Foundry deployment names, which default to the plain model IDs (`claude-sonnet-4-6`, ...). Foundry bills Anthropic's standard API pricing through the Microsoft Marketplace, so costs come from the same pricing table with no multiplier.

### Foundry Auth

The default `AnthropicFoundryEnvAuthProvider` reads an API key from `ANTHROPIC_FOUNDRY_API_KEY`. For Microsoft Entra ID, wrap a bearer-token provider:

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from lmux_anthropic import AnthropicFoundryProvider, AnthropicFoundryTokenAuthProvider

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)
provider = AnthropicFoundryProvider(
    resource="example-resource",
    auth=AnthropicFoundryTokenAuthProvider(token_provider=token_provider),
)
```

Any `AuthProvider` that returns an API key string or a `() -> str` token-provider callable works.

### Foundry Params Caveat

Same as Vertex: `service_tier` and `inference_geo` are dropped from outgoing requests, and the `inference_geo` US cost multiplier never applies.

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
