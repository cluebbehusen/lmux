# lmux-openai

OpenAI provider for [lmux](https://github.com/cluebbehusen/lmux). Talks to the OpenAI REST API directly over [httpx](https://pypi.org/project/httpx/).

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

### Explicit prompt caching

GPT-5.6 and later models accept explicit cache breakpoints in both Chat Completions and Responses. Place `CachePointContent` after the stable content block you want cached:

```python
from lmux import CachePointContent, ResponseInputMessage, TextContent
from lmux_openai import OpenAIParams

input_items = [
    ResponseInputMessage(
        role="developer",
        content=[TextContent(text=stable_instructions), CachePointContent()],
    ),
    ResponseInputMessage(role="user", content="What changed?"),
]
response = provider.create_response(
    "gpt-5.6-terra",
    input_items,
    provider_params=OpenAIParams(prompt_cache_key="knowledge-base-v1"),
)
```

When at least one breakpoint is present, lmux sets `prompt_cache_options.mode` to `"explicit"`, disabling OpenAI's implicit breakpoint so only the marked prefixes are read or written. Cache points are dropped for older models, which continue using automatic prompt caching. `CachePointContent.ttl` is not mapped because OpenAI's TTL is request-wide and currently fixed at `"30m"`.

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

| Parameter                | Type                            | Description                                                      |
| ------------------------ | ------------------------------- | ---------------------------------------------------------------- |
| `service_tier`           | `"auto" \| "default" \| "flex"` | Service tier selection                                           |
| `reasoning_effort`       | `"low" \| "medium" \| "high"`   | Reasoning effort for o-series models                             |
| `seed`                   | `int`                           | Deterministic sampling seed                                      |
| `user`                   | `str`                           | End-user identifier                                              |
| `prompt_cache_key`       | `str`                           | Prompt-cache routing key for better hit rates (chat + responses) |
| `prompt_cache_retention` | `"in_memory" \| "24h"`          | Prompt-cache retention; legacy, pre-gpt-5.6 (chat + responses)   |

## Constructor Options

```python
OpenAIProvider(
    auth=...,             # AuthProvider[str], default: OpenAIEnvAuthProvider()
    base_url=...,         # Optional base URL override
    timeout=...,          # Request timeout in seconds
    max_retries=...,      # Max retry attempts
    data_residency=...,   # bool, default: False — apply 10% uplift for regional endpoints
    organization=...,     # Optional org id -> OpenAI-Organization header
    project=...,          # Optional project id -> OpenAI-Project header
    default_headers=...,  # Optional Mapping[str, str] added to every request
    transport=...,        # Optional httpx.BaseTransport for the sync client (proxies, testing)
    async_transport=...,  # Optional httpx.AsyncBaseTransport for the async client
)
```

lmux does not read OpenAI's `OPENAI_BASE_URL` / `OPENAI_ORG_ID` / `OPENAI_PROJECT_ID`
environment variables (only the API key, via `OpenAIEnvAuthProvider`). Pass `base_url`,
`organization`, and `project` explicitly instead.

### Custom Headers

`default_headers` applies to every request — useful for gateways and proxies (e.g. a
`Helicone-Auth` token). lmux-managed headers (`Authorization`, `Content-Type`,
`OpenAI-Organization`, `OpenAI-Project`) take precedence and cannot be overridden by
`default_headers`; use `organization` / `project` for those.

```python
provider = OpenAIProvider(
    organization="org-abc",
    project="proj-123",
    default_headers={"Helicone-Auth": "Bearer sk-helicone-..."},
)
```

### Data Residency

OpenAI charges a 10% uplift on the `gpt-5.4`, `gpt-5.5`, and `gpt-5.6` families when requests go through a [regional processing (data residency) endpoint](https://developers.openai.com/api/docs/guides/your-data).

Data residency is selected at the _transport_ layer (regional hostname like `eu.api.openai.com`), not via a per-request parameter. Set `data_residency=True` on the provider so lmux applies the uplift to the reported cost.

```python
provider = OpenAIProvider(
    base_url="https://eu.api.openai.com/v1",
    data_residency=True,
)
```

The uplift is only applied to eligible models (checked via `regional_uplift_applies`); other models (e.g. `gpt-4o`, embeddings) return their standard cost even when `data_residency=True`.
