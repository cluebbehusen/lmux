# lmux-azure-foundry

Azure AI Foundry provider for [lmux](https://github.com/cluebbehusen/lmux). Talks to the Azure OpenAI REST API directly with [httpx](https://pypi.org/project/httpx/).

Supports chat completions, streaming, embeddings, and the Responses API.

Part of the [lmux](https://github.com/cluebbehusen/lmux) ecosystem: standardized interface, cost tracking on every response, and registry-based routing across providers.

## Optional Extras

- `lmux-azure-foundry[identity]`: Azure AD token authentication via `azure-identity`

## Auth

Three authentication methods:

### API Key (default)

Set `AZURE_FOUNDRY_API_KEY` in your environment:

```python
from lmux_azure_foundry import AzureFoundryProvider

provider = AzureFoundryProvider(endpoint="https://your-resource.openai.azure.com")
```

### Azure AD Token

```python
from lmux_azure_foundry import AzureFoundryProvider, AzureAdToken

provider = AzureFoundryProvider(
    endpoint="https://your-resource.openai.azure.com",
    auth=my_auth_returning_azure_ad_token,
)
```

### Token Provider

```python
from lmux_azure_foundry import AzureFoundryTokenAuthProvider

provider = AzureFoundryProvider(
    endpoint="https://your-resource.openai.azure.com",
    auth=AzureFoundryTokenAuthProvider(),  # uses azure-identity DefaultAzureCredential
)
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

Required for models that are only served through the Responses API:

```python
response = provider.create_response("gpt-5-pro", "Hello")
print(response.output_text)
```

### Chat over the Responses API

GPT-6 models refuse function tools with reasoning on Chat Completions. Set `api="responses"` to send
chat calls to `/responses` instead, with the same lmux chat types in and out:

```python
params = AzureFoundryParams(api="responses", reasoning_effort="medium")
response = provider.chat("gpt-6-luna", messages, tools=tools, provider_params=params)
messages.append(response.to_assistant_message())
```

Requests are stateless (`store: false`) and always ask for encrypted reasoning, so use this with reasoning
deployments. The encrypted reasoning comes back on `response.continuation`; keep it on the assistant message (as
`to_assistant_message()` does) and the next call replays it. It is replayed to whichever Azure Foundry endpoint gets
the history next, and only on this path; the same history sent through Chat Completions drops it. When a reasoning
effort is set, the request also asks for reasoning summaries, which fill `response.reasoning` and
`chunk.reasoning_delta`. `chat_stream`/`achat_stream` take the same parameter; `stop` and `seed` are not supported
here.

### Async

All methods have async variants: `achat`, `achat_stream`, `aembed`, `acreate_response`.

### Registry

Use with the lmux registry to route across multiple providers:

```python
from lmux import Registry

registry = Registry()
registry.register("azure", provider)
response = registry.chat("azure/gpt-4o", messages)
```

## Provider Params

```python
from lmux_azure_foundry import AzureFoundryParams

response = provider.chat(
    "gpt-4o",
    messages,
    provider_params=AzureFoundryParams(deployment_type="data_zone"),
)
```

| Parameter                | Type                                    | Description                                                       |
| ------------------------ | --------------------------------------- | ----------------------------------------------------------------- |
| `reasoning_effort`       | `"low" \| "medium" \| "high"`           | Reasoning effort for o-series models                              |
| `seed`                   | `int`                                   | Deterministic sampling seed                                       |
| `user`                   | `str`                                   | End-user identifier                                               |
| `prompt_cache_key`       | `str`                                   | Cache key for Azure's automatic prompt caching (chat + responses) |
| `prompt_cache_retention` | `"in_memory" \| "24h"`                  | Prompt cache retention policy (chat + responses)                  |
| `deployment_type`        | `"global" \| "data_zone" \| "regional"` | Affects cost calculation only, not sent to API                    |
| `data_zone`              | `"us" \| "eu"`                          | Pricing zone for Data Zone deployments; defaults to `"us"`        |
| `api`                    | `"chat_completions" \| "responses"`     | Endpoint chat calls use; not sent to API                          |

`deployment_type="data_zone"` uses US pricing when `data_zone` is omitted. Set `data_zone="eu"` for an EU
Data Zone deployment. This selects the applicable model-specific premium for cost reporting; it does not
change API routing. Other deployment types ignore `data_zone`.

## Constructor Options

```python
AzureFoundryProvider(
    endpoint=...,      # required, Azure resource endpoint
    auth=...,          # AuthProvider, default: AzureFoundryKeyAuthProvider()
    api_version=...,   # API version (default: "2025-04-01-preview")
    timeout=...,       # Request timeout in seconds
    max_retries=...,   # Max retry attempts
    default_headers=...,  # Optional headers included with every request
    transport=...,        # Optional httpx.BaseTransport for the sync client (proxies, testing)
    async_transport=...,  # Optional httpx.AsyncBaseTransport for the async client
)
```

`default_headers` is useful for gateway authentication, tracing, and routing. Foundry-managed authentication and
content-type headers take precedence over caller values, case-insensitively.
