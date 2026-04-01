# lmux-azure-foundry

Azure AI Foundry provider for [lmux](https://github.com/cluebbehusen/lmux). Uses the [openai](https://pypi.org/project/openai/) SDK's `AzureOpenAI` client.

Supports chat completions, streaming, and embeddings. Standardized interface, cost tracking on every response, and registry-based routing across providers.

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

### Async

All methods have async variants: `achat`, `achat_stream`, `aembed`.

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

| Parameter | Type | Description |
|---|---|---|
| `reasoning_effort` | `"low" \| "medium" \| "high"` | Reasoning effort for o-series models |
| `seed` | `int` | Deterministic sampling seed |
| `user` | `str` | End-user identifier |
| `deployment_type` | `"global" \| "data_zone" \| "regional"` | Affects cost calculation only, not sent to API |

## Constructor Options

```python
AzureFoundryProvider(
    endpoint=...,      # required, Azure resource endpoint
    auth=...,          # AuthProvider, default: AzureFoundryKeyAuthProvider()
    api_version=...,   # API version (default: "2024-12-01-preview")
    timeout=...,       # Request timeout in seconds
    max_retries=...,   # Max retry attempts
)
```
