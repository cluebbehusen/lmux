# lmux-anthropic

Anthropic provider for [lmux](https://github.com/cluebbehusen/lmux). Talks to the Anthropic Messages API directly over [httpx](https://pypi.org/project/httpx/).

Supports chat completions and streaming.

Part of the [lmux](https://github.com/cluebbehusen/lmux) ecosystem: standardized interface, cost tracking on every response, and registry-based routing across providers.

## Auth

Set `ANTHROPIC_API_KEY` in your environment. The default `AnthropicEnvAuthProvider` reads it automatically.

```python
from lmux_anthropic import AnthropicProvider

provider = AnthropicProvider()
```

### Workload Identity Federation

To authenticate without a static API key, `AnthropicWorkloadIdentityAuthProvider` exchanges an IdP-issued OIDC identity token for a short-lived Anthropic access token, sent as `Authorization: Bearer` on every request and re-exchanged before it expires. Supply the identity token as a callable (invoked fresh on every exchange) or a token file path (e.g. a Kubernetes-projected service-account token):

```python
import boto3
from lmux_anthropic import AnthropicProvider, AnthropicWorkloadIdentityAuthProvider

sts = boto3.client("sts", region_name="us-east-1")

def identity_token() -> str:
    response = sts.get_web_identity_token(
        Audience=["https://api.anthropic.com"], SigningAlgorithm="RS256", DurationSeconds=900
    )
    return response["WebIdentityToken"]

provider = AnthropicProvider(
    auth=AnthropicWorkloadIdentityAuthProvider(
        federation_rule_id="fdrl_...",
        organization_id="...",
        service_account_id="svac_...",
        identity_token_provider=identity_token,
    ),
)
```

`workspace_id` is required when the federation rule covers more than one workspace. The token exchange always targets the Anthropic API (`token_base_url`, default `https://api.anthropic.com`), deliberately independent of the provider's `base_url`, so a gateway that proxies only Messages traffic keeps working. Transient exchange failures (429/5xx, timeouts, connection errors) can be retried with `max_retries` (default 0); each attempt fetches a fresh identity token. Any `AuthProvider` that returns an API key string or a `() -> str` access-token callable works.

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

### Tool continuations

Extended thinking and redacted thinking blocks must be returned unmodified when a tool result continues the assistant turn. `lmux-anthropic` preserves the native ordered blocks in `response.continuation`; use `to_assistant_message()` to keep them with the normalized response:

```python
from lmux import ToolMessage

response = provider.chat(model, messages, tools=tools, reasoning_effort="high")
messages.append(response.to_assistant_message())
messages.append(ToolMessage(content=tool_result, tool_call_id=response.tool_calls[0].id))
```

Continuations are scoped to the Anthropic API surface that produced them, including Vertex AI, Microsoft Foundry, and Amazon Bedrock. A matching continuation is replayed exactly; other providers ignore it and use the normalized content and tool calls.

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

| Parameter       | Type                        | Description                                                                                                          |
| --------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `thinking`      | `dict`                      | Extended thinking configuration                                                                                      |
| `metadata`      | `dict[str, str]`            | Request metadata                                                                                                     |
| `top_k`         | `int`                       | Top-k sampling                                                                                                       |
| `service_tier`  | `"auto" \| "standard_only"` | Service tier selection                                                                                               |
| `inference_geo` | `"us"`                      | Inference geography (affects cost)                                                                                   |
| `cache_control` | `dict`                      | Top-level prompt-cache control — auto-places a breakpoint on the last cacheable block (e.g. `{"type": "ephemeral"}`) |
| `pricing_as_of` | `datetime.date`             | Override the date used for dated pricing, e.g. replaying usage from before a scheduled rate change; defaults to today |

For manual thinking, an integer `budget_tokens` raises the default `max_tokens` when needed. An explicit `max_tokens` is preserved instead, along with the provider-specific thinking configuration. Ensure those explicit values are compatible with the deployed model; manual thinking normally requires `budget_tokens < max_tokens`, except when interleaved thinking applies.

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

Requires the `vertex` extra, which pulls in `google-auth`:

```bash
uv add "lmux-anthropic[vertex]"
```

File-based ADC from `GOOGLE_APPLICATION_CREDENTIALS` or the `gcloud` CLI works with this extra. For ADC from an
attached service account's instance metadata, install the requests transport with
`uv add "lmux-anthropic[vertex,requests]"`.

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

`reasoning_effort` can select the correct thinking mode only when the deployment name contains a recognizable Claude model ID. For an opaque custom deployment name, configure the thinking mode explicitly for the deployed model:

```python
from lmux_anthropic import AnthropicParams

response = provider.chat(
    "claude-prod",
    [UserMessage(content="Hello")],
    provider_params=AnthropicParams(thinking={"type": "enabled", "budget_tokens": 8192}),
)
```

Use `{"type": "adaptive"}` instead when the deployment targets a model that requires adaptive thinking.

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

## Claude on Amazon Bedrock

Requires the `bedrock` extra, which pulls in `boto3` (for the AWS credential chain) and the shared `lmux-bedrock-shared` internals:

```bash
uv add "lmux-anthropic[bedrock]"
```

`AnthropicBedrockProvider` serves Claude through Bedrock's **native Anthropic Messages API** (`InvokeModel` / `InvokeModelWithResponseStream`) with the same chat/streaming interface — distinct from [`lmux-aws-bedrock`](https://pypi.org/project/lmux-aws-bedrock/), which speaks Bedrock's normalized Converse API across all vendors. Use this one when you want Claude on Bedrock with the exact first-party Messages semantics (thinking config, `cache_control`, `output_config` all pass through unchanged):

```python
from lmux_anthropic import AnthropicBedrockProvider

provider = AnthropicBedrockProvider(region="us-east-1")
response = provider.chat("anthropic.claude-opus-4-8", [UserMessage(content="Hello")])
print(response.provider)  # "anthropic-bedrock"
print(response.cost)
```

Model IDs are the Bedrock forms — a bare model ID (`anthropic.claude-opus-4-8`) or a cross-region inference-profile ID (`us.anthropic.claude-opus-4-8`, `eu.anthropic.…`). `region` falls back to the resolved AWS session's region, then `us-east-1`. `endpoint_url` overrides the endpoint; `use_fips=True` selects the FIPS 140-3 endpoint.

Pricing comes from the generated Bedrock table (shared with `lmux-aws-bedrock` via `lmux-bedrock-shared`), keyed by the request's Bedrock ID — so a `us.`-profile request is billed at its regional rate, no multiplier involved.

### Bedrock Auth

Two modes, resolved once on first use:

- **Bearer token** — set `AWS_BEARER_TOKEN_BEDROCK` and the request carries `Authorization: Bearer <token>`; nothing is signed.
- **SigV4** — otherwise AWS credentials are resolved through boto3 (env vars, profile, SSO, instance metadata) and every request is signed. The default `AnthropicBedrockEnvAuthProvider` uses boto3's default credential chain; `AnthropicBedrockSessionAuthProvider` takes explicit `region_name`/`profile_name`/keys:

```python
from lmux_anthropic import AnthropicBedrockProvider, AnthropicBedrockSessionAuthProvider

provider = AnthropicBedrockProvider(
    auth=AnthropicBedrockSessionAuthProvider(profile_name="prod", region_name="us-east-1"),
)
```

### Bedrock Params Caveat

Same as Vertex/Foundry: `service_tier` and `inference_geo` are dropped from outgoing requests, and the `inference_geo` US cost multiplier never applies.

## Constructor Options

```python
AnthropicProvider(
    auth=...,               # AuthProvider[str | () -> str], default: AnthropicEnvAuthProvider()
    base_url=...,           # Optional base URL override
    timeout=...,            # Request timeout in seconds
    max_retries=...,        # Max retry attempts
    default_max_tokens=..., # Default max tokens (default: 4096)
    default_headers=...,    # Optional headers included with every request
    transport=...,          # Optional httpx.BaseTransport for the sync client (proxies, testing)
    async_transport=...,    # Optional httpx.AsyncBaseTransport for the async client
)
```

`default_headers` is also accepted by `AnthropicVertexProvider`, `AnthropicFoundryProvider`, and
`AnthropicBedrockProvider`. It is useful for gateway authentication, tracing, and routing. Provider-managed
authentication, API-version, and content-type headers take precedence over caller values, case-insensitively. Bedrock
custom headers are included in the SigV4 signature.
