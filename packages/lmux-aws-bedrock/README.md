# lmux-aws-bedrock

AWS Bedrock provider for [lmux](https://github.com/cluebbehusen/lmux). Talks to the Bedrock Converse and InvokeModel REST endpoints directly over [httpx](https://pypi.org/project/httpx/), using [boto3](https://pypi.org/project/boto3/) only to resolve AWS credentials for request signing.

Supports chat completions, streaming, and embeddings.

Part of the [lmux](https://github.com/cluebbehusen/lmux) ecosystem: standardized interface, cost tracking on every response, and registry-based routing across providers.

## Optional Extras

- `lmux-aws-bedrock[async]`: `aiobotocore` for async AWS credential resolution in the auth providers. Not required for `achat`/`aembed`/`achat_stream` themselves.

## Auth

Two authentication modes are supported, resolved on the first request:

1. **Bedrock API key (simplest):** set `AWS_BEARER_TOKEN_BEDROCK` and each request is sent with an `Authorization: Bearer <token>` header — no signing required.
2. **SigV4 (fallback):** otherwise AWS credentials are resolved through boto3's default credential chain (env vars, AWS config, instance metadata) and each request is SigV4-signed. No extra setup needed if your AWS credentials are already configured.

```python
from lmux_aws_bedrock import BedrockProvider

provider = BedrockProvider()

# Or specify a region
provider = BedrockProvider(region="us-east-1")
```

For explicit session configuration:

```python
from lmux_aws_bedrock import BedrockSessionAuthProvider

provider = BedrockProvider(auth=BedrockSessionAuthProvider(profile_name="my-profile"))
```

## Usage

### Chat

```python
from lmux import UserMessage

response = provider.chat("anthropic.claude-sonnet-4-20250514-v1:0", [UserMessage(content="Hello")])
print(response.content)
print(response.cost)
```

### Streaming

```python
for chunk in provider.chat_stream("anthropic.claude-sonnet-4-20250514-v1:0", [UserMessage(content="Hello")]):
    if chunk.delta:
        print(chunk.delta, end="")
```

### Embeddings

```python
response = provider.embed("amazon.titan-embed-text-v2:0", "Hello")
print(response.embeddings)
```

### Async

All methods have async variants: `achat`, `achat_stream`, `aembed`. These run over `httpx`'s async client; credentials are resolved synchronously (via boto3) even on the async path.

Bedrock also supports lmux `response_format`, mapped to Converse `outputConfig.textFormat`.

### Registry

Use with the lmux registry to route across multiple providers:

```python
from lmux import Registry

registry = Registry()
registry.register("bedrock", provider)
response = registry.chat("bedrock/anthropic.claude-sonnet-4-20250514-v1:0", messages)
```

## Provider Params

```python
from lmux_aws_bedrock import BedrockParams, GuardrailConfig

response = provider.chat(
    "anthropic.claude-sonnet-4-20250514-v1:0",
    messages,
    provider_params=BedrockParams(
        guardrail_config=GuardrailConfig(
            guardrail_identifier="my-guardrail",
            guardrail_version="1",
        ),
    ),
)
```

| Parameter                               | Type              | Description                                                            |
| --------------------------------------- | ----------------- | ---------------------------------------------------------------------- |
| `guardrail_config`                      | `GuardrailConfig` | Bedrock guardrail to apply                                             |
| `additional_model_request_fields`       | `dict`            | Extra fields passed to the model                                       |
| `additional_model_response_field_paths` | `list[str]`       | Extra response fields to return                                        |
| `pricing_as_of`                         | `datetime.date`   | Override the date used for dated pricing; defaults to the current date |

## Prompt Caching

Place `CachePointContent` parts in `UserMessage` content to emit Converse `cachePoint` blocks marking the end of a stable prompt prefix. A cache point with no preceding block in its message is placed after whatever came before it (the prior message, or the system blocks). Markers with nothing cacheable before them are dropped, and adjacent duplicates are coalesced — the first marker wins.

```python
from lmux import CachePointContent, TextContent, UserMessage

messages = [
    UserMessage(content=[TextContent(text=big_stable_context), CachePointContent()]),
    UserMessage(content="What changed since yesterday?"),
]
```

Cache points are emitted for whatever model the request targets; models without prompt-caching support reject them at request validation. Cache reads/writes are reported on `response.usage` (`cache_read_tokens`, `cache_creation_tokens`, and the per-TTL `cache_creation_tokens_by_ttl` breakdown from `cacheDetails`) and priced into `response.cost`, including per-TTL write rates where the pricing data carries them.

## Constructor Options

```python
BedrockProvider(
    auth=...,          # AuthProvider, default: BedrockEnvAuthProvider()
    region=...,        # AWS region
    endpoint_url=...,  # Custom endpoint URL (overrides region/FIPS host selection)
    use_fips=...,      # bool, default False: target the FIPS 140-3 endpoint (bedrock-runtime-fips.<region>.amazonaws.com)
    timeout=...,       # request timeout in seconds
    max_retries=...,   # retry count for transient failures
    default_headers=...,  # Optional headers included with every request
    transport=...,        # Optional httpx.BaseTransport for the sync client (proxies, testing)
    async_transport=...,  # Optional httpx.AsyncBaseTransport for the async client
)
```

`default_headers` is useful for gateway authentication, tracing, and routing. Bedrock-managed authentication and
content-type headers take precedence over caller values, case-insensitively. With SigV4 authentication, custom headers
are included in the request signature.
