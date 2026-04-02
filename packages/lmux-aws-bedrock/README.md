# lmux-aws-bedrock

AWS Bedrock provider for [lmux](https://github.com/cluebbehusen/lmux). Uses [boto3](https://pypi.org/project/boto3/) and the Converse API.

Supports chat completions, streaming, and embeddings.

Part of the [lmux](https://github.com/cluebbehusen/lmux) ecosystem: standardized interface, cost tracking on every response, and registry-based routing across providers.

## Optional Extras

- `lmux-aws-bedrock[async]`: async support via `aioboto3`

## Auth

Uses boto3's default credential chain (env vars, AWS config, instance metadata). No extra setup needed if your AWS credentials are already configured.

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

Requires the `[async]` extra. All methods have async variants: `achat`, `achat_stream`, `aembed`.

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

| Parameter                               | Type              | Description                      |
| --------------------------------------- | ----------------- | -------------------------------- |
| `guardrail_config`                      | `GuardrailConfig` | Bedrock guardrail to apply       |
| `additional_model_request_fields`       | `dict`            | Extra fields passed to the model |
| `additional_model_response_field_paths` | `list[str]`       | Extra response fields to return  |

## Constructor Options

```python
BedrockProvider(
    auth=...,          # AuthProvider, default: BedrockEnvAuthProvider()
    region=...,        # AWS region
    endpoint_url=...,  # Custom endpoint URL
)
```
