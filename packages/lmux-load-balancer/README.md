# lmux-load-balancer

Weighted, sticky load balancing across [lmux](https://github.com/cluebbehusen/lmux) providers, with failover.

`LoadBalancerProvider` is a meta-provider: it holds no SDK of its own and distributes each chat request across a group of other providers already registered in a `Registry`. Selection is weighted and deterministic per routing key, so one conversation pins to a single endpoint (keeping its provider-side prompt cache warm) while different conversations spread across endpoints.

## Install

```bash
uv add lmux-load-balancer
```

## Usage

Register the underlying providers, then register a `LoadBalancerProvider` under its own prefix with one or more endpoint groups. A group maps a logical model name (the part after this provider's prefix) to `"prefix/model"` endpoints and their weights:

```python
from lmux import Registry, UserMessage
from lmux_anthropic import AnthropicProvider
from lmux_aws_bedrock import BedrockProvider
from lmux_load_balancer import LoadBalancerProvider, LoadBalancerParams, LoadBalancerMetadata

registry = Registry()
registry.register("anthropic", AnthropicProvider())
registry.register("bedrock", BedrockProvider())

registry.register(
    "balanced",
    LoadBalancerProvider(
        registry,
        groups={
            "sonnet": {
                "bedrock/anthropic.claude-sonnet-4": 0.7,
                "anthropic/claude-sonnet-4-20250514": 0.3,
            },
        },
    ),
)

# Distributed by weight, independently per call:
response = registry.chat("balanced/sonnet", [UserMessage(content="Hello")])

# Sticky: the same key always tries the same endpoint first, preserving cache locality:
response = registry.chat(
    "balanced/sonnet",
    [UserMessage(content="Hello")],
    provider_params=LoadBalancerParams(sticky_key="conversation-123"),
)

meta = response.provider_metadata
if isinstance(meta, LoadBalancerMetadata):
    print(meta.primary, meta.served, meta.attempted)
```

Weights are honored in aggregate across many distinct keys, not per call: a sticky conversation's whole traffic lands on the endpoint its key hashes to. A weight of `0` disables an endpoint. The load balancer delegates to each child by model string and does not forward its own `provider_params`, so each child uses the default params it was registered with.

The served endpoint is reported on `provider_metadata` as a `LoadBalancerMetadata` (`primary` selected, `served` actual, `attempted` in order) on the non-streaming response and on the terminal streaming chunk.

## Failover

On a retryable error (rate limit, timeout, or a provider/server error) the request falls through to the next endpoint in a deterministic, per-key order. Non-retryable errors (auth, invalid request, not found, unsupported feature) propagate immediately, and when every candidate is exhausted the underlying error is re-raised unchanged.

`LoadBalancerParams.failover` selects the behavior:

| Value | Behavior |
| --- | --- |
| `"always"` (default) | Fall through to the next endpoint on a retryable error. |
| `"never"` | Try only the selected endpoint; never fall through. |
| `"unless_sticky"` | Fall through only for keyless calls; a sticky call stays pinned to its endpoint. |

For streaming, failover applies only before the first chunk is produced; once output has started it cannot switch endpoints.

Because `"never"` and `"unless_sticky"` can leave a sticky conversation pinned to a down endpoint, keep a direct-provider entry as a last resort where availability matters more than cache locality.
