"""Azure AI Foundry pricing data and cost calculation.

Prices are for Global Standard (pay-as-you-go) deployments.
Use ``register_pricing()`` on ``AzureFoundryProvider`` for provisioned
deployments, regional overrides, or models not listed here.

Pricing as of Mar 2, 2026 (source: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/)
"""

from lmux.cost import ModelPricing, PricingTier, calculate_cost, per_million_tokens
from lmux.types import Cost, Usage

_PRICING: dict[str, ModelPricing] = {
    # GPT-4.1 family
    "gpt-4.1-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.40),
                output_cost_per_token=per_million_tokens(1.60),
                cache_read_cost_per_token=per_million_tokens(0.10),
            )
        ],
    ),
    "gpt-4.1-nano": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.10),
                output_cost_per_token=per_million_tokens(0.40),
                cache_read_cost_per_token=per_million_tokens(0.025),
            )
        ],
    ),
    "gpt-4.1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.00),
                output_cost_per_token=per_million_tokens(8.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
            )
        ],
    ),
    # GPT-4o family
    "gpt-4o-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.15),
                output_cost_per_token=per_million_tokens(0.60),
                cache_read_cost_per_token=per_million_tokens(0.075),
            )
        ],
    ),
    "gpt-4o": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.50),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(1.25),
            )
        ],
    ),
    # Reasoning models
    "o3-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.10),
                output_cost_per_token=per_million_tokens(4.40),
                cache_read_cost_per_token=per_million_tokens(0.55),
            )
        ],
    ),
    "o3": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.00),
                output_cost_per_token=per_million_tokens(8.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
            )
        ],
    ),
    "o4-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.10),
                output_cost_per_token=per_million_tokens(4.40),
                cache_read_cost_per_token=per_million_tokens(0.275),
            )
        ],
    ),
    # Embedding models
    "text-embedding-3-small": ModelPricing(
        tiers=[PricingTier(input_cost_per_token=per_million_tokens(0.02), output_cost_per_token=0.0)]
    ),
    "text-embedding-3-large": ModelPricing(
        tiers=[PricingTier(input_cost_per_token=per_million_tokens(0.13), output_cost_per_token=0.0)]
    ),
    "text-embedding-ada-002": ModelPricing(
        tiers=[PricingTier(input_cost_per_token=per_million_tokens(0.10), output_cost_per_token=0.0)]
    ),
}

# Pre-sorted by key length descending for longest-prefix matching
_PRICING_BY_PREFIX = sorted(_PRICING.items(), key=lambda item: len(item[0]), reverse=True)


def calculate_azure_foundry_cost(model: str, usage: Usage) -> Cost | None:
    """Calculate cost for an Azure AI Foundry API call. Returns None if model pricing is unknown."""
    pricing = _PRICING.get(model)
    if pricing is None:
        for prefix, p in _PRICING_BY_PREFIX:
            if model.startswith(prefix):
                pricing = p
                break
    if pricing is None:
        return None
    return calculate_cost(usage, pricing)
