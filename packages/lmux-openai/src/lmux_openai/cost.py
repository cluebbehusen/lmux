"""OpenAI pricing data and cost calculation."""

from lmux.cost import ModelPricing, PricingTier, calculate_cost, per_million_tokens
from lmux.types import Cost, Usage

# Pricing as of Feb 28, 2026 (source: https://openai.com/api/pricing/)
_PRICING: dict[str, ModelPricing] = {
    # GPT-5 family
    "gpt-5.2-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(21.00), output_cost_per_token=per_million_tokens(168.00)
            )
        ]
    ),
    "gpt-5.2": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(1.75), output_cost_per_token=per_million_tokens(14.00))
        ],
        cache_read_cost_per_token=per_million_tokens(0.175),
    ),
    "gpt-5.1": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(1.25), output_cost_per_token=per_million_tokens(10.00))
        ],
        cache_read_cost_per_token=per_million_tokens(0.125),
    ),
    "gpt-5-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(15.00), output_cost_per_token=per_million_tokens(120.00)
            )
        ]
    ),
    "gpt-5-mini": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(0.25), output_cost_per_token=per_million_tokens(2.00))
        ],
        cache_read_cost_per_token=per_million_tokens(0.025),
    ),
    "gpt-5-nano": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(0.05), output_cost_per_token=per_million_tokens(0.40))
        ],
        cache_read_cost_per_token=per_million_tokens(0.005),
    ),
    "gpt-5": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(1.25), output_cost_per_token=per_million_tokens(10.00))
        ],
        cache_read_cost_per_token=per_million_tokens(0.125),
    ),
    # GPT-4.1 family
    "gpt-4.1-mini": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(0.40), output_cost_per_token=per_million_tokens(1.60))
        ],
        cache_read_cost_per_token=per_million_tokens(0.10),
    ),
    "gpt-4.1-nano": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(0.10), output_cost_per_token=per_million_tokens(0.40))
        ],
        cache_read_cost_per_token=per_million_tokens(0.025),
    ),
    "gpt-4.1": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(2.00), output_cost_per_token=per_million_tokens(8.00))
        ],
        cache_read_cost_per_token=per_million_tokens(0.50),
    ),
    # GPT-4o family
    "gpt-4o-mini": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(0.15), output_cost_per_token=per_million_tokens(0.60))
        ],
        cache_read_cost_per_token=per_million_tokens(0.075),
    ),
    "gpt-4o": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(2.50), output_cost_per_token=per_million_tokens(10.00))
        ],
        cache_read_cost_per_token=per_million_tokens(1.25),
    ),
    # Reasoning models
    "o3-pro": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(20.00), output_cost_per_token=per_million_tokens(80.00))
        ]
    ),
    "o3-mini": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(1.10), output_cost_per_token=per_million_tokens(4.40))
        ],
        cache_read_cost_per_token=per_million_tokens(0.55),
    ),
    "o3": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(2.00), output_cost_per_token=per_million_tokens(8.00))
        ],
        cache_read_cost_per_token=per_million_tokens(0.50),
    ),
    "o4-mini": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(1.10), output_cost_per_token=per_million_tokens(4.40))
        ],
        cache_read_cost_per_token=per_million_tokens(0.275),
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

# Pre-sorted by key length descending for prefix matching (longest match first)
_PRICING_BY_PREFIX = sorted(_PRICING.items(), key=lambda item: len(item[0]), reverse=True)


def calculate_openai_cost(model: str, usage: Usage) -> Cost | None:
    """Calculate cost for an OpenAI API call. Returns None if model pricing is unknown."""
    pricing = _PRICING.get(model)
    if pricing is None:
        for prefix, p in _PRICING_BY_PREFIX:
            if model.startswith(prefix):
                pricing = p
                break
    if pricing is None:
        return None
    return calculate_cost(usage, pricing)
