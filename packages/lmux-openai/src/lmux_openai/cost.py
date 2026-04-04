"""OpenAI pricing data and cost calculation.

Pricing source: https://developers.openai.com/api/docs/pricing
"""

from lmux.cost import ModelPricing, PricingTier, calculate_cost, per_million_tokens
from lmux.types import Cost, Usage

_PRICING: dict[str, ModelPricing] = {
    # GPT-5 family
    "gpt-5.4-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(30.00),
                output_cost_per_token=per_million_tokens(180.00),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(60.00),
                output_cost_per_token=per_million_tokens(270.00),
                min_input_tokens=272_000,
            ),
        ]
    ),
    "gpt-5.4": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.50),
                output_cost_per_token=per_million_tokens(15.00),
                cache_read_cost_per_token=per_million_tokens(0.25),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(5.00),
                output_cost_per_token=per_million_tokens(22.50),
                cache_read_cost_per_token=per_million_tokens(0.50),
                min_input_tokens=272_000,
            ),
        ],
    ),
    "gpt-5.4-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.75),
                output_cost_per_token=per_million_tokens(4.50),
                cache_read_cost_per_token=per_million_tokens(0.075),
            )
        ],
    ),
    "gpt-5.4-nano": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.20),
                output_cost_per_token=per_million_tokens(1.25),
                cache_read_cost_per_token=per_million_tokens(0.02),
            )
        ],
    ),
    "gpt-5.3-codex": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.75),
                output_cost_per_token=per_million_tokens(14.00),
                cache_read_cost_per_token=per_million_tokens(0.175),
            )
        ],
    ),
    "gpt-5.2-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(21.00),
                output_cost_per_token=per_million_tokens(168.00),
            )
        ]
    ),
    "gpt-5.2": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.75),
                output_cost_per_token=per_million_tokens(14.00),
                cache_read_cost_per_token=per_million_tokens(0.175),
            )
        ],
    ),
    "gpt-5.2-codex": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.75),
                output_cost_per_token=per_million_tokens(14.00),
                cache_read_cost_per_token=per_million_tokens(0.175),
            )
        ],
    ),
    "gpt-5.1-codex-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.25),
                output_cost_per_token=per_million_tokens(2.00),
                cache_read_cost_per_token=per_million_tokens(0.025),
            )
        ],
    ),
    "gpt-5.1-codex": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(0.125),
            )
        ],
    ),
    "gpt-5.1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(0.125),
            )
        ],
    ),
    "gpt-5-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(15.00),
                output_cost_per_token=per_million_tokens(120.00),
            )
        ]
    ),
    "gpt-5-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.25),
                output_cost_per_token=per_million_tokens(2.00),
                cache_read_cost_per_token=per_million_tokens(0.025),
            )
        ],
    ),
    "gpt-5-nano": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.05),
                output_cost_per_token=per_million_tokens(0.40),
                cache_read_cost_per_token=per_million_tokens(0.005),
            )
        ],
    ),
    "gpt-5-codex": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(0.125),
            )
        ],
    ),
    "gpt-5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(0.125),
            )
        ],
    ),
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
    "o3-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(20.00),
                output_cost_per_token=per_million_tokens(80.00),
            )
        ]
    ),
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
    "o3-deep-research": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(10.00),
                output_cost_per_token=per_million_tokens(40.00),
                cache_read_cost_per_token=per_million_tokens(2.50),
            )
        ],
    ),
    "o4-mini-deep-research": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.00),
                output_cost_per_token=per_million_tokens(8.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
            )
        ],
    ),
    "codex-mini-latest": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.50),
                output_cost_per_token=per_million_tokens(6.00),
                cache_read_cost_per_token=per_million_tokens(0.375),
            )
        ],
    ),
    "computer-use-preview": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(3.00),
                output_cost_per_token=per_million_tokens(12.00),
            )
        ],
    ),
    # ChatGPT / Chat-latest models
    "gpt-5.3-chat-latest": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.75),
                output_cost_per_token=per_million_tokens(14.00),
                cache_read_cost_per_token=per_million_tokens(0.175),
            )
        ],
    ),
    "gpt-5.2-chat-latest": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.75),
                output_cost_per_token=per_million_tokens(14.00),
                cache_read_cost_per_token=per_million_tokens(0.175),
            )
        ],
    ),
    "gpt-5.1-chat-latest": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(0.125),
            )
        ],
    ),
    "gpt-5-chat-latest": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(0.125),
            )
        ],
    ),
    "chatgpt-4o-latest": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.00),
                output_cost_per_token=per_million_tokens(15.00),
            )
        ],
    ),
    # Codex models
    "gpt-5.1-codex-max": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(0.125),
            )
        ],
    ),
    # Search models
    "gpt-5-search-api": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(0.125),
            )
        ],
    ),
    "gpt-4o-search-preview": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.50),
                output_cost_per_token=per_million_tokens(10.00),
            )
        ],
    ),
    # Legacy reasoning models
    "o1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(15.00),
                output_cost_per_token=per_million_tokens(60.00),
                cache_read_cost_per_token=per_million_tokens(7.50),
            )
        ],
    ),
    "o1-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(150.00),
                output_cost_per_token=per_million_tokens(600.00),
            )
        ]
    ),
    "o1-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.10),
                output_cost_per_token=per_million_tokens(4.40),
                cache_read_cost_per_token=per_million_tokens(0.55),
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
