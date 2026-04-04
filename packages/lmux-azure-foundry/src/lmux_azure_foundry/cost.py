"""Azure AI Foundry pricing data and cost calculation.

Prices are Global Standard (pay-as-you-go) pricing.  Data Zone and Regional
deployments apply a multiplier on top of these base rates.

Use ``register_pricing()`` on ``AzureFoundryProvider`` for provisioned
deployments or models not listed here.

Pricing source: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
"""

from lmux.cost import ModelPricing, PricingTier, calculate_cost, per_million_tokens
from lmux.types import Cost, Usage

# MARK: Deployment-type multipliers

DATA_ZONE_MULTIPLIER = 1.1
"""Data Zone deployments are consistently 1.1x global pricing across all models."""

REGIONAL_MULTIPLIER = 1.1
"""Regional deployments are approximately 1.1x global pricing.

Note: actual regional pricing varies by model (1.1x-1.375x).  This constant
uses the most common multiplier; for exact per-model regional rates, use
``register_pricing()`` to override individual models.
"""

# MARK: Global Standard pricing (base rates)

_PRICING: dict[str, ModelPricing] = {
    # --- OpenAI: GPT-5 family ---
    "gpt-5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(0.125),
            )
        ],
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
    # --- OpenAI: GPT-4.5 ---
    "gpt-4.5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(75.00),
                output_cost_per_token=per_million_tokens(150.00),
                cache_read_cost_per_token=per_million_tokens(37.50),
            )
        ],
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
    "gpt-5.2-chat": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.75),
                output_cost_per_token=per_million_tokens(14.00),
                cache_read_cost_per_token=per_million_tokens(0.175),
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
    "gpt-5.1-chat": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(0.125),
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
    "gpt-5.1-codex-max": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(0.125),
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
    "gpt-5-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(15.00),
                output_cost_per_token=per_million_tokens(120.00),
            )
        ],
    ),
    # --- OpenAI: GPT-4.1 family ---
    "gpt-4.1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.00),
                output_cost_per_token=per_million_tokens(8.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
            )
        ],
    ),
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
    # --- OpenAI: GPT-4o family ---
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
    # --- OpenAI: Reasoning models ---
    "o1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(15.00),
                output_cost_per_token=per_million_tokens(60.00),
                cache_read_cost_per_token=per_million_tokens(7.50),
            )
        ],
    ),
    "o3-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(20.00),
                output_cost_per_token=per_million_tokens(80.00),
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
    "o1-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.10),
                output_cost_per_token=per_million_tokens(4.40),
                cache_read_cost_per_token=per_million_tokens(0.55),
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
    "codex-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.50),
                output_cost_per_token=per_million_tokens(6.00),
                cache_read_cost_per_token=per_million_tokens(0.375),
            )
        ],
    ),
    # --- OpenAI: Embedding models ---
    "text-embedding-3-small": ModelPricing(
        tiers=[PricingTier(input_cost_per_token=per_million_tokens(0.02), output_cost_per_token=0.0)]
    ),
    "text-embedding-3-large": ModelPricing(
        tiers=[PricingTier(input_cost_per_token=per_million_tokens(0.143), output_cost_per_token=0.0)]
    ),
    "text-embedding-ada-002": ModelPricing(
        tiers=[PricingTier(input_cost_per_token=per_million_tokens(0.11), output_cost_per_token=0.0)]
    ),
    # --- DeepSeek ---
    "deepseek-r1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.35),
                output_cost_per_token=per_million_tokens(5.40),
            )
        ],
    ),
    "deepseek-v3": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.14),
                output_cost_per_token=per_million_tokens(4.56),
            )
        ],
    ),
    "deepseek-v3.1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.23),
                output_cost_per_token=per_million_tokens(4.94),
            )
        ],
    ),
    "deepseek-v3.2-sp": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.58),
                output_cost_per_token=per_million_tokens(1.68),
            )
        ],
    ),
    "deepseek-v3.2": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.58),
                output_cost_per_token=per_million_tokens(1.68),
            )
        ],
    ),
    # --- xAI (Grok) ---
    "grok-3": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(3.00),
                output_cost_per_token=per_million_tokens(15.00),
            )
        ],
    ),
    "grok-3-mini": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.25),
                output_cost_per_token=per_million_tokens(1.27),
            )
        ],
    ),
    "grok-4.2": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.00),
                output_cost_per_token=per_million_tokens(6.00),
            )
        ],
    ),
    "grok-4-fast": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.20),
                output_cost_per_token=per_million_tokens(0.50),
            )
        ],
    ),
    "grok-4": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(3.00),
                output_cost_per_token=per_million_tokens(15.00),
            )
        ],
    ),
    "grok-code-fast-1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.20),
                output_cost_per_token=per_million_tokens(1.50),
            )
        ],
    ),
    "grok-4.1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.20),
                output_cost_per_token=per_million_tokens(0.50),
            )
        ],
    ),
    # --- Meta (Llama) ---
    "llama-3.3-70b": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.71),
                output_cost_per_token=per_million_tokens(0.71),
            )
        ],
    ),
    "llama-4-scout": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.16),
                output_cost_per_token=per_million_tokens(0.64),
            )
        ],
    ),
    "llama-4-maverick": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.25),
                output_cost_per_token=per_million_tokens(1.00),
            )
        ],
    ),
    # --- Mistral ---
    "mistral-large": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.50),
                output_cost_per_token=per_million_tokens(1.50),
            )
        ],
    ),
    "codestral": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.30),
                output_cost_per_token=per_million_tokens(0.90),
            )
        ],
    ),
    # --- OpenAI OSS ---
    "gpt-oss-120b": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.15),
                output_cost_per_token=per_million_tokens(0.60),
            )
        ],
    ),
    # --- Cohere ---
    "cohere-command-a": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.50),
                output_cost_per_token=per_million_tokens(10.00),
            )
        ],
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


def apply_cost_multiplier(cost: Cost, multiplier: float) -> Cost:
    """Apply a multiplier to all fields in a cost breakdown."""
    return Cost(
        input_cost=cost.input_cost * multiplier,
        output_cost=cost.output_cost * multiplier,
        total_cost=cost.total_cost * multiplier,
        cache_read_cost=cost.cache_read_cost * multiplier if cost.cache_read_cost is not None else None,
        cache_creation_cost=cost.cache_creation_cost * multiplier if cost.cache_creation_cost is not None else None,
    )
