"""Anthropic pricing data and cost calculation."""

from lmux.cost import ModelPricing, PricingTier, calculate_cost, per_million_tokens
from lmux.types import Cost, Usage

# Standard (global) pricing — uses 5-minute cache write rates.
# 1-hour cache writes (2x input) are set per-content-block and can't be detected from
# the API response, so accurate costing for extended TTL caches is not supported.
_PRICING: dict[str, ModelPricing] = {
    # Claude 4.6 family
    "claude-opus-4-6": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.00),
                output_cost_per_token=per_million_tokens(25.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
                cache_creation_cost_per_token=per_million_tokens(6.25),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(10.00),
                output_cost_per_token=per_million_tokens(37.50),
                cache_read_cost_per_token=per_million_tokens(0.50),
                cache_creation_cost_per_token=per_million_tokens(6.25),
                min_input_tokens=200_000,
            ),
        ],
    ),
    "claude-sonnet-4-6": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(3.00),
                output_cost_per_token=per_million_tokens(15.00),
                cache_read_cost_per_token=per_million_tokens(0.30),
                cache_creation_cost_per_token=per_million_tokens(3.75),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(6.00),
                output_cost_per_token=per_million_tokens(22.50),
                cache_read_cost_per_token=per_million_tokens(0.30),
                cache_creation_cost_per_token=per_million_tokens(3.75),
                min_input_tokens=200_000,
            ),
        ],
    ),
    # Claude 4.5 family
    "claude-opus-4-5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.00),
                output_cost_per_token=per_million_tokens(25.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
                cache_creation_cost_per_token=per_million_tokens(6.25),
            ),
        ],
    ),
    "claude-sonnet-4-5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(3.00),
                output_cost_per_token=per_million_tokens(15.00),
                cache_read_cost_per_token=per_million_tokens(0.30),
                cache_creation_cost_per_token=per_million_tokens(3.75),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(6.00),
                output_cost_per_token=per_million_tokens(22.50),
                cache_read_cost_per_token=per_million_tokens(0.30),
                cache_creation_cost_per_token=per_million_tokens(3.75),
                min_input_tokens=200_000,
            ),
        ],
    ),
    "claude-haiku-4-5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.00),
                output_cost_per_token=per_million_tokens(5.00),
                cache_read_cost_per_token=per_million_tokens(0.10),
                cache_creation_cost_per_token=per_million_tokens(1.25),
            ),
        ],
    ),
    # Claude 4.1 family
    "claude-opus-4-1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(15.00),
                output_cost_per_token=per_million_tokens(75.00),
                cache_read_cost_per_token=per_million_tokens(1.50),
                cache_creation_cost_per_token=per_million_tokens(18.75),
            ),
        ],
    ),
    # Claude 4 family
    "claude-opus-4": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(15.00),
                output_cost_per_token=per_million_tokens(75.00),
                cache_read_cost_per_token=per_million_tokens(1.50),
                cache_creation_cost_per_token=per_million_tokens(18.75),
            ),
        ],
    ),
    "claude-sonnet-4": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(3.00),
                output_cost_per_token=per_million_tokens(15.00),
                cache_read_cost_per_token=per_million_tokens(0.30),
                cache_creation_cost_per_token=per_million_tokens(3.75),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(6.00),
                output_cost_per_token=per_million_tokens(22.50),
                cache_read_cost_per_token=per_million_tokens(0.30),
                cache_creation_cost_per_token=per_million_tokens(3.75),
                min_input_tokens=200_000,
            ),
        ],
    ),
    # Claude 3.7 family
    "claude-3-7-sonnet": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(3.00),
                output_cost_per_token=per_million_tokens(15.00),
                cache_read_cost_per_token=per_million_tokens(0.30),
                cache_creation_cost_per_token=per_million_tokens(3.75),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(6.00),
                output_cost_per_token=per_million_tokens(22.50),
                cache_read_cost_per_token=per_million_tokens(0.30),
                cache_creation_cost_per_token=per_million_tokens(3.75),
                min_input_tokens=200_000,
            ),
        ],
    ),
    # Claude 3.5 family
    "claude-3-5-sonnet": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(3.00),
                output_cost_per_token=per_million_tokens(15.00),
                cache_read_cost_per_token=per_million_tokens(0.30),
                cache_creation_cost_per_token=per_million_tokens(3.75),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(6.00),
                output_cost_per_token=per_million_tokens(22.50),
                cache_read_cost_per_token=per_million_tokens(0.30),
                cache_creation_cost_per_token=per_million_tokens(3.75),
                min_input_tokens=200_000,
            ),
        ],
    ),
    "claude-3-5-haiku": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.80),
                output_cost_per_token=per_million_tokens(4.00),
                cache_read_cost_per_token=per_million_tokens(0.08),
                cache_creation_cost_per_token=per_million_tokens(1.00),
            ),
        ],
    ),
    # Claude 3 family
    "claude-3-opus": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(15.00),
                output_cost_per_token=per_million_tokens(75.00),
                cache_read_cost_per_token=per_million_tokens(1.50),
                cache_creation_cost_per_token=per_million_tokens(18.75),
            ),
        ],
    ),
    "claude-3-haiku": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.25),
                output_cost_per_token=per_million_tokens(1.25),
                cache_read_cost_per_token=per_million_tokens(0.03),
                cache_creation_cost_per_token=per_million_tokens(0.30),
            ),
        ],
    ),
}

_PRICING_BY_PREFIX = sorted(_PRICING.items(), key=lambda item: len(item[0]), reverse=True)

US_INFERENCE_MULTIPLIER = 1.1
FAST_MODE_MULTIPLIER = 6.0


def calculate_anthropic_cost(model: str, usage: Usage) -> Cost | None:
    """Calculate cost for an Anthropic API call. Returns None if model pricing is unknown."""
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
