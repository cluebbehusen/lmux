"""Anthropic pricing data and cost calculation.

Pricing source: https://platform.claude.com/docs/en/about-claude/pricing
"""

from lmux.cost import ModelPricing, PricingTier, calculate_cost, per_million_tokens
from lmux.types import Cost, Usage

# Standard (global) pricing. Cache writes default to the 5-minute rate (1.25x
# input); 1-hour writes (2x input) are billed via the per-TTL breakdown that the
# API reports in usage.cache_creation.
_PRICING: dict[str, ModelPricing] = {
    # Claude 4.8 family
    "claude-opus-4-8": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.00),
                output_cost_per_token=per_million_tokens(25.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
                cache_creation_cost_per_token=per_million_tokens(6.25),
                cache_creation_cost_per_token_by_ttl={"1h": per_million_tokens(10.00)},
            ),
        ],
    ),
    # Claude 4.7 family
    "claude-opus-4-7": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.00),
                output_cost_per_token=per_million_tokens(25.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
                cache_creation_cost_per_token=per_million_tokens(6.25),
                cache_creation_cost_per_token_by_ttl={"1h": per_million_tokens(10.00)},
            ),
        ],
    ),
    # Claude 4.6 family
    "claude-opus-4-6": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.00),
                output_cost_per_token=per_million_tokens(25.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
                cache_creation_cost_per_token=per_million_tokens(6.25),
                cache_creation_cost_per_token_by_ttl={"1h": per_million_tokens(10.00)},
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
                cache_creation_cost_per_token_by_ttl={"1h": per_million_tokens(6.00)},
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
                cache_creation_cost_per_token_by_ttl={"1h": per_million_tokens(10.00)},
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
                cache_creation_cost_per_token_by_ttl={"1h": per_million_tokens(6.00)},
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(6.00),
                output_cost_per_token=per_million_tokens(22.50),
                cache_read_cost_per_token=per_million_tokens(0.60),
                cache_creation_cost_per_token=per_million_tokens(7.50),
                cache_creation_cost_per_token_by_ttl={"1h": per_million_tokens(12.00)},
                min_input_tokens=200000,
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
                cache_creation_cost_per_token_by_ttl={"1h": per_million_tokens(2.00)},
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
                cache_creation_cost_per_token_by_ttl={"1h": per_million_tokens(30.00)},
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
                cache_creation_cost_per_token_by_ttl={"1h": per_million_tokens(30.00)},
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
                cache_creation_cost_per_token_by_ttl={"1h": per_million_tokens(6.00)},
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(6.00),
                output_cost_per_token=per_million_tokens(22.50),
                cache_read_cost_per_token=per_million_tokens(0.60),
                cache_creation_cost_per_token=per_million_tokens(7.50),
                cache_creation_cost_per_token_by_ttl={"1h": per_million_tokens(12.00)},
                min_input_tokens=200000,
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
                cache_creation_cost_per_token_by_ttl={"1h": per_million_tokens(6.00)},
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
                cache_creation_cost_per_token_by_ttl={"1h": per_million_tokens(6.00)},
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
                cache_creation_cost_per_token_by_ttl={"1h": per_million_tokens(1.60)},
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
                cache_creation_cost_per_token_by_ttl={"1h": per_million_tokens(30.00)},
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
                cache_creation_cost_per_token_by_ttl={"1h": per_million_tokens(0.50)},
            ),
        ],
    ),
}

_PRICING_BY_PREFIX = sorted(_PRICING.items(), key=lambda item: len(item[0]), reverse=True)

US_INFERENCE_MULTIPLIER = 1.1


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
