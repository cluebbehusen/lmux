"""Anthropic pricing data and cost calculation.

Pricing source: https://platform.claude.com/docs/en/about-claude/pricing

Claude on Vertex AI bills these same list prices on the global endpoint, so
this table also covers AnthropicVertexProvider. Regional and multi-region
Vertex endpoints carry a 10% premium on Claude Sonnet 4.5/Haiku 4.5/Opus 4.5
and all later models (see VERTEX_REGIONAL_MULTIPLIER); older models are
priced uniformly across all endpoints. Vertex pricing reference:
https://cloud.google.com/vertex-ai/generative-ai/pricing

Claude in Microsoft Foundry bills Anthropic's standard API pricing, so this
table also covers AnthropicFoundryProvider. Global Standard deployments use
these list prices with no multiplier; Foundry's US Data Zone Standard
deployment type (equivalent to inference_geo "us") applies the same 1.1x
premium as US_INFERENCE_MULTIPLIER.
"""

from datetime import date

from lmux.cost import ModelPricing, PricingSchedule, PricingTier, calculate_cost, per_million_tokens
from lmux.types import Cost, Usage

# Standard (global) pricing. Cache writes default to the 5-minute rate (1.25x
# input); 1-hour writes (2x input) are billed via the per-TTL breakdown that the
# API reports in usage.cache_creation.
_PRICING: dict[str, ModelPricing] = {
    # Claude Fable 5 / Mythos 5 family
    "claude-fable-5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(10.00),
                output_cost_per_token=per_million_tokens(50.00),
                cache_read_cost_per_token=per_million_tokens(1.00),
                cache_creation_cost_per_token=per_million_tokens(12.50),
                cache_creation_cost_per_token_by_ttl={"1h": per_million_tokens(20.00)},
            ),
        ],
    ),
    "claude-mythos-5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(10.00),
                output_cost_per_token=per_million_tokens(50.00),
                cache_read_cost_per_token=per_million_tokens(1.00),
                cache_creation_cost_per_token=per_million_tokens(12.50),
                cache_creation_cost_per_token_by_ttl={"1h": per_million_tokens(20.00)},
            ),
        ],
    ),
    # Claude Sonnet 5 family — introductory pricing through 2026-08-31, then
    # standard pricing (matching Sonnet 4.6) from 2026-09-01. Full 1M context
    # at standard pricing, so there is no >200K tier.
    "claude-sonnet-5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.00),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(0.20),
                cache_creation_cost_per_token=per_million_tokens(2.50),
                cache_creation_cost_per_token_by_ttl={"1h": per_million_tokens(4.00)},
            ),
        ],
        schedules=[
            PricingSchedule(
                valid_from=date(2026, 9, 1),
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
        ],
    ),
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

VERTEX_REGIONAL_MULTIPLIER = 1.1

# The 10% Vertex regional/multi-region premium applies to Claude Sonnet 4.5,
# Haiku 4.5, Opus 4.5, and all later models. These older models keep uniform
# pricing across every Vertex endpoint, so they are exempt.
_VERTEX_UNIFORM_PRICING_MODELS = (
    "claude-opus-4-1",
    "claude-opus-4",
    "claude-sonnet-4",
    "claude-3-7-sonnet",
    "claude-3-5-sonnet",
    "claude-3-5-haiku",
    "claude-3-opus",
    "claude-3-haiku",
)
_VERTEX_PREMIUM_PRICING_MODELS = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
)
# Sorted by prefix length descending so the longest match wins — needed to
# tell e.g. claude-opus-4-8 (premium) apart from claude-opus-4 (uniform).
_VERTEX_PREMIUM_BY_PREFIX = sorted(
    [(prefix, True) for prefix in _VERTEX_PREMIUM_PRICING_MODELS]
    + [(prefix, False) for prefix in _VERTEX_UNIFORM_PRICING_MODELS],
    key=lambda item: len(item[0]),
    reverse=True,
)


def has_vertex_regional_premium(model: str) -> bool:
    """Whether the model carries the 10% premium on regional/multi-region Vertex endpoints.

    Unknown models default to True — the premium applies to all models newer
    than the fixed set of exempt older models.
    """
    for prefix, premium in _VERTEX_PREMIUM_BY_PREFIX:
        if model.startswith(prefix):
            return premium
    return True


def calculate_anthropic_cost(model: str, usage: Usage, as_of: date | None = None) -> Cost | None:
    """Calculate cost for an Anthropic API call. Returns None if model pricing is unknown.

    ``as_of`` selects dated pricing for models with scheduled rate changes
    (e.g. Claude Sonnet 5's introductory period); it defaults to the latest
    schedule. See ``lmux.cost.calculate_cost``.
    """
    pricing = _PRICING.get(model)
    if pricing is None:
        for prefix, p in _PRICING_BY_PREFIX:
            if model.startswith(prefix):
                pricing = p
                break
    if pricing is None:
        return None
    return calculate_cost(usage, pricing, as_of)


def apply_cost_multiplier(cost: Cost, multiplier: float) -> Cost:
    """Apply a multiplier to all fields in a cost breakdown."""
    return Cost(
        input_cost=cost.input_cost * multiplier,
        output_cost=cost.output_cost * multiplier,
        total_cost=cost.total_cost * multiplier,
        cache_read_cost=cost.cache_read_cost * multiplier if cost.cache_read_cost is not None else None,
        cache_creation_cost=cost.cache_creation_cost * multiplier if cost.cache_creation_cost is not None else None,
    )
