"""OpenAI pricing data and cost calculation.

Pricing source: https://developers.openai.com/api/docs/pricing
"""

from lmux.cost import (
    ModelPricing,
    PricingTier,
    build_pricing_index,
    calculate_cost,
    per_million_tokens,
    resolve_pricing,
)
from lmux.types import Cost, Usage

_PRICING: dict[str, ModelPricing] = {
    # GPT-5 family
    # gpt-5.6 family (sol/terra/luna). Cache writes are billed on gpt-5.6+ only,
    # at a flat 1.25x the input rate (no per-TTL split), via cache_creation_cost_per_token.
    "gpt-5.6-sol": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.00),
                output_cost_per_token=per_million_tokens(30.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
                cache_creation_cost_per_token=per_million_tokens(6.25),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(10.00),
                output_cost_per_token=per_million_tokens(45.00),
                cache_read_cost_per_token=per_million_tokens(1.00),
                cache_creation_cost_per_token=per_million_tokens(12.50),
                min_input_tokens=272_000,
            ),
        ],
    ),
    "gpt-5.6-terra": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.00),
                output_cost_per_token=per_million_tokens(12.00),
                cache_read_cost_per_token=per_million_tokens(0.20),
                cache_creation_cost_per_token=per_million_tokens(2.50),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(4.00),
                output_cost_per_token=per_million_tokens(18.00),
                cache_read_cost_per_token=per_million_tokens(0.40),
                cache_creation_cost_per_token=per_million_tokens(5.00),
                min_input_tokens=272_000,
            ),
        ],
    ),
    "gpt-5.6-luna": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.20),
                output_cost_per_token=per_million_tokens(1.20),
                cache_read_cost_per_token=per_million_tokens(0.02),
                cache_creation_cost_per_token=per_million_tokens(0.25),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(0.40),
                output_cost_per_token=per_million_tokens(1.80),
                cache_read_cost_per_token=per_million_tokens(0.04),
                cache_creation_cost_per_token=per_million_tokens(0.50),
                min_input_tokens=272_000,
            ),
        ],
    ),
    # The bare "gpt-5.6" alias routes to gpt-5.6-sol; mirror Sol's rates so it does not
    # fall through to the broad "gpt-5" prefix and get under-priced at 1.25/10.
    "gpt-5.6": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.00),
                output_cost_per_token=per_million_tokens(30.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
                cache_creation_cost_per_token=per_million_tokens(6.25),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(10.00),
                output_cost_per_token=per_million_tokens(45.00),
                cache_read_cost_per_token=per_million_tokens(1.00),
                cache_creation_cost_per_token=per_million_tokens(12.50),
                min_input_tokens=272_000,
            ),
        ],
    ),
    "gpt-5.5-pro": ModelPricing(
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
    "gpt-5.5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.00),
                output_cost_per_token=per_million_tokens(30.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(10.00),
                output_cost_per_token=per_million_tokens(45.00),
                cache_read_cost_per_token=per_million_tokens(1.00),
                min_input_tokens=272_000,
            ),
        ],
    ),
    # "cyber" tier is materially pricier than gpt-5.5, so it needs an explicit key
    # (it would otherwise prefix-match gpt-5.5 and be under-priced).
    "gpt-5.5-cyber": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(12.50),
                output_cost_per_token=per_million_tokens(75.00),
                cache_read_cost_per_token=per_million_tokens(1.25),
            ),
        ],
    ),
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
    "gpt-5.3-chat-latest": ModelPricing(
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
    "gpt-5.2-chat-latest": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.75),
                output_cost_per_token=per_million_tokens(14.00),
                cache_read_cost_per_token=per_million_tokens(0.175),
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
    "gpt-5.1-chat-latest": ModelPricing(
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
    "gpt-5-chat-latest": ModelPricing(
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
    # The 2024-05-13 snapshot is priced higher than the current gpt-4o (5/15 vs 2.50/10) and
    # has no cached-input rate; an explicit key stops it inheriting the cheaper gpt-4o prefix.
    "gpt-4o-2024-05-13": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.00),
                output_cost_per_token=per_million_tokens(15.00),
            )
        ],
    ),
    # Search-preview models: same input/output as their base family but OpenAI publishes NO
    # cached-input price, so an explicit key drops the spurious cache rate the prefix would add.
    # (Per-tool-call web-search fees are separate and not modeled here.)
    "gpt-4o-search-preview": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.50),
                output_cost_per_token=per_million_tokens(10.00),
            )
        ],
    ),
    "gpt-4o-mini-search-preview": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.15),
                output_cost_per_token=per_million_tokens(0.60),
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
    "chat-latest": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.00),
                output_cost_per_token=per_million_tokens(30.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
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

# Case-insensitive, longest-prefix index (see lmux.cost.resolve_pricing).
_PRICING_BY_PREFIX = build_pricing_index(_PRICING)

# Models OpenAI lists without a published token price (e.g. gpt-5.4-cyber, shown with blank
# pricing columns). Returning None is correct — do NOT let them fall through to a broad family
# prefix (e.g. gpt-5.4) and inherit a fabricated rate, and do not report a regional uplift for them.
_UNPRICED_MODELS = frozenset({"gpt-5.4-cyber"})

# 10% uplift for regional processing (data residency) endpoints. Per OpenAI, this
# applies to the gpt-5.4 family (gpt-5.4, gpt-5.4-mini, gpt-5.4-nano, gpt-5.4-pro),
# the gpt-5.5 family (gpt-5.5, gpt-5.5-pro), and the gpt-5.6 family (sol/terra/luna).
REGIONAL_UPLIFT = 1.1
_REGIONAL_UPLIFT_PREFIXES = ("gpt-5.4", "gpt-5.5", "gpt-5.6")


def calculate_openai_cost(model: str, usage: Usage) -> Cost | None:
    """Calculate cost for an OpenAI API call. Returns None if model pricing is unknown."""
    if model.lower() in _UNPRICED_MODELS:
        return None
    pricing = resolve_pricing(model, _PRICING_BY_PREFIX)
    if pricing is None:
        return None
    return calculate_cost(usage, pricing)


def regional_uplift_applies(model: str) -> bool:
    """Whether the regional processing (data residency) uplift applies to this model."""
    model_lower = model.lower()
    if model_lower in _UNPRICED_MODELS:
        return False
    return any(model_lower.startswith(prefix) for prefix in _REGIONAL_UPLIFT_PREFIXES)


def apply_cost_multiplier(cost: Cost, multiplier: float) -> Cost:
    """Apply a multiplier to all fields in a cost breakdown."""
    return Cost(
        input_cost=cost.input_cost * multiplier,
        output_cost=cost.output_cost * multiplier,
        total_cost=cost.total_cost * multiplier,
        cache_read_cost=cost.cache_read_cost * multiplier if cost.cache_read_cost is not None else None,
        cache_creation_cost=cost.cache_creation_cost * multiplier if cost.cache_creation_cost is not None else None,
    )
