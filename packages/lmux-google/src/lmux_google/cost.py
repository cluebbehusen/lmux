"""Google model pricing data and cost calculation.

Prices are for standard on-demand (global endpoint) Vertex AI pricing. The
Gemini Developer API paid tier matches these rates for Gemini 2.5 and later,
but diverges on Gemini 2.0: the Developer API bills 2.0 Flash at 0.10/0.40
against Vertex's 0.15/0.60. The Vertex rates are the ones stored here.

These are global-endpoint rates. Non-global Vertex endpoints carry a 10%
premium on the GA Gemini 3 and later families, in effect since 2026-07-01;
GoogleProvider applies it on top of these rates via
VERTEX_NON_GLOBAL_MULTIPLIER when a location other than "global" is set.

Pricing source: https://cloud.google.com/vertex-ai/generative-ai/pricing
"""

from datetime import date

from lmux.cost import (
    ModelPricing,
    PricingSchedule,
    PricingTier,
    build_pricing_index,
    calculate_cost,
    per_million_tokens,
    resolve_pricing,
)
from lmux.types import Cost, Usage

_PRICING: dict[str, ModelPricing] = {
    # ── Google Gemini 3 ────────────────────────────────────────
    "gemini-3.7-flash": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.75),
                output_cost_per_token=per_million_tokens(3.75),
                cache_read_cost_per_token=per_million_tokens(0.075),
            ),
        ],
        schedules=[
            PricingSchedule(
                valid_from=date(2027, 1, 1),
                tiers=[
                    PricingTier(
                        input_cost_per_token=per_million_tokens(1.50),
                        output_cost_per_token=per_million_tokens(7.50),
                        cache_read_cost_per_token=per_million_tokens(0.15),
                    ),
                ],
            ),
        ],
    ),
    "gemini-3.6-flash": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.75),
                output_cost_per_token=per_million_tokens(3.75),
                cache_read_cost_per_token=per_million_tokens(0.075),
            ),
        ],
        schedules=[
            PricingSchedule(
                valid_from=date(2027, 1, 1),
                tiers=[
                    PricingTier(
                        input_cost_per_token=per_million_tokens(1.50),
                        output_cost_per_token=per_million_tokens(7.50),
                        cache_read_cost_per_token=per_million_tokens(0.15),
                    ),
                ],
            ),
        ],
    ),
    "gemini-3.5-flash": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.5),
                output_cost_per_token=per_million_tokens(9.0),
                cache_read_cost_per_token=per_million_tokens(0.15),
            ),
        ],
    ),
    # Flash-Lite must keep its own key: without it the id prefix-matches
    # gemini-3.5-flash and bills 5x the correct input rate.
    "gemini-3.5-flash-lite": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.30),
                output_cost_per_token=per_million_tokens(2.50),
                cache_read_cost_per_token=per_million_tokens(0.03),
            ),
        ],
    ),
    # ── Google Gemini 3 (Preview) ──────────────────────────────
    "gemini-3.1-pro-preview": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.0),
                output_cost_per_token=per_million_tokens(12.0),
                cache_read_cost_per_token=per_million_tokens(0.2),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(4.0),
                output_cost_per_token=per_million_tokens(18.0),
                cache_read_cost_per_token=per_million_tokens(0.4),
                min_input_tokens=200_000,
            ),
        ],
    ),
    # Deprecated: gemini-3-pro-preview was shut down 2026-03-09, superseded by
    # gemini-3.1-pro-preview above. Kept for historical cost lookups.
    "gemini-3-pro-preview": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.0),
                output_cost_per_token=per_million_tokens(12.0),
                cache_read_cost_per_token=per_million_tokens(0.2),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(4.0),
                output_cost_per_token=per_million_tokens(18.0),
                cache_read_cost_per_token=per_million_tokens(0.4),
                min_input_tokens=200_000,
            ),
        ],
    ),
    # GA IDs and their shut-down -preview aliases share the same base rates. The
    # -preview keys are retained for historical cost lookups; new calls use the GA ids.
    "gemini-3.1-flash-lite": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.25),
                output_cost_per_token=per_million_tokens(1.50),
                cache_read_cost_per_token=per_million_tokens(0.025),
            ),
        ],
    ),
    "gemini-3.1-flash-lite-preview": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.25),
                output_cost_per_token=per_million_tokens(1.50),
                cache_read_cost_per_token=per_million_tokens(0.025),
            ),
        ],
    ),
    # Image-output models (gemini-*-image) are intentionally NOT priced — see _UNPRICED_IMAGE_PREFIXES
    # below. Google bills generated-image output far higher than text output ($30-$120/M vs
    # $1.50-$12/M), which a single output rate cannot represent, so they return None.
    "gemini-3-flash-preview": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.5),
                output_cost_per_token=per_million_tokens(3.0),
                cache_read_cost_per_token=per_million_tokens(0.05),
            ),
        ],
    ),
    # ── Google Gemini 2.5 ──────────────────────────────────────
    "gemini-2.5-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.0),
                cache_read_cost_per_token=per_million_tokens(0.125),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(2.5),
                output_cost_per_token=per_million_tokens(15.0),
                cache_read_cost_per_token=per_million_tokens(0.25),
                min_input_tokens=200_000,
            ),
        ],
    ),
    "gemini-2.5-flash": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.3),
                output_cost_per_token=per_million_tokens(2.5),
                cache_read_cost_per_token=per_million_tokens(0.03),
            ),
        ],
    ),
    "gemini-2.5-flash-lite": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.1),
                output_cost_per_token=per_million_tokens(0.4),
                cache_read_cost_per_token=per_million_tokens(0.01),
            ),
        ],
    ),
    # ── Google Gemini 2.0 ──────────────────────────────────────
    "gemini-2.0-flash": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.15),
                output_cost_per_token=per_million_tokens(0.6),
                cache_read_cost_per_token=per_million_tokens(0.0375),
            ),
        ],
    ),
    "gemini-2.0-flash-lite": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.075),
                output_cost_per_token=per_million_tokens(0.3),
                cache_read_cost_per_token=per_million_tokens(0.01875),
            ),
        ],
    ),
    # ── Google Gemini 1.5 ──────────────────────────────────────
    "gemini-1.5-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(5.0),
                cache_read_cost_per_token=per_million_tokens(0.3125),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(2.5),
                output_cost_per_token=per_million_tokens(10.0),
                cache_read_cost_per_token=per_million_tokens(0.625),
                min_input_tokens=128_000,
            ),
        ],
    ),
    "gemini-1.5-flash": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.075),
                output_cost_per_token=per_million_tokens(0.3),
                cache_read_cost_per_token=per_million_tokens(0.01875),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(0.15),
                output_cost_per_token=per_million_tokens(0.6),
                cache_read_cost_per_token=per_million_tokens(0.0375),
                min_input_tokens=128_000,
            ),
        ],
    ),
    # ── Google Gemini (additional) ────────────────────────────
    # gemini-2.5-flash-image is intentionally unpriced (see _UNPRICED_IMAGE_PREFIXES) — its image output
    # is billed far above the modeled text-output rate.
    # Robotics-ER 1.6: text/image/video input $1; audio input ($2) is higher and not modeled.
    "gemini-robotics-er-1.6-preview": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.00),
                output_cost_per_token=per_million_tokens(5.00),
            ),
        ],
    ),
    # Real runtime id is gemini-2.5-computer-use-preview-10-2025 (not "...-pro-...");
    # base rate <=200k, premium rate >200k, no context caching.
    "gemini-2.5-computer-use-preview-10-2025": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.00),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(2.50),
                output_cost_per_token=per_million_tokens(15.00),
                min_input_tokens=200_000,
            ),
        ],
    ),
    # ── Embedding models ───────────────────────────────────────
    # Text-input rate only; image/audio/video embedding inputs are billed at separate
    # rates not modeled by the text embedding path. The -preview key is retained for history.
    "gemini-embedding-2": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.20),
                output_cost_per_token=0.0,
            ),
        ],
    ),
    "gemini-embedding-2-preview": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.20),
                output_cost_per_token=0.0,
            ),
        ],
    ),
    "gemini-embedding-001": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.15),
                output_cost_per_token=0.0,
            ),
        ],
    ),
    "text-embedding-005": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.10),
                output_cost_per_token=0.0,
            ),
        ],
    ),
    "text-embedding-004": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.10),
                output_cost_per_token=0.0,
            ),
        ],
    ),
    "text-multilingual-embedding-002": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.10),
                output_cost_per_token=0.0,
            ),
        ],
    ),
}

# Case-insensitive, longest-prefix index (see lmux.cost.resolve_pricing).
_PRICING_BY_PREFIX = build_pricing_index(_PRICING)

# Image-output model families: Google bills generated-image output tokens far higher than text
# output (e.g. $30-$120/M vs $1.50-$12/M), but the provider collapses all output into one token
# count, so a single output rate would underprice image generation ~10-20x. Return None (unknown)
# rather than a confidently-wrong cost until modality-aware costing exists. Matched as *prefixes*
# (case-insensitively) so every dated/`-preview` variant — e.g. gemini-2.5-flash-image-preview — is
# covered, not just the bare id; a bare prefix would otherwise fall through to the text-priced base.
_UNPRICED_IMAGE_PREFIXES = (
    "gemini-2.5-flash-image",
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
    "gemini-3-pro-image",
)

# MARK: Vertex endpoint premium

VERTEX_NON_GLOBAL_MULTIPLIER = 1.1
"""Non-global Vertex endpoints bill 1.1x the global rate, from VERTEX_NON_GLOBAL_PREMIUM_START.

Applies only to the models Vertex publishes a Non-global row for; the Gemini
Developer API has no equivalent premium.
"""

VERTEX_NON_GLOBAL_PREMIUM_START = date(2026, 7, 1)
"""First day the non-global Vertex premium is billed.

Before this date, global-endpoint pricing applied to every Vertex endpoint, so
costs replayed against an earlier date take no multiplier.
"""

# Models the Vertex pricing page lists a "Non-global" row for. Membership is read
# off that page rather than inferred from a model being GA: Gemini 3.6 and 3.7
# Flash are GA and publish no Non-global rate, so they bill list price everywhere.
_VERTEX_PREMIUM_PRICING_MODELS = (
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)
# Has no Non-global row of its own, but would inherit one by prefix.
_VERTEX_UNIFORM_PRICING_MODELS = ("gemini-3.1-flash-lite-preview",)
# Sorted by prefix length descending so the longest match wins — needed to tell
# e.g. gemini-3.1-flash-lite-preview (uniform) from gemini-3.1-flash-lite (premium).
_VERTEX_PREMIUM_BY_PREFIX = sorted(
    [(prefix, True) for prefix in _VERTEX_PREMIUM_PRICING_MODELS]
    + [(prefix, False) for prefix in _VERTEX_UNIFORM_PRICING_MODELS],
    key=lambda item: len(item[0]),
    reverse=True,
)


def has_vertex_non_global_premium(model: str) -> bool:
    """Whether the model carries the 10% premium on non-global Vertex endpoints.

    Unknown models default to False, so a model Vertex has not published a
    Non-global rate for bills at list price rather than an inferred premium.
    """
    model_lower = model.lower()
    for prefix, premium in _VERTEX_PREMIUM_BY_PREFIX:
        if model_lower.startswith(prefix):
            return premium
    return False


def apply_cost_multiplier(cost: Cost, multiplier: float) -> Cost:
    """Apply a multiplier to all fields in a cost breakdown."""
    return Cost(
        input_cost=cost.input_cost * multiplier,
        output_cost=cost.output_cost * multiplier,
        total_cost=cost.total_cost * multiplier,
        cache_read_cost=cost.cache_read_cost * multiplier if cost.cache_read_cost is not None else None,
        cache_creation_cost=cost.cache_creation_cost * multiplier if cost.cache_creation_cost is not None else None,
    )


def calculate_google_cost(model: str, usage: Usage, as_of: date | None = None) -> Cost | None:
    """Calculate cost for a Google API call. Returns None if model pricing is unknown.

    ``as_of`` selects dated pricing for models with scheduled rate changes
    (e.g. the Gemini Flash introductory windows); it defaults to the latest
    schedule. See ``lmux.cost.calculate_cost``.
    """
    model_lower = model.lower()
    if any(model_lower.startswith(prefix) for prefix in _UNPRICED_IMAGE_PREFIXES):
        return None
    pricing = resolve_pricing(model, _PRICING_BY_PREFIX)
    if pricing is None:
        return None
    return calculate_cost(usage, pricing, as_of)
