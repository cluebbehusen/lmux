"""Google model pricing data and cost calculation.

Prices are for standard on-demand (global endpoint) Vertex AI pricing;
the Gemini Developer API paid tier uses the same per-token rates.
Pricing source: https://cloud.google.com/vertex-ai/generative-ai/pricing
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
    # ── Google Gemini 3 ────────────────────────────────────────
    "gemini-3.5-flash": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.5),
                output_cost_per_token=per_million_tokens(9.0),
                cache_read_cost_per_token=per_million_tokens(0.15),
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
            ),
        ],
    ),
    "gemini-2.0-flash-lite": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.075),
                output_cost_per_token=per_million_tokens(0.3),
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


def calculate_google_cost(model: str, usage: Usage) -> Cost | None:
    """Calculate cost for a Google API call. Returns None if model pricing is unknown."""
    model_lower = model.lower()
    if any(model_lower.startswith(prefix) for prefix in _UNPRICED_IMAGE_PREFIXES):
        return None
    pricing = resolve_pricing(model, _PRICING_BY_PREFIX)
    if pricing is None:
        return None
    return calculate_cost(usage, pricing)
