"""GCP Vertex AI pricing data and cost calculation.

Prices are for standard on-demand (global endpoint) pricing.
Source: https://cloud.google.com/vertex-ai/generative-ai/pricing
"""

from lmux.cost import ModelPricing, PricingTier, calculate_cost, per_million_tokens
from lmux.types import Cost, Usage

_PRICING: dict[str, ModelPricing] = {
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
    # ── Anthropic Claude (via Vertex AI) ───────────────────────
    "claude-opus-4-6": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.0),
                output_cost_per_token=per_million_tokens(25.0),
                cache_read_cost_per_token=per_million_tokens(0.5),
            ),
        ],
    ),
    "claude-sonnet-4-6": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(3.0),
                output_cost_per_token=per_million_tokens(15.0),
                cache_read_cost_per_token=per_million_tokens(0.3),
            ),
        ],
    ),
    "claude-opus-4-5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(5.0),
                output_cost_per_token=per_million_tokens(25.0),
                cache_read_cost_per_token=per_million_tokens(0.5),
            ),
        ],
    ),
    "claude-sonnet-4-5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(3.0),
                output_cost_per_token=per_million_tokens(15.0),
                cache_read_cost_per_token=per_million_tokens(0.3),
            ),
        ],
    ),
    "claude-opus-4-1": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(15.0),
                output_cost_per_token=per_million_tokens(75.0),
                cache_read_cost_per_token=per_million_tokens(1.5),
            ),
        ],
    ),
    "claude-opus-4": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(15.0),
                output_cost_per_token=per_million_tokens(75.0),
                cache_read_cost_per_token=per_million_tokens(1.5),
            ),
        ],
    ),
    "claude-sonnet-4": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(3.0),
                output_cost_per_token=per_million_tokens(15.0),
                cache_read_cost_per_token=per_million_tokens(0.3),
            ),
        ],
    ),
    "claude-haiku-4-5": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.0),
                output_cost_per_token=per_million_tokens(5.0),
                cache_read_cost_per_token=per_million_tokens(0.1),
            ),
        ],
    ),
    "claude-3-5-sonnet-v2": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(3.0),
                output_cost_per_token=per_million_tokens(15.0),
                cache_read_cost_per_token=per_million_tokens(0.3),
            ),
        ],
    ),
    "claude-3-5-haiku": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.8),
                output_cost_per_token=per_million_tokens(4.0),
                cache_read_cost_per_token=per_million_tokens(0.08),
            ),
        ],
    ),
    "claude-3-opus": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(15.0),
                output_cost_per_token=per_million_tokens(75.0),
                cache_read_cost_per_token=per_million_tokens(1.5),
            ),
        ],
    ),
    "claude-3-haiku": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.25),
                output_cost_per_token=per_million_tokens(1.25),
                cache_read_cost_per_token=per_million_tokens(0.03),
            ),
        ],
    ),
    # ── Mistral (via Vertex AI) ────────────────────────────────
    "mistral-medium-3": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.0),
                output_cost_per_token=per_million_tokens(3.0),
            ),
        ],
    ),
    "mistral-small-2503": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.2),
                output_cost_per_token=per_million_tokens(0.6),
            ),
        ],
    ),
    "codestral-2": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.5),
                output_cost_per_token=per_million_tokens(1.5),
            ),
        ],
    ),
    # ── Meta Llama (via Vertex AI) ─────────────────────────────
    "llama-4-maverick-17b-128e-instruct-maas": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.5),
                output_cost_per_token=per_million_tokens(0.77),
            ),
        ],
    ),
    "llama-4-scout-17b-16e-instruct-maas": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.11),
                output_cost_per_token=per_million_tokens(0.34),
            ),
        ],
    ),
    "llama-3.3-70b-instruct-maas": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.2),
                output_cost_per_token=per_million_tokens(0.2),
            ),
        ],
    ),
    # ── Embedding models ───────────────────────────────────────
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
                input_cost_per_token=per_million_tokens(0.025),
                output_cost_per_token=0.0,
            ),
        ],
    ),
    "text-embedding-004": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.025),
                output_cost_per_token=0.0,
            ),
        ],
    ),
    "text-multilingual-embedding-002": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.025),
                output_cost_per_token=0.0,
            ),
        ],
    ),
}

# Pre-sorted by key length descending for prefix matching (longest match first)
_PRICING_BY_PREFIX = sorted(_PRICING.items(), key=lambda item: len(item[0]), reverse=True)


def calculate_gcp_vertex_cost(model: str, usage: Usage) -> Cost | None:
    """Calculate cost for a Vertex AI API call. Returns None if model pricing is unknown."""
    pricing = _PRICING.get(model)
    if pricing is None:
        for prefix, p in _PRICING_BY_PREFIX:
            if model.startswith(prefix):
                pricing = p
                break
    if pricing is None:
        return None
    return calculate_cost(usage, pricing)
