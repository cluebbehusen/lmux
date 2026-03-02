"""Google Vertex AI pricing data and cost calculation."""

from lmux.cost import ModelPricing, PricingTier, calculate_cost, per_million_tokens
from lmux.types import Cost, Usage

# Pricing as of Mar 2, 2026 (source: https://cloud.google.com/vertex-ai/generative-ai/pricing)
_PRICING: dict[str, ModelPricing] = {
    # Gemini 2.5 Pro
    "gemini-2.5-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(0.3125),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(2.50),
                output_cost_per_token=per_million_tokens(15.00),
                cache_read_cost_per_token=per_million_tokens(0.625),
                min_input_tokens=200_000,
            ),
        ],
    ),
    # Gemini 2.5 Flash
    "gemini-2.5-flash": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.15),
                output_cost_per_token=per_million_tokens(0.60),
                cache_read_cost_per_token=per_million_tokens(0.0375),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(0.30),
                output_cost_per_token=per_million_tokens(1.20),
                cache_read_cost_per_token=per_million_tokens(0.075),
                min_input_tokens=200_000,
            ),
        ],
    ),
    # Gemini 2.0 Flash
    "gemini-2.0-flash": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.10),
                output_cost_per_token=per_million_tokens(0.40),
                cache_read_cost_per_token=per_million_tokens(0.025),
            )
        ],
    ),
    # Gemini 2.0 Flash-Lite
    "gemini-2.0-flash-lite": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.075),
                output_cost_per_token=per_million_tokens(0.30),
            )
        ],
    ),
    # Gemini 1.5 Pro
    "gemini-1.5-pro": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.25),
                output_cost_per_token=per_million_tokens(5.00),
                cache_read_cost_per_token=per_million_tokens(0.3125),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(2.50),
                output_cost_per_token=per_million_tokens(10.00),
                cache_read_cost_per_token=per_million_tokens(0.625),
                min_input_tokens=128_000,
            ),
        ],
    ),
    # Gemini 1.5 Flash
    "gemini-1.5-flash": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.075),
                output_cost_per_token=per_million_tokens(0.30),
                cache_read_cost_per_token=per_million_tokens(0.01875),
            ),
            PricingTier(
                input_cost_per_token=per_million_tokens(0.15),
                output_cost_per_token=per_million_tokens(0.60),
                cache_read_cost_per_token=per_million_tokens(0.0375),
                min_input_tokens=128_000,
            ),
        ],
    ),
    # Embedding models
    "text-embedding-005": ModelPricing(
        tiers=[PricingTier(input_cost_per_token=per_million_tokens(0.025), output_cost_per_token=0.0)]
    ),
    "text-embedding-004": ModelPricing(
        tiers=[PricingTier(input_cost_per_token=per_million_tokens(0.025), output_cost_per_token=0.0)]
    ),
    "text-multilingual-embedding-002": ModelPricing(
        tiers=[PricingTier(input_cost_per_token=per_million_tokens(0.025), output_cost_per_token=0.0)]
    ),
}

# Pre-sorted by key length descending for prefix matching (longest match first)
_PRICING_BY_PREFIX = sorted(_PRICING.items(), key=lambda item: len(item[0]), reverse=True)


def calculate_google_cost(model: str, usage: Usage) -> Cost | None:
    """Calculate cost for a Google Vertex AI API call. Returns None if model pricing is unknown."""
    pricing = _PRICING.get(model)
    if pricing is None:
        for prefix, p in _PRICING_BY_PREFIX:
            if model.startswith(prefix):
                pricing = p
                break
    if pricing is None:
        return None
    return calculate_cost(usage, pricing)
