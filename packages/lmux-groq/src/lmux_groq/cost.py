"""Groq pricing data and cost calculation.

Pricing source: https://groq.com/pricing/
"""

from lmux.cost import ModelPricing, PricingTier, calculate_cost, per_million_tokens
from lmux.types import Cost, Usage

_PRICING: dict[str, ModelPricing] = {
    # OpenAI GPT OSS family (prompt caching: 50% discount on cached input tokens)
    "openai/gpt-oss-20b": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.075),
                output_cost_per_token=per_million_tokens(0.30),
                cache_read_cost_per_token=per_million_tokens(0.0375),
            )
        ],
    ),
    "openai/gpt-oss-safeguard-20b": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.075),
                output_cost_per_token=per_million_tokens(0.30),
            )
        ],
    ),
    "openai/gpt-oss-120b": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.15),
                output_cost_per_token=per_million_tokens(0.60),
                cache_read_cost_per_token=per_million_tokens(0.075),
            )
        ],
    ),
    # Moonshot Kimi family (prompt caching: 50% discount on cached input tokens)
    "moonshotai/kimi-k2-instruct": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.00),
                output_cost_per_token=per_million_tokens(3.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
            )
        ],
    ),
    # Meta Llama 4 family
    "meta-llama/llama-4-scout-17b-16e-instruct": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(0.11), output_cost_per_token=per_million_tokens(0.34))
        ],
    ),
    "meta-llama/llama-4-maverick-17b-128e-instruct": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(0.20), output_cost_per_token=per_million_tokens(0.60))
        ],
    ),
    # Qwen family
    "qwen/qwen3-32b": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(0.29), output_cost_per_token=per_million_tokens(0.59))
        ],
    ),
    # Meta Llama 3.x family
    "llama-3.3-70b-versatile": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(0.59), output_cost_per_token=per_million_tokens(0.79))
        ],
    ),
    "llama-3.1-8b-instant": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(0.05), output_cost_per_token=per_million_tokens(0.08))
        ],
    ),
}

# Pre-sorted by key length descending for prefix matching (longest match first)
_PRICING_BY_PREFIX = sorted(_PRICING.items(), key=lambda item: len(item[0]), reverse=True)


def calculate_groq_cost(model: str, usage: Usage) -> Cost | None:
    """Calculate cost for a Groq API call. Returns None if model pricing is unknown."""
    pricing = _PRICING.get(model)
    if pricing is None:
        for prefix, p in _PRICING_BY_PREFIX:
            if model.startswith(prefix):
                pricing = p
                break
    if pricing is None:
        return None
    return calculate_cost(usage, pricing)
