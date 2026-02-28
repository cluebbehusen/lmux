"""Groq pricing data and cost calculation."""

from lmux.cost import ModelPricing, calculate_cost, per_million_tokens
from lmux.types import Cost, Usage

_PRICING: dict[str, ModelPricing] = {
    # GPT OSS family (prompt caching: 50% discount on cached input tokens)
    "gpt-oss-20b-128k": ModelPricing(
        input_cost_per_token=per_million_tokens(0.075),
        output_cost_per_token=per_million_tokens(0.30),
        cache_read_cost_per_token=per_million_tokens(0.0375),
    ),
    "gpt-oss-safeguard-20b": ModelPricing(
        input_cost_per_token=per_million_tokens(0.075),
        output_cost_per_token=per_million_tokens(0.30),
        cache_read_cost_per_token=per_million_tokens(0.0375),
    ),
    "gpt-oss-120b-128k": ModelPricing(
        input_cost_per_token=per_million_tokens(0.15),
        output_cost_per_token=per_million_tokens(0.60),
        cache_read_cost_per_token=per_million_tokens(0.075),
    ),
    # Kimi family (prompt caching: 50% discount on cached input tokens)
    "kimi-k2": ModelPricing(
        input_cost_per_token=per_million_tokens(1.00),
        output_cost_per_token=per_million_tokens(3.00),
        cache_read_cost_per_token=per_million_tokens(0.50),
    ),
    # Llama 4 family
    "llama-4-scout": ModelPricing(
        input_cost_per_token=per_million_tokens(0.11),
        output_cost_per_token=per_million_tokens(0.34),
    ),
    "llama-4-maverick": ModelPricing(
        input_cost_per_token=per_million_tokens(0.20),
        output_cost_per_token=per_million_tokens(0.60),
    ),
    # Qwen family
    "qwen-qwq-32b": ModelPricing(
        input_cost_per_token=per_million_tokens(0.29),
        output_cost_per_token=per_million_tokens(0.59),
    ),
    # Llama 3.x family
    "llama-3.3-70b-versatile": ModelPricing(
        input_cost_per_token=per_million_tokens(0.59),
        output_cost_per_token=per_million_tokens(0.79),
    ),
    "llama-3.1-8b-instant": ModelPricing(
        input_cost_per_token=per_million_tokens(0.05),
        output_cost_per_token=per_million_tokens(0.08),
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
