"""Groq pricing data and cost calculation.

Pricing source: https://console.groq.com/docs/models (per-model cards at
https://console.groq.com/docs/model/<model-id> carry the cached-input rate)
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
    # OpenAI GPT OSS family (prompt caching: discounted cached input tokens). The 2026-12-01
    # gpt-oss-20b/120b rates come from Groq's pricing-change notice ahead of the models page.
    "openai/gpt-oss-20b": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.075),
                output_cost_per_token=per_million_tokens(0.30),
                cache_read_cost_per_token=per_million_tokens(0.0375),
            )
        ],
        schedules=[
            PricingSchedule(
                valid_from=date(2026, 12, 1),
                tiers=[
                    PricingTier(
                        input_cost_per_token=per_million_tokens(0.17),
                        output_cost_per_token=per_million_tokens(0.36),
                        cache_read_cost_per_token=per_million_tokens(0.09),
                    )
                ],
            ),
        ],
    ),
    "openai/gpt-oss-safeguard-20b": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(0.075),
                output_cost_per_token=per_million_tokens(0.30),
                cache_read_cost_per_token=per_million_tokens(0.0375),
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
        schedules=[
            PricingSchedule(
                valid_from=date(2026, 12, 1),
                tiers=[
                    PricingTier(
                        input_cost_per_token=per_million_tokens(0.33),
                        output_cost_per_token=per_million_tokens(0.71),
                        cache_read_cost_per_token=per_million_tokens(0.17),
                    )
                ],
            ),
        ],
    ),
    # Moonshot Kimi family (prompt caching: 50% discount on cached input tokens)
    "moonshotai/kimi-k2-instruct-0905": ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(1.00),
                output_cost_per_token=per_million_tokens(3.00),
                cache_read_cost_per_token=per_million_tokens(0.50),
            )
        ],
    ),
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
    # Llama Prompt Guard 2 (preview classifiers; evaluation-only per Groq)
    "meta-llama/llama-prompt-guard-2-22m": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(0.03), output_cost_per_token=per_million_tokens(0.03))
        ],
    ),
    "meta-llama/llama-prompt-guard-2-86m": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(0.04), output_cost_per_token=per_million_tokens(0.04))
        ],
    ),
    # Qwen family
    "qwen/qwen3-32b": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(0.29), output_cost_per_token=per_million_tokens(0.59))
        ],
    ),
    "qwen/qwen3.8-27b": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(0.80), output_cost_per_token=per_million_tokens(4.00))
        ],
    ),
    "qwen/qwen3.6-27b": ModelPricing(
        tiers=[
            PricingTier(input_cost_per_token=per_million_tokens(0.60), output_cost_per_token=per_million_tokens(3.00))
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

# Case-insensitive, longest-prefix index (see lmux.cost.resolve_pricing).
_PRICING_BY_PREFIX = build_pricing_index(_PRICING)


def calculate_groq_cost(model: str, usage: Usage, as_of: date | None = None) -> Cost | None:
    """Calculate cost for a Groq API call. Returns None if model pricing is unknown.

    ``as_of`` selects dated pricing for models with scheduled rate changes
    (e.g. the gpt-oss repricing on 2026-12-01); it defaults to the latest
    schedule. See ``lmux.cost.calculate_cost``.
    """
    pricing = resolve_pricing(model, _PRICING_BY_PREFIX)
    if pricing is None:
        return None
    return calculate_cost(usage, pricing, as_of)
