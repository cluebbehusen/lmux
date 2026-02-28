"""Cost calculation utility functions."""

from pydantic import BaseModel

from lmux.types import Cost, Usage


def per_million_tokens(price: float) -> float:
    """Convert a per-million-token price to a per-token price."""
    return price / 1_000_000


class PricingTier(BaseModel):
    """A single pricing tier based on total input token count."""

    input_cost_per_token: float
    output_cost_per_token: float
    min_input_tokens: int = 0


class ModelPricing(BaseModel):
    """Pricing data for a specific model.

    ``tiers`` must contain at least one entry with ``min_input_tokens == 0``
    (the base tier).  Additional tiers define premium rates that apply when
    the total input token count exceeds their ``min_input_tokens`` threshold.
    """

    tiers: list[PricingTier]
    cache_read_cost_per_token: float | None = None
    cache_creation_cost_per_token: float | None = None


def _resolve_rates(usage: Usage, pricing: ModelPricing) -> tuple[float, float]:
    """Return (input_cost_per_token, output_cost_per_token), selecting the highest-threshold tier that applies."""
    total_input = usage.input_tokens

    # Iterate tiers in descending min_input_tokens order; pick the first one whose threshold is exceeded.
    for tier in sorted(pricing.tiers, key=lambda t: t.min_input_tokens, reverse=True):
        if total_input > tier.min_input_tokens:
            return tier.input_cost_per_token, tier.output_cost_per_token

    # Fallback to the base tier (min_input_tokens == 0) when total_input == 0.
    base = min(pricing.tiers, key=lambda t: t.min_input_tokens)
    return base.input_cost_per_token, base.output_cost_per_token


def calculate_cost(usage: Usage, pricing: ModelPricing) -> Cost:
    """Calculate the monetary cost from token usage and per-token prices.

    ``usage.input_tokens`` is the **total** prompt token count as reported by
    the provider API.  Cached tokens (read and creation) are subsets of this
    total, so they are subtracted before billing at the regular input rate to
    avoid double-counting.

    When ``pricing`` includes multiple tiers and the total input tokens
    exceed a tier's ``min_input_tokens`` threshold, the higher rates
    are used for all tokens in the request.
    """
    cache_read_tokens = usage.cache_read_tokens or 0
    cache_creation_tokens = usage.cache_creation_tokens or 0

    input_rate, output_rate = _resolve_rates(usage, pricing)

    # Cached tokens are a subset of input_tokens — bill them at their own rate,
    # not the full input rate.
    billable_input = usage.input_tokens - cache_read_tokens - cache_creation_tokens
    input_cost = billable_input * input_rate
    output_cost = usage.output_tokens * output_rate

    cache_read_cost_per_token = pricing.cache_read_cost_per_token or 0.0
    cache_creation_cost_per_token = pricing.cache_creation_cost_per_token or 0.0
    cache_read_cost = cache_read_tokens * cache_read_cost_per_token if cache_read_tokens else None
    cache_creation_cost = cache_creation_tokens * cache_creation_cost_per_token if cache_creation_tokens else None
    total = input_cost + output_cost + (cache_read_cost or 0.0) + (cache_creation_cost or 0.0)

    return Cost(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total,
        cache_read_cost=cache_read_cost,
        cache_creation_cost=cache_creation_cost,
    )
