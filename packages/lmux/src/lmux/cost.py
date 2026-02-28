"""Cost calculation utility functions."""

from pydantic import BaseModel, model_validator

from lmux.types import Cost, Usage


def per_million_tokens(price: float) -> float:
    """Convert a per-million-token price to a per-token price."""
    return price / 1_000_000


class PricingTier(BaseModel):
    """A single pricing tier based on total input token count."""

    input_cost_per_token: float
    output_cost_per_token: float
    cache_read_cost_per_token: float | None = None
    cache_creation_cost_per_token: float | None = None
    min_input_tokens: int = 0


class ModelPricing(BaseModel):
    """Pricing data for a specific model.

    ``tiers`` must contain at least one entry with ``min_input_tokens == 0``
    (the base tier).  Additional tiers define premium rates that apply when
    the total input token count exceeds their ``min_input_tokens`` threshold.
    Tiers must be ordered by ascending ``min_input_tokens``.
    """

    tiers: list[PricingTier]

    @model_validator(mode="after")
    def _validate_tiers(self) -> "ModelPricing":
        if not self.tiers:
            msg = "tiers must not be empty"
            raise ValueError(msg)
        if self.tiers[0].min_input_tokens != 0:
            msg = "first tier must have min_input_tokens == 0 (base tier)"
            raise ValueError(msg)
        for i in range(1, len(self.tiers)):
            if self.tiers[i].min_input_tokens <= self.tiers[i - 1].min_input_tokens:
                msg = "tiers must be ordered by strictly ascending min_input_tokens"
                raise ValueError(msg)
        return self


def _resolve_tier(usage: Usage, pricing: ModelPricing) -> PricingTier:
    """Return the highest-threshold tier whose ``min_input_tokens`` is exceeded by the total input."""
    total_input = usage.input_tokens

    # Iterate tiers in descending min_input_tokens order; pick the first one whose threshold is exceeded.
    for tier in sorted(pricing.tiers, key=lambda t: t.min_input_tokens, reverse=True):
        if total_input > tier.min_input_tokens:
            return tier

    # Fallback to the base tier (min_input_tokens == 0) when total_input == 0.
    return min(pricing.tiers, key=lambda t: t.min_input_tokens)


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

    tier = _resolve_tier(usage, pricing)

    # Cached tokens are a subset of input_tokens — bill them at their own rate,
    # not the full input rate.
    billable_input = usage.input_tokens - cache_read_tokens - cache_creation_tokens
    input_cost = billable_input * tier.input_cost_per_token
    output_cost = usage.output_tokens * tier.output_cost_per_token

    cache_read_cost_per_token = tier.cache_read_cost_per_token or 0.0
    cache_creation_cost_per_token = tier.cache_creation_cost_per_token or 0.0
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
