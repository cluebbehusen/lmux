"""Cost calculation utility functions."""

from pydantic import BaseModel

from lmux.types import Cost, Usage


def per_million_tokens(price: float) -> float:
    """Convert a per-million-token price to a per-token price."""
    return price / 1_000_000


class ModelPricing(BaseModel):
    """Pricing data for a specific model."""

    input_cost_per_token: float
    output_cost_per_token: float
    cache_read_cost_per_token: float | None = None
    cache_creation_cost_per_token: float | None = None
    long_context_input_cost_per_token: float | None = None
    long_context_output_cost_per_token: float | None = None
    long_context_threshold: int = 200_000


def _resolve_rates(usage: Usage, pricing: ModelPricing) -> tuple[float, float]:
    """Return (input_cost_per_token, output_cost_per_token), switching to long-context rates when applicable."""
    if pricing.long_context_input_cost_per_token is None:
        return pricing.input_cost_per_token, pricing.output_cost_per_token

    total_input = usage.input_tokens
    if total_input > pricing.long_context_threshold:
        return (
            pricing.long_context_input_cost_per_token,
            pricing.long_context_output_cost_per_token or pricing.output_cost_per_token,
        )
    return pricing.input_cost_per_token, pricing.output_cost_per_token


def calculate_cost(usage: Usage, pricing: ModelPricing) -> Cost:
    """Calculate the monetary cost from token usage and per-token prices.

    ``usage.input_tokens`` is the **total** prompt token count as reported by
    the provider API.  Cached tokens (read and creation) are subsets of this
    total, so they are subtracted before billing at the regular input rate to
    avoid double-counting.

    When ``pricing`` includes long-context rates and the total input tokens
    (including cached) exceed ``long_context_threshold``, the higher rates
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
