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


def calculate_token_cost(  # noqa: PLR0913
    input_tokens: int,
    output_tokens: int,
    input_cost_per_token: float,
    output_cost_per_token: float,
    *,
    cache_read_tokens: int = 0,
    cache_read_cost_per_token: float = 0.0,
    cache_creation_tokens: int = 0,
    cache_creation_cost_per_token: float = 0.0,
) -> Cost:
    """Calculate the monetary cost from token counts and per-token prices."""
    input_cost = input_tokens * input_cost_per_token
    output_cost = output_tokens * output_cost_per_token
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


def calculate_cost_from_usage(usage: Usage, pricing: ModelPricing) -> Cost:
    """Calculate cost from a Usage object and ModelPricing."""
    return calculate_token_cost(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        input_cost_per_token=pricing.input_cost_per_token,
        output_cost_per_token=pricing.output_cost_per_token,
        cache_read_tokens=usage.cache_read_tokens or 0,
        cache_read_cost_per_token=pricing.cache_read_cost_per_token or 0.0,
        cache_creation_tokens=usage.cache_creation_tokens or 0,
        cache_creation_cost_per_token=pricing.cache_creation_cost_per_token or 0.0,
    )
