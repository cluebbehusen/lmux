"""Cost calculation utility functions."""

from collections.abc import Mapping
from datetime import date

from pydantic import BaseModel, model_validator

from lmux.types import Cost, Usage


def per_million_tokens(price: float) -> float:
    """Convert a per-million-token price to a per-token price."""
    return price / 1_000_000


def build_pricing_index(pricing: Mapping[str, "ModelPricing"]) -> list[tuple[str, "ModelPricing"]]:
    """Build a case-folded, longest-first prefix index for ``resolve_pricing``.

    Keys are lowercased so lookup is case-insensitive — provider ``model`` fields vary in
    capitalization (e.g. Azure returns ``Cohere-command-a`` and ``Phi-4`` but ``deepseek-r1``
    and ``grok-4``). Entries are sorted by lowercased-key length descending so the longest
    prefix wins; an exact match is just the longest possible prefix (a key as long as the id).
    Two keys that lowercase to the same string must carry the same pricing.
    """
    return sorted(
        ((key.lower(), value) for key, value in pricing.items()),
        key=lambda item: len(item[0]),
        reverse=True,
    )


def resolve_pricing(model: str, index: list[tuple[str, "ModelPricing"]]) -> "ModelPricing | None":
    """Resolve a model id to its pricing via case-insensitive longest-prefix matching.

    ``index`` comes from ``build_pricing_index``. The id is lowercased and matched against the
    lowercased keys; the first (longest) prefix hit wins, which yields the exact entry when one
    exists. Returns ``None`` when nothing matches.
    """
    model_lower = model.lower()
    for prefix, pricing in index:
        if model_lower.startswith(prefix):
            return pricing
    return None


class PricingTier(BaseModel):
    """A single pricing tier based on total input token count.

    ``cache_creation_cost_per_token`` is the default cache-write rate.
    ``cache_creation_cost_per_token_by_ttl`` holds per-TTL overrides (e.g.
    ``{"1h": ...}``) for providers whose extended-TTL writes bill at a higher
    rate; write tokens with no matching TTL entry bill at the default rate.
    """

    input_cost_per_token: float
    output_cost_per_token: float
    cache_read_cost_per_token: float | None = None
    cache_creation_cost_per_token: float | None = None
    cache_creation_cost_per_token_by_ttl: dict[str, float] | None = None
    min_input_tokens: int = 0


def _validate_tiers(tiers: list[PricingTier]) -> None:
    """Validate a token-count tier list: non-empty, base tier first, strictly ascending thresholds."""
    if not tiers:
        msg = "tiers must not be empty"
        raise ValueError(msg)
    if tiers[0].min_input_tokens != 0:
        msg = "first tier must have min_input_tokens == 0 (base tier)"
        raise ValueError(msg)
    for i in range(1, len(tiers)):
        if tiers[i].min_input_tokens <= tiers[i - 1].min_input_tokens:
            msg = "tiers must be ordered by strictly ascending min_input_tokens"
            raise ValueError(msg)


class PricingSchedule(BaseModel):
    """A dated pricing override that takes effect on ``valid_from``.

    A schedule's ``tiers`` follow the same rules as ``ModelPricing.tiers`` (a
    base tier with ``min_input_tokens == 0`` plus optional higher tiers).
    Schedules live in ``ModelPricing.schedules``; each one supersedes the
    model's base pricing from its ``valid_from`` date until the next schedule's
    ``valid_from``.
    """

    valid_from: date
    tiers: list[PricingTier]

    @model_validator(mode="after")
    def _validate(self) -> "PricingSchedule":
        _validate_tiers(self.tiers)
        return self


def _validate_schedules(schedules: list[PricingSchedule]) -> None:
    """Validate dated schedules: non-empty and ordered by strictly ascending ``valid_from``."""
    if not schedules:
        msg = "schedules must not be empty when provided"
        raise ValueError(msg)
    previous = schedules[0].valid_from
    for schedule in schedules[1:]:
        if schedule.valid_from <= previous:
            msg = "schedules must be ordered by strictly ascending valid_from"
            raise ValueError(msg)
        previous = schedule.valid_from


class ModelPricing(BaseModel):
    """Pricing data for a specific model.

    ``tiers`` is the base pricing and must contain at least one entry with
    ``min_input_tokens == 0`` (the base tier).  Additional tiers define premium
    rates that apply when the total input token count exceeds their
    ``min_input_tokens`` threshold; tiers must be ordered by ascending
    ``min_input_tokens``.

    ``schedules`` holds optional *dated* overrides for models whose price
    changes on a known date (e.g. an introductory rate that later rises to a
    standard rate).  Each schedule supersedes ``tiers`` from its ``valid_from``
    onward; ``calculate_cost``'s ``as_of`` argument selects which schedule
    applies (defaulting to the latest).  Schedules must be ordered by strictly
    ascending ``valid_from``.
    """

    tiers: list[PricingTier]
    schedules: list[PricingSchedule] | None = None

    @model_validator(mode="after")
    def _validate(self) -> "ModelPricing":
        _validate_tiers(self.tiers)
        if self.schedules is not None:
            _validate_schedules(self.schedules)
        return self


def _active_tiers(pricing: ModelPricing, as_of: date | None) -> list[PricingTier]:
    """Resolve the tier list in effect for ``as_of``.

    Models without ``schedules`` always use their base ``tiers``.  For dated
    models, ``as_of=None`` selects the latest schedule; otherwise the latest
    schedule whose ``valid_from`` is on or before ``as_of`` wins, falling back
    to the base ``tiers`` when ``as_of`` predates every schedule.
    """
    schedules = pricing.schedules
    if schedules is None:
        return pricing.tiers
    if as_of is None:
        return schedules[-1].tiers
    tiers = pricing.tiers
    for schedule in schedules:
        if schedule.valid_from <= as_of:
            tiers = schedule.tiers
    return tiers


def _resolve_tier(usage: Usage, tiers: list[PricingTier]) -> PricingTier:
    """Return the highest-threshold tier whose ``min_input_tokens`` is exceeded by the total input.

    A tier with ``min_input_tokens=200_000`` applies when ``total_input > 200_000``,
    matching provider semantics (e.g. Anthropic bills at premium rates for >200K tokens).
    """
    total_input = usage.input_tokens

    # Iterate tiers in descending min_input_tokens order; pick the first one whose threshold is exceeded.
    for tier in sorted(tiers, key=lambda t: t.min_input_tokens, reverse=True):
        if total_input > tier.min_input_tokens:
            return tier

    # The validated base tier (min_input_tokens == 0) guarantees a match for total_input == 0.
    return tiers[0]


def calculate_cost(usage: Usage, pricing: ModelPricing, as_of: date | None = None) -> Cost:
    """Calculate the monetary cost from token usage and per-token prices.

    ``usage.input_tokens`` is the **total** prompt token count (see ``Usage``
    — providers normalize to this convention).  Cached tokens (read and
    creation) are subsets of this total, so they are subtracted before billing
    at the regular input rate to avoid double-counting.

    Cache-write tokens bill at ``cache_creation_cost_per_token`` by default;
    when ``usage.cache_creation_tokens_by_ttl`` is reported and the tier has a
    matching per-TTL rate, those tokens bill at the per-TTL rate instead.

    When ``pricing`` includes multiple tiers and the total input tokens
    exceed a tier's ``min_input_tokens`` threshold, the higher rates
    are used for all tokens in the request.

    ``as_of`` selects the pricing schedule for models with dated ``schedules``
    (see ``ModelPricing``).  It defaults to ``None``, which uses the latest
    schedule; pass a ``date`` to bill at the rate in effect on that day.
    """
    cache_read_tokens = usage.cache_read_tokens or 0
    cache_creation_tokens = _total_cache_creation_tokens(usage)

    tier = _resolve_tier(usage, _active_tiers(pricing, as_of))

    # Cached tokens are a subset of input_tokens — bill them at their own rate,
    # not the full input rate.
    billable_input = usage.input_tokens - cache_read_tokens - cache_creation_tokens
    input_cost = billable_input * tier.input_cost_per_token
    output_cost = usage.output_tokens * tier.output_cost_per_token

    cache_read_cost_per_token = tier.cache_read_cost_per_token or 0.0
    cache_read_cost = cache_read_tokens * cache_read_cost_per_token if cache_read_tokens else None
    cache_creation_cost = _cache_creation_cost(usage, tier, cache_creation_tokens)
    total = input_cost + output_cost + (cache_read_cost or 0.0) + (cache_creation_cost or 0.0)

    return Cost(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total,
        cache_read_cost=cache_read_cost,
        cache_creation_cost=cache_creation_cost,
    )


def _total_cache_creation_tokens(usage: Usage) -> int:
    """Total cache-write tokens, robust to a breakdown without an aggregate."""
    breakdown_total = sum((usage.cache_creation_tokens_by_ttl or {}).values())
    return max(usage.cache_creation_tokens or 0, breakdown_total)


def _cache_creation_cost(usage: Usage, tier: PricingTier, total_tokens: int) -> float | None:
    """Cache-write cost: per-TTL rates where reported and priced, default rate otherwise."""
    if not total_tokens:
        return None
    default_rate = tier.cache_creation_cost_per_token or 0.0
    ttl_rates = tier.cache_creation_cost_per_token_by_ttl or {}
    cost = 0.0
    covered = 0
    for ttl, tokens in (usage.cache_creation_tokens_by_ttl or {}).items():
        cost += tokens * ttl_rates.get(ttl, default_rate)
        covered += tokens
    # Tokens not covered by the breakdown bill at the default rate.
    cost += (total_tokens - covered) * default_rate
    return cost
