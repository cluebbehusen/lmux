"""Tests for the shared Anthropic-on-Bedrock pricing."""

from datetime import date

import pytest

from lmux.cost import ModelPricing, PricingTier, calculate_cost, per_million_tokens
from lmux.types import Usage
from lmux_bedrock_shared.pricing import ANTHROPIC_REGIONAL_PRICING, calculate_bedrock_anthropic_cost, cost_or_none

_TOKENS_NO_CACHE = ModelPricing(tiers=[PricingTier(input_cost_per_token=1e-6, output_cost_per_token=2e-6)])
_TOKENS_WITH_CACHE = ModelPricing(
    tiers=[
        PricingTier(
            input_cost_per_token=1e-6,
            output_cost_per_token=2e-6,
            cache_read_cost_per_token=1e-7,
            cache_creation_cost_per_token=1e-6,
        )
    ]
)


class TestCostOrNone:
    def test_no_cache_usage_is_priced(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=50)
        assert cost_or_none(_TOKENS_NO_CACHE, usage) == calculate_cost(usage, _TOKENS_NO_CACHE)

    def test_cache_read_without_rate_is_none(self) -> None:
        assert cost_or_none(_TOKENS_NO_CACHE, Usage(input_tokens=100, output_tokens=50, cache_read_tokens=10)) is None

    def test_cache_creation_without_rate_is_none(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=50, cache_creation_tokens=10)
        assert cost_or_none(_TOKENS_NO_CACHE, usage) is None

    def test_cache_creation_by_ttl_without_rate_is_none(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=50, cache_creation_tokens_by_ttl={"1h": 10})
        assert cost_or_none(_TOKENS_NO_CACHE, usage) is None

    def test_cache_usage_with_rate_is_priced(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=50, cache_read_tokens=10, cache_creation_tokens=10)
        assert cost_or_none(_TOKENS_WITH_CACHE, usage) == calculate_cost(usage, _TOKENS_WITH_CACHE)


class TestCalculateBedrockAnthropicCost:
    def test_known_model_exact_match(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_bedrock_anthropic_cost("anthropic.claude-opus-4-8", usage)
        assert cost is not None
        assert cost.input_cost == 5.5
        assert cost.output_cost == 27.5
        assert cost.total_cost == 33.0

    def test_unknown_model_returns_none(self) -> None:
        # Non-anthropic id: no exact key, no prefix match, no profile prefix to strip.
        assert calculate_bedrock_anthropic_cost("openai.gpt-9", Usage(input_tokens=1, output_tokens=1)) is None

    def test_version_suffix_prefix_match(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=1000)
        suffixed = calculate_bedrock_anthropic_cost("anthropic.claude-opus-4-6-v1:0", usage)
        base = calculate_bedrock_anthropic_cost("anthropic.claude-opus-4-6-v1", usage)
        assert suffixed is not None
        assert suffixed == base

    def test_inference_profile_strip_fallback(self) -> None:
        # No dedicated us.anthropic.claude-3-5-haiku entry exists, so the profile
        # prefix is stripped and the bare model prices (exercises the recursion).
        usage = Usage(input_tokens=1000, output_tokens=1000)
        cost = calculate_bedrock_anthropic_cost("us.anthropic.claude-3-5-haiku-20241022-v1:0", usage)
        bare = calculate_bedrock_anthropic_cost("anthropic.claude-3-5-haiku-20241022-v1", usage)
        assert cost is not None
        assert cost == bare

    def test_regional_profile_pricing_beats_bare_where_encoded(self) -> None:
        # Region-profile prices are baked into the table as distinct keys, so no
        # multiplier is needed: the global endpoint prices below the us regional one.
        usage = Usage(input_tokens=1_000_000, output_tokens=0)
        glob = calculate_bedrock_anthropic_cost("global.anthropic.claude-opus-4-6-v1", usage)
        regional = calculate_bedrock_anthropic_cost("us.anthropic.claude-opus-4-6-v1", usage)
        assert glob is not None
        assert regional is not None
        assert glob.input_cost < regional.input_cost

    def test_regional_override_matches_versioned_request_id(self) -> None:
        """A Region with published Claude overrides bills those rates, not us-east-1's.

        Table keys omit the ``:0`` suffix real Bedrock model IDs carry, so the override has to
        survive prefix matching. Driven off the shipped table rather than a hardcoded rate.
        """
        region = next(r for r, models in ANTHROPIC_REGIONAL_PRICING.items() if models)
        model, pricing = next(iter(ANTHROPIC_REGIONAL_PRICING[region].items()))
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_bedrock_anthropic_cost(f"{model}:0", usage, region=region)
        assert cost is not None
        assert cost.total_cost == pytest.approx(calculate_cost(usage, pricing, None).total_cost)

    def test_default_and_unknown_regions_use_the_default_table(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=0)
        base = calculate_bedrock_anthropic_cost("anthropic.claude-opus-4-8", usage)
        assert base is not None
        for region in (None, "us-east-1", "xx-nowhere-99"):
            assert calculate_bedrock_anthropic_cost("anthropic.claude-opus-4-8", usage, region=region) == base

    def test_global_profile_never_takes_a_regional_standard_rate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``global.`` is priced separately (~10% below standard), so a Region carrying only a
        standard override must not capture a global call."""
        regional = {
            "eu-west-1": {
                "anthropic.claude-opus-4-8": ModelPricing(
                    tiers=[
                        PricingTier(
                            input_cost_per_token=per_million_tokens(50.0),
                            output_cost_per_token=per_million_tokens(50.0),
                        )
                    ],
                ),
            },
        }
        monkeypatch.setattr("lmux_bedrock_shared.pricing.ANTHROPIC_REGIONAL_PRICING", regional)
        usage = Usage(input_tokens=1_000_000, output_tokens=0)
        cost = calculate_bedrock_anthropic_cost("global.anthropic.claude-opus-4-8", usage, region="eu-west-1")
        default = calculate_bedrock_anthropic_cost("global.anthropic.claude-opus-4-8", usage)
        assert cost is not None
        assert default is not None
        assert cost.total_cost == pytest.approx(default.total_cost)

    def test_dated_schedule(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=0)
        intro = calculate_bedrock_anthropic_cost("anthropic.claude-sonnet-5", usage, as_of=date(2026, 7, 1))
        standard = calculate_bedrock_anthropic_cost("anthropic.claude-sonnet-5", usage, as_of=date(2026, 9, 1))
        latest = calculate_bedrock_anthropic_cost("anthropic.claude-sonnet-5", usage)
        assert intro is not None
        assert standard is not None
        assert latest is not None
        assert intro.input_cost < standard.input_cost
        assert latest == standard
