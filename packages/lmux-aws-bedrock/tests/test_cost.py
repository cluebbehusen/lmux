"""Tests for AWS Bedrock pricing and cost calculation."""

from datetime import date

import pytest

from lmux.cost import ModelPricing, PricingTier, per_million_tokens
from lmux.types import Usage
from lmux_aws_bedrock.cost import (
    _REGIONAL_PRICING,  # pyright: ignore[reportPrivateUsage]
    calculate_bedrock_cost,
)


class TestCalculateBedrockCost:
    def test_known_model(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_bedrock_cost("meta.llama3-1-70b-instruct-v1", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 0.72 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 0.72 / 1_000_000)
        assert cost.total_cost == pytest.approx(cost.input_cost + cost.output_cost)

    def test_unknown_model_returns_none(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=50)
        cost = calculate_bedrock_cost("unknown-model-xyz", usage)
        assert cost is None

    def test_prefix_matching(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_bedrock_cost("meta.llama3-1-70b-instruct-v1:0", usage)
        assert cost is not None
        base_cost = calculate_bedrock_cost("meta.llama3-1-70b-instruct-v1", usage)
        assert base_cost is not None
        assert cost.total_cost == pytest.approx(base_cost.total_cost)

    def test_with_cache_tokens(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=200)
        cost = calculate_bedrock_cost("amazon.nova-pro-v1", usage)
        assert cost is not None
        assert cost.cache_read_cost is not None
        assert cost.cache_read_cost == pytest.approx(200 * 0.2 / 1_000_000)

    def test_grok_4_3_unit_corrected_pricing(self) -> None:
        """Regression guard: grok-4.3 uses the AWS '1M tokens' unit, not the default 1000x scaling."""
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=200)
        cost = calculate_bedrock_cost("xai.grok-4.3", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(800 * 1.25 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 2.50 / 1_000_000)
        assert cost.cache_read_cost == pytest.approx(200 * 0.20 / 1_000_000)

    def test_claude_3_5_haiku_dated_key_still_prices(self) -> None:
        """The dated 3.5 Haiku key is retained so real calls still price after AWS delisted it."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_bedrock_cost("anthropic.claude-3-5-haiku-20241022-v1:0", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 0.80 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 4.00 / 1_000_000)

    def test_sonnet_5_dated_schedule(self) -> None:
        """Sonnet 5 bills the introductory rate before 2026-09-01 and standard (1.5x) on/after."""
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        intro = calculate_bedrock_cost("anthropic.claude-sonnet-5", usage, as_of=date(2026, 7, 1))
        standard = calculate_bedrock_cost("anthropic.claude-sonnet-5", usage, as_of=date(2026, 9, 1))
        latest = calculate_bedrock_cost("anthropic.claude-sonnet-5", usage)
        assert intro is not None
        assert standard is not None
        assert latest is not None
        assert (intro.input_cost, intro.output_cost) == pytest.approx((2.2, 11.0))
        assert (standard.input_cost, standard.output_cost) == pytest.approx((3.3, 16.5))
        # No as_of defaults to the latest (standard) schedule.
        assert latest.input_cost == pytest.approx(3.3)

    def test_embedding_model(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=0)
        cost = calculate_bedrock_cost("amazon.titan-embed-text-v2", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(100 * 0.02 / 1_000_000)
        assert cost.output_cost == 0.0

    def test_zero_tokens(self) -> None:
        usage = Usage(input_tokens=0, output_tokens=0)
        cost = calculate_bedrock_cost("amazon.nova-micro-v1", usage)
        assert cost is not None
        assert cost.total_cost == 0.0

    def test_prefix_matching_longest_first(self) -> None:
        """Verify that meta.llama3-1-8b-instruct-v1:0 matches llama3-1-8b, not llama3-1-70b."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_bedrock_cost("meta.llama3-1-8b-instruct-v1:0", usage)
        assert cost is not None
        exact_cost = calculate_bedrock_cost("meta.llama3-1-8b-instruct-v1", usage)
        assert exact_cost is not None
        assert cost.total_cost == pytest.approx(exact_cost.total_cost)

    def test_region_none_uses_default(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost_none = calculate_bedrock_cost("meta.llama3-1-70b-instruct-v1", usage, region=None)
        cost_default = calculate_bedrock_cost("meta.llama3-1-70b-instruct-v1", usage)
        assert cost_none is not None
        assert cost_default is not None
        assert cost_none.total_cost == pytest.approx(cost_default.total_cost)

    def test_region_us_east_1_uses_default(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_bedrock_cost("meta.llama3-1-70b-instruct-v1", usage, region="us-east-1")
        cost_default = calculate_bedrock_cost("meta.llama3-1-70b-instruct-v1", usage)
        assert cost is not None
        assert cost_default is not None
        assert cost.total_cost == pytest.approx(cost_default.total_cost)

    def test_unknown_region_falls_back_to_default(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_bedrock_cost("meta.llama3-1-70b-instruct-v1", usage, region="xx-nowhere-99")
        cost_default = calculate_bedrock_cost("meta.llama3-1-70b-instruct-v1", usage)
        assert cost is not None
        assert cost_default is not None
        assert cost.total_cost == pytest.approx(cost_default.total_cost)

    def test_regional_pricing_empty_dict(self) -> None:
        """With empty _REGIONAL_PRICING, all regions fall back to default."""
        assert _REGIONAL_PRICING == {}
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_bedrock_cost("meta.llama3-1-70b-instruct-v1", usage, region="eu-west-1")
        cost_default = calculate_bedrock_cost("meta.llama3-1-70b-instruct-v1", usage)
        assert cost is not None
        assert cost_default is not None
        assert cost.total_cost == pytest.approx(cost_default.total_cost)

    def test_inference_profile_us_prefix(self) -> None:
        """us. prefixed inference profile IDs match and use non-global pricing."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_bedrock_cost("us.anthropic.claude-opus-4-6-v1", usage)
        bare_cost = calculate_bedrock_cost("anthropic.claude-opus-4-6-v1", usage)
        assert cost is not None
        assert bare_cost is not None
        assert cost.total_cost == pytest.approx(bare_cost.total_cost)

    def test_inference_profile_eu_prefix(self) -> None:
        """eu. prefixed inference profile IDs match and use non-global pricing."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_bedrock_cost("eu.anthropic.claude-opus-4-6-v1", usage)
        bare_cost = calculate_bedrock_cost("anthropic.claude-opus-4-6-v1", usage)
        assert cost is not None
        assert bare_cost is not None
        assert cost.total_cost == pytest.approx(bare_cost.total_cost)

    def test_inference_profile_global_prefix(self) -> None:
        """global. prefixed inference profile IDs use global (cheaper) pricing."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_bedrock_cost("global.anthropic.claude-opus-4-6-v1", usage)
        bare_cost = calculate_bedrock_cost("anthropic.claude-opus-4-6-v1", usage)
        assert cost is not None
        assert bare_cost is not None
        # Global pricing is cheaper than non-global
        assert cost.total_cost < bare_cost.total_cost

    def test_inference_profile_prefix_matching_with_version(self) -> None:
        """Inference profile IDs with version suffixes match via prefix."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_bedrock_cost("us.anthropic.claude-opus-4-6-v1:0", usage)
        base_cost = calculate_bedrock_cost("us.anthropic.claude-opus-4-6-v1", usage)
        assert cost is not None
        assert base_cost is not None
        assert cost.total_cost == pytest.approx(base_cost.total_cost)

    def test_inference_profile_model_without_profiles(self) -> None:
        """Models without inference profiles return None for prefixed IDs."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_bedrock_cost("us.ai21.jamba-1-5-large-v1", usage)
        assert cost is None

    def test_regional_pricing_exact_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regional pricing returns different cost when region has overrides."""
        regional = {
            "eu-west-1": {
                "meta.llama3-1-70b-instruct-v1": ModelPricing(
                    tiers=[
                        PricingTier(
                            input_cost_per_token=per_million_tokens(1.0),
                            output_cost_per_token=per_million_tokens(1.0),
                        )
                    ],
                ),
            },
        }
        usage = Usage(input_tokens=1000, output_tokens=500)
        monkeypatch.setattr("lmux_aws_bedrock.cost._REGIONAL_PRICING", regional)
        cost = calculate_bedrock_cost("meta.llama3-1-70b-instruct-v1", usage, region="eu-west-1")
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 1.0 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 1.0 / 1_000_000)

    def test_regional_pricing_prefix_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regional pricing uses prefix matching for versioned model IDs."""
        regional = {
            "eu-west-1": {
                "meta.llama3-1-70b-instruct-v1": ModelPricing(
                    tiers=[
                        PricingTier(
                            input_cost_per_token=per_million_tokens(2.0),
                            output_cost_per_token=per_million_tokens(2.0),
                        )
                    ],
                ),
            },
        }
        usage = Usage(input_tokens=1000, output_tokens=500)
        monkeypatch.setattr("lmux_aws_bedrock.cost._REGIONAL_PRICING", regional)
        cost = calculate_bedrock_cost("meta.llama3-1-70b-instruct-v1:0", usage, region="eu-west-1")
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 2.0 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 2.0 / 1_000_000)

    def test_regional_pricing_falls_back_to_default_for_unlisted_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A model not in regional overrides falls back to us-east-1 default."""
        regional = {
            "eu-west-1": {
                "some.other-model": ModelPricing(
                    tiers=[
                        PricingTier(
                            input_cost_per_token=per_million_tokens(99.0),
                            output_cost_per_token=per_million_tokens(99.0),
                        )
                    ],
                ),
            },
        }
        usage = Usage(input_tokens=1000, output_tokens=500)
        monkeypatch.setattr("lmux_aws_bedrock.cost._REGIONAL_PRICING", regional)
        cost = calculate_bedrock_cost("meta.llama3-1-70b-instruct-v1", usage, region="eu-west-1")
        cost_default = calculate_bedrock_cost("meta.llama3-1-70b-instruct-v1", usage)
        assert cost is not None
        assert cost_default is not None
        assert cost.total_cost == pytest.approx(cost_default.total_cost)
