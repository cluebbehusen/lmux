"""Tests for AWS Bedrock pricing and cost calculation."""

from datetime import date

import pytest

from lmux.cost import ModelPricing, PricingSchedule, PricingTier, calculate_cost, per_million_tokens
from lmux.types import Usage
from lmux_aws_bedrock.cost import (
    _REGIONAL_PRICING,
    calculate_bedrock_cost,
)
from lmux_bedrock_shared.pricing import ANTHROPIC_REGIONAL_PRICING, calculate_bedrock_anthropic_cost


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

    def test_dated_schedule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``as_of`` picks the schedule in effect on that date; no ``as_of`` picks the latest."""
        model = "vendor.scheduled"
        pricing = ModelPricing(
            tiers=[
                PricingTier(
                    input_cost_per_token=per_million_tokens(2.0),
                    output_cost_per_token=per_million_tokens(10.0),
                )
            ],
            schedules=[
                PricingSchedule(
                    valid_from=date(2026, 9, 1),
                    tiers=[
                        PricingTier(
                            input_cost_per_token=per_million_tokens(3.0),
                            output_cost_per_token=per_million_tokens(15.0),
                        )
                    ],
                )
            ],
        )
        monkeypatch.setattr("lmux_aws_bedrock.cost._PRICING", {model: pricing})
        monkeypatch.setattr("lmux_aws_bedrock.cost._PRICING_BY_PREFIX", [(model, pricing)])

        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        intro = calculate_bedrock_cost(model, usage, as_of=date(2026, 7, 1))
        standard = calculate_bedrock_cost(model, usage, as_of=date(2026, 9, 1))
        latest = calculate_bedrock_cost(model, usage)
        assert intro is not None
        assert standard is not None
        assert latest is not None
        assert (intro.input_cost, intro.output_cost) == pytest.approx((2.0, 10.0))
        assert (standard.input_cost, standard.output_cost) == pytest.approx((3.0, 15.0))
        assert latest.input_cost == pytest.approx(3.0)

    def test_regional_profile_falls_back_to_base(self) -> None:
        """A cross-region inference profile with no dedicated entry uses the base model's pricing.

        Guards the 3.5 Haiku regression: AWS delisted its us. profile, so only the base dated key
        is generated; a real ``us.anthropic.claude-3-5-haiku-20241022-v1:0`` call must still price.
        """
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_bedrock_cost("us.anthropic.claude-3-5-haiku-20241022-v1:0", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 0.80 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 4.00 / 1_000_000)

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

    def test_regional_override_matches_versioned_request_id(self) -> None:
        """A Region's override applies to the ID a request actually carries.

        Table keys omit the ``:0`` version suffix that real Bedrock model IDs end with, so the
        override has to survive prefix matching — the case that silently fell through to
        us-east-1 pricing. Driven off the shipped table rather than a hardcoded rate, so a
        pricing refresh doesn't churn the test.
        """
        region = next(r for r, models in _REGIONAL_PRICING.items() if models)
        model, pricing = next(iter(_REGIONAL_PRICING[region].items()))
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_bedrock_cost(f"{model}:0", usage, region=region)
        assert cost is not None
        assert cost.total_cost == pytest.approx(calculate_cost(usage, pricing, None).total_cost)

    def test_no_regional_override_is_cheaper_than_us_east_1(self) -> None:
        """Guard against the global-discount artifact re-entering the regional table.

        Genuine Bedrock regional variation is a premium — niche Regions cost more, not less. The
        only sub-baseline rate is the ~10% Global cross-Region discount, which belongs on the
        ``global.`` keys, not a regional override. A generator that fills a Region's standard rate
        from a Global-only meter re-creates that artifact, and it always lands below us-east-1. So
        every override must be >= the default; a genuinely cheaper Region would trip this and want
        a human to confirm it is real rather than another leak.
        """
        # Input/output only: a partial override omits cache rates and prices a cached call as None,
        # which is not what this guard is about.
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        for region, models in _REGIONAL_PRICING.items():
            for model in models:
                # The premium premise is about standard rates. A per-Region Global rate is a
                # separate axis that legitimately runs cheaper than us-east-1's Global rate.
                if model.startswith("global."):
                    continue
                override = calculate_bedrock_cost(f"{model}:0", usage, region=region)
                default = calculate_bedrock_cost(f"{model}:0", usage, region="us-east-1")
                assert override is not None
                # Some models are Region-exclusive (e.g. a model offered in GovCloud but not
                # us-east-1), so there is no baseline to be cheaper than — nothing to check.
                if default is None:
                    continue
                assert override.total_cost >= default.total_cost - 1e-9, f"{region}/{model} priced below us-east-1"

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

    def test_inference_profile_falls_back_to_base(self) -> None:
        """A regional inference-profile id without its own entry falls back to the base model's pricing."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_bedrock_cost("us.ai21.jamba-1-5-large-v1", usage)
        base = calculate_bedrock_cost("ai21.jamba-1-5-large-v1", usage)
        assert cost is not None
        assert base is not None
        assert cost.total_cost == pytest.approx(base.total_cost)

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

    def test_claude_prices_identically_to_the_native_bedrock_provider(self) -> None:
        """The shared Anthropic table exists so the Converse and native Bedrock providers never
        drift. Assert it holds for every Region that carries a Claude override, including the
        Regions only the shared table knows about.
        """
        # Both a plain call and a cached one: the two providers share one table, so they must agree
        # on the cost — including agreeing that a cached call against a cache-less override is
        # unpriced (both return None).
        for usage in (
            Usage(input_tokens=1000, output_tokens=500),
            Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=100),
        ):
            for region, models in ANTHROPIC_REGIONAL_PRICING.items():
                for model in models:
                    converse = calculate_bedrock_cost(f"{model}:0", usage, region=region)
                    native = calculate_bedrock_anthropic_cost(f"{model}:0", usage, region=region)
                    assert converse == native, f"{region}/{model} drifted"

    def test_partial_override_prices_tokens_but_not_uncovered_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A Region that prices input/output but not cache keeps its token premium; a cached call
        against the missing rate is unpriced (None) rather than billed for free."""
        regional = {
            "eu-west-1": {
                "amazon.nova-pro-v1": ModelPricing(
                    tiers=[
                        PricingTier(
                            input_cost_per_token=per_million_tokens(1.13),
                            output_cost_per_token=per_million_tokens(4.52),
                        )  # no cache rate
                    ],
                ),
            },
        }
        monkeypatch.setattr("lmux_aws_bedrock.cost._REGIONAL_PRICING", regional)
        plain = calculate_bedrock_cost(
            "amazon.nova-pro-v1:0", Usage(input_tokens=1_000_000, output_tokens=0), region="eu-west-1"
        )
        assert plain is not None
        assert plain.input_cost == pytest.approx(1.13)
        cached = calculate_bedrock_cost(
            "amazon.nova-pro-v1:0",
            Usage(input_tokens=1_000_000, output_tokens=0, cache_read_tokens=1000),
            region="eu-west-1",
        )
        assert cached is None

    def test_regional_global_profile_uses_the_regions_global_rate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A global. call is billed by its source Region, and AWS publishes a per-Region global rate.
        The regional global. override must win over the us-east-1 global key."""
        regional = {
            "ap-northeast-1": {
                "global.amazon.nova-2-lite-v1": ModelPricing(
                    tiers=[
                        PricingTier(
                            input_cost_per_token=per_million_tokens(0.36),
                            output_cost_per_token=per_million_tokens(3.01),
                        )
                    ],
                ),
            },
        }
        monkeypatch.setattr("lmux_aws_bedrock.cost._REGIONAL_PRICING", regional)
        usage = Usage(input_tokens=1_000_000, output_tokens=0)
        cost = calculate_bedrock_cost("global.amazon.nova-2-lite-v1:0", usage, region="ap-northeast-1")
        assert cost is not None
        assert cost.input_cost == pytest.approx(0.36)

    def test_geo_profile_uses_the_regions_standard_rate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bedrock bills a geo profile at the standard rate of the Region it is called from, so it
        resolves to that Region's base-model override."""
        regional = {
            "eu-west-1": {
                "meta.llama3-1-70b-instruct-v1": ModelPricing(
                    tiers=[
                        PricingTier(
                            input_cost_per_token=per_million_tokens(3.0),
                            output_cost_per_token=per_million_tokens(3.0),
                        )
                    ],
                ),
            },
        }
        usage = Usage(input_tokens=1000, output_tokens=500)
        monkeypatch.setattr("lmux_aws_bedrock.cost._REGIONAL_PRICING", regional)
        cost = calculate_bedrock_cost("eu.meta.llama3-1-70b-instruct-v1", usage, region="eu-west-1")
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 3.0 / 1_000_000)

    def test_global_profile_never_takes_a_regional_standard_rate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``global.`` is priced separately (~10% below standard), so a Region carrying only a
        standard override must not capture a global call — that would bill the discounted profile
        at the Region's standard rate. Absent regionally means it matches the default table."""
        regional = {
            "eu-west-1": {
                "meta.llama3-1-70b-instruct-v1": ModelPricing(
                    tiers=[
                        PricingTier(
                            input_cost_per_token=per_million_tokens(50.0),
                            output_cost_per_token=per_million_tokens(50.0),
                        )
                    ],
                ),
            },
        }
        usage = Usage(input_tokens=1000, output_tokens=500)
        monkeypatch.setattr("lmux_aws_bedrock.cost._REGIONAL_PRICING", regional)
        cost = calculate_bedrock_cost("global.meta.llama3-1-70b-instruct-v1", usage, region="eu-west-1")
        default = calculate_bedrock_cost("global.meta.llama3-1-70b-instruct-v1", usage)
        assert cost is not None
        assert default is not None
        assert cost.total_cost == pytest.approx(default.total_cost)

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


class TestRetiredModelIdsResolveViaConverse:
    """The Converse table merges the shared Anthropic subset, so it must price the same retired
    IDs. Asserting the IDs a caller presents, not the table's own keys, is what catches a key
    that has silently become unreachable.
    """

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "anthropic.claude-3-7-sonnet-20250219-v1:0",
            "anthropic.claude-3-opus-20240229-v1:0",
            "anthropic.claude-opus-4-20250514-v1:0",
        ],
    )
    def test_retired_dated_id_prices(self, model: str) -> None:
        cost = calculate_bedrock_cost(model, Usage(input_tokens=1_000_000, output_tokens=1_000_000))
        assert cost is not None, f"{model} is unreachable — the table key is not a prefix of it"
        assert cost.total_cost > 0

    def test_opus_5_profiles_price(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        standard = calculate_bedrock_cost("us.anthropic.claude-opus-5", usage)
        glob = calculate_bedrock_cost("global.anthropic.claude-opus-5", usage)
        assert standard is not None
        assert glob is not None
        assert standard.total_cost == pytest.approx(5.5 + 27.5)
        assert glob.total_cost == pytest.approx(5.0 + 25.0)
