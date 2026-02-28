"""Tests for lmux cost calculation utilities."""

import pytest

from lmux.cost import ModelPricing, PricingTier, calculate_cost, per_million_tokens
from lmux.types import Usage


class TestPerMillionTokens:
    def test_basic(self) -> None:
        assert per_million_tokens(2.50) == pytest.approx(2.50 / 1_000_000)

    def test_zero(self) -> None:
        assert per_million_tokens(0.0) == 0.0


class TestPricingTier:
    def test_defaults(self) -> None:
        tier = PricingTier(input_cost_per_token=0.000003, output_cost_per_token=0.000015)
        assert tier.input_cost_per_token == 0.000003
        assert tier.output_cost_per_token == 0.000015
        assert tier.cache_read_cost_per_token is None
        assert tier.cache_creation_cost_per_token is None
        assert tier.min_input_tokens == 0

    def test_with_threshold(self) -> None:
        tier = PricingTier(input_cost_per_token=0.000006, output_cost_per_token=0.00003, min_input_tokens=200_000)
        assert tier.min_input_tokens == 200_000

    def test_with_cache_pricing(self) -> None:
        tier = PricingTier(
            input_cost_per_token=0.000003,
            output_cost_per_token=0.000015,
            cache_read_cost_per_token=0.0000003,
            cache_creation_cost_per_token=0.00000375,
        )
        assert tier.cache_read_cost_per_token == 0.0000003
        assert tier.cache_creation_cost_per_token == 0.00000375


class TestModelPricing:
    def test_basic(self) -> None:
        p = ModelPricing(
            tiers=[PricingTier(input_cost_per_token=0.000003, output_cost_per_token=0.000015)],
        )
        assert p.tiers[0].input_cost_per_token == 0.000003
        assert p.tiers[0].output_cost_per_token == 0.000015
        assert p.tiers[0].cache_read_cost_per_token is None
        assert p.tiers[0].cache_creation_cost_per_token is None

    def test_with_cache_pricing(self) -> None:
        p = ModelPricing(
            tiers=[
                PricingTier(
                    input_cost_per_token=0.000003,
                    output_cost_per_token=0.000015,
                    cache_read_cost_per_token=0.0000003,
                    cache_creation_cost_per_token=0.00000375,
                )
            ],
        )
        assert p.tiers[0].cache_read_cost_per_token == 0.0000003
        assert p.tiers[0].cache_creation_cost_per_token == 0.00000375

    def test_with_multiple_tiers(self) -> None:
        p = ModelPricing(
            tiers=[
                PricingTier(input_cost_per_token=0.000003, output_cost_per_token=0.000015),
                PricingTier(input_cost_per_token=0.000006, output_cost_per_token=0.00003, min_input_tokens=200_000),
            ],
        )
        assert len(p.tiers) == 2
        assert p.tiers[1].min_input_tokens == 200_000

    def test_empty_tiers_rejected(self) -> None:
        with pytest.raises(ValueError, match="tiers must not be empty"):
            ModelPricing(tiers=[])

    def test_missing_base_tier_rejected(self) -> None:
        with pytest.raises(ValueError, match="first tier must have min_input_tokens == 0"):
            ModelPricing(
                tiers=[
                    PricingTier(input_cost_per_token=0.000006, output_cost_per_token=0.00003, min_input_tokens=200_000)
                ]
            )

    def test_unordered_tiers_rejected(self) -> None:
        with pytest.raises(ValueError, match="tiers must be ordered by strictly ascending min_input_tokens"):
            ModelPricing(
                tiers=[
                    PricingTier(input_cost_per_token=0.000003, output_cost_per_token=0.000015),
                    PricingTier(input_cost_per_token=0.000006, output_cost_per_token=0.00003, min_input_tokens=200_000),
                    PricingTier(input_cost_per_token=0.000009, output_cost_per_token=0.00004, min_input_tokens=100_000),
                ],
            )

    def test_duplicate_thresholds_rejected(self) -> None:
        with pytest.raises(ValueError, match="tiers must be ordered by strictly ascending min_input_tokens"):
            ModelPricing(
                tiers=[
                    PricingTier(input_cost_per_token=0.000003, output_cost_per_token=0.000015),
                    PricingTier(input_cost_per_token=0.000006, output_cost_per_token=0.00003, min_input_tokens=200_000),
                    PricingTier(input_cost_per_token=0.000009, output_cost_per_token=0.00004, min_input_tokens=200_000),
                ],
            )


class TestCalculateCost:
    def test_basic(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        pricing = ModelPricing(
            tiers=[PricingTier(input_cost_per_token=0.000003, output_cost_per_token=0.000015)],
        )
        cost = calculate_cost(usage, pricing)
        assert cost.input_cost == pytest.approx(0.003)
        assert cost.output_cost == pytest.approx(0.0075)
        assert cost.total_cost == pytest.approx(0.0105)
        assert cost.cache_read_cost is None
        assert cost.cache_creation_cost is None
        assert cost.currency == "USD"

    def test_with_cache_tokens(self) -> None:
        # input_tokens=1000 total, of which 200 are cache reads and 100 are cache writes.
        # Billable input = 1000 - 200 - 100 = 700 at the full input rate.
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=200, cache_creation_tokens=100)
        pricing = ModelPricing(
            tiers=[
                PricingTier(
                    input_cost_per_token=0.000003,
                    output_cost_per_token=0.000015,
                    cache_read_cost_per_token=0.0000003,
                    cache_creation_cost_per_token=0.00000375,
                )
            ],
        )
        cost = calculate_cost(usage, pricing)
        assert cost.input_cost == pytest.approx(700 * 0.000003)
        assert cost.cache_read_cost == pytest.approx(200 * 0.0000003)
        assert cost.cache_creation_cost == pytest.approx(100 * 0.00000375)
        expected_total = (700 * 0.000003) + (500 * 0.000015) + (200 * 0.0000003) + (100 * 0.00000375)
        assert cost.total_cost == pytest.approx(expected_total)

    def test_zero_tokens(self) -> None:
        usage = Usage(input_tokens=0, output_tokens=0)
        pricing = ModelPricing(
            tiers=[PricingTier(input_cost_per_token=0.000003, output_cost_per_token=0.000015)],
        )
        cost = calculate_cost(usage, pricing)
        assert cost.input_cost == 0.0
        assert cost.output_cost == 0.0
        assert cost.total_cost == 0.0

    def test_zero_cache_tokens_are_none(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=50, cache_read_tokens=0)
        pricing = ModelPricing(
            tiers=[
                PricingTier(
                    input_cost_per_token=0.000003,
                    output_cost_per_token=0.000015,
                    cache_read_cost_per_token=0.0000003,
                )
            ],
        )
        cost = calculate_cost(usage, pricing)
        assert cost.cache_read_cost is None
        assert cost.cache_creation_cost is None

    def test_tier_threshold_does_not_double_count_cache(self) -> None:
        """Cached tokens are a subset of input_tokens — they should not be added again for the threshold check."""
        pricing = ModelPricing(
            tiers=[
                PricingTier(
                    input_cost_per_token=per_million_tokens(3.00),
                    output_cost_per_token=per_million_tokens(15.00),
                    cache_read_cost_per_token=per_million_tokens(0.30),
                ),
                PricingTier(
                    input_cost_per_token=per_million_tokens(6.00),
                    output_cost_per_token=per_million_tokens(22.50),
                    cache_read_cost_per_token=per_million_tokens(0.30),
                    min_input_tokens=200_000,
                ),
            ],
        )
        # 150K input with 100K cache reads — total is still 150K (under threshold)
        usage = Usage(input_tokens=150_000, output_tokens=100, cache_read_tokens=100_000)
        cost = calculate_cost(usage, pricing)
        # Should use standard rate ($3/MTok), not long-context rate ($6/MTok)
        billable_input = 150_000 - 100_000  # 50K at standard rate
        assert cost.input_cost == pytest.approx(billable_input * per_million_tokens(3.00))

    def test_with_cache_read_only(self) -> None:
        # input_tokens=1000 total, 200 cached — billable input = 800
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=200)
        pricing = ModelPricing(
            tiers=[
                PricingTier(
                    input_cost_per_token=0.000003,
                    output_cost_per_token=0.000015,
                    cache_read_cost_per_token=0.0000003,
                )
            ],
        )
        cost = calculate_cost(usage, pricing)
        assert cost.input_cost == pytest.approx(800 * 0.000003)
        assert cost.cache_read_cost == pytest.approx(200 * 0.0000003)

    def test_higher_tier_used_above_threshold(self) -> None:
        """When input exceeds a tier's min_input_tokens, the higher tier rates apply."""
        pricing = ModelPricing(
            tiers=[
                PricingTier(
                    input_cost_per_token=per_million_tokens(3.00),
                    output_cost_per_token=per_million_tokens(15.00),
                ),
                PricingTier(
                    input_cost_per_token=per_million_tokens(6.00),
                    output_cost_per_token=per_million_tokens(30.00),
                    min_input_tokens=200_000,
                ),
            ],
        )
        usage = Usage(input_tokens=250_000, output_tokens=1000)
        cost = calculate_cost(usage, pricing)
        assert cost.input_cost == pytest.approx(250_000 * per_million_tokens(6.00))
        assert cost.output_cost == pytest.approx(1000 * per_million_tokens(30.00))

    def test_base_tier_used_below_threshold(self) -> None:
        """Below the higher tier threshold, the base tier rates apply."""
        pricing = ModelPricing(
            tiers=[
                PricingTier(
                    input_cost_per_token=per_million_tokens(3.00),
                    output_cost_per_token=per_million_tokens(15.00),
                ),
                PricingTier(
                    input_cost_per_token=per_million_tokens(6.00),
                    output_cost_per_token=per_million_tokens(30.00),
                    min_input_tokens=200_000,
                ),
            ],
        )
        usage = Usage(input_tokens=100_000, output_tokens=1000)
        cost = calculate_cost(usage, pricing)
        assert cost.input_cost == pytest.approx(100_000 * per_million_tokens(3.00))
        assert cost.output_cost == pytest.approx(1000 * per_million_tokens(15.00))

    def test_three_tiers(self) -> None:
        """Three tiers — the highest applicable tier wins."""
        pricing = ModelPricing(
            tiers=[
                PricingTier(
                    input_cost_per_token=per_million_tokens(1.00),
                    output_cost_per_token=per_million_tokens(5.00),
                ),
                PricingTier(
                    input_cost_per_token=per_million_tokens(2.00),
                    output_cost_per_token=per_million_tokens(10.00),
                    min_input_tokens=100_000,
                ),
                PricingTier(
                    input_cost_per_token=per_million_tokens(4.00),
                    output_cost_per_token=per_million_tokens(20.00),
                    min_input_tokens=500_000,
                ),
            ],
        )
        # Below first threshold — base tier
        cost_low = calculate_cost(Usage(input_tokens=50_000, output_tokens=100), pricing)
        assert cost_low.input_cost == pytest.approx(50_000 * per_million_tokens(1.00))

        # Between first and second threshold — middle tier
        cost_mid = calculate_cost(Usage(input_tokens=200_000, output_tokens=100), pricing)
        assert cost_mid.input_cost == pytest.approx(200_000 * per_million_tokens(2.00))

        # Above highest threshold — top tier
        cost_high = calculate_cost(Usage(input_tokens=600_000, output_tokens=100), pricing)
        assert cost_high.input_cost == pytest.approx(600_000 * per_million_tokens(4.00))
