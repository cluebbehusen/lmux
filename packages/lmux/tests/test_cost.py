"""Tests for lmux cost calculation utilities."""

from datetime import date

import pytest

from lmux.cost import (
    ModelPricing,
    PricingSchedule,
    PricingTier,
    build_pricing_index,
    calculate_cost,
    per_million_tokens,
    resolve_pricing,
)
from lmux.types import Usage


@pytest.fixture
def dated_pricing() -> ModelPricing:
    """A model with introductory base pricing and a dated standard-rate override."""
    return ModelPricing(
        tiers=[
            PricingTier(
                input_cost_per_token=per_million_tokens(2.00),
                output_cost_per_token=per_million_tokens(10.00),
            )
        ],
        schedules=[
            PricingSchedule(
                valid_from=date(2026, 9, 1),
                tiers=[
                    PricingTier(
                        input_cost_per_token=per_million_tokens(3.00),
                        output_cost_per_token=per_million_tokens(15.00),
                    )
                ],
            )
        ],
    )


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


class TestPricingSchedule:
    def test_basic(self) -> None:
        schedule = PricingSchedule(
            valid_from=date(2026, 9, 1),
            tiers=[PricingTier(input_cost_per_token=0.000003, output_cost_per_token=0.000015)],
        )
        assert schedule.valid_from == date(2026, 9, 1)
        assert schedule.tiers[0].input_cost_per_token == 0.000003

    def test_invalid_tiers_rejected(self) -> None:
        with pytest.raises(ValueError, match="tiers must not be empty"):
            _ = PricingSchedule(valid_from=date(2026, 9, 1), tiers=[])


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
            _ = ModelPricing(tiers=[])

    def test_missing_base_tier_rejected(self) -> None:
        with pytest.raises(ValueError, match="first tier must have min_input_tokens == 0"):
            _ = ModelPricing(
                tiers=[
                    PricingTier(input_cost_per_token=0.000006, output_cost_per_token=0.00003, min_input_tokens=200_000)
                ]
            )

    def test_unordered_tiers_rejected(self) -> None:
        with pytest.raises(ValueError, match="tiers must be ordered by strictly ascending min_input_tokens"):
            _ = ModelPricing(
                tiers=[
                    PricingTier(input_cost_per_token=0.000003, output_cost_per_token=0.000015),
                    PricingTier(input_cost_per_token=0.000006, output_cost_per_token=0.00003, min_input_tokens=200_000),
                    PricingTier(input_cost_per_token=0.000009, output_cost_per_token=0.00004, min_input_tokens=100_000),
                ],
            )

    def test_duplicate_thresholds_rejected(self) -> None:
        with pytest.raises(ValueError, match="tiers must be ordered by strictly ascending min_input_tokens"):
            _ = ModelPricing(
                tiers=[
                    PricingTier(input_cost_per_token=0.000003, output_cost_per_token=0.000015),
                    PricingTier(input_cost_per_token=0.000006, output_cost_per_token=0.00003, min_input_tokens=200_000),
                    PricingTier(input_cost_per_token=0.000009, output_cost_per_token=0.00004, min_input_tokens=200_000),
                ],
            )

    def test_with_single_schedule(self, dated_pricing: ModelPricing) -> None:
        assert dated_pricing.schedules is not None
        assert len(dated_pricing.schedules) == 1
        assert dated_pricing.schedules[0].valid_from == date(2026, 9, 1)
        # The base ``tiers`` are unaffected by the dated override.
        assert dated_pricing.tiers[0].input_cost_per_token == per_million_tokens(2.00)

    def test_with_multiple_schedules(self) -> None:
        p = ModelPricing(
            tiers=[PricingTier(input_cost_per_token=0.000002, output_cost_per_token=0.00001)],
            schedules=[
                PricingSchedule(
                    valid_from=date(2026, 9, 1),
                    tiers=[PricingTier(input_cost_per_token=0.000003, output_cost_per_token=0.000015)],
                ),
                PricingSchedule(
                    valid_from=date(2027, 1, 1),
                    tiers=[PricingTier(input_cost_per_token=0.000004, output_cost_per_token=0.00002)],
                ),
            ],
        )
        assert p.schedules is not None
        assert [s.valid_from for s in p.schedules] == [date(2026, 9, 1), date(2027, 1, 1)]

    def test_empty_schedules_rejected(self) -> None:
        with pytest.raises(ValueError, match="schedules must not be empty when provided"):
            _ = ModelPricing(
                tiers=[PricingTier(input_cost_per_token=0.000002, output_cost_per_token=0.00001)],
                schedules=[],
            )

    def test_unordered_schedules_rejected(self) -> None:
        with pytest.raises(ValueError, match="schedules must be ordered by strictly ascending valid_from"):
            _ = ModelPricing(
                tiers=[PricingTier(input_cost_per_token=0.000002, output_cost_per_token=0.00001)],
                schedules=[
                    PricingSchedule(
                        valid_from=date(2026, 9, 1),
                        tiers=[PricingTier(input_cost_per_token=0.000003, output_cost_per_token=0.000015)],
                    ),
                    PricingSchedule(
                        valid_from=date(2026, 9, 1),
                        tiers=[PricingTier(input_cost_per_token=0.000004, output_cost_per_token=0.00002)],
                    ),
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

    def test_base_tier_used_at_exact_threshold(self) -> None:
        """When input equals a tier's min_input_tokens exactly, the base tier still applies (need to exceed)."""
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
        usage = Usage(input_tokens=200_000, output_tokens=1000)
        cost = calculate_cost(usage, pricing)
        assert cost.input_cost == pytest.approx(200_000 * per_million_tokens(3.00))
        assert cost.output_cost == pytest.approx(1000 * per_million_tokens(15.00))

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

    def test_breakdown_bills_per_ttl_rate(self) -> None:
        pricing = ModelPricing(
            tiers=[
                PricingTier(
                    input_cost_per_token=5.0,
                    output_cost_per_token=25.0,
                    cache_creation_cost_per_token=6.25,
                    cache_creation_cost_per_token_by_ttl={"1h": 10.0},
                )
            ]
        )
        usage = Usage(
            input_tokens=3100,
            output_tokens=0,
            cache_creation_tokens=2000,
            cache_creation_tokens_by_ttl={"1h": 1500, "5m": 500},
        )
        cost = calculate_cost(usage, pricing)
        # 1500 @ 1h rate, 500 @ default rate ("5m" has no per-TTL entry)
        assert cost.cache_creation_cost == pytest.approx(1500 * 10.0 + 500 * 6.25)
        # billable input excludes all cache-write tokens
        assert cost.input_cost == pytest.approx(1100 * 5.0)

    def test_breakdown_without_per_ttl_pricing_uses_default_rate(self) -> None:
        pricing = ModelPricing(
            tiers=[
                PricingTier(
                    input_cost_per_token=5.0,
                    output_cost_per_token=25.0,
                    cache_creation_cost_per_token=6.25,
                )
            ]
        )
        usage = Usage(
            input_tokens=2000,
            output_tokens=0,
            cache_creation_tokens=2000,
            cache_creation_tokens_by_ttl={"1h": 2000},
        )
        cost = calculate_cost(usage, pricing)
        assert cost.cache_creation_cost == pytest.approx(2000 * 6.25)

    def test_aggregate_remainder_bills_at_default_rate(self) -> None:
        pricing = ModelPricing(
            tiers=[
                PricingTier(
                    input_cost_per_token=5.0,
                    output_cost_per_token=25.0,
                    cache_creation_cost_per_token=6.25,
                    cache_creation_cost_per_token_by_ttl={"1h": 10.0},
                )
            ]
        )
        # 800 tokens in the breakdown, 1000 aggregate -> 200 at default rate
        usage = Usage(
            input_tokens=1000,
            output_tokens=0,
            cache_creation_tokens=1000,
            cache_creation_tokens_by_ttl={"1h": 800},
        )
        cost = calculate_cost(usage, pricing)
        assert cost.cache_creation_cost == pytest.approx(800 * 10.0 + 200 * 6.25)

    def test_breakdown_without_aggregate_is_authoritative(self) -> None:
        pricing = ModelPricing(
            tiers=[
                PricingTier(
                    input_cost_per_token=5.0,
                    output_cost_per_token=25.0,
                    cache_creation_cost_per_token=6.25,
                    cache_creation_cost_per_token_by_ttl={"1h": 10.0},
                )
            ]
        )
        usage = Usage(input_tokens=500, output_tokens=0, cache_creation_tokens_by_ttl={"1h": 500})
        cost = calculate_cost(usage, pricing)
        assert cost.cache_creation_cost == pytest.approx(500 * 10.0)
        assert cost.input_cost == pytest.approx(0.0)

    def test_no_cache_creation_returns_none_cost(self) -> None:
        pricing = ModelPricing(
            tiers=[
                PricingTier(
                    input_cost_per_token=5.0,
                    output_cost_per_token=25.0,
                    cache_creation_cost_per_token_by_ttl={"1h": 10.0},
                )
            ]
        )
        usage = Usage(input_tokens=100, output_tokens=10)
        cost = calculate_cost(usage, pricing)
        assert cost.cache_creation_cost is None

    def test_as_of_none_uses_latest_schedule(self, dated_pricing: ModelPricing) -> None:
        """With no ``as_of``, the latest schedule (standard rate) applies."""
        cost = calculate_cost(Usage(input_tokens=1_000_000, output_tokens=0), dated_pricing)
        assert cost.input_cost == pytest.approx(per_million_tokens(3.00) * 1_000_000)

    def test_as_of_before_override_uses_base(self, dated_pricing: ModelPricing) -> None:
        """An ``as_of`` before every schedule's ``valid_from`` falls back to the base tiers."""
        cost = calculate_cost(Usage(input_tokens=1_000_000, output_tokens=0), dated_pricing, as_of=date(2026, 7, 1))
        assert cost.input_cost == pytest.approx(per_million_tokens(2.00) * 1_000_000)

    def test_as_of_on_override_date_uses_override(self, dated_pricing: ModelPricing) -> None:
        """An ``as_of`` on the override's ``valid_from`` uses the override (boundary is inclusive)."""
        cost = calculate_cost(Usage(input_tokens=1_000_000, output_tokens=0), dated_pricing, as_of=date(2026, 9, 1))
        assert cost.input_cost == pytest.approx(per_million_tokens(3.00) * 1_000_000)

    def test_as_of_ignored_without_schedules(self) -> None:
        """``as_of`` has no effect on a model without dated schedules."""
        pricing = ModelPricing(
            tiers=[PricingTier(input_cost_per_token=per_million_tokens(2.00), output_cost_per_token=0.0)]
        )
        cost = calculate_cost(Usage(input_tokens=1_000_000, output_tokens=0), pricing, as_of=date(2026, 9, 1))
        assert cost.input_cost == pytest.approx(per_million_tokens(2.00) * 1_000_000)

    def test_resolves_among_multiple_schedules(self) -> None:
        """With 3+ schedules, the latest whose ``valid_from`` is on or before ``as_of`` wins."""
        # 1M input tokens means input_cost == the per-million rate, so each assertion reads as $/M.
        usage = Usage(input_tokens=1_000_000, output_tokens=0)
        pricing = ModelPricing(
            tiers=[PricingTier(input_cost_per_token=per_million_tokens(2.00), output_cost_per_token=0.0)],
            schedules=[
                PricingSchedule(
                    valid_from=date(2026, 9, 1),
                    tiers=[PricingTier(input_cost_per_token=per_million_tokens(3.00), output_cost_per_token=0.0)],
                ),
                PricingSchedule(
                    valid_from=date(2027, 1, 1),
                    tiers=[PricingTier(input_cost_per_token=per_million_tokens(4.00), output_cost_per_token=0.0)],
                ),
                PricingSchedule(
                    valid_from=date(2027, 6, 1),
                    tiers=[PricingTier(input_cost_per_token=per_million_tokens(5.00), output_cost_per_token=0.0)],
                ),
            ],
        )
        # Before any schedule -> base tiers ($2).
        assert calculate_cost(usage, pricing, as_of=date(2026, 7, 1)).input_cost == pytest.approx(2.00)
        # Strictly between the 2nd and 3rd valid_from -> 2nd schedule ($4): kills both an
        # early-break mutant (would give $3) and an always-use-last mutant (would give $5).
        assert calculate_cost(usage, pricing, as_of=date(2027, 3, 1)).input_cost == pytest.approx(4.00)
        # Exactly on a non-first schedule's valid_from -> that schedule (inclusive boundary).
        assert calculate_cost(usage, pricing, as_of=date(2027, 1, 1)).input_cost == pytest.approx(4.00)
        # On/after the last schedule -> last schedule ($5).
        assert calculate_cost(usage, pricing, as_of=date(2027, 6, 1)).input_cost == pytest.approx(5.00)
        # No as_of -> latest schedule ($5).
        assert calculate_cost(usage, pricing).input_cost == pytest.approx(5.00)


class TestBuildPricingIndexAndResolvePricing:
    @staticmethod
    def _pricing(rate: float) -> ModelPricing:
        return ModelPricing(tiers=[PricingTier(input_cost_per_token=rate, output_cost_per_token=0.0)])

    def test_index_sorted_longest_first(self) -> None:
        p = self._pricing(1.0)
        index = build_pricing_index({"a": p, "aaa": p, "aa": p})
        assert [key for key, _ in index] == ["aaa", "aa", "a"]

    def test_index_lowercases_keys(self) -> None:
        p = self._pricing(1.0)
        index = build_pricing_index({"Cohere-Command-A": p})
        assert index == [("cohere-command-a", p)]

    def test_exact_and_prefix_resolution(self) -> None:
        base = self._pricing(1.0)
        mini = self._pricing(2.0)
        index = build_pricing_index({"gpt-4o": base, "gpt-4o-mini": mini})
        assert resolve_pricing("gpt-4o", index) is base  # exact
        assert resolve_pricing("gpt-4o-2024-11-20", index) is base  # prefix fallback
        assert resolve_pricing("gpt-4o-mini-2024-07-18", index) is mini  # longest prefix wins

    def test_case_insensitive_exact_and_prefix(self) -> None:
        p = self._pricing(1.0)
        index = build_pricing_index({"Cohere-command-a": p})
        assert resolve_pricing("cohere-command-a", index) is p
        assert resolve_pricing("COHERE-COMMAND-A", index) is p
        assert resolve_pricing("Cohere-Command-A-plus-05-2026", index) is p  # mixed-case prefix

    def test_no_match_returns_none(self) -> None:
        index = build_pricing_index({"gpt-4o": self._pricing(1.0)})
        assert resolve_pricing("claude-opus-4-8", index) is None
