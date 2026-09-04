"""Tests for OpenAI pricing and cost calculation."""

import pytest

from lmux.types import Cost, Usage
from lmux_openai.cost import (
    REGIONAL_UPLIFT,
    apply_cost_multiplier,
    calculate_openai_cost,
    regional_uplift_applies,
)


class TestCalculateOpenAICost:
    def test_known_model(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_openai_cost("gpt-4o", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 2.50 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 10.00 / 1_000_000)
        assert cost.total_cost == pytest.approx(cost.input_cost + cost.output_cost)

    def test_unknown_model_returns_none(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=50)
        cost = calculate_openai_cost("unknown-model-xyz", usage)
        assert cost is None

    def test_date_suffixed_model(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_openai_cost("gpt-4o-2024-11-20", usage)
        assert cost is not None
        base_cost = calculate_openai_cost("gpt-4o", usage)
        assert base_cost is not None
        assert cost.total_cost == pytest.approx(base_cost.total_cost)

    def test_with_cache_tokens(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=200)
        cost = calculate_openai_cost("gpt-4o", usage)
        assert cost is not None
        assert cost.cache_read_cost is not None
        assert cost.cache_read_cost == pytest.approx(200 * 1.25 / 1_000_000)

    def test_embedding_model(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=0)
        cost = calculate_openai_cost("text-embedding-3-small", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(100 * 0.02 / 1_000_000)
        assert cost.output_cost == 0.0

    def test_zero_tokens(self) -> None:
        usage = Usage(input_tokens=0, output_tokens=0)
        cost = calculate_openai_cost("gpt-4o", usage)
        assert cost is not None
        assert cost.total_cost == 0.0

    def test_prefix_matching_longest_first(self) -> None:
        """Verify that gpt-4o-mini-2024-07-18 matches gpt-4o-mini, not gpt-4o."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_openai_cost("gpt-4o-mini-2024-07-18", usage)
        assert cost is not None
        mini_cost = calculate_openai_cost("gpt-4o-mini", usage)
        assert mini_cost is not None
        assert cost.total_cost == pytest.approx(mini_cost.total_cost)

    def test_case_insensitive_lookup(self) -> None:
        """A capitalized model id resolves identically to its lowercase form."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        upper = calculate_openai_cost("GPT-4O", usage)
        lower = calculate_openai_cost("gpt-4o", usage)
        assert upper is not None
        assert lower is not None
        assert upper.total_cost == pytest.approx(lower.total_cost)

    def test_gpt_5_5_cyber_has_dedicated_pricing(self) -> None:
        """gpt-5.5-cyber must use its own (pricier) rate, not prefix-match gpt-5.5."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        cyber = calculate_openai_cost("gpt-5.5-cyber", usage)
        base = calculate_openai_cost("gpt-5.5", usage)
        assert cyber is not None
        assert base is not None
        assert cyber.input_cost == pytest.approx(1000 * 12.50 / 1_000_000)
        assert cyber.output_cost == pytest.approx(500 * 75.00 / 1_000_000)
        assert cyber.input_cost > base.input_cost

    def test_gpt_5_6_cyber_has_dedicated_pricing(self) -> None:
        """gpt-5.6-cyber must use its own (pricier) rate, not prefix-match gpt-5.6."""
        usage = Usage(input_tokens=1000, output_tokens=500, cache_creation_tokens=100)
        cyber = calculate_openai_cost("gpt-5.6-cyber", usage)
        base = calculate_openai_cost("gpt-5.6", usage)
        assert cyber is not None
        assert base is not None
        assert cyber.input_cost == pytest.approx((1000 - 100) * 12.50 / 1_000_000)
        assert cyber.output_cost == pytest.approx(500 * 75.00 / 1_000_000)
        assert cyber.cache_creation_cost == pytest.approx(100 * 15.625 / 1_000_000)
        assert cyber.input_cost > base.input_cost

    def test_gpt_6_astra_base_pricing(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=100, cache_creation_tokens=200)
        cost = calculate_openai_cost("gpt-6-astra", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx((1000 - 100 - 200) * 10.00 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 50.00 / 1_000_000)
        assert cost.cache_read_cost == pytest.approx(100 * 1.00 / 1_000_000)
        assert cost.cache_creation_cost == pytest.approx(200 * 12.50 / 1_000_000)

    def test_gpt_6_astra_long_context_tier(self) -> None:
        usage = Usage(
            input_tokens=300_000,
            output_tokens=1000,
            cache_read_tokens=100_000,
            cache_creation_tokens=50_000,
        )
        cost = calculate_openai_cost("gpt-6-astra", usage)
        input_cost = 150_000 * (20.00 / 1_000_000)
        output_cost = 1000 * (75.00 / 1_000_000)
        cache_read_cost = 100_000 * (2.00 / 1_000_000)
        cache_creation_cost = 50_000 * (25.00 / 1_000_000)
        assert cost == Cost(
            input_cost=input_cost,
            output_cost=output_cost,
            cache_read_cost=cache_read_cost,
            cache_creation_cost=cache_creation_cost,
            total_cost=input_cost + output_cost + cache_read_cost + cache_creation_cost,
        )

    @pytest.mark.parametrize(
        ("model", "input_rate", "output_rate", "cache_read_rate", "cache_write_rate"),
        [
            ("gpt-5.6-sol", 4.00, 20.00, 0.40, 5.00),
            ("gpt-5.6-terra", 2.00, 12.00, 0.20, 2.50),
            ("gpt-5.6-luna", 0.20, 1.20, 0.02, 0.25),
        ],
    )
    def test_gpt_5_6_family_base_pricing(
        self,
        model: str,
        input_rate: float,
        output_rate: float,
        cache_read_rate: float,
        cache_write_rate: float,
    ) -> None:
        # gpt-5.6 bills cache writes (1.25x input) on top of the read discount.
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=100, cache_creation_tokens=200)
        cost = calculate_openai_cost(model, usage)
        assert cost is not None
        billable = 1000 - 100 - 200
        assert cost.input_cost == pytest.approx(billable * input_rate / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * output_rate / 1_000_000)
        assert cost.cache_read_cost == pytest.approx(100 * cache_read_rate / 1_000_000)
        assert cost.cache_creation_cost == pytest.approx(200 * cache_write_rate / 1_000_000)

    @pytest.mark.parametrize(
        ("model", "input_rate", "output_rate", "cache_read_rate", "cache_write_rate"),
        [
            ("gpt-5.6-sol", 8.00, 30.00, 0.80, 10.00),
            ("gpt-5.6-terra", 4.00, 18.00, 0.40, 5.00),
            ("gpt-5.6-luna", 0.40, 1.80, 0.04, 0.50),
        ],
    )
    def test_gpt_5_6_long_context_tier(
        self,
        model: str,
        input_rate: float,
        output_rate: float,
        cache_read_rate: float,
        cache_write_rate: float,
    ) -> None:
        usage = Usage(
            input_tokens=300_000,
            output_tokens=1000,
            cache_read_tokens=100_000,
            cache_creation_tokens=50_000,
        )
        cost = calculate_openai_cost(model, usage)
        input_cost = 150_000 * (input_rate / 1_000_000)
        output_cost = 1000 * (output_rate / 1_000_000)
        cache_read_cost = 100_000 * (cache_read_rate / 1_000_000)
        cache_creation_cost = 50_000 * (cache_write_rate / 1_000_000)
        assert cost == Cost(
            input_cost=input_cost,
            output_cost=output_cost,
            cache_read_cost=cache_read_cost,
            cache_creation_cost=cache_creation_cost,
            total_cost=input_cost + output_cost + cache_read_cost + cache_creation_cost,
        )

    def test_gpt_5_6_bare_alias_matches_sol(self) -> None:
        """The bare gpt-5.6 alias mirrors gpt-5.6-sol, not the cheaper gpt-5 prefix."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        bare = calculate_openai_cost("gpt-5.6", usage)
        sol = calculate_openai_cost("gpt-5.6-sol", usage)
        assert bare is not None
        assert sol is not None
        assert bare.input_cost == pytest.approx(1000 * 4.00 / 1_000_000)
        assert bare.total_cost == pytest.approx(sol.total_cost)

    def test_gpt_4o_2024_05_13_snapshot(self) -> None:
        """The 2024-05-13 snapshot is priced above current gpt-4o (5/15) with no cached-input rate."""
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=100)
        cost = calculate_openai_cost("gpt-4o-2024-05-13", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx((1000 - 100) * 5.00 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 15.00 / 1_000_000)
        assert cost.cache_read_cost == pytest.approx(0.0)

    def test_search_preview_models_have_no_cached_input(self) -> None:
        """search-preview models keep base input/output but publish no cached-input price."""
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=100)
        full = calculate_openai_cost("gpt-4o-search-preview", usage)
        mini = calculate_openai_cost("gpt-4o-mini-search-preview", usage)
        assert full is not None
        assert mini is not None
        assert full.input_cost == pytest.approx((1000 - 100) * 2.50 / 1_000_000)
        assert full.output_cost == pytest.approx(500 * 10.00 / 1_000_000)
        assert full.cache_read_cost == pytest.approx(0.0)
        assert mini.input_cost == pytest.approx((1000 - 100) * 0.15 / 1_000_000)
        assert mini.cache_read_cost == pytest.approx(0.0)

    def test_gpt_5_4_cyber_unpriced_returns_none(self) -> None:
        """gpt-5.4-cyber is listed without a price; it must return None while its gpt-5.4 sibling stays priced."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        assert calculate_openai_cost("gpt-5.4-cyber", usage) is None
        sibling = calculate_openai_cost("gpt-5.4", usage)
        assert sibling is not None  # the sentinel is narrow: the family base must still price


class TestRegionalUpliftApplies:
    @pytest.mark.parametrize(
        "model",
        ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4-pro", "gpt-5.4-2025-11-01"],
    )
    def test_applies_to_gpt_5_4_family(self, model: str) -> None:
        assert regional_uplift_applies(model) is True

    @pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"])
    def test_applies_to_gpt_5_6_family(self, model: str) -> None:
        assert regional_uplift_applies(model) is True

    @pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6-astra-2026-09-03"])
    def test_applies_to_gpt_6_family(self, model: str) -> None:
        assert regional_uplift_applies(model) is True

    @pytest.mark.parametrize(
        "model",
        ["gpt-4o", "gpt-5", "gpt-5.3-codex", "o3", "text-embedding-3-small", "unknown-model", "gpt-5.4-cyber"],
    )
    def test_does_not_apply_to_other_models(self, model: str) -> None:
        # gpt-5.4-cyber starts with "gpt-5.4" but is unpriced, so the uplift must not apply.
        assert regional_uplift_applies(model) is False

    @pytest.mark.parametrize("model", ["gpt-5.5-cyber", "gpt-5.6-cyber"])
    def test_does_not_apply_to_priced_cyber_variants(self, model: str) -> None:
        """Cyber models carry a family prefix but are absent from OpenAI's data-residency list."""
        assert regional_uplift_applies(model) is False
        # The sibling that does share the prefix must still take the uplift.
        assert regional_uplift_applies(model.removesuffix("-cyber")) is True

    @pytest.mark.parametrize("model", ["gpt-5.6-cyber-2026-08-01", "gpt-5.5-cyber-2026-04-23"])
    def test_does_not_apply_to_dated_cyber_snapshots(self, model: str) -> None:
        """Pricing resolves cyber snapshots by prefix, so the uplift check must too."""
        assert regional_uplift_applies(model) is False


class TestApplyCostMultiplier:
    def test_applies_multiplier_to_all_fields(self) -> None:
        cost = Cost(
            input_cost=1.0,
            output_cost=2.0,
            total_cost=3.0,
            cache_read_cost=0.5,
            cache_creation_cost=0.25,
        )
        result = apply_cost_multiplier(cost, REGIONAL_UPLIFT)
        assert result.input_cost == pytest.approx(1.1)
        assert result.output_cost == pytest.approx(2.2)
        assert result.total_cost == pytest.approx(3.3)
        assert result.cache_read_cost == pytest.approx(0.55)
        assert result.cache_creation_cost == pytest.approx(0.275)

    def test_preserves_none_cache_costs(self) -> None:
        cost = Cost(input_cost=1.0, output_cost=2.0, total_cost=3.0)
        result = apply_cost_multiplier(cost, 2.0)
        assert result.cache_read_cost is None
        assert result.cache_creation_cost is None
