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

    @pytest.mark.parametrize(
        ("model", "input_rate", "output_rate", "cache_read_rate", "cache_write_rate"),
        [
            ("gpt-5.6-sol", 5.00, 30.00, 0.50, 6.25),
            ("gpt-5.6-terra", 2.50, 15.00, 0.25, 3.125),
            ("gpt-5.6-luna", 1.00, 6.00, 0.10, 1.25),
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

    def test_gpt_5_6_long_context_tier(self) -> None:
        # Above 272k input tokens gpt-5.6-sol switches to the long-context tier (10.00 / 45.00).
        usage = Usage(input_tokens=300_000, output_tokens=1000)
        cost = calculate_openai_cost("gpt-5.6-sol", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(300_000 * 10.00 / 1_000_000)
        assert cost.output_cost == pytest.approx(1000 * 45.00 / 1_000_000)


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

    @pytest.mark.parametrize(
        "model",
        ["gpt-4o", "gpt-5", "gpt-5.3-codex", "o3", "text-embedding-3-small", "unknown-model"],
    )
    def test_does_not_apply_to_other_models(self, model: str) -> None:
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
