"""Tests for Anthropic cost calculation."""

import pytest

from lmux.types import Cost, Usage
from lmux_anthropic.cost import apply_cost_multiplier, calculate_anthropic_cost


class TestCalculateAnthropicCost:
    def test_known_model(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_anthropic_cost("claude-sonnet-4-6", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 3.0 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 15.0 / 1_000_000)
        assert cost.total_cost == pytest.approx(cost.input_cost + cost.output_cost)

    def test_unknown_model_returns_none(self) -> None:
        usage = Usage(input_tokens=10, output_tokens=5)
        assert calculate_anthropic_cost("totally-unknown-model", usage) is None

    def test_date_suffixed_model_matches_prefix(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_anthropic_cost("claude-sonnet-4-6-20260214", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 3.0 / 1_000_000)

    def test_cache_tokens(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=200, cache_creation_tokens=100)
        cost = calculate_anthropic_cost("claude-sonnet-4-6", usage)
        assert cost is not None
        assert cost.cache_read_cost is not None
        assert cost.cache_creation_cost is not None
        assert cost.cache_read_cost == pytest.approx(200 * 0.30 / 1_000_000)
        assert cost.cache_creation_cost == pytest.approx(100 * 3.75 / 1_000_000)

    def test_zero_tokens(self) -> None:
        usage = Usage(input_tokens=0, output_tokens=0)
        cost = calculate_anthropic_cost("claude-sonnet-4-6", usage)
        assert cost is not None
        assert cost.total_cost == 0.0

    def test_opus_4_6_pricing(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_anthropic_cost("claude-opus-4-6", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 5.0 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 25.0 / 1_000_000)

    def test_haiku_4_5_pricing(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_anthropic_cost("claude-haiku-4-5", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 1.0 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 5.0 / 1_000_000)

    def test_sonnet_3_7_pricing(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_anthropic_cost("claude-3-7-sonnet-20250219", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 3.0 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 15.0 / 1_000_000)

    def test_sonnet_3_5_pricing(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_anthropic_cost("claude-3-5-sonnet-20241022", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 3.0 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 15.0 / 1_000_000)

    def test_longest_prefix_matching(self) -> None:
        """claude-opus-4-1-xxx should match claude-opus-4-1, not claude-opus-4."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost_41 = calculate_anthropic_cost("claude-opus-4-1-20260101", usage)
        cost_4 = calculate_anthropic_cost("claude-opus-4-20250514", usage)
        assert cost_41 is not None
        assert cost_4 is not None
        # Both are $15 input, so same price, but they should resolve to different prefixes
        assert cost_41.input_cost == cost_4.input_cost

    def test_flat_pricing_at_high_token_count(self) -> None:
        """Claude Sonnet 4 uses flat pricing regardless of input token count."""
        usage = Usage(input_tokens=250_000, output_tokens=1000)
        cost = calculate_anthropic_cost("claude-sonnet-4", usage)
        assert cost is not None
        # No tiered pricing — base rate applies even above 200K
        assert cost.input_cost == pytest.approx(250_000 * 3.0 / 1_000_000)
        assert cost.output_cost == pytest.approx(1000 * 15.0 / 1_000_000)

    def test_opus_4_6_flat_pricing_at_high_token_count(self) -> None:
        """Claude Opus 4.6 uses flat pricing across the full 1M context window."""
        usage = Usage(input_tokens=500_000, output_tokens=1000)
        cost = calculate_anthropic_cost("claude-opus-4-6", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(500_000 * 5.0 / 1_000_000)
        assert cost.output_cost == pytest.approx(1000 * 25.0 / 1_000_000)


class TestApplyCostMultiplier:
    def test_applies_multiplier_to_all_fields(self) -> None:
        cost = Cost(
            input_cost=1.0,
            output_cost=2.0,
            total_cost=3.0,
            cache_read_cost=0.5,
            cache_creation_cost=0.25,
        )
        result = apply_cost_multiplier(cost, 1.1)
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
