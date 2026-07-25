"""Tests for Anthropic cost calculation."""

from datetime import date

import pytest

from lmux.types import Cost, Usage
from lmux_anthropic.cost import apply_cost_multiplier, calculate_anthropic_cost, has_vertex_regional_premium


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

    def test_opus_5_pricing(self) -> None:
        usage = Usage(
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=200,
            cache_creation_tokens_by_ttl={"5m": 100, "1h": 50},
        )
        cost = calculate_anthropic_cost("claude-opus-5", usage)
        assert cost is not None
        input_cost = (1000 - 200 - 150) * 5.00 / 1_000_000
        output_cost = 500 * 25.00 / 1_000_000
        cache_read_cost = 200 * 0.50 / 1_000_000
        cache_creation_cost = (100 * 6.25 + 50 * 10.00) / 1_000_000
        # Compared as a dict so the whole object is checked while tolerating float drift;
        # pytest.approx cannot be nested inside a Cost constructor.
        assert cost.model_dump() == pytest.approx(
            {
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": input_cost + output_cost + cache_read_cost + cache_creation_cost,
                "cache_read_cost": cache_read_cost,
                "cache_creation_cost": cache_creation_cost,
                "currency": "USD",
            }
        )

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

    def test_case_insensitive_lookup(self) -> None:
        """A capitalized model id resolves identically to its lowercase form."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        upper = calculate_anthropic_cost("Claude-Sonnet-4-6", usage)
        lower = calculate_anthropic_cost("claude-sonnet-4-6", usage)
        assert upper is not None
        assert lower is not None
        assert upper.total_cost == pytest.approx(lower.total_cost)

    def test_long_context_pricing_at_high_token_count(self) -> None:
        """Claude Sonnet 4 uses long-context pricing above 200K input tokens."""
        usage = Usage(input_tokens=250_000, output_tokens=1000)
        cost = calculate_anthropic_cost("claude-sonnet-4", usage)
        assert cost is not None
        # >200K triggers long-context tier: $6/$22.50
        assert cost.input_cost == pytest.approx(250_000 * 6.0 / 1_000_000)
        assert cost.output_cost == pytest.approx(1000 * 22.5 / 1_000_000)

    def test_opus_4_6_flat_pricing_at_high_token_count(self) -> None:
        """Claude Opus 4.6 uses flat pricing across the full 1M context window."""
        usage = Usage(input_tokens=500_000, output_tokens=1000)
        cost = calculate_anthropic_cost("claude-opus-4-6", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(500_000 * 5.0 / 1_000_000)
        assert cost.output_cost == pytest.approx(1000 * 25.0 / 1_000_000)

    def test_sonnet_5_introductory_pricing_before_sep_2026(self) -> None:
        """Before 2026-09-01, Claude Sonnet 5 bills at the introductory rate."""
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=200, cache_creation_tokens=100)
        cost = calculate_anthropic_cost("claude-sonnet-5", usage, as_of=date(2026, 7, 1))
        assert cost is not None
        assert cost.input_cost == pytest.approx((1000 - 200 - 100) * 2.0 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 10.0 / 1_000_000)
        assert cost.cache_read_cost == pytest.approx(200 * 0.20 / 1_000_000)
        assert cost.cache_creation_cost == pytest.approx(100 * 2.50 / 1_000_000)

    def test_sonnet_5_standard_pricing_from_sep_2026(self) -> None:
        """On/after 2026-09-01, Claude Sonnet 5 bills at the standard rate."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_anthropic_cost("claude-sonnet-5", usage, as_of=date(2026, 9, 1))
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 3.0 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 15.0 / 1_000_000)

    def test_sonnet_5_defaults_to_standard_pricing(self) -> None:
        """With no ``as_of``, Claude Sonnet 5 bills at the latest (standard) schedule."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_anthropic_cost("claude-sonnet-5", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 3.0 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 15.0 / 1_000_000)

    def test_sonnet_5_per_ttl_cache_write_rates_differ_by_schedule(self) -> None:
        """The 1h cache-write rate is $4.00/M introductory and $6.00/M standard."""
        usage = Usage(input_tokens=1000, output_tokens=0, cache_creation_tokens_by_ttl={"1h": 1000})
        intro = calculate_anthropic_cost("claude-sonnet-5", usage, as_of=date(2026, 7, 1))
        standard = calculate_anthropic_cost("claude-sonnet-5", usage, as_of=date(2026, 9, 1))
        assert intro is not None
        assert standard is not None
        assert intro.cache_creation_cost == pytest.approx(1000 * 4.0 / 1_000_000)
        assert standard.cache_creation_cost == pytest.approx(1000 * 6.0 / 1_000_000)


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


class TestHasVertexRegionalPremium:
    def test_premium_for_new_models(self) -> None:
        assert has_vertex_regional_premium("claude-sonnet-4-6") is True
        assert has_vertex_regional_premium("claude-haiku-4-5") is True

    def test_premium_for_vertex_model_id_with_version_suffix(self) -> None:
        assert has_vertex_regional_premium("claude-sonnet-4-5@20250929") is True

    def test_uniform_pricing_for_older_models(self) -> None:
        assert has_vertex_regional_premium("claude-3-5-haiku") is False
        assert has_vertex_regional_premium("claude-sonnet-4") is False

    def test_longest_prefix_disambiguates_opus_generations(self) -> None:
        """claude-opus-4-8 (premium) must not be shadowed by the claude-opus-4 (uniform) prefix."""
        assert has_vertex_regional_premium("claude-opus-4-8") is True
        assert has_vertex_regional_premium("claude-opus-4@20250514") is False

    def test_unknown_future_models_default_to_premium(self) -> None:
        assert has_vertex_regional_premium("claude-sonnet-6") is True

    def test_opus_5_is_premium(self) -> None:
        """claude-opus-5 must not be shadowed by the claude-opus-4 (uniform) prefix."""
        assert has_vertex_regional_premium("claude-opus-5") is True

    def test_case_insensitive(self) -> None:
        """Premium classification folds case, like the pricing lookup."""
        assert has_vertex_regional_premium("Claude-Opus-4-8") is True
        assert has_vertex_regional_premium("CLAUDE-3-5-HAIKU") is False
