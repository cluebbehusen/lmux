"""Tests for lmux cost calculation utilities."""

import pytest

from lmux.cost import ModelPricing, calculate_cost, per_million_tokens
from lmux.types import Usage


class TestPerMillionTokens:
    def test_basic(self) -> None:
        assert per_million_tokens(2.50) == pytest.approx(2.50 / 1_000_000)

    def test_zero(self) -> None:
        assert per_million_tokens(0.0) == 0.0


class TestModelPricing:
    def test_basic(self) -> None:
        p = ModelPricing(input_cost_per_token=0.000003, output_cost_per_token=0.000015)
        assert p.input_cost_per_token == 0.000003
        assert p.output_cost_per_token == 0.000015
        assert p.cache_read_cost_per_token is None
        assert p.cache_creation_cost_per_token is None

    def test_with_cache_pricing(self) -> None:
        p = ModelPricing(
            input_cost_per_token=0.000003,
            output_cost_per_token=0.000015,
            cache_read_cost_per_token=0.0000003,
            cache_creation_cost_per_token=0.00000375,
        )
        assert p.cache_read_cost_per_token == 0.0000003
        assert p.cache_creation_cost_per_token == 0.00000375


class TestCalculateCost:
    def test_basic(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        pricing = ModelPricing(input_cost_per_token=0.000003, output_cost_per_token=0.000015)
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
            input_cost_per_token=0.000003,
            output_cost_per_token=0.000015,
            cache_read_cost_per_token=0.0000003,
            cache_creation_cost_per_token=0.00000375,
        )
        cost = calculate_cost(usage, pricing)
        assert cost.input_cost == pytest.approx(700 * 0.000003)
        assert cost.cache_read_cost == pytest.approx(200 * 0.0000003)
        assert cost.cache_creation_cost == pytest.approx(100 * 0.00000375)
        expected_total = (700 * 0.000003) + (500 * 0.000015) + (200 * 0.0000003) + (100 * 0.00000375)
        assert cost.total_cost == pytest.approx(expected_total)

    def test_zero_tokens(self) -> None:
        usage = Usage(input_tokens=0, output_tokens=0)
        pricing = ModelPricing(input_cost_per_token=0.000003, output_cost_per_token=0.000015)
        cost = calculate_cost(usage, pricing)
        assert cost.input_cost == 0.0
        assert cost.output_cost == 0.0
        assert cost.total_cost == 0.0

    def test_zero_cache_tokens_are_none(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=50, cache_read_tokens=0)
        pricing = ModelPricing(
            input_cost_per_token=0.000003,
            output_cost_per_token=0.000015,
            cache_read_cost_per_token=0.0000003,
        )
        cost = calculate_cost(usage, pricing)
        assert cost.cache_read_cost is None
        assert cost.cache_creation_cost is None

    def test_with_cache_read_only(self) -> None:
        # input_tokens=1000 total, 200 cached — billable input = 800
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=200)
        pricing = ModelPricing(
            input_cost_per_token=0.000003,
            output_cost_per_token=0.000015,
            cache_read_cost_per_token=0.0000003,
        )
        cost = calculate_cost(usage, pricing)
        assert cost.input_cost == pytest.approx(800 * 0.000003)
        assert cost.cache_read_cost == pytest.approx(200 * 0.0000003)
