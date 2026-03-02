"""Tests for Google Vertex AI cost calculation."""

import pytest

from lmux.types import Usage
from lmux_google.cost import calculate_google_cost


class TestCalculateGoogleCost:
    def test_known_model(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_google_cost("gemini-2.0-flash", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 0.10 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 0.40 / 1_000_000)
        assert cost.total_cost == cost.input_cost + cost.output_cost

    def test_unknown_model_returns_none(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=50)
        assert calculate_google_cost("totally-unknown-model", usage) is None

    def test_prefix_match_with_date_suffix(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_google_cost("gemini-2.0-flash-001", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 0.10 / 1_000_000)

    def test_tiered_pricing_below_threshold(self) -> None:
        usage = Usage(input_tokens=100_000, output_tokens=500)
        cost = calculate_google_cost("gemini-2.5-pro", usage)
        assert cost is not None
        # Below 200k → lower tier
        assert cost.input_cost == pytest.approx(100_000 * 1.25 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 10.00 / 1_000_000)

    def test_tiered_pricing_above_threshold(self) -> None:
        usage = Usage(input_tokens=300_000, output_tokens=1000)
        cost = calculate_google_cost("gemini-2.5-pro", usage)
        assert cost is not None
        # Above 200k → higher tier for all tokens
        assert cost.input_cost == pytest.approx(300_000 * 2.50 / 1_000_000)
        assert cost.output_cost == pytest.approx(1000 * 15.00 / 1_000_000)

    def test_cache_tokens(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=200)
        cost = calculate_google_cost("gemini-2.0-flash", usage)
        assert cost is not None
        assert cost.cache_read_cost is not None
        assert cost.cache_read_cost == pytest.approx(200 * 0.025 / 1_000_000)

    def test_embedding_model(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=0)
        cost = calculate_google_cost("text-embedding-005", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 0.025 / 1_000_000)
        assert cost.output_cost == 0.0

    def test_zero_tokens(self) -> None:
        usage = Usage(input_tokens=0, output_tokens=0)
        cost = calculate_google_cost("gemini-2.0-flash", usage)
        assert cost is not None
        assert cost.total_cost == 0.0

    def test_longest_prefix_match(self) -> None:
        """gemini-2.0-flash-lite should match before gemini-2.0-flash."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost_lite = calculate_google_cost("gemini-2.0-flash-lite-001", usage)
        cost_flash = calculate_google_cost("gemini-2.0-flash-001", usage)
        assert cost_lite is not None
        assert cost_flash is not None
        # Flash-lite is cheaper than Flash
        assert cost_lite.input_cost < cost_flash.input_cost
