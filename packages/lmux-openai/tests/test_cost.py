"""Tests for OpenAI pricing and cost calculation."""

import pytest

from lmux.types import Usage
from lmux_openai.cost import calculate_openai_cost


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
