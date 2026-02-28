"""Tests for Groq pricing and cost calculation."""

import pytest

from lmux.types import Usage
from lmux_groq.cost import calculate_groq_cost


class TestCalculateGroqCost:
    def test_known_model(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_groq_cost("llama-3.3-70b-versatile", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 0.59 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 0.79 / 1_000_000)
        assert cost.total_cost == pytest.approx(cost.input_cost + cost.output_cost)

    def test_unknown_model_returns_none(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=50)
        cost = calculate_groq_cost("unknown-model-xyz", usage)
        assert cost is None

    def test_prefix_matching(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_groq_cost("llama-3.1-8b-instant-extra", usage)
        assert cost is not None
        base_cost = calculate_groq_cost("llama-3.1-8b-instant", usage)
        assert base_cost is not None
        assert cost.total_cost == pytest.approx(base_cost.total_cost)

    def test_llama4_scout_pricing(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_groq_cost("llama-4-scout-17b-16e-instruct", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 0.11 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 0.34 / 1_000_000)

    def test_llama4_maverick_pricing(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_groq_cost("llama-4-maverick-17b-128e-instruct", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 0.20 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 0.60 / 1_000_000)

    def test_llama31_8b_pricing(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_groq_cost("llama-3.1-8b-instant", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 0.05 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 0.08 / 1_000_000)

    def test_zero_tokens(self) -> None:
        usage = Usage(input_tokens=0, output_tokens=0)
        cost = calculate_groq_cost("llama-3.3-70b-versatile", usage)
        assert cost is not None
        assert cost.total_cost == 0.0

    def test_longest_prefix_matching(self) -> None:
        """llama-4-maverick-xxx should match llama-4-maverick, not llama-4."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost_maverick = calculate_groq_cost("llama-4-maverick-custom", usage)
        cost_scout = calculate_groq_cost("llama-4-scout-custom", usage)
        assert cost_maverick is not None
        assert cost_scout is not None
        # These should resolve to different prefixes with different pricing
        assert cost_maverick.input_cost != cost_scout.input_cost

    def test_qwen_pricing(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_groq_cost("qwen-qwq-32b", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 0.29 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 0.59 / 1_000_000)
