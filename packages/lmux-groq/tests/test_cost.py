"""Tests for Groq pricing and cost calculation."""

from datetime import date

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

    def test_qwen3_6_model(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_groq_cost("qwen/qwen3.6-27b", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 0.60 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 3.00 / 1_000_000)

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
        cost = calculate_groq_cost("meta-llama/llama-4-scout-17b-16e-instruct", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 0.11 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 0.34 / 1_000_000)

    def test_llama4_maverick_pricing(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_groq_cost("meta-llama/llama-4-maverick-17b-128e-instruct", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 0.20 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 0.60 / 1_000_000)

    @pytest.mark.parametrize(
        ("model", "rate"),
        [
            ("meta-llama/llama-prompt-guard-2-22m", 0.03),
            ("meta-llama/llama-prompt-guard-2-86m", 0.04),
        ],
    )
    def test_prompt_guard_pricing(self, model: str, rate: float) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_groq_cost(model, usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * rate / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * rate / 1_000_000)

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
        """meta-llama/llama-4-maverick-xxx should match maverick, not scout."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost_maverick = calculate_groq_cost("meta-llama/llama-4-maverick-17b-128e-instruct-custom", usage)
        cost_scout = calculate_groq_cost("meta-llama/llama-4-scout-17b-16e-instruct-custom", usage)
        assert cost_maverick is not None
        assert cost_scout is not None
        # These should resolve to different prefixes with different pricing
        assert cost_maverick.input_cost != cost_scout.input_cost

    def test_case_insensitive_lookup(self) -> None:
        """A capitalized model id resolves identically to its lowercase form."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        upper = calculate_groq_cost("LLAMA-3.3-70B-VERSATILE", usage)
        lower = calculate_groq_cost("llama-3.3-70b-versatile", usage)
        assert upper is not None
        assert lower is not None
        assert upper.total_cost == pytest.approx(lower.total_cost)

    def test_qwen_pricing(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_groq_cost("qwen/qwen3-32b", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 0.29 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 0.59 / 1_000_000)

    def test_with_cache_tokens_gpt_oss(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=200)
        cost = calculate_groq_cost("openai/gpt-oss-20b", usage, as_of=date(2026, 11, 30))
        assert cost is not None
        # 200 cached tokens at 50% discount ($0.0375/M), 800 regular input at $0.075/M
        assert cost.cache_read_cost is not None
        assert cost.cache_read_cost == pytest.approx(200 * 0.0375 / 1_000_000)
        assert cost.input_cost == pytest.approx(800 * 0.075 / 1_000_000)

    @pytest.mark.parametrize(
        ("model", "as_of", "input_rate", "cache_read_rate", "output_rate"),
        [
            ("openai/gpt-oss-20b", date(2026, 11, 30), 0.075, 0.0375, 0.30),
            ("openai/gpt-oss-20b", date(2026, 12, 1), 0.17, 0.09, 0.36),
            ("openai/gpt-oss-20b", None, 0.17, 0.09, 0.36),
            ("openai/gpt-oss-120b", date(2026, 11, 30), 0.15, 0.075, 0.60),
            ("openai/gpt-oss-120b", date(2026, 12, 1), 0.33, 0.17, 0.71),
            ("openai/gpt-oss-120b", None, 0.33, 0.17, 0.71),
        ],
    )
    def test_gpt_oss_dated_pricing(
        self, model: str, as_of: date | None, input_rate: float, cache_read_rate: float, output_rate: float
    ) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=200)
        cost = calculate_groq_cost(model, usage, as_of=as_of)
        assert cost is not None
        assert cost.input_cost == pytest.approx(800 * input_rate / 1_000_000)
        assert cost.cache_read_cost == pytest.approx(200 * cache_read_rate / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * output_rate / 1_000_000)

    def test_with_cache_tokens_safeguard(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=200)
        cost = calculate_groq_cost("openai/gpt-oss-safeguard-20b", usage)
        assert cost is not None
        assert cost.cache_read_cost == pytest.approx(200 * 0.0375 / 1_000_000)
        assert cost.input_cost == pytest.approx(800 * 0.075 / 1_000_000)

    def test_with_cache_tokens_kimi(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=300)
        cost = calculate_groq_cost("moonshotai/kimi-k2-instruct-0905", usage)
        assert cost is not None
        assert cost.cache_read_cost is not None
        assert cost.cache_read_cost == pytest.approx(300 * 0.50 / 1_000_000)
        assert cost.input_cost == pytest.approx(700 * 1.00 / 1_000_000)

    def test_no_cache_pricing_llama(self) -> None:
        """Llama models don't have cache pricing — cache_read_cost is zero."""
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=200)
        cost = calculate_groq_cost("llama-3.3-70b-versatile", usage)
        assert cost is not None
        # Without cache_read_cost_per_token configured, cached tokens reduce billable input
        # but the cache read cost itself is zero
        assert cost.cache_read_cost == 0.0
        # Only 800 non-cached tokens billed at full input rate
        assert cost.input_cost == pytest.approx(800 * 0.59 / 1_000_000)
