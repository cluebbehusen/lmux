"""Tests for the shared Anthropic-on-Bedrock pricing."""

from datetime import date

from lmux.types import Usage
from lmux_bedrock_shared.pricing import calculate_bedrock_anthropic_cost


class TestCalculateBedrockAnthropicCost:
    def test_known_model_exact_match(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = calculate_bedrock_anthropic_cost("anthropic.claude-opus-4-8", usage)
        assert cost is not None
        assert cost.input_cost == 5.5
        assert cost.output_cost == 27.5
        assert cost.total_cost == 33.0

    def test_unknown_model_returns_none(self) -> None:
        # Non-anthropic id: no exact key, no prefix match, no profile prefix to strip.
        assert calculate_bedrock_anthropic_cost("openai.gpt-9", Usage(input_tokens=1, output_tokens=1)) is None

    def test_version_suffix_prefix_match(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=1000)
        suffixed = calculate_bedrock_anthropic_cost("anthropic.claude-opus-4-6-v1:0", usage)
        base = calculate_bedrock_anthropic_cost("anthropic.claude-opus-4-6-v1", usage)
        assert suffixed is not None
        assert suffixed == base

    def test_inference_profile_strip_fallback(self) -> None:
        # No dedicated us.anthropic.claude-3-5-haiku entry exists, so the profile
        # prefix is stripped and the bare model prices (exercises the recursion).
        usage = Usage(input_tokens=1000, output_tokens=1000)
        cost = calculate_bedrock_anthropic_cost("us.anthropic.claude-3-5-haiku-20241022-v1:0", usage)
        bare = calculate_bedrock_anthropic_cost("anthropic.claude-3-5-haiku-20241022-v1", usage)
        assert cost is not None
        assert cost == bare

    def test_regional_profile_pricing_beats_bare_where_encoded(self) -> None:
        # Region-profile prices are baked into the table as distinct keys, so no
        # multiplier is needed: the global endpoint prices below the us regional one.
        usage = Usage(input_tokens=1_000_000, output_tokens=0)
        glob = calculate_bedrock_anthropic_cost("global.anthropic.claude-opus-4-6-v1", usage)
        regional = calculate_bedrock_anthropic_cost("us.anthropic.claude-opus-4-6-v1", usage)
        assert glob is not None
        assert regional is not None
        assert glob.input_cost < regional.input_cost

    def test_dated_schedule(self) -> None:
        usage = Usage(input_tokens=1_000_000, output_tokens=0)
        intro = calculate_bedrock_anthropic_cost("anthropic.claude-sonnet-5", usage, as_of=date(2026, 7, 1))
        standard = calculate_bedrock_anthropic_cost("anthropic.claude-sonnet-5", usage, as_of=date(2026, 9, 1))
        latest = calculate_bedrock_anthropic_cost("anthropic.claude-sonnet-5", usage)
        assert intro is not None
        assert standard is not None
        assert latest is not None
        assert intro.input_cost < standard.input_cost
        assert latest == standard
