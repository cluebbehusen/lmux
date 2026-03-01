"""Tests for AWS Bedrock pricing and cost calculation."""

import pytest

from lmux.types import Usage
from lmux_aws_bedrock.cost import calculate_bedrock_cost


class TestCalculateBedrockCost:
    def test_known_model(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_bedrock_cost("meta.llama3-1-70b-instruct-v1", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(1000 * 0.72 / 1_000_000)
        assert cost.output_cost == pytest.approx(500 * 0.72 / 1_000_000)
        assert cost.total_cost == pytest.approx(cost.input_cost + cost.output_cost)

    def test_unknown_model_returns_none(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=50)
        cost = calculate_bedrock_cost("unknown-model-xyz", usage)
        assert cost is None

    def test_prefix_matching(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_bedrock_cost("meta.llama3-1-70b-instruct-v1:0", usage)
        assert cost is not None
        base_cost = calculate_bedrock_cost("meta.llama3-1-70b-instruct-v1", usage)
        assert base_cost is not None
        assert cost.total_cost == pytest.approx(base_cost.total_cost)

    def test_with_cache_tokens(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=500, cache_read_tokens=200)
        cost = calculate_bedrock_cost("amazon.nova-pro-v1", usage)
        assert cost is not None
        assert cost.cache_read_cost is not None
        assert cost.cache_read_cost == pytest.approx(200 * 0.20 / 1_000_000)

    def test_embedding_model(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=0)
        cost = calculate_bedrock_cost("amazon.titan-embed-text-v2", usage)
        assert cost is not None
        assert cost.input_cost == pytest.approx(100 * 0.01 / 1_000_000)
        assert cost.output_cost == 0.0

    def test_zero_tokens(self) -> None:
        usage = Usage(input_tokens=0, output_tokens=0)
        cost = calculate_bedrock_cost("amazon.nova-micro-v1", usage)
        assert cost is not None
        assert cost.total_cost == 0.0

    def test_prefix_matching_longest_first(self) -> None:
        """Verify that meta.llama3-1-8b-instruct-v1:0 matches llama3-1-8b, not llama3-1-70b."""
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = calculate_bedrock_cost("meta.llama3-1-8b-instruct-v1:0", usage)
        assert cost is not None
        exact_cost = calculate_bedrock_cost("meta.llama3-1-8b-instruct-v1", usage)
        assert exact_cost is not None
        assert cost.total_cost == pytest.approx(exact_cost.total_cost)
