"""Tests for Anthropic provider params."""

import pytest
from pydantic import ValidationError

from lmux_anthropic.params import AnthropicParams


class TestAnthropicParams:
    def test_defaults(self) -> None:
        params = AnthropicParams()
        assert params.thinking is None
        assert params.metadata is None
        assert params.top_k is None
        assert params.service_tier is None
        assert params.inference_geo is None
        assert params.speed is None

    def test_invalid_service_tier_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AnthropicParams(service_tier="invalid")  # pyright: ignore[reportArgumentType]
