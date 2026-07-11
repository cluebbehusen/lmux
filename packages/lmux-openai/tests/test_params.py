"""Tests for OpenAI provider parameters."""

import pytest
from pydantic import ValidationError

from lmux_openai.params import OpenAIParams


class TestOpenAIParams:
    def test_invalid_service_tier(self) -> None:
        with pytest.raises(ValidationError):
            _ = OpenAIParams(service_tier="invalid")  # ty: ignore[invalid-argument-type]

    def test_invalid_reasoning_effort(self) -> None:
        with pytest.raises(ValidationError):
            _ = OpenAIParams(reasoning_effort="invalid")  # ty: ignore[invalid-argument-type]
