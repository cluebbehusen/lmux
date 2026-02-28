"""Tests for OpenAI provider parameters."""

import pytest
from pydantic import ValidationError

from lmux_openai.params import OpenAIParams


class TestOpenAIParams:
    def test_defaults(self) -> None:
        params = OpenAIParams()
        assert params.service_tier is None
        assert params.reasoning_effort is None
        assert params.seed is None
        assert params.user is None

    def test_service_tier(self) -> None:
        params = OpenAIParams(service_tier="flex")
        assert params.service_tier == "flex"

    def test_reasoning_effort(self) -> None:
        params = OpenAIParams(reasoning_effort="high")
        assert params.reasoning_effort == "high"

    def test_seed(self) -> None:
        params = OpenAIParams(seed=42)
        assert params.seed == 42

    def test_user(self) -> None:
        params = OpenAIParams(user="user-123")
        assert params.user == "user-123"

    def test_all_fields(self) -> None:
        params = OpenAIParams(service_tier="auto", reasoning_effort="low", seed=1, user="u")
        assert params.service_tier == "auto"
        assert params.reasoning_effort == "low"
        assert params.seed == 1
        assert params.user == "u"

    def test_invalid_service_tier(self) -> None:
        with pytest.raises(ValidationError):
            OpenAIParams(service_tier="invalid")  # pyright: ignore[reportArgumentType]

    def test_invalid_reasoning_effort(self) -> None:
        with pytest.raises(ValidationError):
            OpenAIParams(reasoning_effort="invalid")  # pyright: ignore[reportArgumentType]
