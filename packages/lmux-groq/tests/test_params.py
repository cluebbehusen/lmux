"""Tests for Groq provider parameters."""

import pytest
from pydantic import ValidationError

from lmux_groq.params import GroqParams


class TestGroqParams:
    def test_defaults(self) -> None:
        params = GroqParams()
        assert params.service_tier is None
        assert params.seed is None
        assert params.user is None

    def test_groq_specific_reasoning_effort_values(self) -> None:
        assert GroqParams(reasoning_effort="none").reasoning_effort == "none"
        assert GroqParams(reasoning_effort="default").reasoning_effort == "default"

    def test_invalid_service_tier(self) -> None:
        with pytest.raises(ValidationError):
            _ = GroqParams(service_tier="invalid")  # pyright: ignore[reportArgumentType]
