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

    def test_invalid_service_tier(self) -> None:
        with pytest.raises(ValidationError):
            GroqParams(service_tier="invalid")  # pyright: ignore[reportArgumentType]
