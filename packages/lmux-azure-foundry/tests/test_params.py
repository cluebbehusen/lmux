"""Tests for Azure AI Foundry provider parameters."""

import pytest
from pydantic import ValidationError

from lmux_azure_foundry.params import AzureFoundryParams


class TestAzureFoundryParams:
    def test_invalid_reasoning_effort(self) -> None:
        with pytest.raises(ValidationError):
            AzureFoundryParams(reasoning_effort="invalid")  # pyright: ignore[reportArgumentType]

    def test_defaults_all_none(self) -> None:
        params = AzureFoundryParams()
        assert params.reasoning_effort is None
        assert params.seed is None
        assert params.user is None
