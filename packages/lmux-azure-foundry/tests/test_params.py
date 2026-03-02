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
        assert params.deployment_type is None

    def test_invalid_deployment_type(self) -> None:
        with pytest.raises(ValidationError):
            AzureFoundryParams(deployment_type="invalid")  # pyright: ignore[reportArgumentType]

    def test_valid_deployment_types(self) -> None:
        assert AzureFoundryParams(deployment_type="global").deployment_type == "global"
        assert AzureFoundryParams(deployment_type="data_zone").deployment_type == "data_zone"
        assert AzureFoundryParams(deployment_type="regional").deployment_type == "regional"
