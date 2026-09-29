"""Tests for Azure AI Foundry provider parameters."""

import pytest
from pydantic import ValidationError

from lmux_azure_foundry.params import AzureFoundryParams


class TestAzureFoundryParams:
    def test_invalid_reasoning_effort(self) -> None:
        with pytest.raises(ValidationError):
            _ = AzureFoundryParams(reasoning_effort="invalid")  # ty: ignore[invalid-argument-type]

    def test_defaults(self) -> None:
        params = AzureFoundryParams()
        assert params.reasoning_effort is None
        assert params.seed is None
        assert params.user is None
        assert params.deployment_type is None
        assert params.data_zone == "us"

    def test_invalid_data_zone(self) -> None:
        with pytest.raises(ValidationError):
            _ = AzureFoundryParams(data_zone="invalid")  # ty: ignore[invalid-argument-type]

    def test_explicit_eu_data_zone(self) -> None:
        assert AzureFoundryParams(deployment_type="data_zone", data_zone="eu").data_zone == "eu"

    def test_invalid_deployment_type(self) -> None:
        with pytest.raises(ValidationError):
            _ = AzureFoundryParams(deployment_type="invalid")  # ty: ignore[invalid-argument-type]

    def test_valid_deployment_types(self) -> None:
        assert AzureFoundryParams(deployment_type="global").deployment_type == "global"
        assert AzureFoundryParams(deployment_type="data_zone").deployment_type == "data_zone"
        assert AzureFoundryParams(deployment_type="regional").deployment_type == "regional"
