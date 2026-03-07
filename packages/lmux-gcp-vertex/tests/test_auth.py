"""Tests for GCP Vertex AI auth providers."""

from unittest.mock import MagicMock, patch

import pytest

from lmux.exceptions import AuthenticationError
from lmux_gcp_vertex.auth import (
    GCPVertexADCAuthProvider,
    GCPVertexAPIKeyAuthProvider,
    GCPVertexServiceAccountAuthProvider,
)


class TestGCPVertexADCAuthProvider:
    def test_get_credentials(self) -> None:
        mock_creds = MagicMock()
        with patch("google.auth.default", return_value=(mock_creds, "my-project")) as mock_default:
            provider = GCPVertexADCAuthProvider()
            result = provider.get_credentials()

        assert result is mock_creds
        mock_default.assert_called_once()

    async def test_aget_credentials(self) -> None:
        mock_creds = MagicMock()
        with patch("google.auth.default", return_value=(mock_creds, "my-project")) as mock_default:
            provider = GCPVertexADCAuthProvider()
            result = await provider.aget_credentials()

        assert result is mock_creds
        mock_default.assert_called_once()


class TestGCPVertexServiceAccountAuthProvider:
    def test_get_credentials(self) -> None:
        mock_creds = MagicMock()
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=mock_creds,
        ) as mock_from_file:
            provider = GCPVertexServiceAccountAuthProvider(service_account_file="/path/to/key.json")
            result = provider.get_credentials()

        assert result is mock_creds
        mock_from_file.assert_called_once_with(
            "/path/to/key.json", scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    async def test_aget_credentials(self) -> None:
        mock_creds = MagicMock()
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=mock_creds,
        ) as mock_from_file:
            provider = GCPVertexServiceAccountAuthProvider(service_account_file="/path/to/key.json")
            result = await provider.aget_credentials()

        assert result is mock_creds
        mock_from_file.assert_called_once_with(
            "/path/to/key.json", scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    def test_custom_scopes(self) -> None:
        mock_creds = MagicMock()
        custom_scopes = ["https://www.googleapis.com/auth/bigquery"]
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=mock_creds,
        ) as mock_from_file:
            provider = GCPVertexServiceAccountAuthProvider(
                service_account_file="/path/to/key.json", scopes=custom_scopes
            )
            provider.get_credentials()

        mock_from_file.assert_called_once_with("/path/to/key.json", scopes=custom_scopes)


class TestGCPVertexAPIKeyAuthProvider:
    def test_get_credentials_from_explicit_key(self) -> None:
        provider = GCPVertexAPIKeyAuthProvider(api_key="test-key")
        assert provider.get_credentials() == "test-key"

    async def test_aget_credentials_from_explicit_key(self) -> None:
        provider = GCPVertexAPIKeyAuthProvider(api_key="test-key")
        assert await provider.aget_credentials() == "test-key"

    def test_get_credentials_from_env_var(self) -> None:
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "env-key"}):
            provider = GCPVertexAPIKeyAuthProvider()
            assert provider.get_credentials() == "env-key"

    def test_get_credentials_custom_env_var(self) -> None:
        with patch.dict("os.environ", {"MY_KEY": "custom-key"}):
            provider = GCPVertexAPIKeyAuthProvider(env_var="MY_KEY")
            assert provider.get_credentials() == "custom-key"

    def test_get_credentials_missing_env_var_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            provider = GCPVertexAPIKeyAuthProvider()
            with pytest.raises(AuthenticationError, match="GOOGLE_API_KEY environment variable is not set"):
                provider.get_credentials()

    def test_explicit_key_takes_precedence_over_env(self) -> None:
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "env-key"}):
            provider = GCPVertexAPIKeyAuthProvider(api_key="explicit-key")
            assert provider.get_credentials() == "explicit-key"
