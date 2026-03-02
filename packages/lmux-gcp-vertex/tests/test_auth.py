"""Tests for GCP Vertex AI auth providers."""

from unittest.mock import MagicMock, patch

from lmux_gcp_vertex.auth import GCPVertexADCAuthProvider, GCPVertexServiceAccountAuthProvider


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
