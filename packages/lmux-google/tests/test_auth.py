"""Tests for Google auth providers."""

from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from lmux.exceptions import AuthenticationError
from lmux_google.auth import (
    GoogleADCAuthProvider,
    GoogleAPIKeyAuthProvider,
    GoogleServiceAccountAuthProvider,
)


class TestGoogleADCAuthProvider:
    @pytest.fixture
    def mock_creds(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def mock_google_auth_default(self, mock_creds: MagicMock, mocker: MockerFixture) -> MagicMock:
        return mocker.patch("google.auth.default", return_value=(mock_creds, "my-project"))

    def test_get_credentials(self, mock_creds: MagicMock, mock_google_auth_default: MagicMock) -> None:
        provider = GoogleADCAuthProvider()
        result = provider.get_credentials()

        assert result is mock_creds
        mock_google_auth_default.assert_called_once_with(scopes=["https://www.googleapis.com/auth/cloud-platform"])

    async def test_aget_credentials(self, mock_creds: MagicMock, mock_google_auth_default: MagicMock) -> None:
        provider = GoogleADCAuthProvider()
        result = await provider.aget_credentials()

        assert result is mock_creds
        mock_google_auth_default.assert_called_once_with(scopes=["https://www.googleapis.com/auth/cloud-platform"])

    def test_custom_scopes(self, mock_google_auth_default: MagicMock) -> None:
        custom_scopes = ["https://www.googleapis.com/auth/bigquery"]
        provider = GoogleADCAuthProvider(scopes=custom_scopes)
        _ = provider.get_credentials()

        mock_google_auth_default.assert_called_once_with(scopes=custom_scopes)


class TestGoogleServiceAccountAuthProvider:
    @pytest.fixture
    def mock_creds(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def mock_from_service_account_file(self, mock_creds: MagicMock, mocker: MockerFixture) -> MagicMock:
        return mocker.patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            return_value=mock_creds,
        )

    def test_get_credentials(self, mock_creds: MagicMock, mock_from_service_account_file: MagicMock) -> None:
        provider = GoogleServiceAccountAuthProvider(service_account_file="/path/to/key.json")
        result = provider.get_credentials()

        assert result is mock_creds
        mock_from_service_account_file.assert_called_once_with(
            "/path/to/key.json", scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    async def test_aget_credentials(self, mock_creds: MagicMock, mock_from_service_account_file: MagicMock) -> None:
        provider = GoogleServiceAccountAuthProvider(service_account_file="/path/to/key.json")
        result = await provider.aget_credentials()

        assert result is mock_creds
        mock_from_service_account_file.assert_called_once_with(
            "/path/to/key.json", scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    def test_custom_scopes(self, mock_from_service_account_file: MagicMock) -> None:
        custom_scopes = ["https://www.googleapis.com/auth/bigquery"]
        provider = GoogleServiceAccountAuthProvider(service_account_file="/path/to/key.json", scopes=custom_scopes)
        _ = provider.get_credentials()

        mock_from_service_account_file.assert_called_once_with("/path/to/key.json", scopes=custom_scopes)


class TestGoogleAPIKeyAuthProvider:
    def test_get_credentials_from_explicit_key(self) -> None:
        provider = GoogleAPIKeyAuthProvider(api_key="test-key")
        assert provider.get_credentials() == "test-key"

    async def test_aget_credentials_from_explicit_key(self) -> None:
        provider = GoogleAPIKeyAuthProvider(api_key="test-key")
        assert await provider.aget_credentials() == "test-key"

    def test_get_credentials_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        provider = GoogleAPIKeyAuthProvider()
        assert provider.get_credentials() == "env-key"

    def test_get_credentials_custom_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY", "custom-key")
        provider = GoogleAPIKeyAuthProvider(env_var="MY_KEY")
        assert provider.get_credentials() == "custom-key"

    def test_get_credentials_missing_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        provider = GoogleAPIKeyAuthProvider()
        with pytest.raises(AuthenticationError, match="GOOGLE_API_KEY environment variable is not set"):
            _ = provider.get_credentials()

    def test_explicit_key_takes_precedence_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        provider = GoogleAPIKeyAuthProvider(api_key="explicit-key")
        assert provider.get_credentials() == "explicit-key"
