"""Tests for Google auth providers."""

from pathlib import Path
from unittest.mock import MagicMock, sentinel

import pytest
from pytest_mock import MockerFixture

from lmux.exceptions import AuthenticationError
from lmux_google.auth import (
    GoogleADCAuthProvider,
    GoogleAPIKeyAuthProvider,
    GoogleServiceAccountAuthProvider,
)

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class TestGoogleADCAuthProvider:
    @pytest.fixture(autouse=True)
    def clear_adc_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GCLOUD_PROJECT", raising=False)

    @pytest.fixture
    def mock_creds(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def mock_google_auth_default(self, mock_creds: MagicMock, mocker: MockerFixture) -> MagicMock:
        return mocker.patch("google.auth.default", return_value=(mock_creds, "my-project"))

    @pytest.fixture
    def mock_load_credentials(self, mock_creds: MagicMock, mocker: MockerFixture) -> MagicMock:
        return mocker.patch("google.auth.load_credentials_from_file", return_value=(mock_creds, "file-project"))

    @pytest.fixture
    def mock_with_scopes(self, mock_creds: MagicMock, mocker: MockerFixture) -> MagicMock:
        return mocker.patch("google.auth.credentials.with_scopes_if_required", return_value=mock_creds)

    @pytest.fixture
    def mock_cloud_sdk_path(self, mocker: MockerFixture) -> MagicMock:
        return mocker.patch(
            "google.auth._cloud_sdk.get_application_default_credentials_path",
            return_value="/missing/gcloud/application_default_credentials.json",
        )

    @pytest.fixture
    def mock_auth_request(self, mocker: MockerFixture) -> MagicMock:
        return mocker.patch("lmux_google._lazy.HttpxAuthRequest", return_value=sentinel.request)

    def test_get_credentials_falls_back_to_default(
        self,
        mock_creds: MagicMock,
        mock_google_auth_default: MagicMock,
        mock_load_credentials: MagicMock,
        mock_cloud_sdk_path: MagicMock,
    ) -> None:
        provider = GoogleADCAuthProvider()
        result = provider.get_credentials()

        assert result is mock_creds
        mock_google_auth_default.assert_called_once_with(scopes=[CLOUD_PLATFORM_SCOPE])
        mock_load_credentials.assert_not_called()
        mock_cloud_sdk_path.assert_called_once_with()

    async def test_aget_credentials_falls_back_to_default(
        self,
        mock_creds: MagicMock,
        mock_google_auth_default: MagicMock,
        mock_cloud_sdk_path: MagicMock,
    ) -> None:
        provider = GoogleADCAuthProvider()
        result = await provider.aget_credentials()

        assert result is mock_creds
        mock_google_auth_default.assert_called_once_with(scopes=[CLOUD_PLATFORM_SCOPE])
        mock_cloud_sdk_path.assert_called_once_with()

    def test_explicit_adc_file_uses_httpx_request(  # noqa: PLR0913
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_creds: MagicMock,
        mock_google_auth_default: MagicMock,
        mock_load_credentials: MagicMock,
        mock_with_scopes: MagicMock,
        mock_cloud_sdk_path: MagicMock,
        mock_auth_request: MagicMock,
    ) -> None:
        credentials_file = tmp_path / "adc.json"
        credentials_file.touch()
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials_file))

        result = GoogleADCAuthProvider().get_credentials()

        assert result is mock_creds
        mock_load_credentials.assert_called_once_with(str(credentials_file), request=sentinel.request)
        mock_with_scopes.assert_called_once_with(mock_creds, [CLOUD_PLATFORM_SCOPE])
        mock_auth_request.assert_called_once_with()
        mock_google_auth_default.assert_not_called()
        mock_cloud_sdk_path.assert_called_once_with()

    def test_gcloud_adc_file_uses_httpx_request(  # noqa: PLR0913
        self,
        tmp_path: Path,
        mock_creds: MagicMock,
        mock_google_auth_default: MagicMock,
        mock_load_credentials: MagicMock,
        mock_with_scopes: MagicMock,
        mock_cloud_sdk_path: MagicMock,
        mock_auth_request: MagicMock,
    ) -> None:
        credentials_file = tmp_path / "application_default_credentials.json"
        credentials_file.touch()
        mock_cloud_sdk_path.return_value = str(credentials_file)
        mock_load_credentials.return_value = (mock_creds, None)

        result = GoogleADCAuthProvider().get_credentials()

        assert result is mock_creds
        mock_load_credentials.assert_called_once_with(str(credentials_file), request=sentinel.request)
        mock_with_scopes.assert_called_once_with(mock_creds, [CLOUD_PLATFORM_SCOPE])
        mock_google_auth_default.assert_not_called()
        mock_auth_request.assert_called_once_with()

    def test_external_account_does_not_resolve_unused_project(  # noqa: PLR0913
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_creds: MagicMock,
        mock_google_auth_default: MagicMock,
        mock_load_credentials: MagicMock,
        mock_with_scopes: MagicMock,
        mock_cloud_sdk_path: MagicMock,
        mock_auth_request: MagicMock,
    ) -> None:
        credentials_file = tmp_path / "external.json"
        credentials_file.touch()
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials_file))
        scoped_credentials = MagicMock()
        scoped_credentials.get_project_id.return_value = "workload-project"
        mock_load_credentials.return_value = (mock_creds, None)
        mock_with_scopes.return_value = scoped_credentials

        result = GoogleADCAuthProvider().get_credentials()

        assert result is scoped_credentials
        scoped_credentials.get_project_id.assert_not_called()
        mock_google_auth_default.assert_not_called()
        mock_cloud_sdk_path.assert_called_once_with()
        mock_auth_request.assert_called_once_with()

    def test_missing_explicit_file_preserves_default_fallback(  # noqa: PLR0913
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_creds: MagicMock,
        mock_google_auth_default: MagicMock,
        mock_load_credentials: MagicMock,
        mock_cloud_sdk_path: MagicMock,
    ) -> None:
        gcloud_file = tmp_path / "application_default_credentials.json"
        gcloud_file.touch()
        mock_cloud_sdk_path.return_value = str(gcloud_file)
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "missing.json"))

        result = GoogleADCAuthProvider().get_credentials()

        assert result is mock_creds
        mock_google_auth_default.assert_called_once_with(scopes=[CLOUD_PLATFORM_SCOPE])
        mock_load_credentials.assert_not_called()

    def test_custom_scopes_for_file_credentials(  # noqa: PLR0913
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_creds: MagicMock,
        mock_google_auth_default: MagicMock,
        mock_load_credentials: MagicMock,
        mock_with_scopes: MagicMock,
        mock_cloud_sdk_path: MagicMock,
        mock_auth_request: MagicMock,
    ) -> None:
        custom_scopes = ["https://www.googleapis.com/auth/bigquery"]
        credentials_file = tmp_path / "adc.json"
        credentials_file.touch()
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials_file))

        result = GoogleADCAuthProvider(scopes=custom_scopes).get_credentials()

        assert result is mock_creds
        mock_with_scopes.assert_called_once_with(mock_creds, custom_scopes)
        mock_google_auth_default.assert_not_called()
        mock_load_credentials.assert_called_once_with(str(credentials_file), request=sentinel.request)
        mock_cloud_sdk_path.assert_called_once_with()
        mock_auth_request.assert_called_once_with()


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
