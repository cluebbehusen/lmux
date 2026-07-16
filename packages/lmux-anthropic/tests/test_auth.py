"""Tests for Anthropic auth providers."""

from unittest.mock import MagicMock

import boto3
import pytest
from pytest_mock import MockerFixture

from lmux.exceptions import AuthenticationError
from lmux_anthropic.auth import (
    AnthropicBedrockEnvAuthProvider,
    AnthropicBedrockSessionAuthProvider,
    AnthropicEnvAuthProvider,
    AnthropicFoundryEnvAuthProvider,
    AnthropicFoundryTokenAuthProvider,
    AnthropicVertexADCAuthProvider,
    AnthropicVertexServiceAccountAuthProvider,
)

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


@pytest.fixture
def mock_credentials() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_google_auth_default(mock_credentials: MagicMock, mocker: MockerFixture) -> MagicMock:
    return mocker.patch("google.auth.default", return_value=(mock_credentials, "test-project"))


@pytest.fixture
def mock_from_service_account_file(mock_credentials: MagicMock, mocker: MockerFixture) -> MagicMock:
    mock_credentials.project_id = "sa-project"
    return mocker.patch(
        "google.oauth2.service_account.Credentials.from_service_account_file",
        return_value=mock_credentials,
    )


@pytest.fixture
def mock_missing_google_auth(mocker: MockerFixture) -> None:
    mocker.patch.dict("sys.modules", {"google.auth": None})


@pytest.fixture
def mock_missing_google_oauth2(mocker: MockerFixture) -> None:
    mocker.patch.dict("sys.modules", {"google.oauth2": None})


class TestAnthropicEnvAuthProvider:
    def test_get_credentials_returns_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        provider = AnthropicEnvAuthProvider()
        assert provider.get_credentials() == "sk-ant-test-key"

    def test_get_credentials_raises_when_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        provider = AnthropicEnvAuthProvider()
        with pytest.raises(AuthenticationError, match="ANTHROPIC_API_KEY") as exc_info:
            _ = provider.get_credentials()
        assert exc_info.value.provider == "anthropic"

    async def test_aget_credentials_returns_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        provider = AnthropicEnvAuthProvider()
        assert await provider.aget_credentials() == "sk-ant-test-key"


class TestAnthropicVertexADCAuthProvider:
    def test_get_credentials(self, mock_google_auth_default: MagicMock, mock_credentials: MagicMock) -> None:
        provider = AnthropicVertexADCAuthProvider()
        assert provider.get_credentials() == (mock_credentials, "test-project")
        mock_google_auth_default.assert_called_once_with(scopes=[CLOUD_PLATFORM_SCOPE])

    def test_custom_scopes(self, mock_google_auth_default: MagicMock, mock_credentials: MagicMock) -> None:
        provider = AnthropicVertexADCAuthProvider(scopes=["https://www.googleapis.com/auth/custom"])
        assert provider.get_credentials() == (mock_credentials, "test-project")
        mock_google_auth_default.assert_called_once_with(scopes=["https://www.googleapis.com/auth/custom"])

    async def test_aget_credentials(self, mock_google_auth_default: MagicMock, mock_credentials: MagicMock) -> None:
        provider = AnthropicVertexADCAuthProvider()
        assert await provider.aget_credentials() == (mock_credentials, "test-project")
        mock_google_auth_default.assert_called_once_with(scopes=[CLOUD_PLATFORM_SCOPE])

    def test_get_raises_import_error_without_extra(self, mock_missing_google_auth: None) -> None:
        assert mock_missing_google_auth is None  # side-effect fixture: patches sys.modules
        provider = AnthropicVertexADCAuthProvider()
        with pytest.raises(ImportError, match=r"\[vertex\] extra group is required"):
            _ = provider.get_credentials()


class TestAnthropicVertexServiceAccountAuthProvider:
    def test_get_credentials(self, mock_from_service_account_file: MagicMock, mock_credentials: MagicMock) -> None:
        provider = AnthropicVertexServiceAccountAuthProvider(service_account_file="/path/to/key.json")
        assert provider.get_credentials() == (mock_credentials, "sa-project")
        mock_from_service_account_file.assert_called_once_with("/path/to/key.json", scopes=[CLOUD_PLATFORM_SCOPE])

    def test_custom_scopes(self, mock_from_service_account_file: MagicMock, mock_credentials: MagicMock) -> None:
        provider = AnthropicVertexServiceAccountAuthProvider(
            service_account_file="/path/to/key.json",
            scopes=["https://www.googleapis.com/auth/custom"],
        )
        assert provider.get_credentials() == (mock_credentials, "sa-project")
        mock_from_service_account_file.assert_called_once_with(
            "/path/to/key.json", scopes=["https://www.googleapis.com/auth/custom"]
        )

    async def test_aget_credentials(
        self, mock_from_service_account_file: MagicMock, mock_credentials: MagicMock
    ) -> None:
        provider = AnthropicVertexServiceAccountAuthProvider(service_account_file="/path/to/key.json")
        assert await provider.aget_credentials() == (mock_credentials, "sa-project")
        mock_from_service_account_file.assert_called_once_with("/path/to/key.json", scopes=[CLOUD_PLATFORM_SCOPE])

    def test_get_raises_import_error_without_extra(self, mock_missing_google_oauth2: None) -> None:
        assert mock_missing_google_oauth2 is None  # side-effect fixture: patches sys.modules
        provider = AnthropicVertexServiceAccountAuthProvider(service_account_file="/path/to/key.json")
        with pytest.raises(ImportError, match=r"\[vertex\] extra group is required"):
            _ = provider.get_credentials()


class TestAnthropicFoundryEnvAuthProvider:
    def test_get_credentials_returns_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "foundry-test-key")
        provider = AnthropicFoundryEnvAuthProvider()
        assert provider.get_credentials() == "foundry-test-key"

    def test_get_credentials_raises_when_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_FOUNDRY_API_KEY", raising=False)
        provider = AnthropicFoundryEnvAuthProvider()
        with pytest.raises(AuthenticationError, match="ANTHROPIC_FOUNDRY_API_KEY") as exc_info:
            _ = provider.get_credentials()
        assert exc_info.value.provider == "anthropic-foundry"

    async def test_aget_credentials_returns_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "foundry-test-key")
        provider = AnthropicFoundryEnvAuthProvider()
        assert await provider.aget_credentials() == "foundry-test-key"


class TestAnthropicFoundryTokenAuthProvider:
    def test_get_credentials_returns_token_provider(self) -> None:
        def token_provider() -> str:
            return "entra-token"  # pragma: no cover

        provider = AnthropicFoundryTokenAuthProvider(token_provider=token_provider)
        assert provider.get_credentials() is token_provider

    async def test_aget_credentials_returns_token_provider(self) -> None:
        def token_provider() -> str:
            return "entra-token"  # pragma: no cover

        provider = AnthropicFoundryTokenAuthProvider(token_provider=token_provider)
        assert await provider.aget_credentials() is token_provider


class TestAnthropicBedrockEnvAuthProvider:
    def test_get_credentials_returns_session(self) -> None:
        session = AnthropicBedrockEnvAuthProvider().get_credentials()
        assert isinstance(session, boto3.Session)

    async def test_aget_credentials_returns_session(self) -> None:
        session = await AnthropicBedrockEnvAuthProvider().aget_credentials()
        assert isinstance(session, boto3.Session)


class TestAnthropicBedrockSessionAuthProvider:
    def test_get_credentials_applies_config(self) -> None:
        provider = AnthropicBedrockSessionAuthProvider(region_name="us-west-2", profile_name=None)
        session = provider.get_credentials()
        assert isinstance(session, boto3.Session)
        assert session.region_name == "us-west-2"

    async def test_aget_credentials_applies_config(self) -> None:
        provider = AnthropicBedrockSessionAuthProvider(region_name="eu-west-1")
        session = await provider.aget_credentials()
        assert session.region_name == "eu-west-1"
