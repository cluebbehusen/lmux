"""Tests for Azure AI Foundry auth providers."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from lmux.exceptions import AuthenticationError
from lmux_azure_foundry.auth import (
    AzureAdToken,
    AzureFoundryKeyAuthProvider,
    AzureFoundryTokenAuthProvider,
)

# MARK: AzureAdToken


class TestAzureAdToken:
    def test_stores_token(self) -> None:
        token = AzureAdToken(token="eyJhbGciOiJ...")
        assert token.token == "eyJhbGciOiJ..."

    def test_frozen(self) -> None:
        token = AzureAdToken(token="abc")
        with pytest.raises(AttributeError):
            token.token = "xyz"  # pyright: ignore[reportAttributeAccessIssue]

    def test_equality(self) -> None:
        assert AzureAdToken(token="abc") == AzureAdToken(token="abc")
        assert AzureAdToken(token="abc") != AzureAdToken(token="xyz")


# MARK: AzureFoundryKeyAuthProvider


class TestAzureFoundryKeyAuthProvider:
    def test_returns_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "test-key-123")
        provider = AzureFoundryKeyAuthProvider()
        assert provider.get_credentials() == "test-key-123"

    def test_raises_when_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
        provider = AzureFoundryKeyAuthProvider()
        with pytest.raises(AuthenticationError, match="AZURE_FOUNDRY_API_KEY"):
            provider.get_credentials()

    def test_error_has_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
        provider = AzureFoundryKeyAuthProvider()
        with pytest.raises(AuthenticationError) as exc_info:
            provider.get_credentials()
        assert exc_info.value.provider == "azure-foundry"

    async def test_aget_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "test-key-456")
        provider = AzureFoundryKeyAuthProvider()
        assert await provider.aget_credentials() == "test-key-456"

    async def test_aget_raises_when_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
        provider = AzureFoundryKeyAuthProvider()
        with pytest.raises(AuthenticationError):
            await provider.aget_credentials()


# MARK: AzureFoundryTokenAuthProvider


class TestAzureFoundryTokenAuthProvider:
    @patch("azure.identity.get_bearer_token_provider")
    @patch("azure.identity.DefaultAzureCredential")
    def test_get_credentials_returns_callable(
        self, mock_credential_cls: MagicMock, mock_get_provider: MagicMock
    ) -> None:
        mock_token_fn = MagicMock(spec=Callable[[], str])
        mock_get_provider.return_value = mock_token_fn

        provider = AzureFoundryTokenAuthProvider()
        result = provider.get_credentials()

        assert result is mock_token_fn
        mock_credential_cls.assert_called_once()
        mock_get_provider.assert_called_once_with(
            mock_credential_cls.return_value, "https://cognitiveservices.azure.com/.default"
        )

    @patch("azure.identity.get_bearer_token_provider")
    @patch("azure.identity.DefaultAzureCredential")
    def test_custom_scopes(self, mock_credential_cls: MagicMock, mock_get_provider: MagicMock) -> None:
        provider = AzureFoundryTokenAuthProvider(scopes=("https://custom.scope/.default",))
        provider.get_credentials()

        mock_get_provider.assert_called_once_with(mock_credential_cls.return_value, "https://custom.scope/.default")

    @patch("azure.identity.get_bearer_token_provider")
    @patch("azure.identity.DefaultAzureCredential")
    def test_caches_token_provider(self, mock_credential_cls: MagicMock, mock_get_provider: MagicMock) -> None:
        provider = AzureFoundryTokenAuthProvider()
        result1 = provider.get_credentials()
        result2 = provider.get_credentials()

        assert result1 is result2
        mock_credential_cls.assert_called_once()
        mock_get_provider.assert_called_once()

    @patch("azure.identity.get_bearer_token_provider")
    @patch("azure.identity.DefaultAzureCredential")
    async def test_aget_credentials(self, mock_credential_cls: MagicMock, mock_get_provider: MagicMock) -> None:
        mock_token_fn = MagicMock(spec=Callable[[], str])
        mock_get_provider.return_value = mock_token_fn

        provider = AzureFoundryTokenAuthProvider()
        result = await provider.aget_credentials()

        assert result is mock_token_fn
