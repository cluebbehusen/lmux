"""Tests for OpenAI auth provider."""

import pytest

from lmux.exceptions import AuthenticationError
from lmux_openai.auth import OpenAIEnvAuthProvider


class TestOpenAIEnvAuthProvider:
    def test_returns_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
        provider = OpenAIEnvAuthProvider()
        assert provider.get_credentials() == "sk-test-123"

    def test_raises_when_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = OpenAIEnvAuthProvider()
        with pytest.raises(AuthenticationError, match="OPENAI_API_KEY"):
            provider.get_credentials()

    def test_error_has_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = OpenAIEnvAuthProvider()
        with pytest.raises(AuthenticationError) as exc_info:
            provider.get_credentials()
        assert exc_info.value.provider == "openai"

    async def test_aget_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-456")
        provider = OpenAIEnvAuthProvider()
        assert await provider.aget_credentials() == "sk-test-456"

    async def test_aget_raises_when_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = OpenAIEnvAuthProvider()
        with pytest.raises(AuthenticationError):
            await provider.aget_credentials()
