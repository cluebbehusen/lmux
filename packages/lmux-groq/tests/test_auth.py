"""Tests for Groq auth provider."""

import pytest

from lmux.exceptions import AuthenticationError
from lmux_groq.auth import GroqEnvAuthProvider


class TestGroqEnvAuthProvider:
    def test_returns_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test-123")
        provider = GroqEnvAuthProvider()
        assert provider.get_credentials() == "gsk-test-123"

    def test_raises_when_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        provider = GroqEnvAuthProvider()
        with pytest.raises(AuthenticationError, match="GROQ_API_KEY"):
            provider.get_credentials()

    def test_error_has_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        provider = GroqEnvAuthProvider()
        with pytest.raises(AuthenticationError) as exc_info:
            provider.get_credentials()
        assert exc_info.value.provider == "groq"

    async def test_aget_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test-456")
        provider = GroqEnvAuthProvider()
        assert await provider.aget_credentials() == "gsk-test-456"

    async def test_aget_raises_when_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        provider = GroqEnvAuthProvider()
        with pytest.raises(AuthenticationError):
            await provider.aget_credentials()
