"""Tests for Anthropic auth provider."""

import pytest

from lmux.exceptions import AuthenticationError
from lmux_anthropic.auth import AnthropicEnvAuthProvider


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
