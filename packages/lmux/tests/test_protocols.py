"""Tests for lmux protocol definitions."""

from lmux.mock import MockProvider
from lmux.protocols import (
    AuthProvider,
    CompletionProvider,
    EmbeddingProvider,
    PricingProvider,
    ResponsesProvider,
)


class SimpleAuthProvider:
    """A simple auth provider for testing."""

    def get_credentials(self) -> str:
        return "test-key"

    async def aget_credentials(self) -> str:
        return "test-key"


class TestAuthProviderProtocol:
    def test_isinstance_check(self) -> None:
        provider = SimpleAuthProvider()
        assert isinstance(provider, AuthProvider)

    def test_get_credentials(self) -> None:
        provider = SimpleAuthProvider()
        assert provider.get_credentials() == "test-key"

    async def test_aget_credentials(self) -> None:
        provider = SimpleAuthProvider()
        assert await provider.aget_credentials() == "test-key"


class TestCompletionProviderProtocol:
    def test_mock_is_completion_provider(self) -> None:
        provider = MockProvider()
        assert isinstance(provider, CompletionProvider)


class TestEmbeddingProviderProtocol:
    def test_mock_is_embedding_provider(self) -> None:
        provider = MockProvider()
        assert isinstance(provider, EmbeddingProvider)


class TestResponsesProviderProtocol:
    def test_mock_is_responses_provider(self) -> None:
        provider = MockProvider()
        assert isinstance(provider, ResponsesProvider)


class TestPricingProviderProtocol:
    def test_mock_is_pricing_provider(self) -> None:
        provider = MockProvider()
        assert isinstance(provider, PricingProvider)
