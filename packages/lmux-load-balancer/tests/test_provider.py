"""Tests for LoadBalancerProvider."""

import pytest

from lmux import (
    AuthenticationError,
    BaseProviderParams,
    ChatChunk,
    ChatResponse,
    Cost,
    InvalidRequestError,
    MockProvider,
    ProviderError,
    RateLimitError,
    Registry,
    Usage,
    UserMessage,
)
from lmux_load_balancer import LoadBalancerMetadata, LoadBalancerParams, LoadBalancerProvider

# MARK: Fixtures


@pytest.fixture(autouse=True)
def fixed_order(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force candidate order to weight-insertion order so multi-endpoint tests are deterministic.
    def _zero(_sticky_key: str | None) -> float:
        return 0.0

    monkeypatch.setattr("lmux_load_balancer.provider.point_for", _zero)


@pytest.fixture
def response() -> ChatResponse:
    return ChatResponse(
        content="hi",
        usage=Usage(input_tokens=1, output_tokens=1),
        cost=Cost(input_cost=0.0, output_cost=0.0, total_cost=0.0),
        model="model-x",
        provider="child",
    )


@pytest.fixture
def chunks() -> list[ChatChunk]:
    return [
        ChatChunk(delta="he"),
        ChatChunk(delta="llo", usage=Usage(input_tokens=1, output_tokens=1)),
    ]


def _messages() -> list[UserMessage]:
    return [UserMessage(content="hi")]


# MARK: Construction


class TestConstruction:
    def test_rejects_empty_group(self) -> None:
        with pytest.raises(InvalidRequestError, match="no endpoints"):
            _ = LoadBalancerProvider(Registry(), {"m": {}})

    def test_rejects_unregistered_prefix(self) -> None:
        with pytest.raises(InvalidRequestError, match="unregistered prefix"):
            _ = LoadBalancerProvider(Registry(), {"m": {"missing/x": 1.0}})

    def test_accepts_valid_group(self) -> None:
        registry = Registry()
        registry.register("a", MockProvider())
        lb = LoadBalancerProvider(registry, {"m": {"a/x": 1.0}})
        assert isinstance(lb, LoadBalancerProvider)


def _two_endpoint_lb(registry: Registry) -> LoadBalancerProvider:
    return LoadBalancerProvider(registry, {"m": {"a/x": 1.0, "b/x": 1.0}})


# MARK: chat


class TestChat:
    def test_returns_and_stamps_metadata(self, response: ChatResponse) -> None:
        registry = Registry()
        registry.register("a", MockProvider(chat_responses=[response]))
        b = MockProvider(chat_responses=[response])
        registry.register("b", b)
        lb = _two_endpoint_lb(registry)

        result = lb.chat("m", _messages())

        assert result.content == "hi"
        assert result.provider_metadata == LoadBalancerMetadata(primary="a/x", served="a/x", attempted=["a/x"])
        assert b.calls == []  # primary succeeded; fallback untouched

    def test_fails_over_on_retryable(self, response: ChatResponse) -> None:
        registry = Registry()
        registry.register("a", MockProvider(errors=[RateLimitError("rate")]))
        registry.register("b", MockProvider(chat_responses=[response]))
        lb = _two_endpoint_lb(registry)

        result = lb.chat("m", _messages())

        assert result.provider_metadata == LoadBalancerMetadata(primary="a/x", served="b/x", attempted=["a/x", "b/x"])

    def test_non_retryable_propagates(self) -> None:
        registry = Registry()
        registry.register("a", MockProvider(errors=[AuthenticationError("auth")]))
        b = MockProvider(chat_responses=[])
        registry.register("b", b)
        lb = _two_endpoint_lb(registry)

        with pytest.raises(AuthenticationError):
            _ = lb.chat("m", _messages())
        assert b.calls == []  # non-retryable: no failover

    def test_exhaustion_reraises_last(self) -> None:
        registry = Registry()
        registry.register("a", MockProvider(errors=[RateLimitError("first")]))
        registry.register("b", MockProvider(errors=[ProviderError("last")]))
        lb = _two_endpoint_lb(registry)

        with pytest.raises(ProviderError, match="last"):
            _ = lb.chat("m", _messages())

    def test_unknown_model_raises(self) -> None:
        registry = Registry()
        registry.register("a", MockProvider())
        lb = LoadBalancerProvider(registry, {"m": {"a/x": 1.0}})

        with pytest.raises(InvalidRequestError, match="No load-balancer group"):
            _ = lb.chat("unknown", _messages())

    def test_foreign_provider_params_coerced(self, response: ChatResponse) -> None:
        registry = Registry()
        registry.register("a", MockProvider(chat_responses=[response]))
        lb = LoadBalancerProvider(registry, {"m": {"a/x": 1.0}})

        # A non-LoadBalancerParams object must be tolerated (coerced to defaults), not crash.
        result = lb.chat("m", _messages(), provider_params=BaseProviderParams())  # ty: ignore[invalid-argument-type]
        assert result.content == "hi"


# MARK: achat


class TestAchat:
    async def test_returns_and_stamps_metadata(self, response: ChatResponse) -> None:
        registry = Registry()
        registry.register("a", MockProvider(chat_responses=[response]))
        registry.register("b", MockProvider(chat_responses=[response]))
        lb = _two_endpoint_lb(registry)

        result = await lb.achat("m", _messages())

        assert result.provider_metadata == LoadBalancerMetadata(primary="a/x", served="a/x", attempted=["a/x"])

    async def test_fails_over_on_retryable(self, response: ChatResponse) -> None:
        registry = Registry()
        registry.register("a", MockProvider(errors=[RateLimitError("rate")]))
        registry.register("b", MockProvider(chat_responses=[response]))
        lb = _two_endpoint_lb(registry)

        result = await lb.achat("m", _messages())

        assert result.provider_metadata == LoadBalancerMetadata(primary="a/x", served="b/x", attempted=["a/x", "b/x"])

    async def test_non_retryable_propagates(self) -> None:
        registry = Registry()
        registry.register("a", MockProvider(errors=[AuthenticationError("auth")]))
        registry.register("b", MockProvider())
        lb = _two_endpoint_lb(registry)

        with pytest.raises(AuthenticationError):
            _ = await lb.achat("m", _messages())

    async def test_exhaustion_reraises_last(self) -> None:
        registry = Registry()
        registry.register("a", MockProvider(errors=[RateLimitError("first")]))
        registry.register("b", MockProvider(errors=[ProviderError("last")]))
        lb = _two_endpoint_lb(registry)

        with pytest.raises(ProviderError, match="last"):
            _ = await lb.achat("m", _messages())


# MARK: Failover modes


class TestFailoverModes:
    def test_never_does_not_fail_over(self) -> None:
        registry = Registry()
        registry.register("a", MockProvider(errors=[RateLimitError("rate")]))
        b = MockProvider(chat_responses=[])
        registry.register("b", b)
        lb = _two_endpoint_lb(registry)

        with pytest.raises(RateLimitError):
            _ = lb.chat("m", _messages(), provider_params=LoadBalancerParams(failover="never"))
        assert b.calls == []

    def test_unless_sticky_pins_sticky_call(self) -> None:
        registry = Registry()
        registry.register("a", MockProvider(errors=[RateLimitError("rate")]))
        b = MockProvider(chat_responses=[])
        registry.register("b", b)
        lb = _two_endpoint_lb(registry)

        with pytest.raises(RateLimitError):
            _ = lb.chat("m", _messages(), provider_params=LoadBalancerParams(failover="unless_sticky", sticky_key="x"))
        assert b.calls == []

    def test_unless_sticky_fails_over_keyless(self, response: ChatResponse) -> None:
        registry = Registry()
        registry.register("a", MockProvider(errors=[RateLimitError("rate")]))
        registry.register("b", MockProvider(chat_responses=[response]))
        lb = _two_endpoint_lb(registry)

        result = lb.chat("m", _messages(), provider_params=LoadBalancerParams(failover="unless_sticky"))

        assert isinstance(result.provider_metadata, LoadBalancerMetadata)
        assert result.provider_metadata.served == "b/x"


# MARK: chat_stream


class TestChatStream:
    def test_streams_and_stamps_first(self, chunks: list[ChatChunk]) -> None:
        registry = Registry()
        registry.register("a", MockProvider(chat_stream_responses=[chunks]))
        registry.register("b", MockProvider(chat_stream_responses=[chunks]))
        lb = _two_endpoint_lb(registry)

        out = list(lb.chat_stream("m", _messages()))

        assert [c.delta for c in out] == ["he", "llo"]
        assert out[0].provider_metadata == LoadBalancerMetadata(primary="a/x", served="a/x", attempted=["a/x"])
        assert out[-1].provider_metadata is None

    def test_single_chunk_gets_metadata(self) -> None:
        only = [ChatChunk(delta="x", usage=Usage(input_tokens=1, output_tokens=1))]
        registry = Registry()
        registry.register("a", MockProvider(chat_stream_responses=[only]))
        lb = LoadBalancerProvider(registry, {"m": {"a/x": 1.0}})

        out = list(lb.chat_stream("m", _messages()))

        assert len(out) == 1
        assert out[0].provider_metadata == LoadBalancerMetadata(primary="a/x", served="a/x", attempted=["a/x"])

    def test_empty_stream_yields_nothing(self) -> None:
        registry = Registry()
        registry.register("a", MockProvider(chat_stream_responses=[[]]))
        lb = LoadBalancerProvider(registry, {"m": {"a/x": 1.0}})

        assert list(lb.chat_stream("m", _messages())) == []

    def test_fails_over_before_first_chunk(self, chunks: list[ChatChunk]) -> None:
        registry = Registry()
        registry.register("a", MockProvider(errors=[RateLimitError("rate")]))
        registry.register("b", MockProvider(chat_stream_responses=[chunks]))
        lb = _two_endpoint_lb(registry)

        out = list(lb.chat_stream("m", _messages()))

        assert out[0].provider_metadata == LoadBalancerMetadata(primary="a/x", served="b/x", attempted=["a/x", "b/x"])

    def test_non_retryable_propagates(self) -> None:
        registry = Registry()
        registry.register("a", MockProvider(errors=[AuthenticationError("auth")]))
        registry.register("b", MockProvider(chat_stream_responses=[[]]))
        lb = _two_endpoint_lb(registry)

        with pytest.raises(AuthenticationError):
            _ = list(lb.chat_stream("m", _messages()))

    def test_exhaustion_reraises_last(self) -> None:
        registry = Registry()
        registry.register("a", MockProvider(errors=[RateLimitError("first")]))
        registry.register("b", MockProvider(errors=[ProviderError("last")]))
        lb = _two_endpoint_lb(registry)

        with pytest.raises(ProviderError, match="last"):
            _ = list(lb.chat_stream("m", _messages()))


# MARK: achat_stream


class TestAchatStream:
    async def test_streams_and_stamps_first(self, chunks: list[ChatChunk]) -> None:
        registry = Registry()
        registry.register("a", MockProvider(chat_stream_responses=[chunks]))
        registry.register("b", MockProvider(chat_stream_responses=[chunks]))
        lb = _two_endpoint_lb(registry)

        out = [chunk async for chunk in lb.achat_stream("m", _messages())]

        assert [c.delta for c in out] == ["he", "llo"]
        assert out[0].provider_metadata == LoadBalancerMetadata(primary="a/x", served="a/x", attempted=["a/x"])

    async def test_single_chunk_gets_metadata(self) -> None:
        only = [ChatChunk(delta="x", usage=Usage(input_tokens=1, output_tokens=1))]
        registry = Registry()
        registry.register("a", MockProvider(chat_stream_responses=[only]))
        lb = LoadBalancerProvider(registry, {"m": {"a/x": 1.0}})

        out = [chunk async for chunk in lb.achat_stream("m", _messages())]

        assert len(out) == 1
        assert out[0].provider_metadata == LoadBalancerMetadata(primary="a/x", served="a/x", attempted=["a/x"])

    async def test_empty_stream_yields_nothing(self) -> None:
        registry = Registry()
        registry.register("a", MockProvider(chat_stream_responses=[[]]))
        lb = LoadBalancerProvider(registry, {"m": {"a/x": 1.0}})

        assert [chunk async for chunk in lb.achat_stream("m", _messages())] == []

    async def test_fails_over_before_first_chunk(self, chunks: list[ChatChunk]) -> None:
        registry = Registry()
        registry.register("a", MockProvider(errors=[RateLimitError("rate")]))
        registry.register("b", MockProvider(chat_stream_responses=[chunks]))
        lb = _two_endpoint_lb(registry)

        out = [chunk async for chunk in lb.achat_stream("m", _messages())]

        assert out[0].provider_metadata == LoadBalancerMetadata(primary="a/x", served="b/x", attempted=["a/x", "b/x"])

    async def test_non_retryable_propagates(self) -> None:
        registry = Registry()
        registry.register("a", MockProvider(errors=[AuthenticationError("auth")]))
        registry.register("b", MockProvider(chat_stream_responses=[[]]))
        lb = _two_endpoint_lb(registry)

        with pytest.raises(AuthenticationError):
            _ = [chunk async for chunk in lb.achat_stream("m", _messages())]

    async def test_exhaustion_reraises_last(self) -> None:
        registry = Registry()
        registry.register("a", MockProvider(errors=[RateLimitError("first")]))
        registry.register("b", MockProvider(errors=[ProviderError("last")]))
        lb = _two_endpoint_lb(registry)

        with pytest.raises(ProviderError, match="last"):
            _ = [chunk async for chunk in lb.achat_stream("m", _messages())]
