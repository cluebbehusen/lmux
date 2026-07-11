"""Tests for the Groq provider (SDK-lite, respx)."""

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
import respx
from pytest_mock import MockerFixture

from lmux.cost import ModelPricing, PricingTier
from lmux.exceptions import AuthenticationError, InvalidRequestError, ProviderError
from lmux.types import FunctionDefinition, JsonObjectResponseFormat, Tool, UserMessage
from lmux_groq import preload
from lmux_groq.params import GroqParams
from lmux_groq.provider import GroqProvider

_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"


# MARK: Shared Fixtures


class FakeAuth:
    def get_credentials(self) -> str:
        return "gsk-fake-key"

    async def aget_credentials(self) -> str:
        return "gsk-fake-key"


@pytest.fixture
def fake_auth() -> FakeAuth:
    return FakeAuth()


@pytest.fixture
def sync_provider(fake_auth: FakeAuth) -> GroqProvider:
    return GroqProvider(auth=fake_auth)


@pytest.fixture
def async_provider(fake_auth: FakeAuth) -> GroqProvider:
    return GroqProvider(auth=fake_auth)


@pytest.fixture
def sync_create_raises(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_groq.provider.create_sync_client", side_effect=RuntimeError("client init failed"))


@pytest.fixture
def async_create_raises(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_groq.provider.create_async_client", side_effect=RuntimeError("client init failed"))


@pytest.fixture
def async_create_two_clients(mocker: MockerFixture) -> tuple[MagicMock, MagicMock, MagicMock]:
    c1, c2 = MagicMock(), MagicMock()
    create = mocker.patch("lmux_groq.provider.create_async_client", side_effect=[c1, c2])
    return create, c1, c2


@pytest.fixture
def mock_get_running_loop(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_groq.provider.asyncio.get_running_loop")


@pytest.fixture
def completion() -> dict[str, Any]:
    return {
        "id": "chatcmpl-123",
        "model": MODEL,
        "object": "chat.completion",
        "created": 1,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hello!"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _sse_stream() -> bytes:
    chunks = [
        {"model": MODEL, "choices": [{"index": 0, "finish_reason": None, "delta": {"content": "Hel"}}]},
        {"model": MODEL, "choices": [{"index": 0, "finish_reason": None, "delta": {"content": "lo!"}}]},
        {
            "model": MODEL,
            "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    ]
    lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
    return ("\n\n".join(lines) + "\n\n").encode()


def _ok(completion: dict[str, Any], respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(_URL).mock(return_value=httpx.Response(200, json=completion))


# MARK: Chat


class TestChat:
    def test_basic(self, sync_provider: GroqProvider, completion: dict[str, Any], respx_mock: respx.MockRouter) -> None:
        route = _ok(completion, respx_mock)
        result = sync_provider.chat(MODEL, [UserMessage(content="Hi")])
        assert result.content == "Hello!"
        assert result.model == MODEL
        assert result.provider == "groq"
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert route.called

    def test_request_body(
        self, sync_provider: GroqProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(completion, respx_mock)
        sync_provider.chat(MODEL, [UserMessage(content="Hi")], temperature=0.5, max_tokens=100, top_p=0.9, stop=["END"])
        body = json.loads(route.calls.last.request.content)
        assert body == {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "temperature": 0.5,
            "max_tokens": 100,
            "top_p": 0.9,
            "stop": ["END"],
        }

    def test_tools_and_choice(
        self, sync_provider: GroqProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(completion, respx_mock)
        sync_provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            tools=[Tool(function=FunctionDefinition(name="get_weather"))],
            tool_choice="required",
            response_format=JsonObjectResponseFormat(),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["tools"] == [{"type": "function", "function": {"name": "get_weather"}}]
        assert body["tool_choice"] == "required"
        assert body["response_format"] == {"type": "json_object"}

    def test_reasoning_effort(
        self, sync_provider: GroqProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(completion, respx_mock)
        sync_provider.chat(MODEL, [UserMessage(content="Hi")], reasoning_effort="medium")
        body = json.loads(route.calls.last.request.content)
        assert body["reasoning_effort"] == "medium"
        assert body["include_reasoning"] is True

    def test_status_error_mapped(self, sync_provider: GroqProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(return_value=httpx.Response(400, json={"error": {"message": "bad"}}))
        with pytest.raises(InvalidRequestError):
            sync_provider.chat(MODEL, [UserMessage(content="Hi")])

    def test_transport_error_mapped(self, sync_provider: GroqProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            sync_provider.chat(MODEL, [UserMessage(content="Hi")])

    def test_non_json_body_mapped(self, sync_provider: GroqProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(return_value=httpx.Response(200, content=b"not json"))
        with pytest.raises(ProviderError):
            sync_provider.chat(MODEL, [UserMessage(content="Hi")])

    def test_cost_calculated(
        self, sync_provider: GroqProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok(completion, respx_mock)
        result = sync_provider.chat(MODEL, [UserMessage(content="Hi")])
        assert result.cost is not None
        assert result.cost.total_cost > 0


# MARK: Achat


class TestAchat:
    async def test_basic(
        self, async_provider: GroqProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok(completion, respx_mock)
        result = await async_provider.achat(MODEL, [UserMessage(content="Hi")])
        assert result.content == "Hello!"
        assert result.provider == "groq"

    async def test_status_error_mapped(self, async_provider: GroqProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(return_value=httpx.Response(401, json={"error": {"message": "no"}}))
        with pytest.raises(AuthenticationError):
            await async_provider.achat(MODEL, [UserMessage(content="Hi")])

    async def test_transport_error_mapped(self, async_provider: GroqProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            await async_provider.achat(MODEL, [UserMessage(content="Hi")])

    async def test_non_json_body_mapped(self, async_provider: GroqProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(return_value=httpx.Response(200, content=b"not json"))
        with pytest.raises(ProviderError):
            await async_provider.achat(MODEL, [UserMessage(content="Hi")])


# MARK: ChatStream


class TestChatStream:
    def test_yields_and_costs(self, sync_provider: GroqProvider, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(_URL).mock(return_value=httpx.Response(200, content=_sse_stream()))
        chunks = list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))
        assert [c.delta for c in chunks[:2]] == ["Hel", "lo!"]
        assert chunks[2].finish_reason == "stop"
        assert chunks[2].usage is not None
        assert chunks[0].cost is None
        assert chunks[2].cost is not None
        assert chunks[2].cost.total_cost > 0
        body = json.loads(route.calls.last.request.content)
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}

    def test_status_error_on_open(self, sync_provider: GroqProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(return_value=httpx.Response(500, json={"error": {"message": "boom"}}))
        with pytest.raises(ProviderError):
            list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))

    def test_malformed_chunk_mapped(self, sync_provider: GroqProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(return_value=httpx.Response(200, content=b"data: {not json}\n\n"))
        with pytest.raises(ProviderError):
            list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))

    def test_mid_stream_error_raises_after_partial(
        self, sync_provider: GroqProvider, respx_mock: respx.MockRouter
    ) -> None:
        first = {"model": MODEL, "choices": [{"index": 0, "finish_reason": None, "delta": {"content": "Hel"}}]}
        error = {"error": {"message": "mid-stream boom"}}
        sse = (f"data: {json.dumps(first)}\n\ndata: {json.dumps(error)}\n\n").encode()
        respx_mock.post(_URL).mock(return_value=httpx.Response(200, content=sse))
        stream = sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")])
        assert next(stream).delta == "Hel"  # partial output arrives first
        with pytest.raises(ProviderError, match="mid-stream boom"):
            next(stream)

    def test_client_init_failure(self, fake_auth: FakeAuth, sync_create_raises: MagicMock) -> None:
        provider = GroqProvider(auth=fake_auth)
        with pytest.raises(ProviderError, match="client init failed"):
            list(provider.chat_stream(MODEL, [UserMessage(content="Hi")]))
        sync_create_raises.assert_called_once()

    def test_stream_without_done(self, sync_provider: GroqProvider, respx_mock: respx.MockRouter) -> None:
        chunk = {"model": MODEL, "choices": [{"index": 0, "finish_reason": "stop", "delta": {"content": "x"}}]}
        respx_mock.post(_URL).mock(
            return_value=httpx.Response(200, content=(f"data: {json.dumps(chunk)}\n\n").encode())
        )
        chunks = list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))
        assert [c.delta for c in chunks] == ["x"]


# MARK: AchatStream


class TestAchatStream:
    async def test_yields_and_costs(self, async_provider: GroqProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(return_value=httpx.Response(200, content=_sse_stream()))
        chunks = [c async for c in async_provider.achat_stream(MODEL, [UserMessage(content="Hi")])]
        assert [c.delta for c in chunks[:2]] == ["Hel", "lo!"]
        assert chunks[2].cost is not None

    async def test_status_error_on_open(self, async_provider: GroqProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(return_value=httpx.Response(500, json={"error": {"message": "boom"}}))
        with pytest.raises(ProviderError):
            async for _ in async_provider.achat_stream(MODEL, [UserMessage(content="Hi")]):
                pass  # pragma: no cover

    async def test_malformed_chunk_mapped(self, async_provider: GroqProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(return_value=httpx.Response(200, content=b"data: {nope}\n\n"))
        with pytest.raises(ProviderError):
            async for _ in async_provider.achat_stream(MODEL, [UserMessage(content="Hi")]):
                pass  # pragma: no cover

    async def test_mid_stream_error_raises_after_partial(
        self, async_provider: GroqProvider, respx_mock: respx.MockRouter
    ) -> None:
        first = {"model": MODEL, "choices": [{"index": 0, "finish_reason": None, "delta": {"content": "Hel"}}]}
        error = {"error": {"message": "mid-stream boom"}}
        sse = (f"data: {json.dumps(first)}\n\ndata: {json.dumps(error)}\n\n").encode()
        respx_mock.post(_URL).mock(return_value=httpx.Response(200, content=sse))
        stream = async_provider.achat_stream(MODEL, [UserMessage(content="Hi")])
        assert (await anext(stream)).delta == "Hel"  # partial output arrives first
        with pytest.raises(ProviderError, match="mid-stream boom"):
            await anext(stream)

    async def test_client_init_failure(self, fake_auth: FakeAuth, async_create_raises: MagicMock) -> None:
        provider = GroqProvider(auth=fake_auth)
        with pytest.raises(ProviderError, match="client init failed"):
            async for _ in provider.achat_stream(MODEL, [UserMessage(content="Hi")]):
                pass  # pragma: no cover
        async_create_raises.assert_called_once()

    async def test_stream_without_done(self, async_provider: GroqProvider, respx_mock: respx.MockRouter) -> None:
        chunk = {"model": MODEL, "choices": [{"index": 0, "finish_reason": "stop", "delta": {"content": "x"}}]}
        respx_mock.post(_URL).mock(
            return_value=httpx.Response(200, content=(f"data: {json.dumps(chunk)}\n\n").encode())
        )
        chunks = [c async for c in async_provider.achat_stream(MODEL, [UserMessage(content="Hi")])]
        assert [c.delta for c in chunks] == ["x"]


# MARK: Client Management


class TestClientManagement:
    def test_sync_client_reused(
        self, sync_provider: GroqProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok(completion, respx_mock)
        sync_provider.chat(MODEL, [UserMessage(content="a")])
        client = sync_provider._sync_client
        sync_provider.chat(MODEL, [UserMessage(content="b")])
        assert sync_provider._sync_client is client

    async def test_async_client_reused(
        self, async_provider: GroqProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok(completion, respx_mock)
        await async_provider.achat(MODEL, [UserMessage(content="a")])
        client = async_provider._async_client
        await async_provider.achat(MODEL, [UserMessage(content="b")])
        assert async_provider._async_client is client

    def test_custom_base_url(
        self, fake_auth: FakeAuth, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post("https://custom.api/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=completion)
        )
        provider = GroqProvider(auth=fake_auth, base_url="https://custom.api/v1")
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.called

    def test_timeout_and_retries(
        self, fake_auth: FakeAuth, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok(completion, respx_mock)
        provider = GroqProvider(auth=fake_auth, timeout=30.0, max_retries=5)
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert provider._sync_client is not None
        assert provider._sync_client.timeout.read == 30.0

    def test_sync_init_failure_mapped(self, fake_auth: FakeAuth, sync_create_raises: MagicMock) -> None:
        provider = GroqProvider(auth=fake_auth)
        with pytest.raises(ProviderError, match="client init failed"):
            provider.chat(MODEL, [UserMessage(content="Hi")])
        sync_create_raises.assert_called_once()

    async def test_async_init_failure_mapped(self, fake_auth: FakeAuth, async_create_raises: MagicMock) -> None:
        provider = GroqProvider(auth=fake_auth)
        with pytest.raises(ProviderError, match="client init failed"):
            await provider.achat(MODEL, [UserMessage(content="Hi")])
        async_create_raises.assert_called_once()

    async def test_async_client_recreated_on_new_loop(
        self,
        fake_auth: FakeAuth,
        async_create_two_clients: tuple[MagicMock, MagicMock, MagicMock],
        mock_get_running_loop: MagicMock,
    ) -> None:
        create, c1, c2 = async_create_two_clients
        loop1, loop2 = asyncio.new_event_loop(), asyncio.new_event_loop()
        mock_get_running_loop.side_effect = [loop1, loop2]
        provider = GroqProvider(auth=fake_auth)
        r1 = await provider._get_async_client()
        r2 = await provider._get_async_client()
        assert (r1, r2) == (c1, c2)
        assert create.call_count == 2
        loop1.close()
        loop2.close()


# MARK: Register Pricing


class TestRegisterPricing:
    def test_custom_for_unknown_model(self, sync_provider: GroqProvider, respx_mock: respx.MockRouter) -> None:
        completion = {
            "model": "custom-v1",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "x"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        }
        respx_mock.post(_URL).mock(return_value=httpx.Response(200, json=completion))
        sync_provider.register_pricing(
            "custom-v1", ModelPricing(tiers=[PricingTier(input_cost_per_token=5e-6, output_cost_per_token=15e-6)])
        )
        result = sync_provider.chat("custom-v1", [UserMessage(content="Hi")])
        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(1000 * 5e-6)

    def test_unknown_model_none_cost(self, sync_provider: GroqProvider, respx_mock: respx.MockRouter) -> None:
        completion = {
            "model": "totally-unknown",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "x"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        respx_mock.post(_URL).mock(return_value=httpx.Response(200, json=completion))
        result = sync_provider.chat("totally-unknown", [UserMessage(content="Hi")])
        assert result.cost is None


# MARK: Aclose & Preload


class TestAclose:
    async def test_closes_client(
        self, async_provider: GroqProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok(completion, respx_mock)
        await async_provider.achat(MODEL, [UserMessage(content="Hi")])
        assert async_provider._async_client is not None
        await async_provider.aclose()
        assert async_provider._async_client is None

    async def test_noop_when_no_client(self, async_provider: GroqProvider) -> None:
        await async_provider.aclose()


class TestPreload:
    def test_preload(self) -> None:
        preload()


# MARK: Provider Params


class TestProviderParams:
    def test_all_params(
        self, sync_provider: GroqProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(completion, respx_mock)
        sync_provider.chat(
            MODEL, [UserMessage(content="Hi")], provider_params=GroqParams(service_tier="flex", seed=42, user="u1")
        )
        body = json.loads(route.calls.last.request.content)
        assert body["service_tier"] == "flex"
        assert body["seed"] == 42
        assert body["user"] == "u1"

    def test_empty_params(
        self, sync_provider: GroqProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(completion, respx_mock)
        sync_provider.chat(MODEL, [UserMessage(content="Hi")], provider_params=GroqParams())
        body = json.loads(route.calls.last.request.content)
        assert "service_tier" not in body

    def test_params_reasoning_effort(
        self, sync_provider: GroqProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(completion, respx_mock)
        sync_provider.chat(MODEL, [UserMessage(content="Hi")], provider_params=GroqParams(reasoning_effort="high"))
        body = json.loads(route.calls.last.request.content)
        assert body["reasoning_effort"] == "high"
        assert body["include_reasoning"] is True

    def test_params_reasoning_effort_none(
        self, sync_provider: GroqProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(completion, respx_mock)
        sync_provider.chat(MODEL, [UserMessage(content="Hi")], provider_params=GroqParams(reasoning_effort="none"))
        body = json.loads(route.calls.last.request.content)
        assert body["reasoning_effort"] == "none"
        assert "include_reasoning" not in body
