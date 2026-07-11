"""Tests for the OpenAI provider (SDK-lite, respx)."""

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
import respx
from pytest_mock import MockerFixture

from lmux.cost import ModelPricing, PricingTier
from lmux.exceptions import AuthenticationError, InvalidRequestError, NotFoundError, ProviderError
from lmux.types import (
    FunctionDefinition,
    JsonObjectResponseFormat,
    ResponseInputMessage,
    Tool,
    UserMessage,
)
from lmux_openai import preload
from lmux_openai.params import OpenAIParams
from lmux_openai.provider import OpenAIProvider

_CHAT_URL = "https://api.openai.com/v1/chat/completions"
_EMBED_URL = "https://api.openai.com/v1/embeddings"
_RESPONSES_URL = "https://api.openai.com/v1/responses"
MODEL = "gpt-4o"


# MARK: Shared Fixtures


class FakeAuth:
    def get_credentials(self) -> str:
        return "sk-fake-key"

    async def aget_credentials(self) -> str:
        return "sk-fake-key"


@pytest.fixture
def fake_auth() -> FakeAuth:
    return FakeAuth()


@pytest.fixture
def sync_provider(fake_auth: FakeAuth) -> OpenAIProvider:
    return OpenAIProvider(auth=fake_auth)


@pytest.fixture
def async_provider(fake_auth: FakeAuth) -> OpenAIProvider:
    return OpenAIProvider(auth=fake_auth)


def _completion(model: str = MODEL) -> dict[str, Any]:
    return {
        "id": "chatcmpl-123",
        "model": model,
        "object": "chat.completion",
        "created": 1,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hello!"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
def completion() -> dict[str, Any]:
    return _completion()


@pytest.fixture
def embedding() -> dict[str, Any]:
    return {
        "object": "list",
        "model": "text-embedding-3-small",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


@pytest.fixture
def responses_body() -> dict[str, Any]:
    return {
        "id": "resp_123",
        "model": MODEL,
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello!"}]}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _sse_stream(model: str = MODEL) -> bytes:
    chunks = [
        {"model": model, "choices": [{"index": 0, "finish_reason": None, "delta": {"content": "Hel"}}]},
        {"model": model, "choices": [{"index": 0, "finish_reason": None, "delta": {"content": "lo!"}}]},
        {
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    ]
    lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
    return ("\n\n".join(lines) + "\n\n").encode()


def _ok(body: dict[str, Any], url: str, respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(url).mock(return_value=httpx.Response(200, json=body))


# MARK: Chat


class TestChat:
    def test_basic(
        self, sync_provider: OpenAIProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(completion, _CHAT_URL, respx_mock)
        result = sync_provider.chat(MODEL, [UserMessage(content="Hi")])
        assert result.content == "Hello!"
        assert result.model == MODEL
        assert result.provider == "openai"
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert route.called

    def test_request_body(
        self, sync_provider: OpenAIProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(completion, _CHAT_URL, respx_mock)
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

    def test_max_completion_tokens_for_reasoning_models(
        self, sync_provider: OpenAIProvider, respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(_completion("gpt-5"), _CHAT_URL, respx_mock)
        sync_provider.chat("gpt-5", [UserMessage(content="Hi")], max_tokens=50)
        body = json.loads(route.calls.last.request.content)
        assert body["max_completion_tokens"] == 50
        assert "max_tokens" not in body

    def test_tools_and_choice(
        self, sync_provider: OpenAIProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(completion, _CHAT_URL, respx_mock)
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
        self, sync_provider: OpenAIProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(completion, _CHAT_URL, respx_mock)
        sync_provider.chat(MODEL, [UserMessage(content="Hi")], reasoning_effort="medium")
        body = json.loads(route.calls.last.request.content)
        assert body["reasoning_effort"] == "medium"

    def test_status_error_mapped(self, sync_provider: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_CHAT_URL).mock(return_value=httpx.Response(400, json={"error": {"message": "bad"}}))
        with pytest.raises(InvalidRequestError):
            sync_provider.chat(MODEL, [UserMessage(content="Hi")])

    def test_transport_error_mapped(self, sync_provider: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_CHAT_URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            sync_provider.chat(MODEL, [UserMessage(content="Hi")])

    def test_cost_calculated(
        self, sync_provider: OpenAIProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok(completion, _CHAT_URL, respx_mock)
        result = sync_provider.chat(MODEL, [UserMessage(content="Hi")])
        assert result.cost is not None
        assert result.cost.total_cost > 0


# MARK: Achat


class TestAchat:
    async def test_basic(
        self, async_provider: OpenAIProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok(completion, _CHAT_URL, respx_mock)
        result = await async_provider.achat(MODEL, [UserMessage(content="Hi")])
        assert result.content == "Hello!"
        assert result.provider == "openai"

    async def test_status_error_mapped(self, async_provider: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_CHAT_URL).mock(return_value=httpx.Response(401, json={"error": {"message": "no"}}))
        with pytest.raises(AuthenticationError):
            await async_provider.achat(MODEL, [UserMessage(content="Hi")])

    async def test_transport_error_mapped(self, async_provider: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_CHAT_URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            await async_provider.achat(MODEL, [UserMessage(content="Hi")])


# MARK: Data Residency


class TestDataResidency:
    def test_uplift_applied(self, fake_auth: FakeAuth, respx_mock: respx.MockRouter) -> None:
        _ok(_completion("gpt-5.4"), _CHAT_URL, respx_mock)
        base = OpenAIProvider(auth=fake_auth).chat("gpt-5.4", [UserMessage(content="Hi")])
        _ok(_completion("gpt-5.4"), _CHAT_URL, respx_mock)
        residency = OpenAIProvider(auth=fake_auth, data_residency=True).chat("gpt-5.4", [UserMessage(content="Hi")])
        assert base.cost is not None
        assert residency.cost is not None
        assert residency.cost.total_cost == pytest.approx(base.cost.total_cost * 1.1)

    def test_uplift_not_applied_for_non_regional_model(
        self, fake_auth: FakeAuth, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok(completion, _CHAT_URL, respx_mock)
        provider = OpenAIProvider(auth=fake_auth, data_residency=True)
        result = provider.chat(MODEL, [UserMessage(content="Hi")])
        _ok(completion, _CHAT_URL, respx_mock)
        base_result = OpenAIProvider(auth=fake_auth).chat(MODEL, [UserMessage(content="Hi")])
        assert result.cost is not None
        assert base_result.cost is not None
        assert result.cost.total_cost == base_result.cost.total_cost


# MARK: ChatStream


class TestChatStream:
    def test_yields_and_costs(self, sync_provider: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=_sse_stream()))
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

    def test_residency_uplift_in_stream(self, fake_auth: FakeAuth, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=_sse_stream("gpt-5.4")))
        provider = OpenAIProvider(auth=fake_auth, data_residency=True)
        chunks = list(provider.chat_stream("gpt-5.4", [UserMessage(content="Hi")]))
        assert chunks[2].cost is not None
        assert chunks[2].cost.total_cost > 0

    def test_status_error_on_open(self, sync_provider: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_CHAT_URL).mock(return_value=httpx.Response(500, json={"error": {"message": "boom"}}))
        with pytest.raises(ProviderError):
            list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))

    def test_malformed_chunk_mapped(self, sync_provider: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=b"data: {not json}\n\n"))
        with pytest.raises(ProviderError):
            list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))

    def test_client_init_failure(self, fake_auth: FakeAuth, mocker: MockerFixture) -> None:
        mocker.patch("lmux_openai.provider.create_sync_client", side_effect=RuntimeError("boom"))
        provider = OpenAIProvider(auth=fake_auth)
        with pytest.raises(ProviderError, match="boom"):
            list(provider.chat_stream(MODEL, [UserMessage(content="Hi")]))

    def test_stream_without_done(self, sync_provider: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
        chunk = {"model": MODEL, "choices": [{"index": 0, "finish_reason": "stop", "delta": {"content": "x"}}]}
        respx_mock.post(_CHAT_URL).mock(
            return_value=httpx.Response(200, content=(f"data: {json.dumps(chunk)}\n\n").encode())
        )
        chunks = list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))
        assert [c.delta for c in chunks] == ["x"]


# MARK: AchatStream


class TestAchatStream:
    async def test_yields_and_costs(self, async_provider: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=_sse_stream()))
        chunks = [c async for c in async_provider.achat_stream(MODEL, [UserMessage(content="Hi")])]
        assert [c.delta for c in chunks[:2]] == ["Hel", "lo!"]
        assert chunks[2].cost is not None

    async def test_status_error_on_open(self, async_provider: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_CHAT_URL).mock(return_value=httpx.Response(500, json={"error": {"message": "boom"}}))
        with pytest.raises(ProviderError):
            async for _ in async_provider.achat_stream(MODEL, [UserMessage(content="Hi")]):
                pass  # pragma: no cover

    async def test_malformed_chunk_mapped(self, async_provider: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=b"data: {nope}\n\n"))
        with pytest.raises(ProviderError):
            async for _ in async_provider.achat_stream(MODEL, [UserMessage(content="Hi")]):
                pass  # pragma: no cover

    async def test_client_init_failure(self, fake_auth: FakeAuth, mocker: MockerFixture) -> None:
        mocker.patch("lmux_openai.provider.create_async_client", side_effect=RuntimeError("boom"))
        provider = OpenAIProvider(auth=fake_auth)
        with pytest.raises(ProviderError, match="boom"):
            async for _ in provider.achat_stream(MODEL, [UserMessage(content="Hi")]):
                pass  # pragma: no cover

    async def test_stream_without_done(self, async_provider: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
        chunk = {"model": MODEL, "choices": [{"index": 0, "finish_reason": "stop", "delta": {"content": "x"}}]}
        respx_mock.post(_CHAT_URL).mock(
            return_value=httpx.Response(200, content=(f"data: {json.dumps(chunk)}\n\n").encode())
        )
        chunks = [c async for c in async_provider.achat_stream(MODEL, [UserMessage(content="Hi")])]
        assert [c.delta for c in chunks] == ["x"]


# MARK: Embeddings


class TestEmbed:
    def test_basic(
        self, sync_provider: OpenAIProvider, embedding: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(embedding, _EMBED_URL, respx_mock)
        result = sync_provider.embed("text-embedding-3-small", "hello")
        assert result.embeddings == [[0.1, 0.2, 0.3]]
        assert result.cost is not None
        body = json.loads(route.calls.last.request.content)
        assert body == {"model": "text-embedding-3-small", "input": "hello"}

    def test_dimensions_and_params(
        self, sync_provider: OpenAIProvider, embedding: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(embedding, _EMBED_URL, respx_mock)
        sync_provider.embed(
            "text-embedding-3-small", ["a", "b"], dimensions=256, provider_params=OpenAIParams(user="u1")
        )
        body = json.loads(route.calls.last.request.content)
        assert body == {"model": "text-embedding-3-small", "input": ["a", "b"], "dimensions": 256, "user": "u1"}

    def test_status_error_mapped(self, sync_provider: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_EMBED_URL).mock(return_value=httpx.Response(404, json={"error": {"message": "no"}}))
        with pytest.raises(NotFoundError):
            sync_provider.embed("text-embedding-3-small", "hello")

    def test_transport_error_mapped(self, sync_provider: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_EMBED_URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            sync_provider.embed("text-embedding-3-small", "hello")

    async def test_aembed_basic(
        self, async_provider: OpenAIProvider, embedding: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(embedding, _EMBED_URL, respx_mock)
        result = await async_provider.aembed("text-embedding-3-small", "hello", dimensions=128)
        assert result.embeddings == [[0.1, 0.2, 0.3]]
        body = json.loads(route.calls.last.request.content)
        assert body["dimensions"] == 128

    async def test_aembed_transport_error(self, async_provider: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_EMBED_URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            await async_provider.aembed("text-embedding-3-small", "hello")


# MARK: Responses API


class TestCreateResponse:
    def test_basic(
        self, sync_provider: OpenAIProvider, responses_body: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(responses_body, _RESPONSES_URL, respx_mock)
        result = sync_provider.create_response(MODEL, "Say hi")
        assert result.output_text == "Hello!"
        assert result.id == "resp_123"
        assert result.cost is not None
        body = json.loads(route.calls.last.request.content)
        assert body == {"model": MODEL, "input": "Say hi", "stream": False}

    def test_input_list_and_params(
        self, sync_provider: OpenAIProvider, responses_body: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(responses_body, _RESPONSES_URL, respx_mock)
        sync_provider.create_response(
            MODEL,
            [ResponseInputMessage(role="user", content="Hi")],
            provider_params=OpenAIParams(reasoning_effort="high", seed=1),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["input"] == [{"role": "user", "content": "Hi"}]
        assert body["reasoning"] == {"effort": "high"}
        assert body["seed"] == 1
        assert "reasoning_effort" not in body

    def test_params_without_reasoning_effort(
        self, sync_provider: OpenAIProvider, responses_body: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(responses_body, _RESPONSES_URL, respx_mock)
        sync_provider.create_response(MODEL, "Hi", provider_params=OpenAIParams(seed=1))
        body = json.loads(route.calls.last.request.content)
        assert body["seed"] == 1
        assert "reasoning" not in body

    def test_status_error_mapped(self, sync_provider: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_RESPONSES_URL).mock(return_value=httpx.Response(400, json={"error": {"message": "bad"}}))
        with pytest.raises(InvalidRequestError):
            sync_provider.create_response(MODEL, "Hi")

    def test_transport_error_mapped(self, sync_provider: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_RESPONSES_URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            sync_provider.create_response(MODEL, "Hi")

    async def test_acreate_basic(
        self, async_provider: OpenAIProvider, responses_body: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok(responses_body, _RESPONSES_URL, respx_mock)
        result = await async_provider.acreate_response(MODEL, "Say hi")
        assert result.output_text == "Hello!"

    async def test_acreate_transport_error(self, async_provider: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_RESPONSES_URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            await async_provider.acreate_response(MODEL, "Hi")


# MARK: Client Management


class TestClientManagement:
    def test_sync_client_reused(
        self, sync_provider: OpenAIProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok(completion, _CHAT_URL, respx_mock)
        sync_provider.chat(MODEL, [UserMessage(content="a")])
        client = sync_provider._sync_client
        sync_provider.chat(MODEL, [UserMessage(content="b")])
        assert sync_provider._sync_client is client

    async def test_async_client_reused(
        self, async_provider: OpenAIProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok(completion, _CHAT_URL, respx_mock)
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
        provider = OpenAIProvider(auth=fake_auth, base_url="https://custom.api/v1")
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.called

    def test_timeout_and_retries(
        self, fake_auth: FakeAuth, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok(completion, _CHAT_URL, respx_mock)
        provider = OpenAIProvider(auth=fake_auth, timeout=30.0, max_retries=5)
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert provider._sync_client is not None
        assert provider._sync_client.timeout.read == 30.0

    def test_sync_init_failure_mapped(self, fake_auth: FakeAuth, mocker: MockerFixture) -> None:
        mocker.patch("lmux_openai.provider.create_sync_client", side_effect=RuntimeError("connection refused"))
        provider = OpenAIProvider(auth=fake_auth)
        with pytest.raises(ProviderError, match="connection refused"):
            provider.chat(MODEL, [UserMessage(content="Hi")])

    async def test_async_init_failure_mapped(self, fake_auth: FakeAuth, mocker: MockerFixture) -> None:
        mocker.patch("lmux_openai.provider.create_async_client", side_effect=RuntimeError("connection refused"))
        provider = OpenAIProvider(auth=fake_auth)
        with pytest.raises(ProviderError, match="connection refused"):
            await provider.achat(MODEL, [UserMessage(content="Hi")])

    async def test_async_client_recreated_on_new_loop(self, fake_auth: FakeAuth, mocker: MockerFixture) -> None:
        c1, c2 = MagicMock(), MagicMock()
        create = mocker.patch("lmux_openai.provider.create_async_client", side_effect=[c1, c2])
        get_loop = mocker.patch("lmux_openai.provider.asyncio.get_running_loop")
        loop1, loop2 = asyncio.new_event_loop(), asyncio.new_event_loop()
        get_loop.side_effect = [loop1, loop2]
        provider = OpenAIProvider(auth=fake_auth)
        r1 = await provider._get_async_client()
        r2 = await provider._get_async_client()
        assert (r1, r2) == (c1, c2)
        assert create.call_count == 2
        loop1.close()
        loop2.close()


# MARK: Register Pricing


class TestRegisterPricing:
    def test_custom_for_unknown_model(self, sync_provider: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
        completion = _completion("custom-v1")
        completion["usage"] = {"prompt_tokens": 1000, "completion_tokens": 500}
        respx_mock.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=completion))
        sync_provider.register_pricing(
            "custom-v1", ModelPricing(tiers=[PricingTier(input_cost_per_token=5e-6, output_cost_per_token=15e-6)])
        )
        result = sync_provider.chat("custom-v1", [UserMessage(content="Hi")])
        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(1000 * 5e-6)

    def test_unknown_model_none_cost(self, sync_provider: OpenAIProvider, respx_mock: respx.MockRouter) -> None:
        completion = _completion("totally-unknown")
        respx_mock.post(_CHAT_URL).mock(return_value=httpx.Response(200, json=completion))
        result = sync_provider.chat("totally-unknown", [UserMessage(content="Hi")])
        assert result.cost is None


# MARK: Aclose & Preload


class TestAclose:
    async def test_closes_client(
        self, async_provider: OpenAIProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok(completion, _CHAT_URL, respx_mock)
        await async_provider.achat(MODEL, [UserMessage(content="Hi")])
        assert async_provider._async_client is not None
        await async_provider.aclose()
        assert async_provider._async_client is None

    async def test_noop_when_no_client(self, async_provider: OpenAIProvider) -> None:
        await async_provider.aclose()


class TestPreload:
    def test_preload(self) -> None:
        preload()


# MARK: Provider Params


class TestProviderParams:
    def test_all_params(
        self, sync_provider: OpenAIProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(completion, _CHAT_URL, respx_mock)
        sync_provider.chat(
            MODEL, [UserMessage(content="Hi")], provider_params=OpenAIParams(service_tier="flex", seed=42, user="u1")
        )
        body = json.loads(route.calls.last.request.content)
        assert body["service_tier"] == "flex"
        assert body["seed"] == 42
        assert body["user"] == "u1"

    def test_empty_params(
        self, sync_provider: OpenAIProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(completion, _CHAT_URL, respx_mock)
        sync_provider.chat(MODEL, [UserMessage(content="Hi")], provider_params=OpenAIParams())
        body = json.loads(route.calls.last.request.content)
        assert "service_tier" not in body

    def test_params_reasoning_effort_overrides(
        self, sync_provider: OpenAIProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(completion, _CHAT_URL, respx_mock)
        sync_provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            reasoning_effort="low",
            provider_params=OpenAIParams(reasoning_effort="high"),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["reasoning_effort"] == "high"
