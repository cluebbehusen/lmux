"""Tests for the Azure AI Foundry provider (SDK-lite, respx)."""

import asyncio
import json
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
import respx
from pytest_mock import MockerFixture

from lmux.cost import ModelPricing, PricingTier
from lmux.exceptions import AuthenticationError, InvalidRequestError, ProviderError
from lmux.types import (
    CachePointContent,
    FunctionDefinition,
    ImageContent,
    JsonObjectResponseFormat,
    ResponseInputMessage,
    TextContent,
    Tool,
    UserMessage,
)
from lmux_azure_foundry import preload
from lmux_azure_foundry.auth import AzureAdToken
from lmux_azure_foundry.params import AzureFoundryParams
from lmux_azure_foundry.provider import AzureFoundryProvider

ENDPOINT = "https://test.openai.azure.com/"
API_VERSION = "2025-04-01-preview"


def _chat_url(model: str) -> str:
    return f"https://test.openai.azure.com/openai/deployments/{model}/chat/completions"


def _emb_url(model: str) -> str:
    return f"https://test.openai.azure.com/openai/deployments/{model}/embeddings"


RESPONSES_URL = "https://test.openai.azure.com/openai/responses"


# MARK: Shared Fixtures


class FakeAuth:
    def get_credentials(self) -> str:
        return "fake-api-key"

    async def aget_credentials(self) -> str:
        return "fake-api-key"


class FakeTokenAuth:
    def get_credentials(self) -> AzureAdToken:
        return AzureAdToken(token="fake-ad-token")  # noqa: S106

    async def aget_credentials(self) -> AzureAdToken:
        return AzureAdToken(token="fake-ad-token")  # noqa: S106


class FakeTokenProviderAuth:
    @staticmethod
    def _provider() -> str:
        return "fresh-token"

    def get_credentials(self) -> Callable[[], str]:
        return self._provider

    async def aget_credentials(self) -> Callable[[], str]:
        return self._provider


@pytest.fixture
def fake_auth() -> FakeAuth:
    return FakeAuth()


@pytest.fixture
def sync_provider(fake_auth: FakeAuth) -> AzureFoundryProvider:
    return AzureFoundryProvider(endpoint=ENDPOINT, auth=fake_auth)


@pytest.fixture
def async_provider(fake_auth: FakeAuth) -> AzureFoundryProvider:
    return AzureFoundryProvider(endpoint=ENDPOINT, auth=fake_auth)


@pytest.fixture
def sync_create_raises(mocker: MockerFixture) -> MagicMock:
    return mocker.patch(
        "lmux_azure_foundry.provider.create_sync_client", side_effect=RuntimeError("client init failed")
    )


@pytest.fixture
def async_create_raises(mocker: MockerFixture) -> MagicMock:
    return mocker.patch(
        "lmux_azure_foundry.provider.create_async_client", side_effect=RuntimeError("client init failed")
    )


@pytest.fixture
def async_create_two_clients(mocker: MockerFixture) -> tuple[MagicMock, MagicMock, MagicMock]:
    c1, c2 = MagicMock(), MagicMock()
    create = mocker.patch("lmux_azure_foundry.provider.create_async_client", side_effect=[c1, c2])
    return create, c1, c2


@pytest.fixture
def mock_get_running_loop(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_azure_foundry.provider.asyncio.get_running_loop")


@pytest.fixture
def completion() -> dict[str, Any]:
    return {
        "id": "chatcmpl-123",
        "model": "gpt-4o",
        "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hello!"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
def embedding_response() -> dict[str, Any]:
    return {
        "model": "text-embedding-3-small",
        "object": "list",
        "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3], "object": "embedding"}],
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


@pytest.fixture
def responses_body() -> dict[str, Any]:
    return {
        "id": "resp_123",
        "model": "gpt-5-pro",
        "output": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi!"}]},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _sse_stream() -> bytes:
    chunks = [
        {"model": "gpt-4o", "choices": [{"index": 0, "finish_reason": None, "delta": {"content": "Hel"}}]},
        {"model": "gpt-4o", "choices": [{"index": 0, "finish_reason": None, "delta": {"content": "lo!"}}]},
        {
            "model": "gpt-4o",
            "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    ]
    lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
    return ("\n\n".join(lines) + "\n\n").encode()


def _ok_chat(completion: dict[str, Any], respx_mock: respx.MockRouter, model: str = "gpt-4o") -> respx.Route:
    return respx_mock.post(_chat_url(model)).mock(return_value=httpx.Response(200, json=completion))


# MARK: Chat


class TestChat:
    def test_basic(
        self, sync_provider: AzureFoundryProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_chat(completion, respx_mock)
        result = sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])
        assert result.content == "Hello!"
        assert result.model == "gpt-4o"
        assert result.provider == "azure-foundry"
        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert route.called

    def test_request_url_query_and_headers(
        self, sync_provider: AzureFoundryProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_chat(completion, respx_mock)
        sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])
        request = route.calls.last.request
        assert request.url.params.get("api-version") == API_VERSION
        assert request.headers.get("api-key") == "fake-api-key"
        assert "authorization" not in request.headers

    def test_request_body(
        self, sync_provider: AzureFoundryProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_chat(completion, respx_mock)
        sync_provider.chat(
            "gpt-4o", [UserMessage(content="Hi")], temperature=0.5, max_tokens=100, top_p=0.9, stop=["END"]
        )
        body = json.loads(route.calls.last.request.content)
        assert body == {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "temperature": 0.5,
            "max_tokens": 100,
            "top_p": 0.9,
            "stop": ["END"],
        }

    def test_max_completion_tokens_for_newer_models(
        self, sync_provider: AzureFoundryProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(_chat_url("gpt-5-mini")).mock(return_value=httpx.Response(200, json=completion))
        sync_provider.chat("gpt-5-mini", [UserMessage(content="Hi")], max_tokens=100)
        body = json.loads(route.calls.last.request.content)
        assert body["max_completion_tokens"] == 100
        assert "max_tokens" not in body

    def test_tools_choice_and_format(
        self, sync_provider: AzureFoundryProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_chat(completion, respx_mock)
        sync_provider.chat(
            "gpt-4o",
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
        self, sync_provider: AzureFoundryProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(_chat_url("o3")).mock(return_value=httpx.Response(200, json=completion))
        sync_provider.chat("o3", [UserMessage(content="Hi")], reasoning_effort="high")
        body = json.loads(route.calls.last.request.content)
        assert body["reasoning_effort"] == "high"

    def test_prompt_cache_params_sent(
        self, sync_provider: AzureFoundryProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(_chat_url("gpt-4o")).mock(return_value=httpx.Response(200, json=completion))
        sync_provider.chat(
            "gpt-4o",
            [UserMessage(content="Hi")],
            provider_params=AzureFoundryParams(prompt_cache_key="tenant-42", prompt_cache_retention="24h"),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["prompt_cache_key"] == "tenant-42"
        assert body["prompt_cache_retention"] == "24h"

    def test_status_error_mapped(self, sync_provider: AzureFoundryProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_chat_url("gpt-4o")).mock(return_value=httpx.Response(400, json={"error": {"message": "bad"}}))
        with pytest.raises(InvalidRequestError):
            sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])

    def test_transport_error_mapped(self, sync_provider: AzureFoundryProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_chat_url("gpt-4o")).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])

    def test_non_json_body_mapped(self, sync_provider: AzureFoundryProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_chat_url("gpt-4o")).mock(return_value=httpx.Response(200, content=b"not json"))
        with pytest.raises(ProviderError):
            sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])

    def test_client_init_failure(self, fake_auth: FakeAuth, sync_create_raises: MagicMock) -> None:
        provider = AzureFoundryProvider(endpoint=ENDPOINT, auth=fake_auth)
        with pytest.raises(ProviderError, match="client init failed"):
            provider.chat("gpt-4o", [UserMessage(content="Hi")])
        sync_create_raises.assert_called_once()

    def test_cost_calculated(
        self, sync_provider: AzureFoundryProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok_chat(completion, respx_mock)
        result = sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])
        assert result.cost is not None
        assert result.cost.total_cost > 0

    def test_data_zone_multiplier(
        self, sync_provider: AzureFoundryProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok_chat(completion, respx_mock)
        result_global = sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])
        result_dz = sync_provider.chat(
            "gpt-4o", [UserMessage(content="Hi")], provider_params=AzureFoundryParams(deployment_type="data_zone")
        )
        assert result_global.cost is not None
        assert result_dz.cost is not None
        assert result_dz.cost.total_cost == pytest.approx(result_global.cost.total_cost * 1.1)

    def test_regional_multiplier(
        self, sync_provider: AzureFoundryProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok_chat(completion, respx_mock)
        result_global = sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])
        result_regional = sync_provider.chat(
            "gpt-4o", [UserMessage(content="Hi")], provider_params=AzureFoundryParams(deployment_type="regional")
        )
        assert result_global.cost is not None
        assert result_regional.cost is not None
        assert result_regional.cost.total_cost == pytest.approx(result_global.cost.total_cost * 1.1)

    def test_no_multiplier_with_global(
        self, sync_provider: AzureFoundryProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok_chat(completion, respx_mock)
        result1 = sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])
        result2 = sync_provider.chat(
            "gpt-4o", [UserMessage(content="Hi")], provider_params=AzureFoundryParams(deployment_type="global")
        )
        assert result1.cost is not None
        assert result2.cost is not None
        assert result1.cost.total_cost == pytest.approx(result2.cost.total_cost)

    def test_no_multiplier_with_none_cost(
        self, sync_provider: AzureFoundryProvider, respx_mock: respx.MockRouter
    ) -> None:
        unknown = {
            "model": "totally-unknown-model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hi"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        respx_mock.post(_chat_url("totally-unknown-model")).mock(return_value=httpx.Response(200, json=unknown))
        result = sync_provider.chat(
            "totally-unknown-model",
            [UserMessage(content="Hi")],
            provider_params=AzureFoundryParams(deployment_type="data_zone"),
        )
        assert result.cost is None


# MARK: Achat


class TestAchat:
    async def test_basic(
        self, async_provider: AzureFoundryProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok_chat(completion, respx_mock)
        result = await async_provider.achat("gpt-4o", [UserMessage(content="Hi")])
        assert result.content == "Hello!"
        assert result.provider == "azure-foundry"

    async def test_request_query_and_headers(
        self, async_provider: AzureFoundryProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_chat(completion, respx_mock)
        await async_provider.achat("gpt-4o", [UserMessage(content="Hi")])
        request = route.calls.last.request
        assert request.url.params.get("api-version") == API_VERSION
        assert request.headers.get("api-key") == "fake-api-key"

    async def test_status_error_mapped(
        self, async_provider: AzureFoundryProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_chat_url("gpt-4o")).mock(return_value=httpx.Response(401, json={"error": {"message": "no"}}))
        with pytest.raises(AuthenticationError):
            await async_provider.achat("gpt-4o", [UserMessage(content="Hi")])

    async def test_transport_error_mapped(
        self, async_provider: AzureFoundryProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_chat_url("gpt-4o")).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            await async_provider.achat("gpt-4o", [UserMessage(content="Hi")])

    async def test_non_json_body_mapped(
        self, async_provider: AzureFoundryProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_chat_url("gpt-4o")).mock(return_value=httpx.Response(200, content=b"not json"))
        with pytest.raises(ProviderError):
            await async_provider.achat("gpt-4o", [UserMessage(content="Hi")])

    async def test_client_init_failure(self, fake_auth: FakeAuth, async_create_raises: MagicMock) -> None:
        provider = AzureFoundryProvider(endpoint=ENDPOINT, auth=fake_auth)
        with pytest.raises(ProviderError, match="client init failed"):
            await provider.achat("gpt-4o", [UserMessage(content="Hi")])
        async_create_raises.assert_called_once()


# MARK: ChatStream


class TestChatStream:
    def test_yields_and_costs(self, sync_provider: AzureFoundryProvider, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(_chat_url("gpt-4o")).mock(return_value=httpx.Response(200, content=_sse_stream()))
        chunks = list(sync_provider.chat_stream("gpt-4o", [UserMessage(content="Hi")]))
        assert [c.delta for c in chunks[:2]] == ["Hel", "lo!"]
        assert chunks[2].finish_reason == "stop"
        assert chunks[0].cost is None
        assert chunks[2].cost is not None
        assert chunks[2].cost.total_cost > 0
        body = json.loads(route.calls.last.request.content)
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}

    def test_data_zone_multiplier(self, sync_provider: AzureFoundryProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_chat_url("gpt-4o")).mock(return_value=httpx.Response(200, content=_sse_stream()))
        chunks_global = list(sync_provider.chat_stream("gpt-4o", [UserMessage(content="Hi")]))
        respx_mock.post(_chat_url("gpt-4o")).mock(return_value=httpx.Response(200, content=_sse_stream()))
        chunks_dz = list(
            sync_provider.chat_stream(
                "gpt-4o", [UserMessage(content="Hi")], provider_params=AzureFoundryParams(deployment_type="data_zone")
            )
        )
        assert chunks_global[2].cost is not None
        assert chunks_dz[2].cost is not None
        assert chunks_dz[2].cost.total_cost == pytest.approx(chunks_global[2].cost.total_cost * 1.1)

    def test_status_error_on_open(self, sync_provider: AzureFoundryProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_chat_url("gpt-4o")).mock(return_value=httpx.Response(500, json={"error": {"message": "boom"}}))
        with pytest.raises(ProviderError):
            list(sync_provider.chat_stream("gpt-4o", [UserMessage(content="Hi")]))

    def test_malformed_chunk_mapped(self, sync_provider: AzureFoundryProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_chat_url("gpt-4o")).mock(return_value=httpx.Response(200, content=b"data: {nope}\n\n"))
        with pytest.raises(ProviderError):
            list(sync_provider.chat_stream("gpt-4o", [UserMessage(content="Hi")]))

    def test_mid_stream_error_raises_after_partial(
        self, sync_provider: AzureFoundryProvider, respx_mock: respx.MockRouter
    ) -> None:
        first = {"model": "gpt-4o", "choices": [{"index": 0, "finish_reason": None, "delta": {"content": "Hel"}}]}
        error = {"error": {"message": "mid-stream boom"}}
        sse = (f"data: {json.dumps(first)}\n\ndata: {json.dumps(error)}\n\n").encode()
        respx_mock.post(_chat_url("gpt-4o")).mock(return_value=httpx.Response(200, content=sse))
        stream = sync_provider.chat_stream("gpt-4o", [UserMessage(content="Hi")])
        assert next(stream).delta == "Hel"  # partial output arrives first
        with pytest.raises(ProviderError, match="mid-stream boom"):
            next(stream)

    def test_client_init_failure(self, fake_auth: FakeAuth, sync_create_raises: MagicMock) -> None:
        provider = AzureFoundryProvider(endpoint=ENDPOINT, auth=fake_auth)
        with pytest.raises(ProviderError, match="client init failed"):
            list(provider.chat_stream("gpt-4o", [UserMessage(content="Hi")]))
        sync_create_raises.assert_called_once()

    def test_stream_without_done(self, sync_provider: AzureFoundryProvider, respx_mock: respx.MockRouter) -> None:
        chunk = {"model": "gpt-4o", "choices": [{"index": 0, "finish_reason": "stop", "delta": {"content": "x"}}]}
        respx_mock.post(_chat_url("gpt-4o")).mock(
            return_value=httpx.Response(200, content=(f"data: {json.dumps(chunk)}\n\n").encode())
        )
        chunks = list(sync_provider.chat_stream("gpt-4o", [UserMessage(content="Hi")]))
        assert [c.delta for c in chunks] == ["x"]


# MARK: AchatStream


class TestAchatStream:
    async def test_yields_and_costs(self, async_provider: AzureFoundryProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_chat_url("gpt-4o")).mock(return_value=httpx.Response(200, content=_sse_stream()))
        chunks = [c async for c in async_provider.achat_stream("gpt-4o", [UserMessage(content="Hi")])]
        assert [c.delta for c in chunks[:2]] == ["Hel", "lo!"]
        assert chunks[2].cost is not None

    async def test_status_error_on_open(
        self, async_provider: AzureFoundryProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_chat_url("gpt-4o")).mock(return_value=httpx.Response(500, json={"error": {"message": "boom"}}))
        with pytest.raises(ProviderError):
            async for _ in async_provider.achat_stream("gpt-4o", [UserMessage(content="Hi")]):
                pass  # pragma: no cover

    async def test_malformed_chunk_mapped(
        self, async_provider: AzureFoundryProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_chat_url("gpt-4o")).mock(return_value=httpx.Response(200, content=b"data: {nope}\n\n"))
        with pytest.raises(ProviderError):
            async for _ in async_provider.achat_stream("gpt-4o", [UserMessage(content="Hi")]):
                pass  # pragma: no cover

    async def test_mid_stream_error_raises_after_partial(
        self, async_provider: AzureFoundryProvider, respx_mock: respx.MockRouter
    ) -> None:
        first = {"model": "gpt-4o", "choices": [{"index": 0, "finish_reason": None, "delta": {"content": "Hel"}}]}
        error = {"error": {"message": "mid-stream boom"}}
        sse = (f"data: {json.dumps(first)}\n\ndata: {json.dumps(error)}\n\n").encode()
        respx_mock.post(_chat_url("gpt-4o")).mock(return_value=httpx.Response(200, content=sse))
        stream = async_provider.achat_stream("gpt-4o", [UserMessage(content="Hi")])
        assert (await anext(stream)).delta == "Hel"  # partial output arrives first
        with pytest.raises(ProviderError, match="mid-stream boom"):
            await anext(stream)

    async def test_client_init_failure(self, fake_auth: FakeAuth, async_create_raises: MagicMock) -> None:
        provider = AzureFoundryProvider(endpoint=ENDPOINT, auth=fake_auth)
        with pytest.raises(ProviderError, match="client init failed"):
            async for _ in provider.achat_stream("gpt-4o", [UserMessage(content="Hi")]):
                pass  # pragma: no cover
        async_create_raises.assert_called_once()

    async def test_stream_without_done(
        self, async_provider: AzureFoundryProvider, respx_mock: respx.MockRouter
    ) -> None:
        chunk = {"model": "gpt-4o", "choices": [{"index": 0, "finish_reason": "stop", "delta": {"content": "x"}}]}
        respx_mock.post(_chat_url("gpt-4o")).mock(
            return_value=httpx.Response(200, content=(f"data: {json.dumps(chunk)}\n\n").encode())
        )
        chunks = [c async for c in async_provider.achat_stream("gpt-4o", [UserMessage(content="Hi")])]
        assert [c.delta for c in chunks] == ["x"]


# MARK: Embed


class TestEmbed:
    def test_basic(
        self, sync_provider: AzureFoundryProvider, embedding_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(_emb_url("text-embedding-3-small")).mock(
            return_value=httpx.Response(200, json=embedding_response)
        )
        result = sync_provider.embed("text-embedding-3-small", "hello")
        assert result.embeddings == [[0.1, 0.2, 0.3]]
        assert result.provider == "azure-foundry"
        body = json.loads(route.calls.last.request.content)
        assert body == {"model": "text-embedding-3-small", "input": "hello"}
        assert route.calls.last.request.url.params.get("api-version") == API_VERSION

    def test_list_input_and_dimensions(
        self, sync_provider: AzureFoundryProvider, embedding_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(_emb_url("text-embedding-3-small")).mock(
            return_value=httpx.Response(200, json=embedding_response)
        )
        sync_provider.embed("text-embedding-3-small", ["hello", "world"], dimensions=256)
        body = json.loads(route.calls.last.request.content)
        assert body == {"model": "text-embedding-3-small", "input": ["hello", "world"], "dimensions": 256}

    def test_provider_params(
        self, sync_provider: AzureFoundryProvider, embedding_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(_emb_url("text-embedding-3-small")).mock(
            return_value=httpx.Response(200, json=embedding_response)
        )
        sync_provider.embed(
            "text-embedding-3-small",
            "hello",
            provider_params=AzureFoundryParams(user="u1", prompt_cache_key="k", prompt_cache_retention="24h"),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["user"] == "u1"
        # prompt-cache fields are Chat-Completions-only and must not leak onto the embeddings body.
        assert "prompt_cache_key" not in body
        assert "prompt_cache_retention" not in body

    def test_status_error_mapped(self, sync_provider: AzureFoundryProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_emb_url("text-embedding-3-small")).mock(
            return_value=httpx.Response(400, json={"error": {"message": "bad"}})
        )
        with pytest.raises(InvalidRequestError):
            sync_provider.embed("text-embedding-3-small", "hello")

    def test_transport_error_mapped(self, sync_provider: AzureFoundryProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_emb_url("text-embedding-3-small")).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            sync_provider.embed("text-embedding-3-small", "hello")

    def test_client_init_failure(self, fake_auth: FakeAuth, sync_create_raises: MagicMock) -> None:
        provider = AzureFoundryProvider(endpoint=ENDPOINT, auth=fake_auth)
        with pytest.raises(ProviderError, match="client init failed"):
            provider.embed("text-embedding-3-small", "hello")
        sync_create_raises.assert_called_once()

    def test_data_zone_multiplier(
        self, sync_provider: AzureFoundryProvider, embedding_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_emb_url("text-embedding-3-small")).mock(
            return_value=httpx.Response(200, json=embedding_response)
        )
        result_global = sync_provider.embed("text-embedding-3-small", "hello")
        result_dz = sync_provider.embed(
            "text-embedding-3-small", "hello", provider_params=AzureFoundryParams(deployment_type="data_zone")
        )
        assert result_global.cost is not None
        assert result_dz.cost is not None
        assert result_dz.cost.total_cost == pytest.approx(result_global.cost.total_cost * 1.1)

    async def test_aembed(
        self, async_provider: AzureFoundryProvider, embedding_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(_emb_url("text-embedding-3-small")).mock(
            return_value=httpx.Response(200, json=embedding_response)
        )
        result = await async_provider.aembed("text-embedding-3-small", "hello", dimensions=128)
        assert result.embeddings == [[0.1, 0.2, 0.3]]
        body = json.loads(route.calls.last.request.content)
        assert body["dimensions"] == 128

    async def test_aembed_status_error(
        self, async_provider: AzureFoundryProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_emb_url("text-embedding-3-small")).mock(
            return_value=httpx.Response(400, json={"error": {"message": "bad"}})
        )
        with pytest.raises(InvalidRequestError):
            await async_provider.aembed("text-embedding-3-small", "hello")

    async def test_aembed_client_init_failure(self, fake_auth: FakeAuth, async_create_raises: MagicMock) -> None:
        provider = AzureFoundryProvider(endpoint=ENDPOINT, auth=fake_auth)
        with pytest.raises(ProviderError, match="client init failed"):
            await provider.aembed("text-embedding-3-small", "hello")
        async_create_raises.assert_called_once()


# MARK: CreateResponse


class TestCreateResponse:
    def test_basic(
        self, sync_provider: AzureFoundryProvider, responses_body: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(RESPONSES_URL).mock(return_value=httpx.Response(200, json=responses_body))
        result = sync_provider.create_response("gpt-5-pro", "Hello")
        assert result.id == "resp_123"
        assert result.output_text == "Hi!"
        assert result.provider == "azure-foundry"
        assert result.usage is not None
        assert result.cost is not None
        body = json.loads(route.calls.last.request.content)
        assert body == {"model": "gpt-5-pro", "input": "Hello", "stream": False}
        assert route.calls.last.request.url.params.get("api-version") == API_VERSION

    def test_input_items_mapped(
        self, sync_provider: AzureFoundryProvider, responses_body: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(RESPONSES_URL).mock(return_value=httpx.Response(200, json=responses_body))
        sync_provider.create_response("gpt-5-pro", [ResponseInputMessage(role="user", content="Hello")])
        body = json.loads(route.calls.last.request.content)
        assert body["input"] == [{"role": "user", "content": "Hello"}]

    def test_structured_input_items_mapped(
        self, sync_provider: AzureFoundryProvider, responses_body: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(RESPONSES_URL).mock(return_value=httpx.Response(200, json=responses_body))
        sync_provider.create_response(
            "gpt-5-pro",
            [
                ResponseInputMessage(
                    role="user",
                    content=[
                        TextContent(text="Describe this image"),
                        CachePointContent(),
                        ImageContent(url="https://example.com/image.png"),
                    ],
                )
            ],
        )
        body = json.loads(route.calls.last.request.content)
        assert body["input"] == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this image"},
                    {"type": "input_image", "image_url": "https://example.com/image.png", "detail": "auto"},
                ],
            }
        ]

    def test_provider_params(
        self, sync_provider: AzureFoundryProvider, responses_body: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(RESPONSES_URL).mock(return_value=httpx.Response(200, json=responses_body))
        sync_provider.create_response(
            "gpt-5-pro", "Hello", provider_params=AzureFoundryParams(reasoning_effort="low", seed=42, user="u1")
        )
        body = json.loads(route.calls.last.request.content)
        assert body["reasoning"] == {"effort": "low"}
        assert body["seed"] == 42
        assert body["user"] == "u1"

    def test_prompt_cache_params(
        self, sync_provider: AzureFoundryProvider, responses_body: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(RESPONSES_URL).mock(return_value=httpx.Response(200, json=responses_body))
        sync_provider.create_response(
            "gpt-5-pro",
            "Hello",
            provider_params=AzureFoundryParams(prompt_cache_key="tenant-42", prompt_cache_retention="24h"),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["prompt_cache_key"] == "tenant-42"
        assert body["prompt_cache_retention"] == "24h"

    def test_deployment_multiplier(
        self, sync_provider: AzureFoundryProvider, responses_body: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(RESPONSES_URL).mock(return_value=httpx.Response(200, json=responses_body))
        result_global = sync_provider.create_response("gpt-5-pro", "Hello")
        result_dz = sync_provider.create_response(
            "gpt-5-pro", "Hello", provider_params=AzureFoundryParams(deployment_type="data_zone")
        )
        assert result_global.cost is not None
        assert result_dz.cost is not None
        assert result_dz.cost.total_cost == pytest.approx(result_global.cost.total_cost * 1.1)

    def test_status_error_mapped(self, sync_provider: AzureFoundryProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(RESPONSES_URL).mock(return_value=httpx.Response(401, json={"error": {"message": "no"}}))
        with pytest.raises(AuthenticationError):
            sync_provider.create_response("gpt-5-pro", "Hello")

    def test_transport_error_mapped(self, sync_provider: AzureFoundryProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(RESPONSES_URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            sync_provider.create_response("gpt-5-pro", "Hello")

    async def test_acreate_response(
        self, async_provider: AzureFoundryProvider, responses_body: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(RESPONSES_URL).mock(return_value=httpx.Response(200, json=responses_body))
        result = await async_provider.acreate_response(
            "gpt-5-pro", "Hello", provider_params=AzureFoundryParams(reasoning_effort="high")
        )
        assert result.output_text == "Hi!"
        body = json.loads(route.calls.last.request.content)
        assert body["reasoning"] == {"effort": "high"}

    async def test_acreate_response_status_error(
        self, async_provider: AzureFoundryProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(RESPONSES_URL).mock(return_value=httpx.Response(401, json={"error": {"message": "no"}}))
        with pytest.raises(AuthenticationError):
            await async_provider.acreate_response("gpt-5-pro", "Hello")

    async def test_acreate_response_client_init_failure(
        self, fake_auth: FakeAuth, async_create_raises: MagicMock
    ) -> None:
        provider = AzureFoundryProvider(endpoint=ENDPOINT, auth=fake_auth)
        with pytest.raises(ProviderError, match="client init failed"):
            await provider.acreate_response("gpt-5-pro", "Hello")
        async_create_raises.assert_called_once()


# MARK: Auth


class TestAuth:
    def test_api_key_header(
        self, fake_auth: FakeAuth, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_chat(completion, respx_mock)
        provider = AzureFoundryProvider(endpoint=ENDPOINT, auth=fake_auth)
        provider.chat("gpt-4o", [UserMessage(content="Hi")])
        assert route.calls.last.request.headers.get("api-key") == "fake-api-key"

    def test_static_ad_token_header(self, completion: dict[str, Any], respx_mock: respx.MockRouter) -> None:
        route = _ok_chat(completion, respx_mock)
        provider = AzureFoundryProvider(endpoint=ENDPOINT, auth=FakeTokenAuth())
        provider.chat("gpt-4o", [UserMessage(content="Hi")])
        request = route.calls.last.request
        assert request.headers.get("authorization") == "Bearer fake-ad-token"
        assert "api-key" not in request.headers

    def test_token_provider_header(self, completion: dict[str, Any], respx_mock: respx.MockRouter) -> None:
        route = _ok_chat(completion, respx_mock)
        provider = AzureFoundryProvider(endpoint=ENDPOINT, auth=FakeTokenProviderAuth())
        provider.chat("gpt-4o", [UserMessage(content="Hi")])
        assert route.calls.last.request.headers.get("authorization") == "Bearer fresh-token"

    def test_default_auth_used_when_none(
        self, monkeypatch: pytest.MonkeyPatch, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "env-key")
        route = _ok_chat(completion, respx_mock)
        provider = AzureFoundryProvider(endpoint=ENDPOINT)
        provider.chat("gpt-4o", [UserMessage(content="Hi")])
        assert route.calls.last.request.headers.get("api-key") == "env-key"


# MARK: Client Management


class TestClientManagement:
    def test_sync_client_reused(
        self, sync_provider: AzureFoundryProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok_chat(completion, respx_mock)
        sync_provider.chat("gpt-4o", [UserMessage(content="a")])
        client = sync_provider._sync_client
        sync_provider.chat("gpt-4o", [UserMessage(content="b")])
        assert sync_provider._sync_client is client

    async def test_async_client_reused(
        self, async_provider: AzureFoundryProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok_chat(completion, respx_mock)
        await async_provider.achat("gpt-4o", [UserMessage(content="a")])
        client = async_provider._async_client
        await async_provider.achat("gpt-4o", [UserMessage(content="b")])
        assert async_provider._async_client is client

    def test_custom_endpoint_and_api_version(
        self, fake_auth: FakeAuth, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        url = "https://my.openai.azure.com/openai/deployments/gpt-4o/chat/completions"
        route = respx_mock.post(url).mock(return_value=httpx.Response(200, json=completion))
        provider = AzureFoundryProvider(
            endpoint="https://my.openai.azure.com/", auth=fake_auth, api_version="2025-01-01"
        )
        provider.chat("gpt-4o", [UserMessage(content="Hi")])
        assert route.called
        assert route.calls.last.request.url.params.get("api-version") == "2025-01-01"

    def test_timeout_and_retries(
        self, fake_auth: FakeAuth, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok_chat(completion, respx_mock)
        provider = AzureFoundryProvider(endpoint=ENDPOINT, auth=fake_auth, timeout=30.0, max_retries=5)
        provider.chat("gpt-4o", [UserMessage(content="Hi")])
        assert provider._sync_client is not None
        assert provider._sync_client.timeout.read == 30.0

    async def test_async_client_recreated_on_new_loop(
        self,
        fake_auth: FakeAuth,
        async_create_two_clients: tuple[MagicMock, MagicMock, MagicMock],
        mock_get_running_loop: MagicMock,
    ) -> None:
        create, c1, c2 = async_create_two_clients
        loop1, loop2 = asyncio.new_event_loop(), asyncio.new_event_loop()
        mock_get_running_loop.side_effect = [loop1, loop2]
        provider = AzureFoundryProvider(endpoint=ENDPOINT, auth=fake_auth)
        r1 = await provider._get_async_client()
        r2 = await provider._get_async_client()
        assert (r1, r2) == (c1, c2)
        assert create.call_count == 2
        loop1.close()
        loop2.close()

    def test_custom_transport_used(self, fake_auth: FakeAuth, completion: dict[str, Any]) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=completion)

        provider = AzureFoundryProvider(endpoint=ENDPOINT, auth=fake_auth, transport=httpx.MockTransport(handler))
        resp = provider.chat("gpt-4o", [UserMessage(content="Hi")])
        assert len(requests) == 1
        assert resp.provider == "azure-foundry"

    async def test_custom_async_transport_used(self, fake_auth: FakeAuth, completion: dict[str, Any]) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=completion)

        provider = AzureFoundryProvider(endpoint=ENDPOINT, auth=fake_auth, async_transport=httpx.MockTransport(handler))
        resp = await provider.achat("gpt-4o", [UserMessage(content="Hi")])
        assert len(requests) == 1
        assert resp.provider == "azure-foundry"


# MARK: Provider Params Kwargs


class TestProviderParamsKwargs:
    def test_empty_params(
        self, sync_provider: AzureFoundryProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_chat(completion, respx_mock)
        sync_provider.chat("gpt-4o", [UserMessage(content="Hi")], provider_params=AzureFoundryParams())
        body = json.loads(route.calls.last.request.content)
        assert "reasoning_effort" not in body
        assert "seed" not in body
        assert "user" not in body
        assert "deployment_type" not in body

    def test_all_params(
        self, sync_provider: AzureFoundryProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_chat(completion, respx_mock)
        sync_provider.chat(
            "gpt-4o",
            [UserMessage(content="Hi")],
            provider_params=AzureFoundryParams(reasoning_effort="low", seed=42, user="u1"),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["reasoning_effort"] == "low"
        assert body["seed"] == 42
        assert body["user"] == "u1"
        assert "deployment_type" not in body


# MARK: Register Pricing


class TestRegisterPricing:
    def test_custom_for_unknown_model(self, sync_provider: AzureFoundryProvider, respx_mock: respx.MockRouter) -> None:
        completion = {
            "model": "ft:custom-model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "x"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        }
        respx_mock.post(_chat_url("ft:custom-model")).mock(return_value=httpx.Response(200, json=completion))
        sync_provider.register_pricing(
            "ft:custom-model", ModelPricing(tiers=[PricingTier(input_cost_per_token=5e-6, output_cost_per_token=15e-6)])
        )
        result = sync_provider.chat("ft:custom-model", [UserMessage(content="Hi")])
        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(1000 * 5e-6)

    def test_custom_overrides_builtin(
        self, sync_provider: AzureFoundryProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok_chat(completion, respx_mock)
        sync_provider.register_pricing(
            "gpt-4o", ModelPricing(tiers=[PricingTier(input_cost_per_token=99e-6, output_cost_per_token=199e-6)])
        )
        result = sync_provider.chat("gpt-4o", [UserMessage(content="Hi")])
        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(10 * 99e-6)

    def test_unknown_model_none_cost(self, sync_provider: AzureFoundryProvider, respx_mock: respx.MockRouter) -> None:
        completion = {
            "model": "totally-unknown",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "x"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        respx_mock.post(_chat_url("totally-unknown")).mock(return_value=httpx.Response(200, json=completion))
        result = sync_provider.chat("totally-unknown", [UserMessage(content="Hi")])
        assert result.cost is None


# MARK: Aclose & Preload


class TestAclose:
    async def test_closes_client(
        self, async_provider: AzureFoundryProvider, completion: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok_chat(completion, respx_mock)
        await async_provider.achat("gpt-4o", [UserMessage(content="Hi")])
        assert async_provider._async_client is not None
        await async_provider.aclose()
        assert async_provider._async_client is None

    async def test_noop_when_no_client(self, async_provider: AzureFoundryProvider) -> None:
        await async_provider.aclose()


class TestPreload:
    def test_preload(self) -> None:
        preload()
