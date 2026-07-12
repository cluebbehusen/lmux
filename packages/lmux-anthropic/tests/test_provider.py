"""Tests for the Anthropic provider (SDK-lite, respx)."""

import asyncio
import json
import threading
from collections.abc import Callable
from datetime import date
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import httpx
import pytest
import respx
from pytest_mock import MockerFixture

if TYPE_CHECKING:
    from google.auth.credentials import Credentials

from lmux.cost import ModelPricing, PricingTier
from lmux.exceptions import AuthenticationError, InvalidRequestError, ProviderError, UnsupportedFeatureError
from lmux.types import (
    FunctionDefinition,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    SystemMessage,
    TextResponseFormat,
    Tool,
    UserMessage,
)
from lmux_anthropic import preload
from lmux_anthropic.auth import AnthropicFoundryEnvAuthProvider, AnthropicVertexADCAuthProvider
from lmux_anthropic.params import AnthropicParams
from lmux_anthropic.provider import AnthropicFoundryProvider, AnthropicProvider, AnthropicVertexProvider

MODEL = "claude-sonnet-4-6"
_URL = "https://api.anthropic.com/v1/messages"


# MARK: Fakes & builders


class FakeAuth:
    def get_credentials(self) -> str:
        return "sk-ant-fake-key"

    async def aget_credentials(self) -> str:
        return "sk-ant-fake-key"


class FakeCredentials:
    """Minimal google.auth-style credentials for Vertex tests."""

    def __init__(self, *, access: str | None = "vertex-token", expired: bool = False) -> None:
        self.token = access
        self.expired = expired
        self.refreshed = False
        self.refresh_thread: int | None = None
        self._refreshed_value = "refreshed-token"

    def refresh(self, request: object) -> None:  # noqa: ARG002
        self.refreshed = True
        self.refresh_thread = threading.get_ident()
        self.token = self._refreshed_value


class FakeVertexAuth:
    def __init__(self) -> None:
        self.credentials = FakeCredentials()

    def get_credentials(self) -> "tuple[Credentials, str]":
        return (cast("Credentials", self.credentials), "auth-project")

    async def aget_credentials(self) -> "tuple[Credentials, str]":
        return (cast("Credentials", self.credentials), "auth-project")


class FakeBareVertexAuth:
    def __init__(self) -> None:
        self.credentials = FakeCredentials()

    def get_credentials(self) -> "Credentials":
        return cast("Credentials", self.credentials)

    async def aget_credentials(self) -> "Credentials":
        return cast("Credentials", self.credentials)


class FakeFoundryAuth:
    def get_credentials(self) -> str:
        return "foundry-key"

    async def aget_credentials(self) -> str:
        return "foundry-key"


class FakeFoundryTokenAuth:
    def __init__(self) -> None:
        self.invocations = 0

        def _token_provider() -> str:
            self.invocations += 1
            return "entra-token"

        self.token_provider: Callable[[], str] = _token_provider

    def get_credentials(self) -> Callable[[], str]:
        return self.token_provider

    async def aget_credentials(self) -> Callable[[], str]:
        return self.token_provider


def _message(  # noqa: PLR0913
    *,
    text: str = "Hello!",
    model: str = MODEL,
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 5,
    content: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "stop_reason": stop_reason,
        "content": content if content is not None else [{"type": "text", "text": text}],
        "usage": usage if usage is not None else {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _sse(events: list[tuple[str, dict[str, Any]]]) -> bytes:
    blocks = [f"event: {etype}\ndata: {json.dumps(data)}" for etype, data in events]
    return ("\n\n".join(blocks) + "\n\n").encode()


def _default_stream(model: str = MODEL) -> bytes:
    return _sse(
        [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {"model": model, "usage": {"input_tokens": 10, "output_tokens": 0}},
                },
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo!"}},
            ),
            (
                "message_delta",
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
    )


@pytest.fixture
def fake_auth() -> FakeAuth:
    return FakeAuth()


@pytest.fixture
def sync_provider(fake_auth: FakeAuth) -> AnthropicProvider:
    return AnthropicProvider(auth=fake_auth)


@pytest.fixture
def async_provider(fake_auth: FakeAuth) -> AnthropicProvider:
    return AnthropicProvider(auth=fake_auth)


@pytest.fixture
def sync_create_raises(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_anthropic.provider.create_sync_client", side_effect=RuntimeError("client init failed"))


@pytest.fixture
def async_create_raises(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_anthropic.provider.create_async_client", side_effect=RuntimeError("client init failed"))


@pytest.fixture
def async_create_two_clients(mocker: MockerFixture) -> tuple[MagicMock, MagicMock, MagicMock]:
    c1, c2 = MagicMock(), MagicMock()
    create = mocker.patch("lmux_anthropic.provider.create_async_client", side_effect=[c1, c2])
    return create, c1, c2


@pytest.fixture
def mock_get_running_loop(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_anthropic.provider.asyncio.get_running_loop")


def _ok(respx_mock: respx.MockRouter, message: dict[str, Any] | None = None) -> respx.Route:
    return respx_mock.post(_URL).mock(
        return_value=httpx.Response(200, json=message if message is not None else _message())
    )


# MARK: Pricing as-of


class TestPricingAsOf:
    def test_intro_window_rate(
        self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("lmux_anthropic.provider._today", lambda: date(2026, 7, 1))
        _ok(respx_mock, _message(model="claude-sonnet-5", input_tokens=1000, output_tokens=500))
        result = sync_provider.chat("claude-sonnet-5", [UserMessage(content="Hi")])
        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(1000 * 2.0 / 1_000_000)
        assert result.cost.output_cost == pytest.approx(500 * 10.0 / 1_000_000)

    def test_rate_after_switch(
        self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("lmux_anthropic.provider._today", lambda: date(2026, 9, 15))
        _ok(respx_mock, _message(model="claude-sonnet-5", input_tokens=1000, output_tokens=500))
        result = sync_provider.chat("claude-sonnet-5", [UserMessage(content="Hi")])
        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(1000 * 3.0 / 1_000_000)

    def test_override_wins_over_clock(
        self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("lmux_anthropic.provider._today", lambda: date(2026, 9, 15))
        _ok(respx_mock, _message(model="claude-sonnet-5", input_tokens=1000, output_tokens=500))
        result = sync_provider.chat(
            "claude-sonnet-5",
            [UserMessage(content="Hi")],
            provider_params=AnthropicParams(pricing_as_of=date(2026, 7, 1)),
        )
        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(1000 * 2.0 / 1_000_000)

    async def test_achat_applies_dated_pricing(
        self, async_provider: AnthropicProvider, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("lmux_anthropic.provider._today", lambda: date(2026, 7, 1))
        _ok(respx_mock, _message(model="claude-sonnet-5", input_tokens=1000, output_tokens=500))
        result = await async_provider.achat("claude-sonnet-5", [UserMessage(content="Hi")])
        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(1000 * 2.0 / 1_000_000)


# MARK: Chat


class TestChat:
    def test_basic(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        route = _ok(respx_mock)
        result = sync_provider.chat(MODEL, [UserMessage(content="Hi")])
        assert result.content == "Hello!"
        assert result.model == MODEL
        assert result.provider == "anthropic"
        assert result.cost is not None
        assert route.called

    def test_request_shape_and_headers(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        route = _ok(respx_mock)
        sync_provider.chat(
            MODEL,
            [SystemMessage(content="Be nice"), UserMessage(content="Hi")],
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            stop="END",
        )
        request = route.calls.last.request
        assert request.headers["x-api-key"] == "sk-ant-fake-key"
        assert request.headers["anthropic-version"] == "2023-06-01"
        body = json.loads(request.content)
        assert body == {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100,
            "stream": False,
            "system": "Be nice",
            "temperature": 0.5,
            "top_p": 0.9,
            "stop_sequences": ["END"],
        }

    def test_stop_list(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        route = _ok(respx_mock)
        sync_provider.chat(MODEL, [UserMessage(content="Hi")], stop=["A", "B"])
        assert json.loads(route.calls.last.request.content)["stop_sequences"] == ["A", "B"]

    def test_default_max_tokens(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        route = _ok(respx_mock)
        sync_provider.chat(MODEL, [UserMessage(content="Hi")])
        assert json.loads(route.calls.last.request.content)["max_tokens"] == 4096

    def test_tools_and_tool_choice(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        route = _ok(respx_mock)
        sync_provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            tools=[Tool(function=FunctionDefinition(name="get_weather"))],
            tool_choice="required",
        )
        body = json.loads(route.calls.last.request.content)
        assert body["tools"] == [{"name": "get_weather", "input_schema": {"type": "object"}}]
        assert body["tool_choice"] == {"type": "any"}

    def test_json_schema_response_format(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        route = _ok(respx_mock)
        sync_provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            response_format=JsonSchemaResponseFormat(name="out", json_schema={"type": "object", "properties": {}}),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["output_config"]["format"]["type"] == "json_schema"

    def test_json_schema_response_format_full_body(
        self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock)
        schema = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}
        sync_provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            response_format=JsonSchemaResponseFormat(name="ans", json_schema=schema),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["output_config"] == {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "integer"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            }
        }

    def test_text_response_format_noop(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        route = _ok(respx_mock)
        sync_provider.chat(MODEL, [UserMessage(content="Hi")], response_format=TextResponseFormat())
        assert "output_config" not in json.loads(route.calls.last.request.content)

    def test_json_object_raises(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        _ok(respx_mock)
        with pytest.raises(UnsupportedFeatureError):
            sync_provider.chat(MODEL, [UserMessage(content="Hi")], response_format=JsonObjectResponseFormat())

    def test_reasoning_effort_budget_model(
        self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock)
        sync_provider.chat("claude-sonnet-4-5", [UserMessage(content="Hi")], reasoning_effort="medium")
        assert json.loads(route.calls.last.request.content)["thinking"] == {"type": "enabled", "budget_tokens": 4095}

    def test_reasoning_effort_high_max_tokens(
        self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock)
        sync_provider.chat(
            "claude-sonnet-4-5", [UserMessage(content="Hi")], max_tokens=100000, reasoning_effort="medium"
        )
        assert json.loads(route.calls.last.request.content)["thinking"] == {"type": "enabled", "budget_tokens": 8192}

    def test_reasoning_effort_adaptive_model(
        self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock)
        sync_provider.chat(MODEL, [UserMessage(content="Hi")], reasoning_effort="high")
        body = json.loads(route.calls.last.request.content)
        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"] == {"effort": "high"}

    def test_reasoning_effort_adaptive_merges_response_format(
        self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock)
        sync_provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            reasoning_effort="low",
            response_format=JsonSchemaResponseFormat(name="out", json_schema={"type": "object", "properties": {}}),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["output_config"]["effort"] == "low"
        assert body["output_config"]["format"]["type"] == "json_schema"

    def test_reasoning_effort_ignored_when_provider_thinking(
        self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter
    ) -> None:
        route = _ok(respx_mock)
        sync_provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            reasoning_effort="high",
            provider_params=AnthropicParams(thinking={"type": "enabled", "budget_tokens": 2048}),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert "output_config" not in body

    def test_provider_params(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        route = _ok(respx_mock)
        sync_provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            provider_params=AnthropicParams(
                thinking={"type": "enabled", "budget_tokens": 5000},
                metadata={"user_id": "u1"},
                top_k=40,
                service_tier="auto",
                inference_geo="us",
                cache_control={"type": "ephemeral"},
            ),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 5000}
        assert body["metadata"] == {"user_id": "u1"}
        assert body["top_k"] == 40
        assert body["service_tier"] == "auto"
        assert body["inference_geo"] == "us"
        assert body["cache_control"] == {"type": "ephemeral"}

    def test_empty_provider_params(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        route = _ok(respx_mock)
        sync_provider.chat(MODEL, [UserMessage(content="Hi")], provider_params=AnthropicParams())
        body = json.loads(route.calls.last.request.content)
        assert "service_tier" not in body
        assert "thinking" not in body

    def test_status_error_mapped(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(return_value=httpx.Response(400, json={"error": {"message": "bad"}}))
        with pytest.raises(InvalidRequestError):
            sync_provider.chat(MODEL, [UserMessage(content="Hi")])

    def test_transport_error_mapped(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            sync_provider.chat(MODEL, [UserMessage(content="Hi")])

    def test_non_json_body_mapped(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(return_value=httpx.Response(200, content=b"not json"))
        with pytest.raises(ProviderError):
            sync_provider.chat(MODEL, [UserMessage(content="Hi")])

    def test_us_inference_multiplier(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        _ok(respx_mock)
        standard = sync_provider.chat(MODEL, [UserMessage(content="Hi")])
        us = sync_provider.chat(MODEL, [UserMessage(content="Hi")], provider_params=AnthropicParams(inference_geo="us"))
        assert standard.cost is not None
        assert us.cost is not None
        assert us.cost.total_cost == pytest.approx(standard.cost.total_cost * 1.1)

    def test_no_multiplier_with_empty_params(
        self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter
    ) -> None:
        _ok(respx_mock)
        standard = sync_provider.chat(MODEL, [UserMessage(content="Hi")])
        empty = sync_provider.chat(MODEL, [UserMessage(content="Hi")], provider_params=AnthropicParams())
        assert standard.cost is not None
        assert empty.cost is not None
        assert standard.cost.total_cost == empty.cost.total_cost


# MARK: Achat


class TestAchat:
    async def test_basic(self, async_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        _ok(respx_mock)
        result = await async_provider.achat(MODEL, [UserMessage(content="Hi")])
        assert result.content == "Hello!"
        assert result.provider == "anthropic"

    async def test_status_error_mapped(self, async_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(return_value=httpx.Response(401, json={"error": {"message": "no"}}))
        with pytest.raises(AuthenticationError):
            await async_provider.achat(MODEL, [UserMessage(content="Hi")])

    async def test_transport_error_mapped(
        self, async_provider: AnthropicProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            await async_provider.achat(MODEL, [UserMessage(content="Hi")])

    async def test_non_json_body_mapped(self, async_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(return_value=httpx.Response(200, content=b"not json"))
        with pytest.raises(ProviderError):
            await async_provider.achat(MODEL, [UserMessage(content="Hi")])


# MARK: ChatStream


class TestChatStream:
    def test_yields_and_costs(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(_URL).mock(return_value=httpx.Response(200, content=_default_stream()))
        chunks = list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))
        assert [c.delta for c in chunks[:2]] == ["Hel", "lo!"]
        assert chunks[-1].finish_reason == "stop"
        assert chunks[0].model is None
        assert chunks[0].provider is None
        assert chunks[0].cost is None
        assert chunks[-1].model == MODEL
        assert chunks[-1].provider == "anthropic"
        assert chunks[-1].cost is not None
        assert chunks[-1].cost.total_cost > 0
        assert json.loads(route.calls.last.request.content)["stream"] is True

    def test_mid_stream_error_raises_after_partial(
        self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter
    ) -> None:
        sse = _sse(
            [
                (
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
                ),
                ("error", {"type": "error", "error": {"type": "overloaded_error", "message": "mid-stream boom"}}),
            ]
        )
        respx_mock.post(_URL).mock(return_value=httpx.Response(200, content=sse))
        stream = sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")])
        assert next(stream).delta == "Hel"  # partial output arrives first
        with pytest.raises(ProviderError, match="mid-stream boom"):
            next(stream)

    def test_cost_bills_resolved_model(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(return_value=httpx.Response(200, content=_default_stream()))
        chunks = list(sync_provider.chat_stream("opaque-alias", [UserMessage(content="Hi")]))
        assert chunks[-1].model == MODEL
        assert chunks[-1].cost is not None

    def test_content_block_start_and_deltas(
        self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter
    ) -> None:
        events = _sse(
            [
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {"model": MODEL, "usage": {"input_tokens": 10, "output_tokens": 0}},
                    },
                ),
                (
                    "content_block_start",
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                ),
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {}},
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "thinking_delta", "thinking": "Let me think..."},
                    },
                ),
                (
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "some_future_delta"}},
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                (
                    "message_delta",
                    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}},
                ),
            ]
        )
        respx_mock.post(_URL).mock(return_value=httpx.Response(200, content=events))
        chunks = list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))
        assert len(chunks) == 3
        assert chunks[0].tool_call_deltas is not None
        assert chunks[0].tool_call_deltas[0].function is not None
        assert chunks[0].tool_call_deltas[0].function.name == "get_weather"
        assert chunks[1].reasoning_delta == "Let me think..."
        assert chunks[2].finish_reason == "tool_calls"

    def test_message_delta_without_start_is_dropped(
        self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter
    ) -> None:
        events = _sse(
            [
                (
                    "message_delta",
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
                )
            ]
        )
        respx_mock.post(_URL).mock(return_value=httpx.Response(200, content=events))
        assert list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")])) == []

    def test_reasoning_effort_passthrough(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(_URL).mock(return_value=httpx.Response(200, content=_default_stream()))
        list(sync_provider.chat_stream("claude-sonnet-4-5", [UserMessage(content="Hi")], reasoning_effort="medium"))
        assert json.loads(route.calls.last.request.content)["thinking"] == {"type": "enabled", "budget_tokens": 4095}

    def test_status_error_on_open(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(return_value=httpx.Response(500, json={"error": {"message": "boom"}}))
        with pytest.raises(ProviderError):
            list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))

    def test_malformed_chunk_mapped(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(
            return_value=httpx.Response(200, content=b"event: message_start\ndata: {not json}\n\n")
        )
        with pytest.raises(ProviderError):
            list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))

    def test_client_init_failure(self, fake_auth: FakeAuth, sync_create_raises: MagicMock) -> None:
        provider = AnthropicProvider(auth=fake_auth)
        with pytest.raises(ProviderError, match="client init failed"):
            list(provider.chat_stream(MODEL, [UserMessage(content="Hi")]))
        sync_create_raises.assert_called_once()


# MARK: AchatStream


class TestAchatStream:
    async def test_yields_and_costs(self, async_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(return_value=httpx.Response(200, content=_default_stream()))
        chunks = [c async for c in async_provider.achat_stream(MODEL, [UserMessage(content="Hi")])]
        assert [c.delta for c in chunks[:2]] == ["Hel", "lo!"]
        assert chunks[-1].finish_reason == "stop"
        assert chunks[-1].model == MODEL
        assert chunks[-1].provider == "anthropic"
        assert chunks[-1].cost is not None

    async def test_mid_stream_error_raises_after_partial(
        self, async_provider: AnthropicProvider, respx_mock: respx.MockRouter
    ) -> None:
        sse = _sse(
            [
                (
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
                ),
                ("error", {"type": "error", "error": {"type": "overloaded_error", "message": "mid-stream boom"}}),
            ]
        )
        respx_mock.post(_URL).mock(return_value=httpx.Response(200, content=sse))
        stream = async_provider.achat_stream(MODEL, [UserMessage(content="Hi")])
        assert (await anext(stream)).delta == "Hel"  # partial output arrives first
        with pytest.raises(ProviderError, match="mid-stream boom"):
            await anext(stream)

    async def test_status_error_on_open(self, async_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_URL).mock(return_value=httpx.Response(500, json={"error": {"message": "boom"}}))
        with pytest.raises(ProviderError):
            async for _ in async_provider.achat_stream(MODEL, [UserMessage(content="Hi")]):
                pass  # pragma: no cover

    async def test_malformed_chunk_mapped(
        self, async_provider: AnthropicProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_URL).mock(return_value=httpx.Response(200, content=b"event: message_start\ndata: {nope}\n\n"))
        with pytest.raises(ProviderError):
            async for _ in async_provider.achat_stream(MODEL, [UserMessage(content="Hi")]):
                pass  # pragma: no cover

    async def test_client_init_failure(self, fake_auth: FakeAuth, async_create_raises: MagicMock) -> None:
        provider = AnthropicProvider(auth=fake_auth)
        with pytest.raises(ProviderError, match="client init failed"):
            async for _ in provider.achat_stream(MODEL, [UserMessage(content="Hi")]):
                pass  # pragma: no cover
        async_create_raises.assert_called_once()


# MARK: Client management


class TestClientManagement:
    def test_sync_client_reused(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        _ok(respx_mock)
        sync_provider.chat(MODEL, [UserMessage(content="a")])
        client = sync_provider._sync_client
        sync_provider.chat(MODEL, [UserMessage(content="b")])
        assert sync_provider._sync_client is client

    async def test_async_client_reused(self, async_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        _ok(respx_mock)
        await async_provider.achat(MODEL, [UserMessage(content="a")])
        client = async_provider._async_client
        await async_provider.achat(MODEL, [UserMessage(content="b")])
        assert async_provider._async_client is client

    def test_custom_base_url(self, fake_auth: FakeAuth, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post("https://custom.api/v1/messages").mock(
            return_value=httpx.Response(200, json=_message())
        )
        provider = AnthropicProvider(auth=fake_auth, base_url="https://custom.api")
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.called

    def test_custom_transport_used(self, fake_auth: FakeAuth) -> None:
        # No respx here: the injected transport must be the one that serves the request.
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=_message())

        provider = AnthropicProvider(auth=fake_auth, transport=httpx.MockTransport(handler))
        resp = provider.chat(MODEL, [UserMessage(content="Hi")])
        assert len(requests) == 1
        assert resp.provider == "anthropic"

    async def test_custom_async_transport_used(self, fake_auth: FakeAuth) -> None:
        # The injected async transport must be the one that serves the request.
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=_message())

        provider = AnthropicProvider(auth=fake_auth, async_transport=httpx.MockTransport(handler))
        resp = await provider.achat(MODEL, [UserMessage(content="Hi")])
        assert len(requests) == 1
        assert resp.provider == "anthropic"

    def test_timeout_and_retries(self, fake_auth: FakeAuth, respx_mock: respx.MockRouter) -> None:
        _ok(respx_mock)
        provider = AnthropicProvider(auth=fake_auth, timeout=30.0, max_retries=5)
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert provider._sync_client is not None
        assert provider._sync_client.timeout.read == 30.0

    def test_custom_default_max_tokens(self, fake_auth: FakeAuth, respx_mock: respx.MockRouter) -> None:
        route = _ok(respx_mock)
        provider = AnthropicProvider(auth=fake_auth, default_max_tokens=8192)
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert json.loads(route.calls.last.request.content)["max_tokens"] == 8192

    def test_sync_init_failure_mapped(self, fake_auth: FakeAuth, sync_create_raises: MagicMock) -> None:
        provider = AnthropicProvider(auth=fake_auth)
        with pytest.raises(ProviderError, match="client init failed"):
            provider.chat(MODEL, [UserMessage(content="Hi")])
        sync_create_raises.assert_called_once()

    async def test_async_init_failure_mapped(self, fake_auth: FakeAuth, async_create_raises: MagicMock) -> None:
        provider = AnthropicProvider(auth=fake_auth)
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
        provider = AnthropicProvider(auth=fake_auth)
        r1 = await provider._get_async_client()
        r2 = await provider._get_async_client()
        assert (r1, r2) == (c1, c2)
        assert create.call_count == 2
        loop1.close()
        loop2.close()


# MARK: Register pricing & aclose & preload


class TestRegisterPricing:
    def test_custom_for_unknown_model(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        _ok(respx_mock, _message(model="claude-custom-v1", input_tokens=1000, output_tokens=500))
        sync_provider.register_pricing(
            "claude-custom-v1",
            ModelPricing(tiers=[PricingTier(input_cost_per_token=5e-6, output_cost_per_token=15e-6)]),
        )
        result = sync_provider.chat("claude-custom-v1", [UserMessage(content="Hi")])
        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(1000 * 5e-6)

    def test_custom_overrides_builtin(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        _ok(respx_mock)
        sync_provider.register_pricing(
            MODEL, ModelPricing(tiers=[PricingTier(input_cost_per_token=99e-6, output_cost_per_token=199e-6)])
        )
        result = sync_provider.chat(MODEL, [UserMessage(content="Hi")])
        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(10 * 99e-6)

    def test_unknown_model_none_cost(self, sync_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        _ok(respx_mock, _message(model="totally-unknown"))
        result = sync_provider.chat("totally-unknown", [UserMessage(content="Hi")])
        assert result.cost is None


class TestAclose:
    async def test_closes_client(self, async_provider: AnthropicProvider, respx_mock: respx.MockRouter) -> None:
        _ok(respx_mock)
        await async_provider.achat(MODEL, [UserMessage(content="Hi")])
        assert async_provider._async_client is not None
        await async_provider.aclose()
        assert async_provider._async_client is None

    async def test_noop_when_no_client(self, async_provider: AnthropicProvider) -> None:
        await async_provider.aclose()


class TestPreload:
    def test_preload(self) -> None:
        preload()


# MARK: Vertex


def _vertex_url(model: str, *, region: str = "us-east5", project: str = "my-proj", stream: bool = False) -> str:
    if region == "global":
        host = "aiplatform.googleapis.com"
    elif region in ("us", "eu"):
        host = f"aiplatform.{region}.rep.googleapis.com"
    else:
        host = f"{region}-aiplatform.googleapis.com"
    specifier = "streamRawPredict" if stream else "rawPredict"
    return f"https://{host}/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{model}:{specifier}"


@pytest.fixture
def vertex_auth() -> FakeVertexAuth:
    return FakeVertexAuth()


@pytest.fixture
def vertex_sync_provider(vertex_auth: FakeVertexAuth) -> AnthropicVertexProvider:
    return AnthropicVertexProvider(auth=vertex_auth, project_id="my-proj", region="us-east5")


class TestVertexChat:
    def test_basic(self, vertex_sync_provider: AnthropicVertexProvider, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(_vertex_url(MODEL)).mock(return_value=httpx.Response(200, json=_message()))
        result = vertex_sync_provider.chat(MODEL, [UserMessage(content="Hi")])
        assert result.content == "Hello!"
        assert result.provider == "anthropic-vertex"
        assert result.cost is not None
        request = route.calls.last.request
        assert request.headers["authorization"] == "Bearer vertex-token"
        body = json.loads(request.content)
        assert "model" not in body
        assert body["anthropic_version"] == "vertex-2023-10-16"

    async def test_basic_achat(self, vertex_auth: FakeVertexAuth, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(_vertex_url(MODEL)).mock(return_value=httpx.Response(200, json=_message()))
        provider = AnthropicVertexProvider(auth=vertex_auth, project_id="my-proj", region="us-east5")
        result = await provider.achat(MODEL, [UserMessage(content="Hi")])
        assert result.provider == "anthropic-vertex"
        assert route.calls.last.request.headers["authorization"] == "Bearer vertex-token"

    def test_multi_region_us_routes_to_rep_endpoint(
        self, vertex_auth: FakeVertexAuth, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(_vertex_url(MODEL, region="us")).mock(return_value=httpx.Response(200, json=_message()))
        provider = AnthropicVertexProvider(auth=vertex_auth, project_id="my-proj", region="us")
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.called  # https://aiplatform.us.rep.googleapis.com/... not us-aiplatform.googleapis.com

    def test_token_refreshed_per_request(self, vertex_auth: FakeVertexAuth, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(_vertex_url(MODEL)).mock(return_value=httpx.Response(200, json=_message()))
        provider = AnthropicVertexProvider(auth=vertex_auth, project_id="my-proj", region="us-east5")
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.calls[-1].request.headers["authorization"] == "Bearer vertex-token"
        # The token expires between requests; the next call must refresh rather than reuse a frozen client header.
        vertex_auth.credentials.expired = True
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert vertex_auth.credentials.refreshed is True
        assert route.calls[-1].request.headers["authorization"] == "Bearer refreshed-token"

    async def test_async_refresh_runs_off_the_event_loop(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_vertex_url(MODEL)).mock(return_value=httpx.Response(200, json=_message()))
        auth = FakeVertexAuth()
        auth.credentials = FakeCredentials(access=None)  # empty token forces a blocking refresh
        provider = AnthropicVertexProvider(auth=auth, project_id="my-proj", region="us-east5")
        await provider.achat(MODEL, [UserMessage(content="Hi")])
        assert auth.credentials.refreshed is True
        # The refresh must run on a worker thread (asyncio.to_thread), not stall the event loop.
        assert auth.credentials.refresh_thread is not None
        assert auth.credentials.refresh_thread != threading.get_ident()

    def test_model_prefix_pricing(
        self, vertex_sync_provider: AnthropicVertexProvider, respx_mock: respx.MockRouter
    ) -> None:
        model = "claude-sonnet-4-5@20250929"
        respx_mock.post(_vertex_url(model)).mock(return_value=httpx.Response(200, json=_message(model=model)))
        result = vertex_sync_provider.chat(model, [UserMessage(content="Hi")])
        assert result.cost is not None
        assert result.cost.total_cost > 0

    def test_api_only_params_dropped(
        self, vertex_sync_provider: AnthropicVertexProvider, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(_vertex_url(MODEL)).mock(return_value=httpx.Response(200, json=_message()))
        vertex_sync_provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            provider_params=AnthropicParams(
                top_k=40, service_tier="auto", inference_geo="us", cache_control={"type": "ephemeral"}
            ),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["top_k"] == 40
        assert body["cache_control"] == {"type": "ephemeral"}
        assert "service_tier" not in body
        assert "inference_geo" not in body

    def test_inference_geo_multiplier_not_applied(
        self, vertex_sync_provider: AnthropicVertexProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_vertex_url(MODEL)).mock(return_value=httpx.Response(200, json=_message()))
        standard = vertex_sync_provider.chat(MODEL, [UserMessage(content="Hi")])
        us = vertex_sync_provider.chat(
            MODEL, [UserMessage(content="Hi")], provider_params=AnthropicParams(inference_geo="us")
        )
        assert standard.cost is not None
        assert us.cost is not None
        assert standard.cost.total_cost == us.cost.total_cost

    def test_regional_premium_applied(self, vertex_auth: FakeVertexAuth, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_vertex_url(MODEL, region="global", project="p")).mock(
            return_value=httpx.Response(200, json=_message())
        )
        respx_mock.post(_vertex_url(MODEL, region="us-east5", project="p")).mock(
            return_value=httpx.Response(200, json=_message())
        )
        global_provider = AnthropicVertexProvider(auth=vertex_auth, project_id="p", region="global")
        regional_provider = AnthropicVertexProvider(auth=vertex_auth, project_id="p", region="us-east5")
        result_global = global_provider.chat(MODEL, [UserMessage(content="Hi")])
        result_regional = regional_provider.chat(MODEL, [UserMessage(content="Hi")])
        assert result_global.cost is not None
        assert result_regional.cost is not None
        assert result_regional.cost.total_cost == pytest.approx(result_global.cost.total_cost * 1.1)

    def test_no_premium_uniform_model(self, vertex_auth: FakeVertexAuth, respx_mock: respx.MockRouter) -> None:
        model = "claude-3-5-haiku"
        respx_mock.post(_vertex_url(model, region="global", project="p")).mock(
            return_value=httpx.Response(200, json=_message(model=model))
        )
        respx_mock.post(_vertex_url(model, region="us-east5", project="p")).mock(
            return_value=httpx.Response(200, json=_message(model=model))
        )
        result_global = AnthropicVertexProvider(auth=vertex_auth, project_id="p", region="global").chat(
            model, [UserMessage(content="Hi")]
        )
        result_regional = AnthropicVertexProvider(auth=vertex_auth, project_id="p", region="us-east5").chat(
            model, [UserMessage(content="Hi")]
        )
        assert result_global.cost is not None
        assert result_regional.cost is not None
        assert result_global.cost.total_cost == result_regional.cost.total_cost

    def test_region_falls_back_to_env(
        self, monkeypatch: pytest.MonkeyPatch, vertex_auth: FakeVertexAuth, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_vertex_url(MODEL, region="global", project="p")).mock(
            return_value=httpx.Response(200, json=_message())
        )
        respx_mock.post(_vertex_url(MODEL, region="us-east5", project="p")).mock(
            return_value=httpx.Response(200, json=_message())
        )
        provider = AnthropicVertexProvider(auth=vertex_auth, project_id="p")
        monkeypatch.setenv("CLOUD_ML_REGION", "global")
        result_global = provider.chat(MODEL, [UserMessage(content="Hi")])
        provider._sync_client = None  # force client recreation for the new region
        monkeypatch.setenv("CLOUD_ML_REGION", "us-east5")
        result_regional = provider.chat(MODEL, [UserMessage(content="Hi")])
        assert result_global.cost is not None
        assert result_regional.cost is not None
        assert result_regional.cost.total_cost == pytest.approx(result_global.cost.total_cost * 1.1)

    def test_missing_region_raises(self, monkeypatch: pytest.MonkeyPatch, vertex_auth: FakeVertexAuth) -> None:
        monkeypatch.delenv("CLOUD_ML_REGION", raising=False)
        provider = AnthropicVertexProvider(auth=vertex_auth, project_id="p")
        with pytest.raises(ProviderError, match="region"):
            provider.chat(MODEL, [UserMessage(content="Hi")])

    def test_missing_project_raises(self, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter) -> None:  # noqa: ARG002
        monkeypatch.delenv("ANTHROPIC_VERTEX_PROJECT_ID", raising=False)

        class _NoProjectAuth:
            def get_credentials(self) -> "Credentials":
                return cast("Credentials", FakeCredentials())

            async def aget_credentials(self) -> "Credentials":
                return cast("Credentials", FakeCredentials())

        provider = AnthropicVertexProvider(auth=_NoProjectAuth(), region="us-east5")
        with pytest.raises(ProviderError, match="project_id"):
            provider.chat(MODEL, [UserMessage(content="Hi")])

    def test_project_falls_back_to_auth(
        self, monkeypatch: pytest.MonkeyPatch, vertex_auth: FakeVertexAuth, respx_mock: respx.MockRouter
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_VERTEX_PROJECT_ID", raising=False)
        route = respx_mock.post(_vertex_url(MODEL, region="global", project="auth-project")).mock(
            return_value=httpx.Response(200, json=_message())
        )
        provider = AnthropicVertexProvider(auth=vertex_auth, region="global")
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.called

    def test_env_project_beats_auth(
        self, monkeypatch: pytest.MonkeyPatch, vertex_auth: FakeVertexAuth, respx_mock: respx.MockRouter
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "env-project")
        route = respx_mock.post(_vertex_url(MODEL, region="global", project="env-project")).mock(
            return_value=httpx.Response(200, json=_message())
        )
        provider = AnthropicVertexProvider(auth=vertex_auth, region="global")
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.called

    def test_bare_credentials_supported(self, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter) -> None:
        monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "env-project")
        route = respx_mock.post(_vertex_url(MODEL, region="global", project="env-project")).mock(
            return_value=httpx.Response(200, json=_message())
        )
        provider = AnthropicVertexProvider(auth=FakeBareVertexAuth(), region="global")
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.called

    async def test_bare_credentials_supported_async(
        self, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_VERTEX_PROJECT_ID", "env-project")
        route = respx_mock.post(_vertex_url(MODEL, region="global", project="env-project")).mock(
            return_value=httpx.Response(200, json=_message())
        )
        provider = AnthropicVertexProvider(auth=FakeBareVertexAuth(), region="global")
        await provider.achat(MODEL, [UserMessage(content="Hi")])
        assert route.called

    def test_exception_reports_vertex_provider(
        self, vertex_sync_provider: AnthropicVertexProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_vertex_url(MODEL)).mock(return_value=httpx.Response(400, json={"error": {"message": "bad"}}))
        with pytest.raises(InvalidRequestError) as exc_info:
            vertex_sync_provider.chat(MODEL, [UserMessage(content="Hi")])
        assert exc_info.value.provider == "anthropic-vertex"

    def test_default_auth_is_adc(self) -> None:
        provider = AnthropicVertexProvider()
        assert isinstance(provider._vertex_auth, AnthropicVertexADCAuthProvider)


class TestVertexChatStream:
    def test_stream_stamps_vertex_identity(
        self, vertex_sync_provider: AnthropicVertexProvider, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(_vertex_url(MODEL, stream=True)).mock(
            return_value=httpx.Response(200, content=_default_stream())
        )
        chunks = list(vertex_sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))
        assert chunks[-1].finish_reason == "stop"
        assert chunks[-1].model == MODEL
        assert chunks[-1].provider == "anthropic-vertex"
        assert chunks[-1].cost is not None
        assert route.calls.last.request.headers["authorization"] == "Bearer vertex-token"

    async def test_async_stream(self, vertex_auth: FakeVertexAuth, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(_vertex_url(MODEL, stream=True)).mock(
            return_value=httpx.Response(200, content=_default_stream())
        )
        provider = AnthropicVertexProvider(auth=vertex_auth, project_id="my-proj", region="us-east5")
        chunks = [c async for c in provider.achat_stream(MODEL, [UserMessage(content="Hi")])]
        assert chunks[-1].provider == "anthropic-vertex"
        assert route.calls.last.request.headers["authorization"] == "Bearer vertex-token"


# MARK: Foundry


def _foundry_url(*, resource: str = "my-resource") -> str:
    return f"https://{resource}.services.ai.azure.com/anthropic/v1/messages"


@pytest.fixture
def foundry_sync_provider() -> AnthropicFoundryProvider:
    return AnthropicFoundryProvider(auth=FakeFoundryAuth(), resource="my-resource")


class TestFoundryChat:
    def test_basic(self, foundry_sync_provider: AnthropicFoundryProvider, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(_foundry_url()).mock(return_value=httpx.Response(200, json=_message()))
        result = foundry_sync_provider.chat(MODEL, [UserMessage(content="Hi")])
        assert result.content == "Hello!"
        assert result.provider == "anthropic-foundry"
        assert result.cost is not None
        request = route.calls.last.request
        assert request.headers["api-key"] == "foundry-key"
        assert json.loads(request.content)["model"] == MODEL

    async def test_basic_achat(self, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_foundry_url()).mock(return_value=httpx.Response(200, json=_message()))
        provider = AnthropicFoundryProvider(auth=FakeFoundryAuth(), resource="my-resource")
        result = await provider.achat(MODEL, [UserMessage(content="Hi")])
        assert result.provider == "anthropic-foundry"

    def test_api_key_sends_both_headers(
        self, foundry_sync_provider: AnthropicFoundryProvider, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(_foundry_url()).mock(return_value=httpx.Response(200, json=_message()))
        foundry_sync_provider.chat(MODEL, [UserMessage(content="Hi")])
        headers = route.calls.last.request.headers
        # The Anthropic-on-Foundry endpoint authenticates with x-api-key; api-key is kept for compatibility.
        assert headers["x-api-key"] == "foundry-key"
        assert headers["api-key"] == "foundry-key"

    def test_token_auth_uses_bearer(self, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(_foundry_url()).mock(return_value=httpx.Response(200, json=_message()))
        provider = AnthropicFoundryProvider(auth=FakeFoundryTokenAuth(), resource="my-resource")
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.calls.last.request.headers["authorization"] == "Bearer entra-token"

    def test_token_provider_invoked_per_request(self, respx_mock: respx.MockRouter) -> None:
        auth = FakeFoundryTokenAuth()
        respx_mock.post(_foundry_url()).mock(return_value=httpx.Response(200, json=_message()))
        provider = AnthropicFoundryProvider(auth=auth, resource="my-resource")
        provider.chat(MODEL, [UserMessage(content="Hi")])
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert auth.invocations == 2  # invoked per request, not frozen at client creation

    def test_api_only_params_dropped(
        self, foundry_sync_provider: AnthropicFoundryProvider, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(_foundry_url()).mock(return_value=httpx.Response(200, json=_message()))
        foundry_sync_provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            provider_params=AnthropicParams(top_k=40, service_tier="auto", inference_geo="us"),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["top_k"] == 40
        assert "service_tier" not in body
        assert "inference_geo" not in body

    def test_inference_geo_multiplier_not_applied(
        self, foundry_sync_provider: AnthropicFoundryProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_foundry_url()).mock(return_value=httpx.Response(200, json=_message()))
        standard = foundry_sync_provider.chat(MODEL, [UserMessage(content="Hi")])
        us = foundry_sync_provider.chat(
            MODEL, [UserMessage(content="Hi")], provider_params=AnthropicParams(inference_geo="us")
        )
        assert standard.cost is not None
        assert us.cost is not None
        assert standard.cost.total_cost == us.cost.total_cost

    def test_exception_reports_foundry_provider(
        self, foundry_sync_provider: AnthropicFoundryProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_foundry_url()).mock(return_value=httpx.Response(400, json={"error": {"message": "bad"}}))
        with pytest.raises(InvalidRequestError) as exc_info:
            foundry_sync_provider.chat(MODEL, [UserMessage(content="Hi")])
        assert exc_info.value.provider == "anthropic-foundry"

    def test_default_auth_is_env(self) -> None:
        provider = AnthropicFoundryProvider(resource="my-resource")
        assert isinstance(provider._foundry_auth, AnthropicFoundryEnvAuthProvider)
