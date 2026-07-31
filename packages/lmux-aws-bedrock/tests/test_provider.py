"""Tests for the AWS Bedrock provider (SDK-lite, respx)."""

import asyncio
import json
import struct
import zlib
from datetime import date
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import botocore.exceptions
import httpx
import pytest
import respx
from pytest_mock import MockerFixture

if TYPE_CHECKING:
    import boto3
    from aiobotocore.session import AioSession

from lmux.cost import ModelPricing, PricingTier, per_million_tokens
from lmux.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
    TimeoutError,  # noqa: A004
    UnsupportedFeatureError,
)
from lmux.types import (
    ChatChunk,
    ChatResponse,
    EmbeddingResponse,
    FunctionDefinition,
    JsonObjectResponseFormat,
    JsonSchemaResponseFormat,
    SystemMessage,
    TextResponseFormat,
    Tool,
    Usage,
    UserMessage,
)
from lmux_aws_bedrock import preload
from lmux_aws_bedrock.params import BedrockParams, GuardrailConfig
from lmux_aws_bedrock.provider import BedrockProvider

_BASE = "https://bedrock-runtime.us-east-1.amazonaws.com"
MODEL = "anthropic.claude-sonnet-4"
EMBED_MODEL = "amazon.titan-embed-text-v2"


def _url(model: str, action: str, *, base: str = _BASE) -> str:
    return f"{base}/model/{model}/{action}"


# MARK: Fake Auth (SigV4 credential resolution)


class _FrozenCreds:
    access_key = "AKIAFAKEEXAMPLE"
    secret_key = "fake-secret-key"  # noqa: S105
    token = None


class _Credentials:
    def get_frozen_credentials(self) -> _FrozenCreds:
        return _FrozenCreds()


_DEFAULT_CREDENTIALS = _Credentials()


class _Session:
    def __init__(
        self, *, region_name: str | None = None, credentials: _Credentials | None = _DEFAULT_CREDENTIALS
    ) -> None:
        self.region_name = region_name
        self._credentials = credentials

    def get_credentials(self) -> _Credentials | None:
        return self._credentials


class FakeAuth:
    """Fake auth provider returning a boto3-Session-like object for SigV4 signing.

    Credentials are resolved synchronously, so only ``get_credentials`` is exercised;
    ``aget_credentials`` exists to satisfy the ``AuthProvider`` protocol.
    """

    def __init__(self, session: _Session | None = None) -> None:
        self._session = session if session is not None else _Session()
        self.get_calls = 0

    def get_credentials(self) -> "boto3.Session":
        self.get_calls += 1
        return cast("boto3.Session", self._session)

    async def aget_credentials(self) -> "AioSession":  # pragma: no cover - protocol conformance only
        return cast("AioSession", self._session)


class _RaisingSessionAuth:
    """Auth provider whose session construction fails, e.g. a stale AWS_PROFILE."""

    def get_credentials(self) -> "boto3.Session":
        raise botocore.exceptions.ProfileNotFound(profile="missing-profile")

    async def aget_credentials(self) -> "AioSession":  # pragma: no cover - protocol conformance only
        raise botocore.exceptions.ProfileNotFound(profile="missing-profile")


# MARK: Event-stream framing (mirrors tests/test_eventstream.py)


def _encode(headers: dict[str, str], payload: bytes) -> bytes:
    hb = b""
    for name, value in headers.items():
        nb, vb = name.encode(), value.encode()
        hb += bytes([len(nb)]) + nb + bytes([7]) + struct.pack(">H", len(vb)) + vb
    total = 16 + len(hb) + len(payload)
    prelude = struct.pack(">II", total, len(hb))
    body = prelude + struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF) + hb + payload
    return body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def _frame(event_type: str, payload: dict[str, Any] | None, *, message_type: str = "event") -> bytes:
    data = json.dumps(payload).encode() if payload is not None else b""
    if message_type == "exception":
        # Real exception frames carry a camelCase :exception-type and no :event-type.
        return _encode({":message-type": "exception", ":exception-type": event_type}, data)
    return _encode({":event-type": event_type, ":message-type": message_type}, data)


def _stream_bytes(events: list[tuple[str, dict[str, Any] | None]]) -> bytes:
    return b"".join(_frame(event_type, payload) for event_type, payload in events)


_STREAM_EVENTS: list[tuple[str, dict[str, Any] | None]] = [
    ("messageStart", {"role": "assistant"}),
    ("contentBlockDelta", {"delta": {"text": "Hello"}, "contentBlockIndex": 0}),
    ("contentBlockStop", None),  # empty payload -> skipped, exercises the empty-payload branch
    ("messageStop", {"stopReason": "end_turn"}),
    ("metadata", {"usage": {"inputTokens": 10, "outputTokens": 5}, "metrics": {"latencyMs": 100}}),
]


# MARK: Shared Fixtures


@pytest.fixture(autouse=True)
def _no_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test to the SigV4 path; bearer-token tests opt back in."""
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)


@pytest.fixture
def fake_auth() -> FakeAuth:
    return FakeAuth()


@pytest.fixture
def sync_provider(fake_auth: FakeAuth) -> BedrockProvider:
    return BedrockProvider(auth=fake_auth)


@pytest.fixture
def async_provider(fake_auth: FakeAuth) -> BedrockProvider:
    return BedrockProvider(auth=fake_auth)


@pytest.fixture
def sync_create_raises(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_aws_bedrock.provider.create_sync_client", side_effect=RuntimeError("client init failed"))


@pytest.fixture
def async_create_raises(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_aws_bedrock.provider.create_async_client", side_effect=RuntimeError("client init failed"))


@pytest.fixture
def async_create_two_clients(mocker: MockerFixture) -> tuple[MagicMock, MagicMock, MagicMock]:
    c1, c2 = MagicMock(), MagicMock()
    create = mocker.patch("lmux_aws_bedrock.provider.create_async_client", side_effect=[c1, c2])
    return create, c1, c2


@pytest.fixture
def mock_create_sync(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_aws_bedrock.provider.create_sync_client", return_value=MagicMock())


@pytest.fixture
def mock_create_async(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_aws_bedrock.provider.create_async_client", return_value=MagicMock())


@pytest.fixture
def mock_get_running_loop(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux_aws_bedrock.provider.asyncio.get_running_loop")


@pytest.fixture
def converse_response() -> dict[str, Any]:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": "Hello!"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }


@pytest.fixture
def embedding_response() -> dict[str, Any]:
    return {"embedding": [0.1, 0.2, 0.3], "inputTextTokenCount": 5}


def _ok_converse(response: dict[str, Any], respx_mock: respx.MockRouter, *, model: str = MODEL) -> respx.Route:
    return respx_mock.post(_url(model, "converse")).mock(return_value=httpx.Response(200, json=response))


def _ok_embed(response: dict[str, Any], respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(_url(EMBED_MODEL, "invoke")).mock(return_value=httpx.Response(200, json=response))


# MARK: Pricing As-Of


class TestPricingAsOf:
    def test_live_cost_uses_current_date(
        self,
        sync_provider: BedrockProvider,
        converse_response: dict[str, Any],
        respx_mock: respx.MockRouter,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("lmux_aws_bedrock.provider._today", lambda: date(2026, 7, 1))
        _ok_converse(converse_response, respx_mock, model="anthropic.claude-sonnet-5")
        result = sync_provider.chat("anthropic.claude-sonnet-5", [UserMessage(content="Hi")])
        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(10 * 2.2 / 1_000_000)

    def test_pricing_as_of_override_wins_over_clock(
        self,
        sync_provider: BedrockProvider,
        converse_response: dict[str, Any],
        respx_mock: respx.MockRouter,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("lmux_aws_bedrock.provider._today", lambda: date(2026, 9, 15))
        _ok_converse(converse_response, respx_mock, model="anthropic.claude-sonnet-5")
        result = sync_provider.chat(
            "anthropic.claude-sonnet-5",
            [UserMessage(content="Hi")],
            provider_params=BedrockParams(pricing_as_of=date(2026, 7, 1)),
        )
        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(10 * 2.2 / 1_000_000)

    def test_live_cost_uses_current_date_after_switch(
        self,
        sync_provider: BedrockProvider,
        converse_response: dict[str, Any],
        respx_mock: respx.MockRouter,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("lmux_aws_bedrock.provider._today", lambda: date(2026, 9, 15))
        _ok_converse(converse_response, respx_mock, model="anthropic.claude-sonnet-5")
        result = sync_provider.chat("anthropic.claude-sonnet-5", [UserMessage(content="Hi")])
        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(10 * 3.3 / 1_000_000)


# MARK: Chat


class TestChat:
    def test_basic_chat(
        self, sync_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_converse(converse_response, respx_mock, model="amazon.nova-pro-v1")
        result = sync_provider.chat("amazon.nova-pro-v1", [UserMessage(content="Hi")])

        assert result == ChatResponse(
            content="Hello!",
            tool_calls=None,
            usage=Usage(input_tokens=10, output_tokens=5),
            cost=result.cost,
            model="amazon.nova-pro-v1",
            provider="aws-bedrock",
            finish_reason="stop",
        )
        assert result.cost is not None
        assert result.cost.total_cost > 0
        assert route.called
        # Request is signed with SigV4 and carries no bearer token.
        assert route.calls.last.request.headers["authorization"].startswith("AWS4-HMAC-SHA256")
        assert "x-amz-date" in route.calls.last.request.headers

    def test_request_body_omits_model_id(
        self, sync_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_converse(converse_response, respx_mock)
        sync_provider.chat(MODEL, [UserMessage(content="Hi")], temperature=0.5, max_tokens=100, top_p=0.9, stop=["END"])
        body = json.loads(route.calls.last.request.content)
        assert body == {
            "messages": [{"role": "user", "content": [{"text": "Hi"}]}],
            "inferenceConfig": {"temperature": 0.5, "maxTokens": 100, "topP": 0.9, "stopSequences": ["END"]},
        }

    def test_default_headers_are_signed_and_managed_headers_win(
        self, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_converse(converse_response, respx_mock)
        provider = BedrockProvider(
            auth=FakeAuth(),
            default_headers={
                "X-Trace-ID": "first",
                "x-trace-id": "trace-123",
                "Authorization": "Bearer wrong",
                "CONTENT-TYPE": "text/plain",
                "Transfer-Encoding": "chunked",
                "User-Agent": "lmux-test",
                "X-AMZ-DATE": "wrong",
            },
        )
        provider.chat(MODEL, [UserMessage(content="Hi")])
        headers = route.calls.last.request.headers
        signed_headers = headers["authorization"].split("SignedHeaders=", 1)[1].split(",", 1)[0].split(";")
        assert headers["x-trace-id"] == "trace-123"
        assert headers["content-type"] == "application/json"
        assert headers["x-amz-date"] != "wrong"
        assert headers.get_list("x-trace-id") == ["trace-123"]
        assert headers["user-agent"] == "lmux-test"
        assert "transfer-encoding" not in headers
        assert "x-trace-id" in signed_headers
        assert "user-agent" not in signed_headers

    def test_chat_with_tools_and_choice(
        self, sync_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_converse(converse_response, respx_mock)
        sync_provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            tools=[Tool(function=FunctionDefinition(name="get_weather"))],
            tool_choice="required",
        )
        body = json.loads(route.calls.last.request.content)
        assert body["toolConfig"] == {
            "tools": [{"toolSpec": {"name": "get_weather", "inputSchema": {"json": {"type": "object"}}}}],
            "toolChoice": {"any": {}},
        }

    def test_chat_with_tools_no_choice(
        self, sync_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_converse(converse_response, respx_mock)
        sync_provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            tools=[Tool(function=FunctionDefinition(name="get_weather"))],
        )
        body = json.loads(route.calls.last.request.content)
        assert body["toolConfig"] == {
            "tools": [{"toolSpec": {"name": "get_weather", "inputSchema": {"json": {"type": "object"}}}}],
        }

    def test_chat_with_text_response_format(
        self, sync_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_converse(converse_response, respx_mock)
        result = sync_provider.chat(MODEL, [UserMessage(content="Hi")], response_format=TextResponseFormat())
        assert result.content == "Hello!"
        body = json.loads(route.calls.last.request.content)
        assert "outputConfig" not in body

    def test_chat_json_object_raises(self, sync_provider: BedrockProvider) -> None:
        with pytest.raises(UnsupportedFeatureError, match="JsonObjectResponseFormat is not supported"):
            sync_provider.chat(MODEL, [UserMessage(content="Hi")], response_format=JsonObjectResponseFormat())

    def test_chat_with_json_schema_response_format(
        self, sync_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_converse(converse_response, respx_mock)
        sync_provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            response_format=JsonSchemaResponseFormat(
                name="weather_response",
                description="Structured weather payload",
                json_schema={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
        )
        body = json.loads(route.calls.last.request.content)
        assert body["outputConfig"] == {
            "textFormat": {
                "type": "json_schema",
                "structure": {
                    "jsonSchema": {
                        "schema": json.dumps(
                            {
                                "additionalProperties": False,
                                "properties": {"city": {"type": "string"}},
                                "type": "object",
                            },
                            sort_keys=True,
                        ),
                        "name": "weather_response",
                        "description": "Structured weather payload",
                    }
                },
            }
        }

    def test_chat_with_system_message(
        self, sync_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_converse(converse_response, respx_mock)
        sync_provider.chat(MODEL, [SystemMessage(content="Be helpful."), UserMessage(content="Hi")])
        body = json.loads(route.calls.last.request.content)
        assert body["system"] == [{"text": "Be helpful."}]

    def test_chat_with_stop_string(
        self, sync_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_converse(converse_response, respx_mock)
        sync_provider.chat(MODEL, [UserMessage(content="Hi")], stop="STOP")
        body = json.loads(route.calls.last.request.content)
        assert body["inferenceConfig"]["stopSequences"] == ["STOP"]

    def test_chat_with_reasoning_effort(
        self, sync_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_converse(converse_response, respx_mock)
        sync_provider.chat(MODEL, [UserMessage(content="Hi")], reasoning_effort="medium")
        body = json.loads(route.calls.last.request.content)
        assert body["additionalModelRequestFields"]["thinking"] == {"type": "enabled", "budget_tokens": 8192}

    def test_reasoning_effort_deep_merges_with_provider_params(
        self, sync_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_converse(converse_response, respx_mock)
        sync_provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            reasoning_effort="high",
            provider_params=BedrockParams(additional_model_request_fields={"some_field": "value"}),
        )
        additional = json.loads(route.calls.last.request.content)["additionalModelRequestFields"]
        assert additional["thinking"] == {"type": "enabled", "budget_tokens": 32768}
        assert additional["some_field"] == "value"

    def test_provider_params_thinking_overrides_reasoning_effort(
        self, sync_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_converse(converse_response, respx_mock)
        sync_provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            reasoning_effort="high",
            provider_params=BedrockParams(
                additional_model_request_fields={"thinking": {"type": "enabled", "budget_tokens": 99999}}
            ),
        )
        additional = json.loads(route.calls.last.request.content)["additionalModelRequestFields"]
        assert additional["thinking"] == {"type": "enabled", "budget_tokens": 99999}

    def test_reasoning_effort_does_not_mutate_provider_params(
        self, sync_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok_converse(converse_response, respx_mock)
        fields: dict[str, Any] = {"some_field": "value"}
        sync_provider.chat(
            MODEL,
            [UserMessage(content="Hi")],
            reasoning_effort="high",
            provider_params=BedrockParams(additional_model_request_fields=fields),
        )
        assert "thinking" not in fields

    def test_reasoning_effort_adaptive_model(
        self, sync_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_converse(converse_response, respx_mock, model="anthropic.claude-opus-4-8")
        sync_provider.chat("anthropic.claude-opus-4-8", [UserMessage(content="Hi")], reasoning_effort="high")
        additional = json.loads(route.calls.last.request.content)["additionalModelRequestFields"]
        assert additional["thinking"] == {"type": "adaptive"}
        assert additional["output_config"] == {"effort": "high"}

    def test_status_error_mapped(self, sync_provider: BedrockProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_url(MODEL, "converse")).mock(return_value=httpx.Response(400, json={"message": "bad"}))
        with pytest.raises(InvalidRequestError):
            sync_provider.chat(MODEL, [UserMessage(content="Hi")])

    def test_transport_error_mapped(self, sync_provider: BedrockProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_url(MODEL, "converse")).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            sync_provider.chat(MODEL, [UserMessage(content="Hi")])

    def test_timeout_error_mapped(self, sync_provider: BedrockProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_url(MODEL, "converse")).mock(side_effect=httpx.ConnectTimeout("slow"))
        with pytest.raises(TimeoutError):
            sync_provider.chat(MODEL, [UserMessage(content="Hi")])

    def test_non_json_body_mapped(self, sync_provider: BedrockProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_url(MODEL, "converse")).mock(return_value=httpx.Response(200, content=b"not json"))
        with pytest.raises(ProviderError):
            sync_provider.chat(MODEL, [UserMessage(content="Hi")])


# MARK: Achat


class TestAchat:
    async def test_basic_achat(
        self, async_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok_converse(converse_response, respx_mock)
        result = await async_provider.achat(MODEL, [UserMessage(content="Hi")])
        assert result.content == "Hello!"
        assert result.provider == "aws-bedrock"

    async def test_status_error_mapped(self, async_provider: BedrockProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_url(MODEL, "converse")).mock(return_value=httpx.Response(401, json={"message": "no"}))
        with pytest.raises(AuthenticationError):
            await async_provider.achat(MODEL, [UserMessage(content="Hi")])

    async def test_transport_error_mapped(self, async_provider: BedrockProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_url(MODEL, "converse")).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            await async_provider.achat(MODEL, [UserMessage(content="Hi")])

    async def test_non_json_body_mapped(self, async_provider: BedrockProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_url(MODEL, "converse")).mock(return_value=httpx.Response(200, content=b"not json"))
        with pytest.raises(ProviderError):
            await async_provider.achat(MODEL, [UserMessage(content="Hi")])


# MARK: ChatStream


class TestChatStream:
    def test_yields_chunks(self, sync_provider: BedrockProvider, respx_mock: respx.MockRouter) -> None:
        route = respx_mock.post(_url(MODEL, "converse-stream")).mock(
            return_value=httpx.Response(200, content=_stream_bytes(_STREAM_EVENTS))
        )
        chunks = list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))
        assert len(chunks) == 3
        assert chunks[0].delta == "Hello"
        assert chunks[1].finish_reason == "stop"
        assert chunks[2].usage is not None
        assert chunks[2].usage.input_tokens == 10
        assert chunks[2].usage.output_tokens == 5
        body = json.loads(route.calls.last.request.content)
        assert "modelId" not in body

    def test_terminal_chunk_stamps_model_and_provider(
        self, sync_provider: BedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_url(MODEL, "converse-stream")).mock(
            return_value=httpx.Response(200, content=_stream_bytes(_STREAM_EVENTS))
        )
        chunks = list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))
        assert chunks[0].model is None
        assert chunks[0].provider is None
        assert chunks[2].model == MODEL
        assert chunks[2].provider == "aws-bedrock"

    def test_cost_on_metadata_chunk(self, sync_provider: BedrockProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_url("amazon.nova-pro-v1", "converse-stream")).mock(
            return_value=httpx.Response(200, content=_stream_bytes(_STREAM_EVENTS))
        )
        chunks = list(sync_provider.chat_stream("amazon.nova-pro-v1", [UserMessage(content="Hi")]))
        assert chunks[0].cost is None
        assert chunks[1].cost is None
        assert chunks[2].cost is not None
        assert chunks[2].cost.total_cost > 0

    def test_chunk_without_usage_has_no_cost(
        self, sync_provider: BedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        events: list[tuple[str, dict[str, Any] | None]] = [
            ("contentBlockDelta", {"delta": {"text": "Hi"}, "contentBlockIndex": 0}),
            ("messageStop", {"stopReason": "end_turn"}),
        ]
        respx_mock.post(_url(MODEL, "converse-stream")).mock(
            return_value=httpx.Response(200, content=_stream_bytes(events))
        )
        chunks = list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))
        assert chunks[0] == ChatChunk(delta="Hi")
        assert chunks[0].cost is None
        assert chunks[1].cost is None

    def test_status_error_on_open(self, sync_provider: BedrockProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_url(MODEL, "converse-stream")).mock(return_value=httpx.Response(500, json={"message": "boom"}))
        with pytest.raises(ProviderError):
            list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))

    def test_transport_error_on_open(self, sync_provider: BedrockProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_url(MODEL, "converse-stream")).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))

    def test_malformed_frame_mapped(self, sync_provider: BedrockProvider, respx_mock: respx.MockRouter) -> None:
        bad = _encode({":event-type": "contentBlockDelta", ":message-type": "event"}, b"{not json}")
        respx_mock.post(_url(MODEL, "converse-stream")).mock(return_value=httpx.Response(200, content=bad))
        with pytest.raises(ProviderError):
            list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))

    def test_stream_exception_frame_maps_to_typed_error(
        self, sync_provider: BedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        # A real exception frame carries a camelCase :exception-type and no :event-type; it must both
        # avoid a KeyError on :event-type and classify to the typed error (RateLimitError here).
        content = _frame("contentBlockDelta", {"delta": {"text": "Hi"}, "contentBlockIndex": 0}) + _frame(
            "throttlingException", {"message": "slow down"}, message_type="exception"
        )
        respx_mock.post(_url(MODEL, "converse-stream")).mock(return_value=httpx.Response(200, content=content))
        with pytest.raises(RateLimitError, match="slow down"):
            list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))

    def test_stream_error_frame_surfaces_service_error(
        self, sync_provider: BedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        # An unmodeled error frame carries :error-code/:error-message and no :event-type; the real
        # service error must surface, not a KeyError-wrapped "':event-type'".
        content = _encode(
            {":message-type": "error", ":error-code": "internalServerException", ":error-message": "boom"}, b""
        )
        respx_mock.post(_url(MODEL, "converse-stream")).mock(return_value=httpx.Response(200, content=content))
        with pytest.raises(ProviderError, match="boom"):
            list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))

    def test_client_init_failure(self, fake_auth: FakeAuth, sync_create_raises: MagicMock) -> None:
        provider = BedrockProvider(auth=fake_auth)
        with pytest.raises(ProviderError, match="client init failed"):
            list(provider.chat_stream(MODEL, [UserMessage(content="Hi")]))
        sync_create_raises.assert_called_once()


# MARK: AchatStream


class TestAchatStream:
    async def test_yields_chunks(self, async_provider: BedrockProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_url("amazon.nova-pro-v1", "converse-stream")).mock(
            return_value=httpx.Response(200, content=_stream_bytes(_STREAM_EVENTS))
        )
        chunks = [c async for c in async_provider.achat_stream("amazon.nova-pro-v1", [UserMessage(content="Hi")])]
        assert len(chunks) == 3
        assert chunks[0].delta == "Hello"
        assert chunks[1].finish_reason == "stop"
        assert chunks[2].cost is not None

    async def test_terminal_chunk_stamps_model_and_provider(
        self, async_provider: BedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_url(MODEL, "converse-stream")).mock(
            return_value=httpx.Response(200, content=_stream_bytes(_STREAM_EVENTS))
        )
        chunks = [c async for c in async_provider.achat_stream(MODEL, [UserMessage(content="Hi")])]
        assert chunks[0].model is None
        assert chunks[2].model == MODEL
        assert chunks[2].provider == "aws-bedrock"

    async def test_transport_error_on_open(self, async_provider: BedrockProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_url(MODEL, "converse-stream")).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            async for _ in async_provider.achat_stream(MODEL, [UserMessage(content="Hi")]):
                pass  # pragma: no cover

    async def test_status_error_on_open(self, async_provider: BedrockProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_url(MODEL, "converse-stream")).mock(return_value=httpx.Response(500, json={"message": "boom"}))
        with pytest.raises(ProviderError):
            async for _ in async_provider.achat_stream(MODEL, [UserMessage(content="Hi")]):
                pass  # pragma: no cover

    async def test_malformed_frame_mapped(self, async_provider: BedrockProvider, respx_mock: respx.MockRouter) -> None:
        bad = _encode({":event-type": "contentBlockDelta", ":message-type": "event"}, b"{not json}")
        respx_mock.post(_url(MODEL, "converse-stream")).mock(return_value=httpx.Response(200, content=bad))
        with pytest.raises(ProviderError):
            async for _ in async_provider.achat_stream(MODEL, [UserMessage(content="Hi")]):
                pass  # pragma: no cover

    async def test_exception_frame_maps_to_typed_error(
        self, async_provider: BedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        content = _frame("contentBlockDelta", {"delta": {"text": "Hi"}, "contentBlockIndex": 0}) + _frame(
            "validationException", {"message": "invalid"}, message_type="exception"
        )
        respx_mock.post(_url(MODEL, "converse-stream")).mock(return_value=httpx.Response(200, content=content))
        with pytest.raises(InvalidRequestError, match="invalid"):
            async for _ in async_provider.achat_stream(MODEL, [UserMessage(content="Hi")]):
                pass

    async def test_client_init_failure(self, fake_auth: FakeAuth, async_create_raises: MagicMock) -> None:
        provider = BedrockProvider(auth=fake_auth)
        with pytest.raises(ProviderError, match="client init failed"):
            async for _ in provider.achat_stream(MODEL, [UserMessage(content="Hi")]):
                pass  # pragma: no cover
        async_create_raises.assert_called_once()


# MARK: Embed


class TestEmbed:
    def test_basic_embed(
        self, sync_provider: BedrockProvider, embedding_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_embed(embedding_response, respx_mock)
        result = sync_provider.embed(EMBED_MODEL, "hello")

        assert result == EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3]],
            usage=Usage(input_tokens=5, output_tokens=0),
            cost=result.cost,
            model=EMBED_MODEL,
            provider="aws-bedrock",
        )
        assert json.loads(route.calls.last.request.content) == {"inputText": "hello"}

    def test_embed_list_input(self, sync_provider: BedrockProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_url(EMBED_MODEL, "invoke")).mock(
            side_effect=[
                httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3], "inputTextTokenCount": 3}),
                httpx.Response(200, json={"embedding": [0.4, 0.5, 0.6], "inputTextTokenCount": 4}),
            ]
        )
        result = sync_provider.embed(EMBED_MODEL, ["hello", "world"])
        assert result == EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            usage=Usage(input_tokens=7, output_tokens=0),
            cost=result.cost,
            model=EMBED_MODEL,
            provider="aws-bedrock",
        )

    def test_embed_with_dimensions(
        self, sync_provider: BedrockProvider, embedding_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_embed(embedding_response, respx_mock)
        sync_provider.embed(EMBED_MODEL, "hello", dimensions=256)
        assert json.loads(route.calls.last.request.content) == {"inputText": "hello", "dimensions": 256}

    def test_embed_status_error_mapped(self, sync_provider: BedrockProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_url(EMBED_MODEL, "invoke")).mock(return_value=httpx.Response(400, json={"message": "bad"}))
        with pytest.raises(InvalidRequestError):
            sync_provider.embed(EMBED_MODEL, "hello")

    def test_embed_transport_error_mapped(self, sync_provider: BedrockProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_url(EMBED_MODEL, "invoke")).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            sync_provider.embed(EMBED_MODEL, "hello")

    def test_embed_client_init_failure_mapped(self, fake_auth: FakeAuth, sync_create_raises: MagicMock) -> None:
        provider = BedrockProvider(auth=fake_auth)
        with pytest.raises(ProviderError, match="client init failed"):
            provider.embed(EMBED_MODEL, "hello")
        sync_create_raises.assert_called_once()


# MARK: Aembed


class TestAembed:
    async def test_basic_aembed(
        self, async_provider: BedrockProvider, embedding_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_embed(embedding_response, respx_mock)
        result = await async_provider.aembed(EMBED_MODEL, "hello")
        assert result.embeddings == [[0.1, 0.2, 0.3]]
        assert result.usage == Usage(input_tokens=5, output_tokens=0)
        assert result.provider == "aws-bedrock"
        assert json.loads(route.calls.last.request.content) == {"inputText": "hello"}

    async def test_aembed_list_input(self, async_provider: BedrockProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_url(EMBED_MODEL, "invoke")).mock(
            side_effect=[
                httpx.Response(200, json={"embedding": [0.1, 0.2], "inputTextTokenCount": 3}),
                httpx.Response(200, json={"embedding": [0.3, 0.4], "inputTextTokenCount": 4}),
            ]
        )
        result = await async_provider.aembed(EMBED_MODEL, ["hello", "world"])
        assert result.embeddings == [[0.1, 0.2], [0.3, 0.4]]
        assert result.usage == Usage(input_tokens=7, output_tokens=0)

    async def test_aembed_with_dimensions(
        self, async_provider: BedrockProvider, embedding_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_embed(embedding_response, respx_mock)
        await async_provider.aembed(EMBED_MODEL, "hello", dimensions=256)
        assert json.loads(route.calls.last.request.content) == {"inputText": "hello", "dimensions": 256}

    async def test_aembed_status_error_mapped(
        self, async_provider: BedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_url(EMBED_MODEL, "invoke")).mock(return_value=httpx.Response(400, json={"message": "no"}))
        with pytest.raises(InvalidRequestError):
            await async_provider.aembed(EMBED_MODEL, "hello")

    async def test_aembed_transport_error_mapped(
        self, async_provider: BedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_url(EMBED_MODEL, "invoke")).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError, match="refused"):
            await async_provider.aembed(EMBED_MODEL, "hello")


# MARK: Client & Auth Management


class TestClientManagement:
    def test_sync_client_reused(
        self, sync_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok_converse(converse_response, respx_mock)
        sync_provider.chat(MODEL, [UserMessage(content="a")])
        client = sync_provider._sync_client
        sync_provider.chat(MODEL, [UserMessage(content="b")])
        assert sync_provider._sync_client is client

    def test_auth_resolved_once(
        self, fake_auth: FakeAuth, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok_converse(converse_response, respx_mock)
        provider = BedrockProvider(auth=fake_auth)
        provider.chat(MODEL, [UserMessage(content="a")])
        provider.chat(MODEL, [UserMessage(content="b")])
        assert fake_auth.get_calls == 1

    def test_region_selects_endpoint_host(
        self, fake_auth: FakeAuth, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        base = "https://bedrock-runtime.us-west-2.amazonaws.com"
        route = respx_mock.post(_url(MODEL, "converse", base=base)).mock(
            return_value=httpx.Response(200, json=converse_response)
        )
        provider = BedrockProvider(auth=fake_auth, region="us-west-2")
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.called

    def test_session_region_selects_endpoint_host(
        self, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        base = "https://bedrock-runtime.ap-south-1.amazonaws.com"
        route = respx_mock.post(_url(MODEL, "converse", base=base)).mock(
            return_value=httpx.Response(200, json=converse_response)
        )
        provider = BedrockProvider(auth=FakeAuth(_Session(region_name="ap-south-1")))
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.called

    def test_session_region_prices_regionally(
        self, converse_response: dict[str, Any], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cost follows the Region the request was sent to even when no ``region=`` was passed.

        Leaving it unset and letting the session resolve it (AWS_DEFAULT_REGION, a profile) is the
        common configuration, so pricing must read the resolved Region rather than the constructor
        argument, which is None here.
        """
        regional = {
            "ap-south-1": {
                MODEL: ModelPricing(
                    tiers=[
                        PricingTier(
                            input_cost_per_token=per_million_tokens(9.0),
                            output_cost_per_token=per_million_tokens(9.0),
                        )
                    ],
                ),
            },
        }
        monkeypatch.setattr("lmux_aws_bedrock.cost._REGIONAL_PRICING", regional)
        base = "https://bedrock-runtime.ap-south-1.amazonaws.com"
        respx_mock.post(_url(MODEL, "converse", base=base)).mock(
            return_value=httpx.Response(200, json=converse_response)
        )
        provider = BedrockProvider(auth=FakeAuth(_Session(region_name="ap-south-1")))
        response = provider.chat(MODEL, [UserMessage(content="Hi")])
        assert response.cost is not None
        assert response.cost.input_cost == pytest.approx(10 * 9.0 / 1_000_000)

    def test_endpoint_url_override(
        self, fake_auth: FakeAuth, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        base = "https://custom.bedrock.endpoint"
        route = respx_mock.post(_url(MODEL, "converse", base=base)).mock(
            return_value=httpx.Response(200, json=converse_response)
        )
        provider = BedrockProvider(auth=fake_auth, endpoint_url=base)
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.called

    def test_use_fips_selects_fips_host(
        self, fake_auth: FakeAuth, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        base = "https://bedrock-runtime-fips.us-east-1.amazonaws.com"
        route = respx_mock.post(_url(MODEL, "converse", base=base)).mock(
            return_value=httpx.Response(200, json=converse_response)
        )
        provider = BedrockProvider(auth=fake_auth, use_fips=True)
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.called

    def test_timeout_and_retries_forwarded(self, fake_auth: FakeAuth, mock_create_sync: MagicMock) -> None:
        provider = BedrockProvider(auth=fake_auth, timeout=30.0, max_retries=5)
        provider._get_sync_client()
        mock_create_sync.assert_called_once_with(
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com", timeout=30.0, max_retries=5, transport=None
        )

    async def test_async_timeout_and_retries_forwarded(self, fake_auth: FakeAuth, mock_create_async: MagicMock) -> None:
        provider = BedrockProvider(auth=fake_auth, timeout=30.0, max_retries=5)
        await provider._get_async_client()
        mock_create_async.assert_called_once_with(
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com", timeout=30.0, max_retries=5, transport=None
        )

    def test_no_credentials_raises_auth_error(
        self, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok_converse(converse_response, respx_mock)
        provider = BedrockProvider(auth=FakeAuth(_Session(credentials=None)))
        with pytest.raises(AuthenticationError):
            provider.chat(MODEL, [UserMessage(content="Hi")])

    async def test_async_client_reused(
        self, async_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok_converse(converse_response, respx_mock)
        await async_provider.achat(MODEL, [UserMessage(content="a")])
        client = async_provider._async_client
        await async_provider.achat(MODEL, [UserMessage(content="b")])
        assert async_provider._async_client is client

    async def test_async_client_recreated_on_new_loop(
        self,
        fake_auth: FakeAuth,
        async_create_two_clients: tuple[MagicMock, MagicMock, MagicMock],
        mock_get_running_loop: MagicMock,
    ) -> None:
        create, c1, c2 = async_create_two_clients
        loop1, loop2 = asyncio.new_event_loop(), asyncio.new_event_loop()
        mock_get_running_loop.side_effect = [loop1, loop2]
        provider = BedrockProvider(auth=fake_auth)
        r1 = await provider._get_async_client()
        r2 = await provider._get_async_client()
        assert (r1, r2) == (c1, c2)
        assert create.call_count == 2
        loop1.close()
        loop2.close()

    def test_sync_client_init_failure_mapped(self, fake_auth: FakeAuth, sync_create_raises: MagicMock) -> None:
        provider = BedrockProvider(auth=fake_auth)
        with pytest.raises(ProviderError, match="client init failed"):
            provider.chat(MODEL, [UserMessage(content="Hi")])
        sync_create_raises.assert_called_once()

    def test_custom_transport_used(self, fake_auth: FakeAuth, converse_response: dict[str, Any]) -> None:
        # The injected transport must be the one that serves the request.
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=converse_response)

        provider = BedrockProvider(auth=fake_auth, transport=httpx.MockTransport(handler))
        resp = provider.chat(MODEL, [UserMessage(content="Hi")])
        assert len(requests) == 1
        assert resp.provider == "aws-bedrock"

    async def test_custom_async_transport_used(self, fake_auth: FakeAuth, converse_response: dict[str, Any]) -> None:
        # The injected async transport must be the one that serves the request.
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=converse_response)

        provider = BedrockProvider(auth=fake_auth, async_transport=httpx.MockTransport(handler))
        resp = await provider.achat(MODEL, [UserMessage(content="Hi")])
        assert len(requests) == 1
        assert resp.provider == "aws-bedrock"


# MARK: SigV4 Auth


class TestSigV4Auth:
    def test_freezes_credentials_per_request(
        self, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        class _RotatingCredentials:
            def __init__(self) -> None:
                self.calls = 0

            def get_frozen_credentials(self) -> Any:  # noqa: ANN401
                self.calls += 1
                return SimpleNamespace(
                    access_key="AKIAFAKEEXAMPLE",
                    secret_key="fake-secret-key",  # noqa: S106
                    token=f"session-token-{self.calls}",
                )

        creds = _RotatingCredentials()
        provider = BedrockProvider(auth=FakeAuth(_Session(credentials=cast("_Credentials", creds))))
        route = _ok_converse(converse_response, respx_mock)
        provider.chat(MODEL, [UserMessage(content="Hi")])
        provider.chat(MODEL, [UserMessage(content="Hi")])
        # Credentials are frozen once per request (not cached at client creation), so a rotating
        # source flows a fresh token into each signature.
        assert creds.calls == 2
        assert route.calls[0].request.headers["x-amz-security-token"] == "session-token-1"
        assert route.calls[1].request.headers["x-amz-security-token"] == "session-token-2"


# MARK: Bearer Token Auth


class TestBearerAuth:
    def test_bearer_token_sets_authorization(
        self, converse_response: dict[str, Any], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bt-secret")
        route = _ok_converse(converse_response, respx_mock)
        # A session with no region falls back to us-east-1 (the default _url region).
        provider = BedrockProvider(auth=FakeAuth())
        result = provider.chat(MODEL, [UserMessage(content="Hi")])
        assert result.content == "Hello!"
        assert route.calls.last.request.headers["authorization"] == "Bearer bt-secret"

    def test_default_headers_preserve_bearer_auth(
        self, converse_response: dict[str, Any], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bt-secret")
        route = _ok_converse(converse_response, respx_mock)
        provider = BedrockProvider(
            auth=FakeAuth(), default_headers={"X-Trace-ID": "trace-123", "authorization": "Bearer wrong"}
        )
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.calls.last.request.headers["x-trace-id"] == "trace-123"
        assert route.calls.last.request.headers["authorization"] == "Bearer bt-secret"

    def test_bearer_token_honors_session_region(
        self, converse_response: dict[str, Any], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bt-secret")
        base = "https://bedrock-runtime.eu-west-1.amazonaws.com"
        route = respx_mock.post(_url(MODEL, "converse", base=base)).mock(
            return_value=httpx.Response(200, json=converse_response)
        )
        # No constructor region, but the session is configured for eu-west-1 — bearer mode must use it
        # instead of silently defaulting to us-east-1.
        provider = BedrockProvider(auth=FakeAuth(_Session(region_name="eu-west-1")))
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.called

    def test_bearer_token_respects_region(
        self, converse_response: dict[str, Any], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bt-secret")
        base = "https://bedrock-runtime.eu-central-1.amazonaws.com"
        route = respx_mock.post(_url(MODEL, "converse", base=base)).mock(
            return_value=httpx.Response(200, json=converse_response)
        )
        provider = BedrockProvider(auth=FakeAuth(), region="eu-central-1")
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.called

    def test_bearer_token_survives_unconstructable_session(
        self, converse_response: dict[str, Any], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bt-secret")
        route = _ok_converse(converse_response, respx_mock)  # default us-east-1 endpoint
        # A stale AWS_PROFILE makes boto3.Session() raise at construction; bearer mode must not fail —
        # it falls back to the default region since it never needs the session.
        provider = BedrockProvider(auth=_RaisingSessionAuth())
        result = provider.chat(MODEL, [UserMessage(content="Hi")])
        assert result.content == "Hello!"
        assert route.calls.last.request.headers["authorization"] == "Bearer bt-secret"


# MARK: Register Pricing


class TestRegisterPricing:
    def test_custom_pricing_for_unknown_model(
        self, sync_provider: BedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        response = {
            "output": {"message": {"role": "assistant", "content": [{"text": "Hello!"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1000, "outputTokens": 500, "totalTokens": 1500},
        }
        respx_mock.post(_url("custom.my-model-v1", "converse")).mock(return_value=httpx.Response(200, json=response))
        sync_provider.register_pricing(
            "custom.my-model-v1",
            ModelPricing(
                tiers=[PricingTier(input_cost_per_token=5.0 / 1_000_000, output_cost_per_token=15.0 / 1_000_000)]
            ),
        )
        result = sync_provider.chat("custom.my-model-v1", [UserMessage(content="Hi")])
        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(1000 * 5.0 / 1_000_000)
        assert result.cost.output_cost == pytest.approx(500 * 15.0 / 1_000_000)

    def test_custom_pricing_overrides_builtin(
        self, sync_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok_converse(converse_response, respx_mock)
        sync_provider.register_pricing(
            MODEL,
            ModelPricing(
                tiers=[PricingTier(input_cost_per_token=99.0 / 1_000_000, output_cost_per_token=199.0 / 1_000_000)]
            ),
        )
        result = sync_provider.chat(MODEL, [UserMessage(content="Hi")])
        assert result.cost is not None
        assert result.cost.input_cost == pytest.approx(10 * 99.0 / 1_000_000)
        assert result.cost.output_cost == pytest.approx(5 * 199.0 / 1_000_000)

    def test_unregistered_unknown_model_returns_none_cost(
        self, sync_provider: BedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        response = {
            "output": {"message": {"role": "assistant", "content": [{"text": "Hello!"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }
        respx_mock.post(_url("totally-unknown-model", "converse")).mock(return_value=httpx.Response(200, json=response))
        result = sync_provider.chat("totally-unknown-model", [UserMessage(content="Hi")])
        assert result.cost is None


# MARK: Provider Params Kwargs


class TestProviderParamsKwargs:
    def test_empty_params(
        self, sync_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_converse(converse_response, respx_mock)
        sync_provider.chat(MODEL, [UserMessage(content="Hi")], provider_params=BedrockParams())
        body = json.loads(route.calls.last.request.content)
        assert "guardrailConfig" not in body
        assert "additionalModelRequestFields" not in body
        assert "additionalModelResponseFieldPaths" not in body

    def test_all_params(
        self, sync_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_converse(converse_response, respx_mock)
        params = BedrockParams(
            guardrail_config=GuardrailConfig(guardrail_identifier="g1", guardrail_version="1", trace="enabled"),
            additional_model_request_fields={"custom_field": "value"},
            additional_model_response_field_paths=["$.path.to.field"],
        )
        sync_provider.chat(MODEL, [UserMessage(content="Hi")], provider_params=params)
        body = json.loads(route.calls.last.request.content)
        assert body["guardrailConfig"] == {"guardrailIdentifier": "g1", "guardrailVersion": "1", "trace": "enabled"}
        assert body["additionalModelRequestFields"] == {"custom_field": "value"}
        assert body["additionalModelResponseFieldPaths"] == ["$.path.to.field"]

    def test_guardrail_without_trace(
        self, sync_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        route = _ok_converse(converse_response, respx_mock)
        params = BedrockParams(guardrail_config=GuardrailConfig(guardrail_identifier="g1", guardrail_version="1"))
        sync_provider.chat(MODEL, [UserMessage(content="Hi")], provider_params=params)
        body = json.loads(route.calls.last.request.content)
        assert body["guardrailConfig"] == {"guardrailIdentifier": "g1", "guardrailVersion": "1"}


# MARK: Aclose & Preload


class TestAclose:
    async def test_closes_client(
        self, async_provider: BedrockProvider, converse_response: dict[str, Any], respx_mock: respx.MockRouter
    ) -> None:
        _ok_converse(converse_response, respx_mock)
        await async_provider.achat(MODEL, [UserMessage(content="Hi")])
        assert async_provider._async_client is not None
        await async_provider.aclose()
        assert async_provider._async_client is None

    async def test_noop_when_no_client(self, async_provider: BedrockProvider) -> None:
        await async_provider.aclose()


class TestPreload:
    def test_preload(self) -> None:
        preload()
