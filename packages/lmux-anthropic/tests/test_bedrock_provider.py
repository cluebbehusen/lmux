"""Tests for the native Anthropic-on-Bedrock provider (SDK-lite, respx)."""

import base64
import json
import struct
import zlib
from datetime import date
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
import respx

if TYPE_CHECKING:
    import boto3

from lmux.cost import ModelPricing, PricingTier
from lmux.exceptions import AuthenticationError, InvalidRequestError, ProviderError, RateLimitError
from lmux.types import Cost, UserMessage
from lmux_anthropic import AnthropicBedrockProvider
from lmux_anthropic.params import AnthropicParams

_BASE = "https://bedrock-runtime.us-east-1.amazonaws.com"
MODEL = "anthropic.claude-opus-4-8"
_RESPONSE_MODEL = "claude-opus-4-8"  # what the native response echoes (region-less)


def _url(model: str, action: str, *, base: str = _BASE) -> str:
    return f"{base}/model/{model}/{action}"


# MARK: Fake SigV4 auth (boto3-Session-like, resolved synchronously)


class _FrozenCreds:
    access_key = "AKIAFAKEEXAMPLE"
    secret_key = "fake-secret-key"  # noqa: S105
    token = None


class _Credentials:
    def get_frozen_credentials(self) -> _FrozenCreds:
        return _FrozenCreds()


class _Session:
    def __init__(self, *, region_name: str | None = None, credentials: _Credentials | None = None) -> None:
        self.region_name = region_name
        self._credentials = _Credentials() if credentials is None else credentials

    def get_credentials(self) -> _Credentials | None:
        return self._credentials


class FakeAuth:
    def __init__(self, session: _Session | None = None) -> None:
        self._session = session if session is not None else _Session()

    def get_credentials(self) -> "boto3.Session":
        return cast("boto3.Session", self._session)


class _NoCredsSession(_Session):
    def get_credentials(self) -> _Credentials | None:
        return None


# MARK: Response + event-stream builders


def _message(*, model: str = _RESPONSE_MODEL, input_tokens: int = 10, output_tokens: int = 5) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "Hello!"}],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _encode(headers: dict[str, str], payload: bytes) -> bytes:
    hb = b""
    for name, value in headers.items():
        nb, vb = name.encode(), value.encode()
        hb += bytes([len(nb)]) + nb + bytes([7]) + struct.pack(">H", len(vb)) + vb
    total = 16 + len(hb) + len(payload)
    prelude = struct.pack(">II", total, len(hb))
    body = prelude + struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF) + hb + payload
    return body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def _chunk(event: dict[str, Any]) -> bytes:
    """A Bedrock ``chunk`` frame wrapping a native Anthropic streaming event as base64 under ``bytes``."""
    inner = base64.b64encode(json.dumps(event).encode()).decode()
    payload = json.dumps({"bytes": inner}).encode()
    return _encode({":event-type": "chunk", ":message-type": "event", ":content-type": "application/json"}, payload)


def _exception_frame(exception_type: str, message: str) -> bytes:
    return _encode(
        {":message-type": "exception", ":exception-type": exception_type}, json.dumps({"message": message}).encode()
    )


def _error_frame(error_code: str, message: str) -> bytes:
    return _encode({":message-type": "error", ":error-code": error_code, ":error-message": message}, b"")


_ANTHROPIC_STREAM_EVENTS: list[dict[str, Any]] = [
    {"type": "message_start", "message": {"model": _RESPONSE_MODEL, "usage": {"input_tokens": 10, "output_tokens": 0}}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo!"}},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
    {"type": "message_stop"},
]


def _default_stream() -> bytes:
    return b"".join(_chunk(event) for event in _ANTHROPIC_STREAM_EVENTS)


# MARK: Fixtures


@pytest.fixture(autouse=True)
def _no_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default every test to the SigV4 path; bearer-token tests opt back in."""
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)


@pytest.fixture
def sync_provider() -> AnthropicBedrockProvider:
    return AnthropicBedrockProvider(auth=FakeAuth())


# MARK: Chat (unary)


class TestChat:
    def test_signs_and_maps_response(
        self, sync_provider: AnthropicBedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(_url(MODEL, "invoke")).mock(return_value=httpx.Response(200, json=_message()))
        response = sync_provider.chat(MODEL, [UserMessage(content="Hi")])

        assert response.content == "Hello!"
        assert response.provider == "anthropic-bedrock"
        assert response.model == _RESPONSE_MODEL
        assert response.cost is not None
        # Body drops model/stream and carries anthropic_version; SigV4 header is attached.
        request = route.calls.last.request
        body = json.loads(request.content)
        assert body["anthropic_version"] == "bedrock-2023-05-31"
        assert "model" not in body
        assert "stream" not in body
        assert request.headers["Authorization"].startswith("AWS4-HMAC-SHA256")

    def test_cost_uses_request_model_pricing(
        self, sync_provider: AnthropicBedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        # us. regional profile prices above the bare/us-east-1 entry; cost must reflect the request id.
        respx_mock.post(_url("us.anthropic.claude-opus-4-8", "invoke")).mock(
            return_value=httpx.Response(200, json=_message(input_tokens=1_000_000, output_tokens=0))
        )
        response = sync_provider.chat("us.anthropic.claude-opus-4-8", [UserMessage(content="Hi")])
        assert response.cost is not None
        assert response.cost.input_cost == 5.5  # us.anthropic.claude-opus-4-8 == list x1.1

    async def test_achat_signs_and_maps(self, respx_mock: respx.MockRouter) -> None:
        provider = AnthropicBedrockProvider(auth=FakeAuth())
        respx_mock.post(_url(MODEL, "invoke")).mock(return_value=httpx.Response(200, json=_message()))
        response = await provider.achat(MODEL, [UserMessage(content="Hi")])
        assert response.content == "Hello!"
        assert response.cost is not None
        await provider.aclose()

    def test_drops_service_tier_and_inference_geo(
        self, sync_provider: AnthropicBedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        route = respx_mock.post(_url(MODEL, "invoke")).mock(return_value=httpx.Response(200, json=_message()))
        params = AnthropicParams(service_tier="standard_only", inference_geo="us", top_k=5)
        sync_provider.chat(MODEL, [UserMessage(content="Hi")], provider_params=params)
        body = json.loads(route.calls.last.request.content)
        assert "service_tier" not in body
        assert "inference_geo" not in body
        assert body["top_k"] == 5

    def test_inference_geo_multiplier_not_applied(
        self, sync_provider: AnthropicBedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_url(MODEL, "invoke")).mock(return_value=httpx.Response(200, json=_message()))
        standard = sync_provider.chat(MODEL, [UserMessage(content="Hi")])
        us = sync_provider.chat(MODEL, [UserMessage(content="Hi")], provider_params=AnthropicParams(inference_geo="us"))
        assert standard.cost is not None
        assert us.cost is not None
        assert standard.cost.total_cost == us.cost.total_cost

    def test_custom_pricing_overrides_table(
        self, sync_provider: AnthropicBedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        sync_provider.register_pricing(
            MODEL,
            ModelPricing(
                tiers=[PricingTier(input_cost_per_token=1.0, output_cost_per_token=2.0)],
            ),
        )
        respx_mock.post(_url(MODEL, "invoke")).mock(
            return_value=httpx.Response(200, json=_message(input_tokens=1, output_tokens=1))
        )
        response = sync_provider.chat(MODEL, [UserMessage(content="Hi")])
        assert response.cost == Cost(input_cost=1.0, output_cost=2.0, total_cost=3.0)

    def test_endpoint_url_and_fips_override_base(self, respx_mock: respx.MockRouter) -> None:
        fips = AnthropicBedrockProvider(auth=FakeAuth(), use_fips=True)
        fips_url = _url(MODEL, "invoke", base="https://bedrock-runtime-fips.us-east-1.amazonaws.com")
        route = respx_mock.post(fips_url).mock(return_value=httpx.Response(200, json=_message()))
        fips.chat(MODEL, [UserMessage(content="Hi")])
        assert route.called

    def test_region_from_session(self, respx_mock: respx.MockRouter) -> None:
        provider = AnthropicBedrockProvider(auth=FakeAuth(_Session(region_name="eu-west-1")))
        route = respx_mock.post(_url(MODEL, "invoke", base="https://bedrock-runtime.eu-west-1.amazonaws.com")).mock(
            return_value=httpx.Response(200, json=_message())
        )
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.called

    def test_error_response_maps_to_lmux_error(
        self, sync_provider: AnthropicBedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_url(MODEL, "invoke")).mock(
            return_value=httpx.Response(
                400, headers={"x-amzn-errortype": "ValidationException"}, json={"message": "bad"}
            )
        )
        with pytest.raises(InvalidRequestError):
            sync_provider.chat(MODEL, [UserMessage(content="Hi")])

    def test_transport_error_maps_to_provider_error(
        self, sync_provider: AnthropicBedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_url(MODEL, "invoke")).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ProviderError):
            sync_provider.chat(MODEL, [UserMessage(content="Hi")])

    def test_missing_credentials_maps_to_auth_error(self, respx_mock: respx.MockRouter) -> None:  # noqa: ARG002
        provider = AnthropicBedrockProvider(auth=FakeAuth(_NoCredsSession()))
        with pytest.raises(AuthenticationError):
            provider.chat(MODEL, [UserMessage(content="Hi")])


# MARK: Bearer-token auth


class TestBearerAuth:
    def test_bearer_token_sets_authorization_header(
        self, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bt-secret")
        provider = AnthropicBedrockProvider(auth=FakeAuth(_Session(region_name="us-east-1")))
        route = respx_mock.post(_url(MODEL, "invoke")).mock(return_value=httpx.Response(200, json=_message()))
        provider.chat(MODEL, [UserMessage(content="Hi")])
        assert route.calls.last.request.headers["Authorization"] == "Bearer bt-secret"


# MARK: Streaming


class TestChatStream:
    def test_streams_events_and_bills_request_model(
        self, sync_provider: AnthropicBedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        respx_mock.post(_url(MODEL, "invoke-with-response-stream")).mock(
            return_value=httpx.Response(200, content=_default_stream())
        )
        chunks = list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))
        text = "".join(c.delta for c in chunks if c.delta)
        assert text == "Hello!"
        final = chunks[-1]
        assert final.model == _RESPONSE_MODEL
        assert final.provider == "anthropic-bedrock"
        assert final.cost is not None  # priced by the request id despite the region-less response model

    async def test_achat_stream(self, respx_mock: respx.MockRouter) -> None:
        provider = AnthropicBedrockProvider(auth=FakeAuth())
        respx_mock.post(_url(MODEL, "invoke-with-response-stream")).mock(
            return_value=httpx.Response(200, content=_default_stream())
        )
        chunks = [c async for c in provider.achat_stream(MODEL, [UserMessage(content="Hi")])]
        assert "".join(c.delta for c in chunks if c.delta) == "Hello!"
        await provider.aclose()

    def test_stream_http_error(self, sync_provider: AnthropicBedrockProvider, respx_mock: respx.MockRouter) -> None:
        respx_mock.post(_url(MODEL, "invoke-with-response-stream")).mock(
            return_value=httpx.Response(
                429, headers={"x-amzn-errortype": "ThrottlingException"}, json={"message": "slow down"}
            )
        )
        with pytest.raises(RateLimitError):
            list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))

    def test_stream_exception_frame(
        self, sync_provider: AnthropicBedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        body = _chunk(_ANTHROPIC_STREAM_EVENTS[0]) + _exception_frame("throttlingException", "slow down")
        respx_mock.post(_url(MODEL, "invoke-with-response-stream")).mock(return_value=httpx.Response(200, content=body))
        with pytest.raises(RateLimitError):
            list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))

    def test_stream_error_frame(self, sync_provider: AnthropicBedrockProvider, respx_mock: respx.MockRouter) -> None:
        body = _chunk(_ANTHROPIC_STREAM_EVENTS[0]) + _error_frame("ModelStreamErrorException", "boom")
        respx_mock.post(_url(MODEL, "invoke-with-response-stream")).mock(return_value=httpx.Response(200, content=body))
        with pytest.raises(ProviderError):
            list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))

    def test_stream_anthropic_error_event(
        self, sync_provider: AnthropicBedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        # A mid-stream Anthropic error rides in an ordinary chunk frame on a 200 response, not an
        # AWS exception frame; raise rather than truncate the stream silently.
        error_event = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        body = _chunk(_ANTHROPIC_STREAM_EVENTS[0]) + _chunk(_ANTHROPIC_STREAM_EVENTS[1]) + _chunk(error_event)
        respx_mock.post(_url(MODEL, "invoke-with-response-stream")).mock(return_value=httpx.Response(200, content=body))
        with pytest.raises(ProviderError, match="Overloaded"):
            list(sync_provider.chat_stream(MODEL, [UserMessage(content="Hi")]))

    async def test_astream_http_error(self, respx_mock: respx.MockRouter) -> None:
        provider = AnthropicBedrockProvider(auth=FakeAuth())
        respx_mock.post(_url(MODEL, "invoke-with-response-stream")).mock(
            return_value=httpx.Response(400, headers={"x-amzn-errortype": "ValidationException"}, json={"message": "x"})
        )
        with pytest.raises(InvalidRequestError):
            [c async for c in provider.achat_stream(MODEL, [UserMessage(content="Hi")])]
        await provider.aclose()


# MARK: Pricing as-of / cost model


class TestCost:
    def test_pricing_as_of_dated_schedule(
        self, sync_provider: AnthropicBedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        url = _url("anthropic.claude-sonnet-5", "invoke")
        respx_mock.post(url).mock(
            return_value=httpx.Response(200, json=_message(input_tokens=1_000_000, output_tokens=0))
        )
        intro = sync_provider.chat(
            "anthropic.claude-sonnet-5",
            [UserMessage(content="Hi")],
            provider_params=AnthropicParams(pricing_as_of=date(2026, 7, 1)),
        )
        assert intro.cost is not None
        assert intro.cost.input_cost == 2.2  # us-east-1 intro rate (2.0 list x1.1)

    def test_unknown_model_has_no_cost(
        self, sync_provider: AnthropicBedrockProvider, respx_mock: respx.MockRouter
    ) -> None:
        url = _url("anthropic.unknown-model", "invoke")
        respx_mock.post(url).mock(return_value=httpx.Response(200, json=_message(model="unknown-model")))
        response = sync_provider.chat("anthropic.unknown-model", [UserMessage(content="Hi")])
        assert response.cost is None
