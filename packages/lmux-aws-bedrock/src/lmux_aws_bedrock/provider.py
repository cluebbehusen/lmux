"""AWS Bedrock provider implementation (SDK-lite, httpx transport).

Requests go straight to the Bedrock runtime REST endpoints over httpx. Authentication
is one of two modes, resolved once on first use:

* **Bearer token** — if ``AWS_BEARER_TOKEN_BEDROCK`` is set, each request carries an
  ``Authorization: Bearer <token>`` header and nothing is signed.
* **SigV4** — otherwise classic AWS credentials are resolved through boto3 (kept purely
  for the credential chain) and each request is signed with :mod:`._sigv4`.

Credentials are resolved synchronously even on the async path — reimplementing the AWS
credential chain asynchronously is out of scope, and boto3 stays a dependency only for
this step.
"""

import asyncio
import json
import os
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast, override
from urllib.parse import quote

if TYPE_CHECKING:
    import boto3
    import httpx
    from aiobotocore.session import AioSession
    from mypy_boto3_bedrock_runtime.type_defs import ConverseResponseTypeDef, ConverseStreamOutputTypeDef

from lmux.cost import ModelPricing, calculate_cost
from lmux.exceptions import LmuxError, ProviderError
from lmux.protocols import AuthProvider, CompletionProvider, EmbeddingProvider, PricingProvider
from lmux.types import (
    ChatChunk,
    ChatResponse,
    Cost,
    EmbeddingResponse,
    Message,
    ResponseFormat,
    Tool,
    ToolChoice,
    Usage,
)
from lmux_aws_bedrock._eventstream import decode_messages
from lmux_aws_bedrock._exceptions import PROVIDER, map_transport_error, parse_json, raise_for_status
from lmux_aws_bedrock._lazy import DEFAULT_REGION, bedrock_base_url, create_async_client, create_sync_client
from lmux_aws_bedrock._mappers import (
    build_embedding_request_body,
    map_converse_response,
    map_messages,
    map_response_format,
    map_stream_event,
    map_tool_choice,
    map_tools,
    model_uses_adaptive_thinking,
)
from lmux_aws_bedrock._sigv4 import sign
from lmux_aws_bedrock.auth import BedrockEnvAuthProvider
from lmux_aws_bedrock.cost import calculate_bedrock_cost
from lmux_aws_bedrock.params import BedrockParams

PROVIDER_NAME = PROVIDER
_SERVICE = "bedrock"
_BEARER_TOKEN_ENV = "AWS_BEARER_TOKEN_BEDROCK"  # noqa: S105
_JSON_CONTENT_TYPE = "application/json"

_CONVERSE = "converse"
_CONVERSE_STREAM = "converse-stream"
_INVOKE = "invoke"

_EXCEPTION_MESSAGE_TYPE = "exception"


def _today() -> date:
    """Return today's date, indirected so tests can pin the pricing clock."""
    return date.today()


@dataclass(frozen=True)
class _AuthContext:
    """Resolved per-provider auth: a region plus either a bearer token or SigV4 creds."""

    region: str
    bearer_token: str | None = None
    access_key: str | None = None
    secret_key: str | None = None
    session_token: str | None = None

    def apply(self, request: "httpx.Request") -> None:
        """Attach the auth header(s) to a fully built request."""
        if self.bearer_token is not None:
            request.headers["Authorization"] = f"Bearer {self.bearer_token}"
            return
        signed = sign(
            method=request.method,
            url=str(request.url),
            headers={"content-type": _JSON_CONTENT_TYPE},
            body=request.content,
            access_key=self.access_key or "",
            secret_key=self.secret_key or "",
            region=self.region,
            service=_SERVICE,
            now=datetime.now(UTC),
            session_token=self.session_token,
        )
        request.headers.update(signed)


def _resolve_auth_context(auth: "AuthProvider[boto3.Session, AioSession]", region_override: str | None) -> _AuthContext:
    """Resolve the auth mode once: bearer token if present, else SigV4 credentials."""
    bearer = os.environ.get(_BEARER_TOKEN_ENV)
    if bearer:
        return _AuthContext(region=region_override or DEFAULT_REGION, bearer_token=bearer)

    session = auth.get_credentials()
    credentials = session.get_credentials()
    if credentials is None:
        _raise_no_credentials()
    frozen = credentials.get_frozen_credentials()
    region = region_override or session.region_name or DEFAULT_REGION
    return _AuthContext(
        region=region,
        access_key=frozen.access_key,
        secret_key=frozen.secret_key,
        session_token=frozen.token,
    )


def _raise_no_credentials() -> NoReturn:
    """Raise botocore's ``NoCredentialsError`` so it maps to ``AuthenticationError``."""
    import botocore.exceptions  # noqa: PLC0415

    raise botocore.exceptions.NoCredentialsError


class BedrockProvider(
    CompletionProvider[BedrockParams],
    EmbeddingProvider[BedrockParams],
    PricingProvider,
):
    """AWS Bedrock API provider over httpx (Converse API + InvokeModel embeddings)."""

    def __init__(
        self,
        *,
        auth: "AuthProvider[boto3.Session, AioSession] | None" = None,
        region: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        self._auth: AuthProvider[boto3.Session, AioSession] = auth or BedrockEnvAuthProvider()
        self._region: str | None = region
        self._endpoint_url: str | None = endpoint_url
        self._auth_ctx: _AuthContext | None = None
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._custom_pricing: dict[str, ModelPricing] = {}

    # MARK: Pricing

    @override
    def register_pricing(self, model: str, pricing: ModelPricing) -> None:
        self._custom_pricing[model] = pricing

    @staticmethod
    def _resolve_pricing_as_of(provider_params: BedrockParams | None) -> date:
        """Effective pricing date: an explicit ``pricing_as_of`` override, else today."""
        if provider_params is not None and provider_params.pricing_as_of is not None:
            return provider_params.pricing_as_of
        return _today()

    def _calculate_cost(self, model: str, usage: Usage, as_of: date) -> Cost | None:
        pricing = self._custom_pricing.get(model)
        if pricing is not None:
            return calculate_cost(usage, pricing, as_of)
        return calculate_bedrock_cost(model, usage, region=self._region, as_of=as_of)

    # MARK: Client & Auth Management

    def _resolve_auth(self) -> _AuthContext:
        if self._auth_ctx is None:
            self._auth_ctx = _resolve_auth_context(self._auth, self._region)
        return self._auth_ctx

    def _base_url(self, auth: _AuthContext) -> str:
        return self._endpoint_url or bedrock_base_url(auth.region)

    def _get_sync_client(self) -> "httpx.Client":
        if self._sync_client is None:
            auth = self._resolve_auth()
            self._sync_client = create_sync_client(base_url=self._base_url(auth))
        return self._sync_client

    async def _get_async_client(self) -> "httpx.AsyncClient":
        loop = asyncio.get_running_loop()
        if self._async_client is None or self._async_loop is not loop:
            auth = self._resolve_auth()
            self._async_client = create_async_client(base_url=self._base_url(auth))
            self._async_loop = loop
        return self._async_client

    async def aclose(self) -> None:
        """Close the underlying async HTTP client."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
            self._async_loop = None

    def _build_request(self, client: "httpx.Client | httpx.AsyncClient", path: str, body: bytes) -> "httpx.Request":
        """Build a signed/authorized POST request for a Bedrock endpoint."""
        request = client.build_request("POST", path, content=body, headers={"content-type": _JSON_CONTENT_TYPE})
        self._resolve_auth().apply(request)
        return request

    @staticmethod
    def _endpoint_path(model: str, action: str) -> str:
        return f"/model/{quote(model, safe='')}/{action}"

    @staticmethod
    def _converse_body(kwargs: dict[str, Any]) -> bytes:
        """Serialize converse kwargs to a JSON body, dropping ``modelId`` (it is in the path)."""
        return json.dumps({k: v for k, v in kwargs.items() if k != "modelId"}).encode()

    # MARK: Chat

    @override
    def chat(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: BedrockParams | None = None,
    ) -> ChatResponse:
        kwargs = self._build_converse_kwargs(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
        )  # fmt: skip
        as_of = self._resolve_pricing_as_of(provider_params)
        try:
            client = self._get_sync_client()
            request = self._build_request(client, self._endpoint_path(model, _CONVERSE), self._converse_body(kwargs))
            response = client.send(request)
        except Exception as e:
            raise map_transport_error(e) from e
        raise_for_status(response)
        return map_converse_response(
            cast("ConverseResponseTypeDef", parse_json(response)),
            model,
            PROVIDER_NAME,
            lambda m, u: self._calculate_cost(m, u, as_of),
        )

    @override
    async def achat(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: BedrockParams | None = None,
    ) -> ChatResponse:
        kwargs = self._build_converse_kwargs(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
        )  # fmt: skip
        as_of = self._resolve_pricing_as_of(provider_params)
        try:
            client = await self._get_async_client()
            request = self._build_request(client, self._endpoint_path(model, _CONVERSE), self._converse_body(kwargs))
            response = await client.send(request)
        except Exception as e:
            raise map_transport_error(e) from e
        raise_for_status(response)
        return map_converse_response(
            cast("ConverseResponseTypeDef", parse_json(response)),
            model,
            PROVIDER_NAME,
            lambda m, u: self._calculate_cost(m, u, as_of),
        )

    @override
    def chat_stream(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: BedrockParams | None = None,
    ) -> Iterator[ChatChunk]:
        kwargs = self._build_converse_kwargs(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
        )  # fmt: skip
        as_of = self._resolve_pricing_as_of(provider_params)
        try:
            client = self._get_sync_client()
            path = self._endpoint_path(model, _CONVERSE_STREAM)
            response = client.send(self._build_request(client, path, self._converse_body(kwargs)))
        except Exception as e:
            raise map_transport_error(e) from e
        raise_for_status(response)
        yield from self._iter_stream(response.content, model, as_of)

    @override
    async def achat_stream(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | None = None,
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: BedrockParams | None = None,
    ) -> AsyncIterator[ChatChunk]:
        kwargs = self._build_converse_kwargs(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
        )  # fmt: skip
        as_of = self._resolve_pricing_as_of(provider_params)
        try:
            client = await self._get_async_client()
            path = self._endpoint_path(model, _CONVERSE_STREAM)
            response = await client.send(self._build_request(client, path, self._converse_body(kwargs)))
        except Exception as e:
            raise map_transport_error(e) from e
        raise_for_status(response)
        for chunk in self._iter_stream(response.content, model, as_of):
            yield chunk

    def _iter_stream(self, content: bytes, model: str, as_of: date) -> Iterator[ChatChunk]:
        """Decode an event-stream body into ChatChunks, stamping cost on the usage chunk."""
        try:
            for event in _decode_stream(content):
                chunk = map_stream_event(cast("ConverseStreamOutputTypeDef", event))
                if chunk is not None:
                    yield self._finalize_chunk(chunk, model, as_of)
        except LmuxError:
            raise
        except Exception as e:
            raise map_transport_error(e) from e

    def _finalize_chunk(self, chunk: ChatChunk, model: str, as_of: date) -> ChatChunk:
        if chunk.usage is None:
            return chunk
        return chunk.model_copy(
            update={
                "cost": self._calculate_cost(model, chunk.usage, as_of),
                "model": model,
                "provider": PROVIDER_NAME,
            }
        )

    # MARK: Embeddings

    @override
    def embed(
        self,
        model: str,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        provider_params: BedrockParams | None = None,
    ) -> EmbeddingResponse:
        texts = [input] if isinstance(input, str) else input
        all_embeddings: list[list[float]] = []
        total_input_tokens = 0
        path = self._endpoint_path(model, _INVOKE)
        try:
            client = self._get_sync_client()
            for text in texts:
                body = build_embedding_request_body(text, dimensions=dimensions).encode()
                response = client.send(self._build_request(client, path, body))
                raise_for_status(response)
                result: dict[str, Any] = parse_json(response)
                all_embeddings.append(result.get("embedding", []))
                total_input_tokens += result.get("inputTextTokenCount", 0)
        except LmuxError:
            raise
        except Exception as e:
            raise map_transport_error(e) from e

        return self._embedding_response(model, all_embeddings, total_input_tokens, provider_params)

    @override
    async def aembed(
        self,
        model: str,
        input: str | list[str],
        *,
        dimensions: int | None = None,
        provider_params: BedrockParams | None = None,
    ) -> EmbeddingResponse:
        texts = [input] if isinstance(input, str) else input
        all_embeddings: list[list[float]] = []
        total_input_tokens = 0
        path = self._endpoint_path(model, _INVOKE)
        try:
            client = await self._get_async_client()
            for text in texts:
                body = build_embedding_request_body(text, dimensions=dimensions).encode()
                response = await client.send(self._build_request(client, path, body))
                raise_for_status(response)
                result: dict[str, Any] = parse_json(response)
                all_embeddings.append(result.get("embedding", []))
                total_input_tokens += result.get("inputTextTokenCount", 0)
        except LmuxError:
            raise
        except Exception as e:
            raise map_transport_error(e) from e

        return self._embedding_response(model, all_embeddings, total_input_tokens, provider_params)

    def _embedding_response(
        self,
        model: str,
        embeddings: list[list[float]],
        input_tokens: int,
        provider_params: BedrockParams | None,
    ) -> EmbeddingResponse:
        usage = Usage(input_tokens=input_tokens, output_tokens=0)
        cost = self._calculate_cost(model, usage, self._resolve_pricing_as_of(provider_params))
        return EmbeddingResponse(
            embeddings=embeddings,
            usage=usage,
            cost=cost,
            model=model,
            provider=PROVIDER_NAME,
        )

    # MARK: Internal Helpers

    @staticmethod
    def _build_converse_kwargs(  # noqa: PLR0913, PLR0912
        model: str,
        messages: Sequence[Message],
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        stop: str | list[str] | None,
        tools: list[Tool] | None,
        tool_choice: ToolChoice | None,
        response_format: ResponseFormat | None,
        reasoning_effort: Literal["low", "medium", "high"] | None,
        provider_params: BedrockParams | None,
    ) -> dict[str, Any]:
        system, mapped_messages = map_messages(messages)
        kwargs: dict[str, Any] = {
            "modelId": model,
            "messages": mapped_messages,
        }
        if system is not None:
            kwargs["system"] = system

        # Build inferenceConfig
        inference_config: dict[str, Any] = {}
        if temperature is not None:
            inference_config["temperature"] = temperature
        if max_tokens is not None:
            inference_config["maxTokens"] = max_tokens
        if top_p is not None:
            inference_config["topP"] = top_p
        if stop is not None:
            inference_config["stopSequences"] = [stop] if isinstance(stop, str) else stop
        if inference_config:
            kwargs["inferenceConfig"] = inference_config

        if tools is not None:
            tool_config = map_tools(tools)
            if tool_choice is not None:
                tool_config["toolChoice"] = map_tool_choice(tool_choice)  # ty: ignore[invalid-assignment]
            kwargs["toolConfig"] = tool_config

        if response_format is not None:
            output_config = map_response_format(response_format)
            if output_config is not None:
                kwargs["outputConfig"] = output_config

        if provider_params is not None:
            kwargs.update(BedrockProvider._provider_params_kwargs(provider_params))

        # Apply reasoning_effort AFTER provider_params so we merge into any existing
        # additionalModelRequestFields rather than being clobbered by them.
        # If provider_params also sets a "thinking" key, it wins (already in the dict).
        if reasoning_effort is not None:
            existing = {**kwargs.get("additionalModelRequestFields", {})}
            if "thinking" not in existing:
                if model_uses_adaptive_thinking(model):
                    existing["thinking"] = {"type": "adaptive"}
                    existing["output_config"] = {**existing.get("output_config", {}), "effort": reasoning_effort}
                else:
                    budget = {"low": 1024, "medium": 8192, "high": 32768}[reasoning_effort]
                    existing["thinking"] = {"type": "enabled", "budget_tokens": budget}
            kwargs["additionalModelRequestFields"] = existing

        return kwargs

    @staticmethod
    def _provider_params_kwargs(params: BedrockParams) -> dict[str, Any]:
        """Convert BedrockParams to kwargs for the Converse API."""
        kwargs: dict[str, Any] = {}
        if params.guardrail_config is not None:
            gc: dict[str, str] = {
                "guardrailIdentifier": params.guardrail_config.guardrail_identifier,
                "guardrailVersion": params.guardrail_config.guardrail_version,
            }
            if params.guardrail_config.trace is not None:
                gc["trace"] = params.guardrail_config.trace
            kwargs["guardrailConfig"] = gc
        if params.additional_model_request_fields is not None:
            kwargs["additionalModelRequestFields"] = params.additional_model_request_fields
        if params.additional_model_response_field_paths is not None:
            kwargs["additionalModelResponseFieldPaths"] = params.additional_model_response_field_paths
        return kwargs


def _decode_stream(content: bytes) -> Iterator[dict[str, Any]]:
    """Decode an AWS event-stream body into ``{event_type: payload}`` dicts.

    Frames whose ``:message-type`` is ``exception`` (a mid-stream service error) are
    raised rather than yielded, preserving the boto3 client's fail-on-error behavior.
    """
    for headers, payload in decode_messages(content):
        event_type = headers[":event-type"]
        data: dict[str, Any] = json.loads(payload) if payload else {}
        if headers.get(":message-type") == _EXCEPTION_MESSAGE_TYPE:
            message = data.get("message") or data.get("Message") or event_type
            raise ProviderError(str(message), provider=PROVIDER_NAME)
        yield {event_type: data}
