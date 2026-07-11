"""Groq provider implementation (SDK-lite, httpx transport)."""

import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal, override

if TYPE_CHECKING:
    import httpx

from lmux._http import aiter_sse, iter_sse
from lmux.cost import ModelPricing, calculate_cost
from lmux.exceptions import LmuxError
from lmux.protocols import AuthProvider, CompletionProvider, PricingProvider
from lmux.types import ChatChunk, ChatResponse, Cost, Message, ResponseFormat, Tool, ToolChoice, Usage
from lmux_groq._exceptions import (
    error_from_response,
    error_from_stream,
    map_transport_error,
    parse_completion,
    raise_for_status,
)
from lmux_groq._lazy import create_async_client, create_sync_client
from lmux_groq._mappers import (
    map_chat_chunk,
    map_chat_completion,
    map_messages,
    map_response_format,
    map_tool_choice,
    map_tools,
)
from lmux_groq._wire import WireChunk
from lmux_groq.auth import GroqEnvAuthProvider
from lmux_groq.cost import calculate_groq_cost
from lmux_groq.params import GroqParams

PROVIDER_NAME = "groq"
_CHAT_PATH = "/chat/completions"


class GroqProvider(
    CompletionProvider[GroqParams],
    PricingProvider,
):
    """Groq API provider over httpx (OpenAI-compatible endpoint)."""

    def __init__(
        self,
        *,
        auth: AuthProvider[str] | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._auth: AuthProvider[str] = auth or GroqEnvAuthProvider()
        self._base_url: str | None = base_url
        self._timeout: float | None = timeout
        self._max_retries: int | None = max_retries
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._custom_pricing: dict[str, ModelPricing] = {}

    # MARK: Pricing

    @override
    def register_pricing(self, model: str, pricing: ModelPricing) -> None:
        self._custom_pricing[model] = pricing

    def _calculate_cost(self, model: str, usage: Usage) -> Cost | None:
        pricing = self._custom_pricing.get(model)
        if pricing is not None:
            return calculate_cost(usage, pricing)
        return calculate_groq_cost(model, usage)

    def _get_sync_client(self) -> "httpx.Client":
        if self._sync_client is None:
            self._sync_client = create_sync_client(
                api_key=self._auth.get_credentials(),
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        return self._sync_client

    async def _get_async_client(self) -> "httpx.AsyncClient":
        loop = asyncio.get_running_loop()
        if self._async_client is None or self._async_loop is not loop:
            self._async_client = create_async_client(
                api_key=await self._auth.aget_credentials(),
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
            self._async_loop = loop
        return self._async_client

    async def aclose(self) -> None:
        """Close the underlying async HTTP client."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
            self._async_loop = None

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
        provider_params: GroqParams | None = None,
    ) -> ChatResponse:
        body = self._build_body(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
        )  # fmt: skip
        try:
            client = self._get_sync_client()
            response = client.post(_CHAT_PATH, json={**body, "stream": False})
        except Exception as e:
            raise map_transport_error(e) from e
        raise_for_status(response)
        return map_chat_completion(parse_completion(response), PROVIDER_NAME, self._calculate_cost)

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
        provider_params: GroqParams | None = None,
    ) -> ChatResponse:
        body = self._build_body(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
        )  # fmt: skip
        try:
            client = await self._get_async_client()
            response = await client.post(_CHAT_PATH, json={**body, "stream": False})
        except Exception as e:
            raise map_transport_error(e) from e
        raise_for_status(response)
        return map_chat_completion(parse_completion(response), PROVIDER_NAME, self._calculate_cost)

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
        provider_params: GroqParams | None = None,
    ) -> Iterator[ChatChunk]:
        body = self._stream_body(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
        )  # fmt: skip
        try:
            client = self._get_sync_client()
        except Exception as e:
            raise map_transport_error(e) from e
        try:
            with client.stream("POST", _CHAT_PATH, json=body) as response:
                if response.status_code >= _HTTP_ERROR:
                    response.read()
                    raise error_from_response(response)  # noqa: TRY301
                for _event, data in iter_sse(response):
                    if data == _SSE_DONE:
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise error_from_stream(chunk)  # noqa: TRY301
                    yield self._map_stream_chunk(chunk, model)
        except LmuxError:
            raise
        except Exception as e:
            raise map_transport_error(e) from e

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
        provider_params: GroqParams | None = None,
    ) -> AsyncIterator[ChatChunk]:
        body = self._stream_body(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
        )  # fmt: skip
        try:
            client = await self._get_async_client()
        except Exception as e:
            raise map_transport_error(e) from e
        try:
            async with client.stream("POST", _CHAT_PATH, json=body) as response:
                if response.status_code >= _HTTP_ERROR:
                    await response.aread()
                    raise error_from_response(response)  # noqa: TRY301
                async for _event, data in aiter_sse(response):
                    if data == _SSE_DONE:
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise error_from_stream(chunk)  # noqa: TRY301
                    yield self._map_stream_chunk(chunk, model)
        except LmuxError:
            raise
        except Exception as e:
            raise map_transport_error(e) from e

    # MARK: Internal Helpers

    def _map_stream_chunk(self, chunk: dict[str, Any], model: str) -> ChatChunk:
        wire = WireChunk.model_validate(chunk)
        mapped = map_chat_chunk(wire, PROVIDER_NAME)
        if mapped.usage is not None:
            mapped = mapped.model_copy(update={"cost": self._calculate_cost(wire.model or model, mapped.usage)})
        return mapped

    def _stream_body(  # noqa: PLR0913
        self,
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
        provider_params: GroqParams | None,
    ) -> dict[str, Any]:
        body = self._build_body(
            model, messages, temperature, max_tokens, top_p, stop,
            tools, tool_choice, response_format, reasoning_effort, provider_params,
        )  # fmt: skip
        return {**body, "stream": True, "stream_options": {"include_usage": True}}

    @staticmethod
    def _build_body(  # noqa: PLR0913
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
        provider_params: GroqParams | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "messages": map_messages(messages)}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if stop is not None:
            body["stop"] = stop
        if tools is not None:
            body["tools"] = map_tools(tools)
        if tool_choice is not None:
            body["tool_choice"] = map_tool_choice(tool_choice)
        if response_format is not None:
            body["response_format"] = map_response_format(response_format)
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
            body["include_reasoning"] = True
        if provider_params is not None:
            body.update(GroqProvider._provider_params_kwargs(provider_params))
        return body

    @staticmethod
    def _provider_params_kwargs(params: GroqParams) -> dict[str, Any]:
        """Convert GroqParams to request-body kwargs."""
        kwargs: dict[str, Any] = {}
        if params.service_tier is not None:
            kwargs["service_tier"] = params.service_tier
        if params.reasoning_effort is not None:
            kwargs["reasoning_effort"] = params.reasoning_effort
            if params.reasoning_effort != "none":
                kwargs["include_reasoning"] = True
        if params.seed is not None:
            kwargs["seed"] = params.seed
        if params.user is not None:
            kwargs["user"] = params.user
        return kwargs


_HTTP_ERROR = 400
_SSE_DONE = "[DONE]"
