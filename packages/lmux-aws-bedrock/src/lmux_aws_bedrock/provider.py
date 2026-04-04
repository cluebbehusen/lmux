"""AWS Bedrock provider implementation."""

import json
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any, Literal, override

if TYPE_CHECKING:
    import boto3
    from aiobotocore.session import AioSession
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from types_aiobotocore_bedrock_runtime import BedrockRuntimeClient as AsyncBedrockRuntimeClient

from lmux.cost import ModelPricing, calculate_cost
from lmux.protocols import AuthProvider, CompletionProvider, EmbeddingProvider, PricingProvider
from lmux.types import (
    ChatChunk,
    ChatResponse,
    Cost,
    EmbeddingResponse,
    Message,
    ResponseFormat,
    Tool,
    Usage,
)
from lmux_aws_bedrock._exceptions import map_bedrock_error
from lmux_aws_bedrock._lazy import create_async_client, create_sync_client
from lmux_aws_bedrock._mappers import (
    build_embedding_request_body,
    map_converse_response,
    map_messages,
    map_response_format,
    map_stream_event,
    map_tools,
)
from lmux_aws_bedrock.auth import BedrockEnvAuthProvider
from lmux_aws_bedrock.cost import calculate_bedrock_cost
from lmux_aws_bedrock.params import BedrockParams

PROVIDER_NAME = "aws-bedrock"


class BedrockProvider(
    CompletionProvider[BedrockParams],
    EmbeddingProvider[BedrockParams],
    PricingProvider,
):
    """AWS Bedrock API provider using the Converse API."""

    def __init__(
        self,
        *,
        auth: AuthProvider["boto3.Session", "AioSession"] | None = None,
        region: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        self._auth: AuthProvider[boto3.Session, AioSession] = auth or BedrockEnvAuthProvider()
        self._region: str | None = region
        self._endpoint_url: str | None = endpoint_url
        self._sync_client: BedrockRuntimeClient | None = None
        self._async_session: AioSession | None = None
        self._custom_pricing: dict[str, ModelPricing] = {}

    # MARK: Pricing

    @override
    def register_pricing(self, model: str, pricing: ModelPricing) -> None:
        self._custom_pricing[model] = pricing

    def _calculate_cost(self, model: str, usage: Usage) -> Cost | None:
        pricing = self._custom_pricing.get(model)
        if pricing is not None:
            return calculate_cost(usage, pricing)
        return calculate_bedrock_cost(model, usage, region=self._region)

    # MARK: Client Management

    def _get_sync_client(self) -> "BedrockRuntimeClient":
        if self._sync_client is None:
            session = self._auth.get_credentials()
            self._sync_client = create_sync_client(session, region_name=self._region, endpoint_url=self._endpoint_url)
        return self._sync_client

    async def _get_async_client_ctx(self) -> AbstractAsyncContextManager["AsyncBedrockRuntimeClient"]:
        """Return an aiobotocore async client context manager."""
        if self._async_session is None:
            self._async_session = await self._auth.aget_credentials()
        return create_async_client(
            self._async_session,
            region_name=self._region,
            endpoint_url=self._endpoint_url,
        )

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
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: BedrockParams | None = None,
    ) -> ChatResponse:
        kwargs = self._build_converse_kwargs(
            model,
            messages,
            temperature,
            max_tokens,
            top_p,
            stop,
            tools,
            response_format,
            reasoning_effort,
            provider_params,
        )
        try:
            client = self._get_sync_client()
            response = client.converse(**kwargs)
        except Exception as e:
            raise map_bedrock_error(e) from e
        return map_converse_response(response, model, PROVIDER_NAME, self._calculate_cost)

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
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: BedrockParams | None = None,
    ) -> ChatResponse:
        kwargs = self._build_converse_kwargs(
            model,
            messages,
            temperature,
            max_tokens,
            top_p,
            stop,
            tools,
            response_format,
            reasoning_effort,
            provider_params,
        )
        try:
            async with await self._get_async_client_ctx() as client:
                response = await client.converse(**kwargs)
        except Exception as e:
            raise map_bedrock_error(e) from e
        return map_converse_response(response, model, PROVIDER_NAME, self._calculate_cost)

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
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: BedrockParams | None = None,
    ) -> Iterator[ChatChunk]:
        kwargs = self._build_converse_kwargs(
            model,
            messages,
            temperature,
            max_tokens,
            top_p,
            stop,
            tools,
            response_format,
            reasoning_effort,
            provider_params,
        )
        try:
            client = self._get_sync_client()
            response = client.converse_stream(**kwargs)
        except Exception as e:
            raise map_bedrock_error(e) from e

        try:
            event_stream = response.get("stream", [])
            for event in event_stream:
                chunk = map_stream_event(event)
                if chunk is not None:
                    if chunk.usage is not None:
                        chunk = chunk.model_copy(update={"cost": self._calculate_cost(model, chunk.usage)})
                    yield chunk
        except Exception as e:
            raise map_bedrock_error(e) from e

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
        response_format: ResponseFormat | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        provider_params: BedrockParams | None = None,
    ) -> AsyncIterator[ChatChunk]:
        kwargs = self._build_converse_kwargs(
            model,
            messages,
            temperature,
            max_tokens,
            top_p,
            stop,
            tools,
            response_format,
            reasoning_effort,
            provider_params,
        )
        try:
            async with await self._get_async_client_ctx() as client:
                response = await client.converse_stream(**kwargs)
                event_stream = response.get("stream", [])
                async for event in event_stream:
                    chunk = map_stream_event(event)
                    if chunk is not None:
                        if chunk.usage is not None:
                            chunk = chunk.model_copy(update={"cost": self._calculate_cost(model, chunk.usage)})
                        yield chunk
        except Exception as e:
            raise map_bedrock_error(e) from e

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

        try:
            client = self._get_sync_client()
        except Exception as e:
            raise map_bedrock_error(e) from e

        for text in texts:
            body = build_embedding_request_body(text, dimensions=dimensions)
            try:
                response = client.invoke_model(
                    modelId=model,
                    contentType="application/json",
                    body=body,
                )
            except Exception as e:
                raise map_bedrock_error(e) from e

            result: dict[str, Any] = json.loads(response["body"].read())
            all_embeddings.append(result.get("embedding", []))
            total_input_tokens += result.get("inputTextTokenCount", 0)

        usage = Usage(input_tokens=total_input_tokens, output_tokens=0)
        cost = self._calculate_cost(model, usage)

        return EmbeddingResponse(
            embeddings=all_embeddings,
            usage=usage,
            cost=cost,
            model=model,
            provider=PROVIDER_NAME,
        )

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

        async with await self._get_async_client_ctx() as client:
            for text in texts:
                body = build_embedding_request_body(text, dimensions=dimensions)
                try:
                    response = await client.invoke_model(
                        modelId=model,
                        contentType="application/json",
                        body=body,
                    )
                except Exception as e:
                    raise map_bedrock_error(e) from e

                raw = await response["body"].read()
                result: dict[str, Any] = json.loads(raw)
                all_embeddings.append(result.get("embedding", []))
                total_input_tokens += result.get("inputTextTokenCount", 0)

        usage = Usage(input_tokens=total_input_tokens, output_tokens=0)
        cost = self._calculate_cost(model, usage)

        return EmbeddingResponse(
            embeddings=all_embeddings,
            usage=usage,
            cost=cost,
            model=model,
            provider=PROVIDER_NAME,
        )

    # MARK: Internal Helpers

    @staticmethod
    def _build_converse_kwargs(  # noqa: PLR0913
        model: str,
        messages: Sequence[Message],
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        stop: str | list[str] | None,
        tools: list[Tool] | None,
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
            kwargs["toolConfig"] = map_tools(tools)

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
            budget = {"low": 1024, "medium": 8192, "high": 32768}[reasoning_effort]
            existing = {**kwargs.get("additionalModelRequestFields", {})}
            if "thinking" not in existing:
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
