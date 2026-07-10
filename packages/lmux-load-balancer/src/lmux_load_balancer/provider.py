"""A load-balancing meta-provider that distributes across other registered providers."""

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Literal

from lmux import (
    ChatChunk,
    ChatResponse,
    InvalidRequestError,
    LmuxError,
    Message,
    Registry,
    ResponseFormat,
    Tool,
    ToolChoice,
)
from lmux_load_balancer._retryable import is_retryable
from lmux_load_balancer._selection import ordered_candidates, point_for, validate_group
from lmux_load_balancer.metadata import LoadBalancerMetadata
from lmux_load_balancer.params import LoadBalancerParams

# ruff: noqa: PLR0913


class LoadBalancerProvider:
    """Distributes chat requests across a group of registered provider endpoints.

    Selection is weighted and deterministic per ``sticky_key`` (so a conversation pins
    to one endpoint and keeps its provider-side cache warm), while keyless calls are
    distributed by weight independently. On a retryable failure the request falls through
    to the next endpoint (subject to :attr:`LoadBalancerParams.failover`), and the result
    carries a :class:`LoadBalancerMetadata` recording which endpoint served it.

    It holds no SDK client; each call is delegated back through the ``Registry`` to a
    child provider by model string, without forwarding its own ``provider_params`` (so
    each child uses the default params it was registered with). ``groups`` maps a logical
    model name (the part after this provider's own prefix) to a mapping of
    ``"prefix/model"`` endpoints and their weights; a weight of ``0`` disables an endpoint.
    """

    def __init__(
        self,
        registry: Registry,
        groups: Mapping[str, Mapping[str, float]],
    ) -> None:
        self._registry: Registry = registry
        self._groups: dict[str, dict[str, float]] = {logical: dict(weights) for logical, weights in groups.items()}
        for logical, weights in self._groups.items():
            validate_group(logical, weights)
            for endpoint in weights:
                prefix = endpoint.split("/", maxsplit=1)[0]
                if prefix not in registry.registered_prefixes:
                    msg = f"Load-balancer group {logical!r} endpoint {endpoint!r} uses unregistered prefix {prefix!r}"
                    raise InvalidRequestError(msg)

    @staticmethod
    def _coerce_params(provider_params: LoadBalancerParams | None) -> LoadBalancerParams:
        """Tolerate a missing or foreign params object rather than raising ``AttributeError``."""
        if isinstance(provider_params, LoadBalancerParams):
            return provider_params
        return LoadBalancerParams()

    def _candidates(self, logical_model: str, params: LoadBalancerParams) -> list[str]:
        weights = self._groups.get(logical_model)
        if weights is None:
            msg = f"No load-balancer group registered for {logical_model!r}"
            raise InvalidRequestError(msg)
        order = ordered_candidates(weights, point_for(params.sticky_key))
        if params.failover == "never":
            return order[:1]
        if params.failover == "unless_sticky" and params.sticky_key is not None:
            return order[:1]
        return order

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
        provider_params: LoadBalancerParams | None = None,
    ) -> ChatResponse:
        params = self._coerce_params(provider_params)
        candidates = self._candidates(model, params)
        attempted: list[str] = []
        for index, child in enumerate(candidates):
            attempted.append(child)
            try:
                response = self._registry.chat(
                    child,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    reasoning_effort=reasoning_effort,
                    provider_params=None,
                )
            except LmuxError as exc:
                if index == len(candidates) - 1 or not is_retryable(exc):
                    raise
                continue
            meta = LoadBalancerMetadata(primary=candidates[0], served=child, attempted=list(attempted))
            return response.model_copy(update={"provider_metadata": meta})
        raise AssertionError  # pragma: no cover

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
        provider_params: LoadBalancerParams | None = None,
    ) -> ChatResponse:
        params = self._coerce_params(provider_params)
        candidates = self._candidates(model, params)
        attempted: list[str] = []
        for index, child in enumerate(candidates):
            attempted.append(child)
            try:
                response = await self._registry.achat(
                    child,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    reasoning_effort=reasoning_effort,
                    provider_params=None,
                )
            except LmuxError as exc:
                if index == len(candidates) - 1 or not is_retryable(exc):
                    raise
                continue
            meta = LoadBalancerMetadata(primary=candidates[0], served=child, attempted=list(attempted))
            return response.model_copy(update={"provider_metadata": meta})
        raise AssertionError  # pragma: no cover

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
        provider_params: LoadBalancerParams | None = None,
    ) -> Iterator[ChatChunk]:
        params = self._coerce_params(provider_params)
        candidates = self._candidates(model, params)
        attempted: list[str] = []
        for index, child in enumerate(candidates):
            attempted.append(child)
            stream = self._registry.chat_stream(
                child,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                reasoning_effort=reasoning_effort,
                provider_params=None,
            )
            try:
                first = next(stream)
            except StopIteration:
                return  # a successful but empty stream
            except LmuxError as exc:
                if index == len(candidates) - 1 or not is_retryable(exc):
                    raise
                continue
            # Committed to this endpoint: failover cannot happen once output has started.
            meta = LoadBalancerMetadata(primary=candidates[0], served=child, attempted=list(attempted))
            previous = first
            for chunk in stream:
                yield previous
                previous = chunk
            yield previous.model_copy(update={"provider_metadata": meta})
            return
        raise AssertionError  # pragma: no cover

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
        provider_params: LoadBalancerParams | None = None,
    ) -> AsyncIterator[ChatChunk]:
        params = self._coerce_params(provider_params)
        candidates = self._candidates(model, params)
        attempted: list[str] = []
        for index, child in enumerate(candidates):
            attempted.append(child)
            stream = self._registry.achat_stream(
                child,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                reasoning_effort=reasoning_effort,
                provider_params=None,
            )
            try:
                first = await stream.__anext__()
            except StopAsyncIteration:
                return  # a successful but empty stream
            except LmuxError as exc:
                if index == len(candidates) - 1 or not is_retryable(exc):
                    raise
                continue
            # Committed to this endpoint: failover cannot happen once output has started.
            meta = LoadBalancerMetadata(primary=candidates[0], served=child, attempted=list(attempted))
            previous = first
            async for chunk in stream:
                yield previous
                previous = chunk
            yield previous.model_copy(update={"provider_metadata": meta})
            return
        raise AssertionError  # pragma: no cover
