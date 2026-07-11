"""Internal HTTP transport helpers for SDK-lite providers.

Provider packages talk to their REST APIs with ``httpx`` directly instead of a
vendor SDK. This module holds the shared pieces — client factories and a
Server-Sent Events parser. ``httpx`` is imported lazily inside the functions so
importing lmux core never requires it; the dependency is declared by each
provider package that uses this module.

Retries are **opt-in**. By default (``max_retries`` unset or ``0``) lmux issues
each request exactly once and lets the caller handle failures — no hidden retry
loops. Passing a positive ``max_retries`` enables status-aware retries: transient
responses (408/409/429/5xx) and connection errors are retried with exponential
backoff, honoring a numeric ``Retry-After`` header. Retries are applied when a
request is *established*; a stream that has begun emitting events is never
retried.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

# A long read timeout suits slow LLM generations; a short connect timeout fails fast.
_DEFAULT_READ_TIMEOUT = 600.0
_CONNECT_TIMEOUT = 10.0

# Transient statuses worth retrying: a few 4xx plus the whole 5xx range (matches the
# vendor SDKs — covers e.g. 529 Anthropic overloaded and Cloudflare 520-524).
_RETRYABLE_4XX = frozenset({408, 409, 429})
_SERVER_ERROR = 500
_BASE_BACKOFF = 0.5
_MAX_BACKOFF = 8.0


def _is_retryable(status: int) -> bool:
    """Whether an HTTP status is a transient failure worth retrying."""
    return status in _RETRYABLE_4XX or status >= _SERVER_ERROR


def _timeout(timeout: float | None) -> "httpx.Timeout":
    import httpx  # noqa: PLC0415

    if timeout is not None:
        # An explicit caller timeout applies to every phase, including connect.
        return httpx.Timeout(timeout)
    return httpx.Timeout(_DEFAULT_READ_TIMEOUT, connect=_CONNECT_TIMEOUT)


def _backoff(attempt: int) -> float:
    """Exponential backoff for retry ``attempt`` (1-based)."""
    return min(_MAX_BACKOFF, _BASE_BACKOFF * (2 ** (attempt - 1)))


def _retry_after(response: "httpx.Response") -> float | None:
    """Seconds to wait per a numeric ``Retry-After`` header, if present and valid."""
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _sync_retry_client(
    *, max_retries: int, base_url: str, headers: dict[str, str], timeout: "httpx.Timeout"
) -> "httpx.Client":
    """A retrying ``httpx.Client`` instance.

    Overriding ``send`` covers both ``post`` and ``stream`` without touching call
    sites, and — unlike a custom transport — leaves httpx's env-proxy discovery
    intact. The subclass captures its config via closure so its ``__init__``
    forwards only well-typed args to ``httpx.Client``.
    """
    import httpx  # noqa: PLC0415

    class _RetryClient(httpx.Client):
        def __init__(self) -> None:
            super().__init__(base_url=base_url, headers=headers, timeout=timeout, follow_redirects=True)

        def send(self, request: "httpx.Request", **kwargs: object) -> "httpx.Response":
            attempt = 0
            while True:
                try:
                    response = super().send(request, **kwargs)  # ty: ignore[invalid-argument-type]
                except httpx.TransportError:
                    if attempt >= max_retries:
                        raise
                    attempt += 1
                    time.sleep(_backoff(attempt))
                    continue
                if _is_retryable(response.status_code) and attempt < max_retries:
                    attempt += 1
                    delay = _retry_after(response)
                    if delay is None:
                        delay = _backoff(attempt)
                    response.close()
                    time.sleep(delay)
                    continue
                return response

    return _RetryClient()


def _async_retry_client(
    *, max_retries: int, base_url: str, headers: dict[str, str], timeout: "httpx.Timeout"
) -> "httpx.AsyncClient":
    """Async counterpart of :func:`_sync_retry_client`."""
    import httpx  # noqa: PLC0415

    class _RetryAsyncClient(httpx.AsyncClient):
        def __init__(self) -> None:
            super().__init__(base_url=base_url, headers=headers, timeout=timeout, follow_redirects=True)

        async def send(self, request: "httpx.Request", **kwargs: object) -> "httpx.Response":
            attempt = 0
            while True:
                try:
                    response = await super().send(request, **kwargs)  # ty: ignore[invalid-argument-type]
                except httpx.TransportError:
                    if attempt >= max_retries:
                        raise
                    attempt += 1
                    await asyncio.sleep(_backoff(attempt))
                    continue
                if _is_retryable(response.status_code) and attempt < max_retries:
                    attempt += 1
                    delay = _retry_after(response)
                    if delay is None:
                        delay = _backoff(attempt)
                    await response.aclose()
                    await asyncio.sleep(delay)
                    continue
                return response

    return _RetryAsyncClient()


def create_sync_client(
    *,
    base_url: str,
    headers: Mapping[str, str],
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "httpx.Client":
    """Create an ``httpx.Client`` for a provider, lazily importing httpx.

    A positive ``max_retries`` opts into status-aware retries (see the module
    docstring); otherwise a plain client is returned, so env-proxy discovery and
    default behavior are untouched.
    """
    import httpx  # noqa: PLC0415

    if max_retries:
        return _sync_retry_client(
            max_retries=max_retries, base_url=base_url, headers=dict(headers), timeout=_timeout(timeout)
        )
    return httpx.Client(base_url=base_url, headers=dict(headers), timeout=_timeout(timeout), follow_redirects=True)


def create_async_client(
    *,
    base_url: str,
    headers: Mapping[str, str],
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "httpx.AsyncClient":
    """Create an ``httpx.AsyncClient`` for a provider, lazily importing httpx."""
    import httpx  # noqa: PLC0415

    if max_retries:
        return _async_retry_client(
            max_retries=max_retries, base_url=base_url, headers=dict(headers), timeout=_timeout(timeout)
        )
    return httpx.AsyncClient(base_url=base_url, headers=dict(headers), timeout=_timeout(timeout), follow_redirects=True)


class _SSEAccumulator:
    """Line-oriented Server-Sent Events state machine, shared by sync/async iterators."""

    def __init__(self) -> None:
        self._event: str | None = None
        self._data: list[str] = []

    def feed(self, line: str) -> tuple[str | None, str] | None:
        """Feed one raw line; return an ``(event, data)`` pair when one completes."""
        line = line.rstrip("\r")
        if line == "":
            return self.flush()
        if line.startswith(":"):
            return None  # comment/heartbeat
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            self._event = value
        elif field == "data":
            self._data.append(value)
        return None

    def flush(self) -> tuple[str | None, str] | None:
        """Emit the buffered event, if any, and reset."""
        if not self._data:
            self._event = None
            return None
        out = (self._event, "\n".join(self._data))
        self._event = None
        self._data = []
        return out


def iter_sse(response: "httpx.Response") -> Iterator[tuple[str | None, str]]:
    """Yield ``(event, data)`` tuples from a Server-Sent Events response."""
    acc = _SSEAccumulator()
    for line in response.iter_lines():
        item = acc.feed(line)
        if item is not None:
            yield item
    final = acc.flush()
    if final is not None:
        yield final


async def aiter_sse(response: "httpx.Response") -> AsyncIterator[tuple[str | None, str]]:
    """Async variant of :func:`iter_sse`."""
    acc = _SSEAccumulator()
    async for line in response.aiter_lines():
        item = acc.feed(line)
        if item is not None:
            yield item
    final = acc.flush()
    if final is not None:
        yield final
