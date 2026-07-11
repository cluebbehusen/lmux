"""Internal HTTP transport helpers for SDK-lite providers.

Provider packages talk to their REST APIs with ``httpx`` directly instead of a
vendor SDK. This module holds the shared pieces — client factories and a
Server-Sent Events parser. ``httpx`` is imported lazily inside the functions so
importing lmux core never requires it; the dependency is declared by each
provider package that uses this module.
"""

from collections.abc import AsyncIterator, Iterator, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

# A long read timeout suits slow LLM generations; a short connect timeout fails fast.
_DEFAULT_READ_TIMEOUT = 600.0
_CONNECT_TIMEOUT = 10.0


def _timeout(timeout: float | None) -> "httpx.Timeout":
    import httpx  # noqa: PLC0415

    read = timeout if timeout is not None else _DEFAULT_READ_TIMEOUT
    return httpx.Timeout(read, connect=_CONNECT_TIMEOUT)


def create_sync_client(
    *,
    base_url: str,
    headers: Mapping[str, str],
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "httpx.Client":
    """Create an ``httpx.Client`` for a provider, lazily importing httpx.

    ``max_retries`` maps to httpx transport-level retries, which cover connection
    errors only (not retryable status codes) — a known SDK-lite limitation.
    """
    import httpx  # noqa: PLC0415

    transport = httpx.HTTPTransport(retries=max_retries) if max_retries else None
    return httpx.Client(base_url=base_url, headers=dict(headers), timeout=_timeout(timeout), transport=transport)


def create_async_client(
    *,
    base_url: str,
    headers: Mapping[str, str],
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "httpx.AsyncClient":
    """Create an ``httpx.AsyncClient`` for a provider, lazily importing httpx."""
    import httpx  # noqa: PLC0415

    transport = httpx.AsyncHTTPTransport(retries=max_retries) if max_retries else None
    return httpx.AsyncClient(base_url=base_url, headers=dict(headers), timeout=_timeout(timeout), transport=transport)


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
