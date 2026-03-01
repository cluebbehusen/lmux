"""Lazy Groq SDK loading internals."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import groq


def create_sync_client(
    *,
    api_key: str,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "groq.Groq":
    """Create a groq.Groq client, lazily importing the SDK."""
    import groq  # noqa: PLC0415

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        kwargs["base_url"] = base_url
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return groq.Groq(**kwargs)


def create_async_client(
    *,
    api_key: str,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "groq.AsyncGroq":
    """Create a groq.AsyncGroq client, lazily importing the SDK."""
    import groq  # noqa: PLC0415

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        kwargs["base_url"] = base_url
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return groq.AsyncGroq(**kwargs)
