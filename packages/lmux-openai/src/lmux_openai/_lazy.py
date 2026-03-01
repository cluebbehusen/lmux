"""Lazy OpenAI SDK loading internals.

Client creation is isolated here so tests can easily mock it
without patching sys.modules or using TYPE_CHECKING tricks.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import openai


def create_sync_client(
    *,
    api_key: str,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "openai.OpenAI":
    """Create an openai.OpenAI client, lazily importing the SDK."""
    import openai  # noqa: PLC0415

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        kwargs["base_url"] = base_url
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return openai.OpenAI(**kwargs)


def create_async_client(
    *,
    api_key: str,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "openai.AsyncOpenAI":
    """Create an openai.AsyncOpenAI client, lazily importing the SDK."""
    import openai  # noqa: PLC0415

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        kwargs["base_url"] = base_url
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return openai.AsyncOpenAI(**kwargs)
