"""Lazy OpenAI SDK loading internals.

Client creation is isolated here so tests can easily mock it
without patching sys.modules or using TYPE_CHECKING tricks.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import openai


def create_sync_client(**kwargs: Any) -> "openai.OpenAI":  # noqa: ANN401
    """Create an openai.OpenAI client, lazily importing the SDK."""
    import openai  # noqa: PLC0415  # pragma: no cover

    return openai.OpenAI(**kwargs)  # pragma: no cover


def create_async_client(**kwargs: Any) -> "openai.AsyncOpenAI":  # noqa: ANN401
    """Create an openai.AsyncOpenAI client, lazily importing the SDK."""
    import openai  # noqa: PLC0415  # pragma: no cover

    return openai.AsyncOpenAI(**kwargs)  # pragma: no cover
