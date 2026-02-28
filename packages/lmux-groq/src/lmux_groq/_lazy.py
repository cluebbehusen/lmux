"""Lazy Groq SDK loading internals."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import groq


def create_sync_client(**kwargs: Any) -> "groq.Groq":  # noqa: ANN401
    """Create a groq.Groq client, lazily importing the SDK."""
    import groq  # noqa: PLC0415  # pragma: no cover

    return groq.Groq(**kwargs)  # pragma: no cover


def create_async_client(**kwargs: Any) -> "groq.AsyncGroq":  # noqa: ANN401
    """Create a groq.AsyncGroq client, lazily importing the SDK."""
    import groq  # noqa: PLC0415  # pragma: no cover

    return groq.AsyncGroq(**kwargs)  # pragma: no cover
