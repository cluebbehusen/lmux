"""Lazy Anthropic SDK loading internals."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import anthropic


def create_sync_client(**kwargs: Any) -> "anthropic.Anthropic":  # noqa: ANN401
    """Create an anthropic.Anthropic client, lazily importing the SDK."""
    import anthropic  # noqa: PLC0415  # pragma: no cover

    return anthropic.Anthropic(**kwargs)  # pragma: no cover


def create_async_client(**kwargs: Any) -> "anthropic.AsyncAnthropic":  # noqa: ANN401
    """Create an anthropic.AsyncAnthropic client, lazily importing the SDK."""
    import anthropic  # noqa: PLC0415  # pragma: no cover

    return anthropic.AsyncAnthropic(**kwargs)  # pragma: no cover
