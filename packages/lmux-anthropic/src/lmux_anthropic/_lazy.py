"""Lazy Anthropic SDK loading internals."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import anthropic
    from google.auth.credentials import Credentials


def create_sync_client(
    *,
    api_key: str,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "anthropic.Anthropic":
    """Create an anthropic.Anthropic client, lazily importing the SDK."""
    import anthropic  # noqa: PLC0415

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        kwargs["base_url"] = base_url
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return anthropic.Anthropic(**kwargs)


def create_async_client(
    *,
    api_key: str,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "anthropic.AsyncAnthropic":
    """Create an anthropic.AsyncAnthropic client, lazily importing the SDK."""
    import anthropic  # noqa: PLC0415

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        kwargs["base_url"] = base_url
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return anthropic.AsyncAnthropic(**kwargs)


def _vertex_client_kwargs(  # noqa: PLR0913
    *,
    credentials: "Credentials",
    project_id: str | None,
    region: str | None,
    base_url: str | None,
    timeout: float | None,
    max_retries: int | None,
) -> dict[str, Any]:
    """Build AnthropicVertex constructor kwargs, omitting None values.

    Omitted region/project_id fall back to the SDK's CLOUD_ML_REGION and
    ANTHROPIC_VERTEX_PROJECT_ID environment variables.
    """
    kwargs: dict[str, Any] = {"credentials": credentials}
    if project_id is not None:
        kwargs["project_id"] = project_id
    if region is not None:
        kwargs["region"] = region
    if base_url is not None:
        kwargs["base_url"] = base_url
    if timeout is not None:
        kwargs["timeout"] = timeout
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    return kwargs


def create_sync_vertex_client(  # noqa: PLR0913
    *,
    credentials: "Credentials",
    project_id: str | None = None,
    region: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "anthropic.AnthropicVertex":
    """Create an anthropic.AnthropicVertex client, lazily importing the SDK."""
    import anthropic  # noqa: PLC0415

    kwargs = _vertex_client_kwargs(
        credentials=credentials,
        project_id=project_id,
        region=region,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    return anthropic.AnthropicVertex(**kwargs)


def create_async_vertex_client(  # noqa: PLR0913
    *,
    credentials: "Credentials",
    project_id: str | None = None,
    region: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "anthropic.AsyncAnthropicVertex":
    """Create an anthropic.AsyncAnthropicVertex client, lazily importing the SDK."""
    import anthropic  # noqa: PLC0415

    kwargs = _vertex_client_kwargs(
        credentials=credentials,
        project_id=project_id,
        region=region,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    return anthropic.AsyncAnthropicVertex(**kwargs)
