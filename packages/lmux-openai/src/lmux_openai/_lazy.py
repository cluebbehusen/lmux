"""HTTP client factories for the OpenAI provider."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

from lmux._http import create_async_client as _create_async
from lmux._http import create_sync_client as _create_sync

if TYPE_CHECKING:
    import httpx

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _headers(
    api_key: str,
    organization: str | None,
    project: str | None,
    default_headers: Mapping[str, str] | None,
) -> dict[str, str]:
    # default_headers form the base layer; lmux-managed headers are applied on top and win on conflict.
    headers: dict[str, str] = dict(default_headers or {})
    headers["Authorization"] = f"Bearer {api_key}"
    headers["Content-Type"] = "application/json"
    if organization is not None:
        headers["OpenAI-Organization"] = organization
    if project is not None:
        headers["OpenAI-Project"] = project
    return headers


def create_sync_client(  # noqa: PLR0913
    *,
    api_key: str,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    organization: str | None = None,
    project: str | None = None,
    default_headers: Mapping[str, str] | None = None,
) -> "httpx.Client":
    """Create an httpx client for the OpenAI API."""
    return _create_sync(
        base_url=base_url or DEFAULT_BASE_URL,
        headers=_headers(api_key, organization, project, default_headers),
        timeout=timeout,
        max_retries=max_retries,
    )


def create_async_client(  # noqa: PLR0913
    *,
    api_key: str,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    organization: str | None = None,
    project: str | None = None,
    default_headers: Mapping[str, str] | None = None,
) -> "httpx.AsyncClient":
    """Create an async httpx client for the OpenAI API."""
    return _create_async(
        base_url=base_url or DEFAULT_BASE_URL,
        headers=_headers(api_key, organization, project, default_headers),
        timeout=timeout,
        max_retries=max_retries,
    )
