"""HTTP client factories for the Groq provider."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

from lmux._http import create_async_client as _create_async
from lmux._http import create_sync_client as _create_sync

if TYPE_CHECKING:
    import httpx

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


def _headers(api_key: str, default_headers: Mapping[str, str] | None) -> Mapping[str, str]:
    headers = dict(default_headers or {})
    for name, value in {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}.items():
        for existing in [key for key in headers if key.lower() == name.lower()]:
            del headers[existing]
        headers[name] = value
    return headers


def create_sync_client(  # noqa: PLR0913
    *,
    api_key: str,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    default_headers: Mapping[str, str] | None = None,
    transport: "httpx.BaseTransport | None" = None,
) -> "httpx.Client":
    """Create an httpx client for the Groq (OpenAI-compatible) API."""
    return _create_sync(
        base_url=base_url or DEFAULT_BASE_URL,
        headers=_headers(api_key, default_headers),
        timeout=timeout,
        max_retries=max_retries,
        transport=transport,
    )


def create_async_client(  # noqa: PLR0913
    *,
    api_key: str,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    default_headers: Mapping[str, str] | None = None,
    transport: "httpx.AsyncBaseTransport | None" = None,
) -> "httpx.AsyncClient":
    """Create an async httpx client for the Groq (OpenAI-compatible) API."""
    return _create_async(
        base_url=base_url or DEFAULT_BASE_URL,
        headers=_headers(api_key, default_headers),
        timeout=timeout,
        max_retries=max_retries,
        transport=transport,
    )
