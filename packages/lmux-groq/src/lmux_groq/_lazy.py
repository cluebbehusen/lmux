"""HTTP client factories for the Groq provider."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

from lmux._http import create_async_client as _create_async
from lmux._http import create_sync_client as _create_sync

if TYPE_CHECKING:
    import httpx

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


def _headers(api_key: str) -> Mapping[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def create_sync_client(
    *,
    api_key: str,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "httpx.Client":
    """Create an httpx client for the Groq (OpenAI-compatible) API."""
    return _create_sync(
        base_url=base_url or DEFAULT_BASE_URL, headers=_headers(api_key), timeout=timeout, max_retries=max_retries
    )


def create_async_client(
    *,
    api_key: str,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "httpx.AsyncClient":
    """Create an async httpx client for the Groq (OpenAI-compatible) API."""
    return _create_async(
        base_url=base_url or DEFAULT_BASE_URL, headers=_headers(api_key), timeout=timeout, max_retries=max_retries
    )
