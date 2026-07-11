"""HTTP client factories and endpoint resolution for the Google provider.

The Gemini REST API is reachable through two transports that share the same
request/response JSON shape but differ in base URL and authentication:

* **Gemini Developer API** — ``https://generativelanguage.googleapis.com`` with
  an API key sent in the ``x-goog-api-key`` header. Model paths look like
  ``/v1beta/models/<model>:generateContent``.
* **Vertex AI** — ``https://<location>-aiplatform.googleapis.com`` (or the global
  ``https://aiplatform.googleapis.com``) with a google-auth bearer token. Model
  paths look like
  ``/v1/projects/<project>/locations/<location>/publishers/google/models/<model>:generateContent``.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from lmux._http import create_async_client as _create_async
from lmux._http import create_sync_client as _create_sync

if TYPE_CHECKING:
    import httpx
    from google.auth.credentials import Credentials

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"


def vertex_base_url(location: str | None) -> str:
    """Resolve the Vertex AI regional (or global) base URL for a location."""
    if location and location != "global":
        return f"https://{location}-aiplatform.googleapis.com"
    return "https://aiplatform.googleapis.com"


def api_key_headers(api_key: str) -> Mapping[str, str]:
    """Auth headers for the Gemini Developer API (API-key transport)."""
    return {"x-goog-api-key": api_key, "Content-Type": "application/json"}


def bearer_headers(token: str) -> Mapping[str, str]:
    """Auth headers for the Vertex AI transport (OAuth bearer token)."""
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def bearer_token(credentials: "Credentials") -> str:
    """Return a valid access token from google-auth credentials, refreshing if needed."""
    from google.auth.transport.requests import Request  # noqa: PLC0415

    if not credentials.valid:
        credentials.refresh(Request())
    return cast("str", credentials.token)


def create_sync_client(
    *,
    base_url: str,
    headers: Mapping[str, str],
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "httpx.Client":
    """Create an httpx client for the Gemini REST API."""
    return _create_sync(base_url=base_url, headers=headers, timeout=timeout, max_retries=max_retries)


def create_async_client(
    *,
    base_url: str,
    headers: Mapping[str, str],
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "httpx.AsyncClient":
    """Create an async httpx client for the Gemini REST API."""
    return _create_async(base_url=base_url, headers=headers, timeout=timeout, max_retries=max_retries)
