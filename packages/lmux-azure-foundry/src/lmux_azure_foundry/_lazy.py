"""HTTP client factories and auth-header helpers for the Azure AI Foundry provider.

Azure AI Foundry / Azure OpenAI is OpenAI-compatible on the wire, but the URL
layout and auth differ from vanilla OpenAI:

- Base URL is ``{endpoint}/openai`` and every request carries an ``api-version``
  query parameter.
- Deployment-routed endpoints (chat completions, embeddings) live under
  ``/deployments/{model}/...``; the Responses API is served from ``/responses``
  with the deployment named in the request body.
- Auth is either an ``api-key`` header (API key) or an ``Authorization: Bearer``
  header (static Entra token or a per-request token-provider callable).
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

from lmux._http import create_async_client as _create_async
from lmux._http import create_sync_client as _create_sync
from lmux_azure_foundry.auth import AzureAdToken, AzureFoundryCredential

if TYPE_CHECKING:
    import httpx


def build_base_url(endpoint: str) -> str:
    """Build the OpenAI-on-Azure base URL from a resource endpoint."""
    return f"{endpoint.rstrip('/')}/openai"


def auth_headers(credential: AzureFoundryCredential) -> dict[str, str]:
    """Build the request auth headers for a resolved credential.

    - API key (``str``) -> ``api-key`` header (Azure convention).
    - Static Entra token (``AzureAdToken``) -> ``Authorization: Bearer``.
    - Token provider (``Callable[[], str]``) -> ``Authorization: Bearer`` with a
      freshly minted token; the callable is invoked on every request.
    """
    if isinstance(credential, str):
        return {"api-key": credential}
    if isinstance(credential, AzureAdToken):
        return {"Authorization": f"Bearer {credential.token}"}
    # Token provider callable — invoked on every request for a fresh bearer token.
    return {"Authorization": f"Bearer {credential()}"}


def _headers(default_headers: Mapping[str, str] | None) -> dict[str, str]:
    headers = dict(default_headers or {})
    for existing in [key for key in headers if key.lower() == "content-type"]:
        del headers[existing]
    headers["Content-Type"] = "application/json"
    return headers


def create_sync_client(
    *,
    endpoint: str,
    timeout: float | None = None,
    max_retries: int | None = None,
    default_headers: Mapping[str, str] | None = None,
    transport: "httpx.BaseTransport | None" = None,
) -> "httpx.Client":
    """Create an httpx client for the Azure AI Foundry (OpenAI-compatible) API."""
    return _create_sync(
        base_url=build_base_url(endpoint),
        headers=_headers(default_headers),
        timeout=timeout,
        max_retries=max_retries,
        transport=transport,
    )


def create_async_client(
    *,
    endpoint: str,
    timeout: float | None = None,
    max_retries: int | None = None,
    default_headers: Mapping[str, str] | None = None,
    transport: "httpx.AsyncBaseTransport | None" = None,
) -> "httpx.AsyncClient":
    """Create an async httpx client for the Azure AI Foundry (OpenAI-compatible) API."""
    return _create_async(
        base_url=build_base_url(endpoint),
        headers=_headers(default_headers),
        timeout=timeout,
        max_retries=max_retries,
        transport=transport,
    )
