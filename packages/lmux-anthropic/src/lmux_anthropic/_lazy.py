"""HTTP client factories for the Anthropic, Vertex, and Foundry transports."""

import os
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from lmux._http import create_async_client as _create_async
from lmux._http import create_sync_client as _create_sync

if TYPE_CHECKING:
    import httpx
    from google.auth.credentials import Credentials

DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
VERTEX_ANTHROPIC_VERSION = "vertex-2023-10-16"

_JSON = "application/json"
_REFRESH_TIMEOUT = 120.0


# MARK: Direct Anthropic API


def _api_headers(api_key: str) -> Mapping[str, str]:
    return {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION, "content-type": _JSON}


def create_sync_client(
    *,
    api_key: str,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    transport: "httpx.BaseTransport | None" = None,
) -> "httpx.Client":
    """Create an httpx client for the Anthropic Messages API."""
    return _create_sync(
        base_url=base_url or DEFAULT_BASE_URL,
        headers=_api_headers(api_key),
        timeout=timeout,
        max_retries=max_retries,
        transport=transport,
    )


def create_async_client(
    *,
    api_key: str,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    transport: "httpx.AsyncBaseTransport | None" = None,
) -> "httpx.AsyncClient":
    """Create an async httpx client for the Anthropic Messages API."""
    return _create_async(
        base_url=base_url or DEFAULT_BASE_URL,
        headers=_api_headers(api_key),
        timeout=timeout,
        max_retries=max_retries,
        transport=transport,
    )


# MARK: Vertex AI


class _HttpxTransportResponse:
    """Adapt an httpx response to the ``google.auth.transport.Response`` interface."""

    def __init__(self, response: "httpx.Response") -> None:
        self._response = response

    @property
    def status(self) -> int:
        return self._response.status_code

    @property
    def headers(self) -> "httpx.Headers":
        return self._response.headers

    @property
    def data(self) -> bytes:
        return self._response.content


class HttpxTransportRequest:
    """Minimal ``google.auth.transport.Request`` backed by httpx.

    Lets Vertex credential refresh reuse httpx instead of pulling in the
    ``requests`` library that ``google.auth.transport.requests`` needs.
    """

    def __call__(
        self,
        url: str,
        method: str = "GET",
        body: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        **kwargs: object,  # noqa: ARG002
    ) -> _HttpxTransportResponse:
        import httpx  # noqa: PLC0415

        response = httpx.request(
            method,
            url,
            content=body,
            headers=dict(headers) if headers else None,
            timeout=timeout if timeout is not None else _REFRESH_TIMEOUT,
        )
        return _HttpxTransportResponse(response)


def vertex_base_url(region: str) -> str:
    """Return the Vertex AI base URL for a region.

    ``global`` has no region prefix; the ``us`` and ``eu`` multi-regions use their
    dedicated ``rep`` endpoints; anything else is a standard regional host.
    """
    if region == "global":
        return "https://aiplatform.googleapis.com/v1"
    if region in ("us", "eu"):
        return f"https://aiplatform.{region}.rep.googleapis.com/v1"
    return f"https://{region}-aiplatform.googleapis.com/v1"


def resolve_vertex_token(credentials: "Credentials") -> str:
    """Resolve a bearer token from Google credentials, refreshing them if needed."""
    if not credentials.token or credentials.expired:
        credentials.refresh(HttpxTransportRequest())
    if not credentials.token:
        raise RuntimeError("Could not resolve a Vertex access token from the credentials")  # noqa: TRY003
    return credentials.token


def vertex_auth_headers(credentials: "Credentials") -> dict[str, str]:
    """Per-request Vertex auth header, refreshing the (short-lived) access token if needed."""
    return {"Authorization": f"Bearer {resolve_vertex_token(credentials)}"}


def create_sync_vertex_client(
    *,
    region: str,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "httpx.Client":
    """Create an httpx client for the Vertex AI ``rawPredict`` endpoint (auth is applied per request)."""
    return _create_sync(
        base_url=base_url or vertex_base_url(region),
        headers={"content-type": _JSON},
        timeout=timeout,
        max_retries=max_retries,
    )


def create_async_vertex_client(
    *,
    region: str,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "httpx.AsyncClient":
    """Create an async httpx client for the Vertex AI ``rawPredict`` endpoint (auth is applied per request)."""
    return _create_async(
        base_url=base_url or vertex_base_url(region),
        headers={"content-type": _JSON},
        timeout=timeout,
        max_retries=max_retries,
    )


# MARK: Microsoft Foundry


def foundry_base_url(resource: str) -> str:
    """Return the Foundry base URL for a resource name."""
    return f"https://{resource}.services.ai.azure.com/anthropic"


def foundry_auth_headers(api_key: str | None, azure_ad_token_provider: "Callable[[], str] | None") -> dict[str, str]:
    """Per-request Foundry auth headers.

    An Entra ID token provider is invoked on every call for a fresh bearer token. For
    API-key auth the endpoint authenticates with ``x-api-key``; ``api-key`` is also sent
    for backwards compatibility.
    """
    if azure_ad_token_provider is not None:
        return {"Authorization": f"Bearer {azure_ad_token_provider()}"}
    if api_key is not None:
        return {"x-api-key": api_key, "api-key": api_key}
    raise ValueError("Foundry requires either an api_key or an azure_ad_token_provider")  # noqa: TRY003


def _resolve_foundry_base_url(base_url: str | None, resource: str | None) -> str:
    # base_url and resource are mutually exclusive; resolve both (explicit or env) so a stale
    # ANTHROPIC_FOUNDRY_BASE_URL can't silently override an explicitly selected resource.
    base_url = base_url or os.environ.get("ANTHROPIC_FOUNDRY_BASE_URL")
    resource = resource or os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE")
    if base_url is not None:
        if resource is not None:
            msg = "Foundry base_url and resource are mutually exclusive"
            raise ValueError(msg)
        return base_url
    if resource is not None:
        return foundry_base_url(resource)
    msg = (
        "Foundry requires a base_url or resource "
        "(or the ANTHROPIC_FOUNDRY_BASE_URL / ANTHROPIC_FOUNDRY_RESOURCE env var)"
    )
    raise ValueError(msg)


def create_sync_foundry_client(
    *,
    resource: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "httpx.Client":
    """Create an httpx client for the Microsoft Foundry Anthropic endpoint (auth is applied per request)."""
    return _create_sync(
        base_url=_resolve_foundry_base_url(base_url, resource),
        headers={"anthropic-version": ANTHROPIC_VERSION, "content-type": _JSON},
        timeout=timeout,
        max_retries=max_retries,
    )


def create_async_foundry_client(
    *,
    resource: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "httpx.AsyncClient":
    """Create an async httpx client for the Microsoft Foundry Anthropic endpoint (auth is applied per request)."""
    return _create_async(
        base_url=_resolve_foundry_base_url(base_url, resource),
        headers={"anthropic-version": ANTHROPIC_VERSION, "content-type": _JSON},
        timeout=timeout,
        max_retries=max_retries,
    )


# MARK: Amazon Bedrock (native Anthropic Messages API via InvokeModel)


def bedrock_base_url(region: str, *, use_fips: bool = False) -> str:
    """Return the bedrock-runtime endpoint for a region, optionally the FIPS 140-3 variant.

    FIPS endpoints (``bedrock-runtime-fips.<region>.amazonaws.com``) force FIPS-validated
    in-transit cryptography in the commercial and GovCloud regions where Bedrock runs.
    """
    service = "bedrock-runtime-fips" if use_fips else "bedrock-runtime"
    return f"https://{service}.{region}.amazonaws.com"


def create_sync_bedrock_client(
    *,
    base_url: str,
    timeout: float | None = None,
    max_retries: int | None = None,
    transport: "httpx.BaseTransport | None" = None,
) -> "httpx.Client":
    """Create an httpx client for the Bedrock runtime endpoint.

    Auth (SigV4 signature or a bearer token) and ``content-type`` are attached per request on the
    fully built request, so the client carries no default headers.
    """
    return _create_sync(base_url=base_url, headers={}, timeout=timeout, max_retries=max_retries, transport=transport)


def create_async_bedrock_client(
    *,
    base_url: str,
    timeout: float | None = None,
    max_retries: int | None = None,
    transport: "httpx.AsyncBaseTransport | None" = None,
) -> "httpx.AsyncClient":
    """Create an async httpx client for the Bedrock runtime endpoint (auth is signed per request)."""
    return _create_async(base_url=base_url, headers={}, timeout=timeout, max_retries=max_retries, transport=transport)
