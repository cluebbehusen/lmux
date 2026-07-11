"""HTTP client factories for the Anthropic, Vertex, and Foundry transports."""

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
) -> "httpx.Client":
    """Create an httpx client for the Anthropic Messages API."""
    return _create_sync(
        base_url=base_url or DEFAULT_BASE_URL, headers=_api_headers(api_key), timeout=timeout, max_retries=max_retries
    )


def create_async_client(
    *,
    api_key: str,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "httpx.AsyncClient":
    """Create an async httpx client for the Anthropic Messages API."""
    return _create_async(
        base_url=base_url or DEFAULT_BASE_URL, headers=_api_headers(api_key), timeout=timeout, max_retries=max_retries
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
    """Return the Vertex AI base URL for a region (``global`` has no region prefix)."""
    if region == "global":
        return "https://aiplatform.googleapis.com/v1"
    return f"https://{region}-aiplatform.googleapis.com/v1"


def resolve_vertex_token(credentials: "Credentials") -> str:
    """Resolve a bearer token from Google credentials, refreshing them if needed."""
    if not credentials.token or credentials.expired:
        credentials.refresh(HttpxTransportRequest())
    if not credentials.token:
        raise RuntimeError("Could not resolve a Vertex access token from the credentials")  # noqa: TRY003
    return credentials.token


def _vertex_headers(token: str) -> Mapping[str, str]:
    return {"Authorization": f"Bearer {token}", "content-type": _JSON}


def create_sync_vertex_client(
    *,
    credentials: "Credentials",
    region: str,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "httpx.Client":
    """Create an httpx client for the Vertex AI ``rawPredict`` endpoint."""
    return _create_sync(
        base_url=base_url or vertex_base_url(region),
        headers=_vertex_headers(resolve_vertex_token(credentials)),
        timeout=timeout,
        max_retries=max_retries,
    )


def create_async_vertex_client(
    *,
    credentials: "Credentials",
    region: str,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "httpx.AsyncClient":
    """Create an async httpx client for the Vertex AI ``rawPredict`` endpoint."""
    return _create_async(
        base_url=base_url or vertex_base_url(region),
        headers=_vertex_headers(resolve_vertex_token(credentials)),
        timeout=timeout,
        max_retries=max_retries,
    )


# MARK: Microsoft Foundry


def foundry_base_url(resource: str) -> str:
    """Return the Foundry base URL for a resource name."""
    return f"https://{resource}.services.ai.azure.com/anthropic"


def _foundry_headers(api_key: str | None, azure_ad_token_provider: "Callable[[], str] | None") -> Mapping[str, str]:
    headers: dict[str, str] = {"anthropic-version": ANTHROPIC_VERSION, "content-type": _JSON}
    if azure_ad_token_provider is not None:
        headers["Authorization"] = f"Bearer {azure_ad_token_provider()}"
    elif api_key is not None:
        headers["api-key"] = api_key
    else:
        raise ValueError("Foundry requires either an api_key or an azure_ad_token_provider")  # noqa: TRY003
    return headers


def _resolve_foundry_base_url(base_url: str | None, resource: str | None) -> str:
    if base_url is not None:
        return base_url
    if resource is not None:
        return foundry_base_url(resource)
    raise ValueError("Foundry requires either a base_url or a resource")  # noqa: TRY003


def create_sync_foundry_client(  # noqa: PLR0913
    *,
    api_key: str | None = None,
    azure_ad_token_provider: "Callable[[], str] | None" = None,
    resource: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "httpx.Client":
    """Create an httpx client for the Microsoft Foundry Anthropic endpoint."""
    return _create_sync(
        base_url=_resolve_foundry_base_url(base_url, resource),
        headers=_foundry_headers(api_key, azure_ad_token_provider),
        timeout=timeout,
        max_retries=max_retries,
    )


def create_async_foundry_client(  # noqa: PLR0913
    *,
    api_key: str | None = None,
    azure_ad_token_provider: "Callable[[], str] | None" = None,
    resource: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> "httpx.AsyncClient":
    """Create an async httpx client for the Microsoft Foundry Anthropic endpoint."""
    return _create_async(
        base_url=_resolve_foundry_base_url(base_url, resource),
        headers=_foundry_headers(api_key, azure_ad_token_provider),
        timeout=timeout,
        max_retries=max_retries,
    )
