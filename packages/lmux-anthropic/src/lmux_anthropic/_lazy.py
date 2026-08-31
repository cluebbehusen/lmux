"""HTTP client factories for the Anthropic, Vertex, and Foundry transports."""

import os
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from lmux._http import create_async_client as _create_async
from lmux._http import create_sync_client as _create_sync
from lmux.exceptions import AuthenticationError, LmuxError, ProviderError, RateLimitError

if TYPE_CHECKING:
    import httpx
    from google.auth.credentials import Credentials

DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
VERTEX_ANTHROPIC_VERSION = "vertex-2023-10-16"

_JSON = "application/json"
_REFRESH_TIMEOUT = 120.0


# MARK: Direct Anthropic API


def _merge_headers(default_headers: Mapping[str, str] | None, managed_headers: Mapping[str, str]) -> Mapping[str, str]:
    managed_names = {name.lower() for name in managed_headers}
    headers = {name: value for name, value in (default_headers or {}).items() if name.lower() not in managed_names}
    headers.update(managed_headers)
    return headers


def _api_headers(api_key: str | None, default_headers: Mapping[str, str] | None) -> Mapping[str, str]:
    # x-api-key is always provider-managed: under bearer auth (api_key None) a caller-supplied
    # key is dropped so a request never carries both credentials (the API can prefer the static
    # key over the bearer token when both are present).
    managed = {"anthropic-version": ANTHROPIC_VERSION, "content-type": _JSON}
    if api_key is not None:
        managed["x-api-key"] = api_key
    headers = _merge_headers(default_headers, managed)
    if api_key is None:
        headers = {name: value for name, value in headers.items() if name.lower() != "x-api-key"}
    return headers


def create_sync_client(  # noqa: PLR0913
    *,
    api_key: str | None,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    default_headers: Mapping[str, str] | None = None,
    transport: "httpx.BaseTransport | None" = None,
) -> "httpx.Client":
    """Create an httpx client for the Anthropic Messages API.

    ``api_key`` is baked into the client headers; pass ``None`` for bearer-token auth,
    which is applied per request instead.
    """
    return _create_sync(
        base_url=base_url or DEFAULT_BASE_URL,
        headers=_api_headers(api_key, default_headers),
        timeout=timeout,
        max_retries=max_retries,
        transport=transport,
    )


def create_async_client(  # noqa: PLR0913
    *,
    api_key: str | None,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    default_headers: Mapping[str, str] | None = None,
    transport: "httpx.AsyncBaseTransport | None" = None,
) -> "httpx.AsyncClient":
    """Create an async httpx client for the Anthropic Messages API.

    ``api_key`` is baked into the client headers; pass ``None`` for bearer-token auth,
    which is applied per request instead.
    """
    return _create_async(
        base_url=base_url or DEFAULT_BASE_URL,
        headers=_api_headers(api_key, default_headers),
        timeout=timeout,
        max_retries=max_retries,
        transport=transport,
    )


# Beta flags the first-party SDKs send for OAuth bearer auth (undocumented in the public API
# docs). oauth-2025-04-20 is required on any request authenticated with a bearer token, and
# oidc-federation-2026-04-01 routes a /v1/oauth/token jwt-bearer grant to the federation
# handler; it must only be sent on jwt-bearer exchanges.
OAUTH_BETA_FLAG = "oauth-2025-04-20"
_FEDERATION_BETA_FLAG = "oidc-federation-2026-04-01"
_JWT_BEARER_BETA_FLAGS = f"{OAUTH_BETA_FLAG},{_FEDERATION_BETA_FLAG}"


def _merged_beta_flags(default_headers: Mapping[str, str] | None) -> str:
    existing = next((value for name, value in (default_headers or {}).items() if name.lower() == "anthropic-beta"), "")
    flags = [flag.strip() for flag in existing.split(",") if flag.strip()]
    if OAUTH_BETA_FLAG not in flags:
        flags.append(OAUTH_BETA_FLAG)
    return ",".join(flags)


def bearer_auth_headers(
    token_provider: "Callable[[], str]", default_headers: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Per-request bearer auth headers; the token provider is invoked on every call for a fresh token.

    The required OAuth beta flag is merged into any caller-supplied ``anthropic-beta`` value,
    which this per-request header would otherwise replace.
    """
    return {
        "Authorization": f"Bearer {token_provider()}",
        "anthropic-beta": _merged_beta_flags(default_headers),
    }


# MARK: Workload Identity Federation


_OAUTH_TOKEN_PATH = "/v1/oauth/token"  # noqa: S105
_JWT_BEARER_GRANT = "urn:ietf:params:oauth:grant-type:jwt-bearer"
_EXCHANGE_TIMEOUT = 30.0
_EXCHANGE_ERROR_DETAIL_LIMIT = 200
_TOO_MANY_REQUESTS = 429
_SERVER_ERROR = 500


def _exchange_error_message(response: "httpx.Response") -> str:
    """Bounded message for a failed exchange: only the API's ``error.message`` is retained, so
    an arbitrary response body (e.g. from a gateway) is never propagated into exception text.
    """
    base = "Workload identity token exchange failed"
    try:
        payload = response.json()
    except ValueError:
        return base
    error = payload.get("error") if isinstance(payload, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if isinstance(message, str) and message:
        return f"{base}: {message[:_EXCHANGE_ERROR_DETAIL_LIMIT]}"
    return base


def _exchange_error(response: "httpx.Response") -> LmuxError:
    """Typed error for a failed exchange, so transient statuses stay retryable downstream."""
    message = _exchange_error_message(response)
    status = response.status_code
    if status == _TOO_MANY_REQUESTS:
        return RateLimitError(message, provider="anthropic", status_code=status)
    if status >= _SERVER_ERROR:
        return ProviderError(message, provider="anthropic", status_code=status)
    return AuthenticationError(message, provider="anthropic", status_code=status)


def exchange_workload_identity_token(  # noqa: PLR0913
    *,
    assertion: str,
    federation_rule_id: str,
    organization_id: str,
    service_account_id: str | None = None,
    workspace_id: str | None = None,
    base_url: str | None = None,
) -> tuple[str, float]:
    """Exchange an IdP-issued OIDC identity token for an Anthropic access token.

    Returns the access token together with its lifetime in seconds (``expires_in``).
    """
    import httpx  # noqa: PLC0415

    from lmux_anthropic._exceptions import map_transport_error  # noqa: PLC0415

    body: dict[str, str] = {
        "grant_type": _JWT_BEARER_GRANT,
        "assertion": assertion,
        "federation_rule_id": federation_rule_id,
        "organization_id": organization_id,
    }
    if service_account_id is not None:
        body["service_account_id"] = service_account_id
    if workspace_id is not None:
        body["workspace_id"] = workspace_id
    try:
        response = httpx.post(
            f"{(base_url or DEFAULT_BASE_URL).rstrip('/')}{_OAUTH_TOKEN_PATH}",
            json=body,
            headers={"anthropic-beta": _JWT_BEARER_BETA_FLAGS, "content-type": _JSON},
            timeout=_EXCHANGE_TIMEOUT,
        )
    except httpx.HTTPError as e:
        raise map_transport_error(e) from e
    if response.is_error:
        raise _exchange_error(response)
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if not isinstance(payload, dict):
        msg = "Workload identity token exchange returned a malformed response body"
        raise AuthenticationError(msg, provider="anthropic")
    access_token = payload.get("access_token")
    expires_in = payload.get("expires_in")
    if not isinstance(access_token, str) or not isinstance(expires_in, int | float):
        msg = "Workload identity token exchange response is missing access_token or expires_in"
        raise AuthenticationError(msg, provider="anthropic")
    return access_token, float(expires_in)


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
    default_headers: Mapping[str, str] | None = None,
) -> "httpx.Client":
    """Create an httpx client for the Vertex AI ``rawPredict`` endpoint (auth is applied per request)."""
    return _create_sync(
        base_url=base_url or vertex_base_url(region),
        headers=_merge_headers(default_headers, {"content-type": _JSON}),
        timeout=timeout,
        max_retries=max_retries,
    )


def create_async_vertex_client(
    *,
    region: str,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    default_headers: Mapping[str, str] | None = None,
) -> "httpx.AsyncClient":
    """Create an async httpx client for the Vertex AI ``rawPredict`` endpoint (auth is applied per request)."""
    return _create_async(
        base_url=base_url or vertex_base_url(region),
        headers=_merge_headers(default_headers, {"content-type": _JSON}),
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
    default_headers: Mapping[str, str] | None = None,
) -> "httpx.Client":
    """Create an httpx client for the Microsoft Foundry Anthropic endpoint (auth is applied per request)."""
    return _create_sync(
        base_url=_resolve_foundry_base_url(base_url, resource),
        headers=_merge_headers(default_headers, {"anthropic-version": ANTHROPIC_VERSION, "content-type": _JSON}),
        timeout=timeout,
        max_retries=max_retries,
    )


def create_async_foundry_client(
    *,
    resource: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    default_headers: Mapping[str, str] | None = None,
) -> "httpx.AsyncClient":
    """Create an async httpx client for the Microsoft Foundry Anthropic endpoint (auth is applied per request)."""
    return _create_async(
        base_url=_resolve_foundry_base_url(base_url, resource),
        headers=_merge_headers(default_headers, {"anthropic-version": ANTHROPIC_VERSION, "content-type": _JSON}),
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
