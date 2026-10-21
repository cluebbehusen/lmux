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
from time import monotonic
from typing import TYPE_CHECKING, cast

from lmux._http import create_async_client as _create_async
from lmux._http import create_sync_client as _create_sync

if TYPE_CHECKING:
    import httpx
    from google.auth.credentials import Credentials

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
_REFRESH_TIMEOUT = 120.0


def vertex_base_url(location: str | None) -> str:
    """Resolve the Vertex AI base URL for a location.

    ``global`` has no region prefix; the ``us`` and ``eu`` multi-regions use their dedicated
    ``rep`` endpoints; anything else is a standard regional host.
    """
    if location in ("us", "eu"):
        return f"https://aiplatform.{location}.rep.googleapis.com"
    if location and location != "global":
        return f"https://{location}-aiplatform.googleapis.com"
    return "https://aiplatform.googleapis.com"


def merge_headers(default_headers: Mapping[str, str] | None, managed_headers: Mapping[str, str]) -> Mapping[str, str]:
    """Merge caller headers while keeping provider-managed headers authoritative."""
    managed_names = {name.lower() for name in managed_headers}
    headers = {name: value for name, value in (default_headers or {}).items() if name.lower() not in managed_names}
    headers.update(managed_headers)
    return headers


def api_key_headers(api_key: str, default_headers: Mapping[str, str] | None = None) -> Mapping[str, str]:
    """Auth headers for the Gemini Developer API (API-key transport)."""
    return merge_headers(
        default_headers,
        {"x-goog-api-key": api_key, "Content-Type": "application/json"},
    )


def bearer_headers(token: str, quota_project: str | None = None) -> Mapping[str, str]:
    """Auth headers for the Vertex AI transport (OAuth bearer token).

    ADC credentials that carry a ``quota_project_id`` must send it as ``x-goog-user-project`` so
    quota and billing are attributed to the configured project (this is what ``Credentials.apply``
    would add); omitting it can cause a quota-related 403 for user ADC.
    """
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if quota_project is not None:
        headers["x-goog-user-project"] = quota_project
    return headers


class _HttpxAuthResponse:
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


class HttpxAuthRequest:
    """Minimal ``google.auth.transport.Request`` backed by httpx.

    Lets credential refresh reuse httpx instead of pulling in the ``requests`` library that
    ``google.auth.transport.requests`` (and the ``google-auth[requests]`` extra) needs.

    Requests made through one instance share a deadline ``timeout`` seconds after it is created, so a
    refresh that makes several requests in a row (e.g. a workload identity token exchange, then service
    account impersonation) can't spend the full timeout on each. Each request's httpx timeout is capped
    at the time left, but httpx applies it per connect, send, and read, so this bounds stalls rather than
    total elapsed time.
    """

    def __init__(self, timeout: float | None = None) -> None:
        self._deadline = monotonic() + (timeout if timeout is not None else _REFRESH_TIMEOUT)

    def __call__(
        self,
        url: str,
        method: str = "GET",
        body: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        **kwargs: object,  # noqa: ARG002
    ) -> _HttpxAuthResponse:
        import httpx  # noqa: PLC0415

        remaining = self._deadline - monotonic()
        if remaining <= 0:
            msg = "Credential refresh deadline exceeded"
            raise httpx.TimeoutException(msg)
        response = httpx.request(
            method,
            url,
            content=body,
            headers=dict(headers) if headers else None,
            timeout=min(timeout, remaining) if timeout is not None else remaining,
        )
        return _HttpxAuthResponse(response)


def bearer_token(credentials: "Credentials", timeout: float | None = None) -> str:
    """Return a valid access token from google-auth credentials, refreshing if needed.

    ``timeout`` sets the refresh deadline (default 120 seconds); see ``HttpxAuthRequest``.
    """
    if not credentials.valid:
        credentials.refresh(HttpxAuthRequest(timeout=timeout))
    return cast("str", credentials.token)


def create_sync_client(
    *,
    base_url: str,
    headers: Mapping[str, str],
    timeout: float | None = None,
    max_retries: int | None = None,
    transport: "httpx.BaseTransport | None" = None,
) -> "httpx.Client":
    """Create an httpx client for the Gemini REST API."""
    return _create_sync(
        base_url=base_url, headers=headers, timeout=timeout, max_retries=max_retries, transport=transport
    )


def create_async_client(
    *,
    base_url: str,
    headers: Mapping[str, str],
    timeout: float | None = None,
    max_retries: int | None = None,
    transport: "httpx.AsyncBaseTransport | None" = None,
) -> "httpx.AsyncClient":
    """Create an async httpx client for the Gemini REST API."""
    return _create_async(
        base_url=base_url, headers=headers, timeout=timeout, max_retries=max_retries, transport=transport
    )
