"""Map HTTP responses and transport errors to the lmux exception hierarchy."""

from typing import TYPE_CHECKING

from lmux.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    LmuxError,
    NotFoundError,
    ProviderError,
    RateLimitError,
    TimeoutError,  # noqa: A004
)

if TYPE_CHECKING:
    import httpx

PROVIDER = "google"

_BAD_REQUEST = 400
_AUTH = 401
_FORBIDDEN = 403
_NOT_FOUND = 404
_TIMEOUT = 408
_RATE_LIMIT = 429


def raise_for_status(response: "httpx.Response") -> None:
    """Raise the mapped lmux error for a non-streamed error response."""
    if response.status_code >= _BAD_REQUEST:
        raise error_from_response(response)


def error_from_response(response: "httpx.Response") -> LmuxError:
    """Map an HTTP error response to an lmux exception (body must be readable)."""
    code = response.status_code
    msg = _message(response)
    if code in (_AUTH, _FORBIDDEN):
        return AuthenticationError(msg, provider=PROVIDER, status_code=code)
    if code == _RATE_LIMIT:
        return RateLimitError(msg, provider=PROVIDER, status_code=code)
    if code == _BAD_REQUEST:
        return InvalidRequestError(msg, provider=PROVIDER, status_code=code)
    if code == _NOT_FOUND:
        return NotFoundError(msg, provider=PROVIDER, status_code=code)
    if code == _TIMEOUT:
        return TimeoutError(msg, provider=PROVIDER, status_code=code)
    return ProviderError(msg, provider=PROVIDER, status_code=code)


def map_transport_error(error: Exception) -> LmuxError:
    """Map an httpx transport error, or a google-auth credential failure, to an lmux error."""
    import httpx  # noqa: PLC0415

    if isinstance(error, httpx.TimeoutException):
        return TimeoutError(str(error), provider=PROVIDER)

    auth_error = _check_auth_error(error)
    if auth_error is not None:
        return auth_error

    return ProviderError(str(error), provider=PROVIDER)


def _check_auth_error(error: Exception) -> AuthenticationError | None:
    """Map a google-auth credential/refresh failure to an AuthenticationError."""
    import google.auth.exceptions  # noqa: PLC0415

    if isinstance(error, google.auth.exceptions.DefaultCredentialsError | google.auth.exceptions.RefreshError):
        return AuthenticationError(str(error), provider=PROVIDER)
    return None


def _message(response: "httpx.Response") -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        return str(data["error"].get("message") or response.text)
    return response.text
