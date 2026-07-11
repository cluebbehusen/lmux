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

PROVIDER = "anthropic"

_AUTH = 401
_FORBIDDEN = 403
_BAD_REQUEST = 400
_NOT_FOUND = 404
_RATE_LIMIT = 429


def raise_for_status(response: "httpx.Response", provider: str = PROVIDER) -> None:
    """Raise the mapped lmux error for a non-streamed error response."""
    if response.status_code >= _BAD_REQUEST:
        raise error_from_response(response, provider)


def error_from_response(response: "httpx.Response", provider: str = PROVIDER) -> LmuxError:
    """Map an HTTP error response to an lmux exception (body must be readable)."""
    code = response.status_code
    msg = _message(response)
    if code in (_AUTH, _FORBIDDEN):
        return AuthenticationError(msg, provider=provider, status_code=code)
    if code == _RATE_LIMIT:
        return RateLimitError(msg, provider=provider, status_code=code, retry_after=_retry_after(response))
    if code == _BAD_REQUEST:
        return InvalidRequestError(msg, provider=provider, status_code=code)
    if code == _NOT_FOUND:
        return NotFoundError(msg, provider=provider, status_code=code)
    return ProviderError(msg, provider=provider, status_code=code)


def map_transport_error(error: Exception, provider: str = PROVIDER) -> LmuxError:
    """Map an httpx transport error (or client-creation failure) to an lmux error."""
    import httpx  # noqa: PLC0415

    if isinstance(error, LmuxError):
        return error
    if isinstance(error, httpx.TimeoutException):
        return TimeoutError(str(error), provider=provider)
    return ProviderError(str(error), provider=provider)


def _message(response: "httpx.Response") -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        return str(data["error"].get("message") or response.text)
    return response.text


def _retry_after(response: "httpx.Response") -> float | None:
    header = response.headers.get("retry-after")
    if header is None:
        return None
    try:
        return float(header)
    except ValueError:
        return None
