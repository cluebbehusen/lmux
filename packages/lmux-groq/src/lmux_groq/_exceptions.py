"""Map HTTP responses and transport errors to the lmux exception hierarchy."""

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from lmux.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    LmuxError,
    NotFoundError,
    PermissionDeniedError,
    ProviderError,
    RateLimitError,
    TimeoutError,  # noqa: A004
)
from lmux_groq._wire import WireCompletion

if TYPE_CHECKING:
    import httpx

PROVIDER = "groq"

_AUTH = 401
_FORBIDDEN = 403
_BAD_REQUEST = 400
_NOT_FOUND = 404
_RATE_LIMIT = 429


def raise_for_status(response: "httpx.Response") -> None:
    """Raise the mapped lmux error for a non-streamed error response."""
    if response.status_code >= _BAD_REQUEST:
        raise error_from_response(response)


def parse_completion(response: "httpx.Response") -> WireCompletion:
    """Validate a success body into a WireCompletion, mapping a malformed/mis-shaped body to a ProviderError."""
    try:
        return WireCompletion.model_validate_json(response.content)
    except ValidationError as e:
        msg = "malformed response body"
        raise ProviderError(msg, provider=PROVIDER, status_code=response.status_code) from e


def error_from_stream(payload: dict[str, Any]) -> LmuxError:
    """Map an error object embedded in a streamed response to an lmux error."""
    error = payload.get("error")
    message = error.get("message") if isinstance(error, dict) else error
    return ProviderError(str(message), provider=PROVIDER)


def error_from_response(response: "httpx.Response") -> LmuxError:
    """Map an HTTP error response to an lmux exception (body must be readable)."""
    code = response.status_code
    msg = _message(response)
    if code == _AUTH:
        return AuthenticationError(msg, provider=PROVIDER, status_code=code)
    if code == _FORBIDDEN:
        return PermissionDeniedError(msg, provider=PROVIDER, status_code=code)
    if code == _RATE_LIMIT:
        return RateLimitError(msg, provider=PROVIDER, status_code=code, retry_after=_retry_after(response))
    if code == _BAD_REQUEST:
        return InvalidRequestError(msg, provider=PROVIDER, status_code=code)
    if code == _NOT_FOUND:
        return NotFoundError(msg, provider=PROVIDER, status_code=code)
    return ProviderError(msg, provider=PROVIDER, status_code=code)


def map_transport_error(error: Exception) -> LmuxError:
    """Map an httpx transport error (or client-creation failure) to an lmux error."""
    import httpx  # noqa: PLC0415

    if isinstance(error, httpx.TimeoutException):
        return TimeoutError(str(error), provider=PROVIDER)
    return ProviderError(str(error), provider=PROVIDER)


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
