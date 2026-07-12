"""Map HTTP responses and transport errors to the lmux exception hierarchy."""

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

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


def parse_body[T: BaseModel](response: "httpx.Response", model: type[T]) -> T:
    """Validate a success body into the given wire model, mapping a malformed or mis-shaped body to a ProviderError."""
    try:
        return model.model_validate_json(response.content)
    except ValidationError as e:
        msg = "malformed response body"
        raise ProviderError(msg, provider=PROVIDER, status_code=response.status_code) from e


def error_from_stream(payload: dict[str, Any]) -> LmuxError:
    """Map an error object embedded in a streamed (HTTP 200) response to an lmux error.

    A mid-stream error carries an RPC status code (e.g. ``{"error": {"code": 429, ...}}``); routing
    it through the same classifier as non-streamed errors keeps RateLimitError/AuthenticationError/etc.
    catchable instead of collapsing every mid-stream failure into a generic ProviderError.
    """
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        message = str(error.get("message") or error)
        return _error_for_code(code if isinstance(code, int) else None, message)
    return ProviderError(str(error), provider=PROVIDER)


def error_from_response(response: "httpx.Response") -> LmuxError:
    """Map an HTTP error response to an lmux exception (body must be readable)."""
    return _error_for_code(response.status_code, _message(response))


def _error_for_code(code: int | None, message: str) -> LmuxError:  # noqa: PLR0911 — one return per mapped status
    """Map an HTTP/RPC status code to the lmux exception hierarchy."""
    if code == _AUTH:
        return AuthenticationError(message, provider=PROVIDER, status_code=code)
    if code == _FORBIDDEN:
        return PermissionDeniedError(message, provider=PROVIDER, status_code=code)
    if code == _RATE_LIMIT:
        return RateLimitError(message, provider=PROVIDER, status_code=code)
    if code == _BAD_REQUEST:
        return InvalidRequestError(message, provider=PROVIDER, status_code=code)
    if code == _NOT_FOUND:
        return NotFoundError(message, provider=PROVIDER, status_code=code)
    if code == _TIMEOUT:
        return TimeoutError(message, provider=PROVIDER, status_code=code)
    return ProviderError(message, provider=PROVIDER, status_code=code)


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
