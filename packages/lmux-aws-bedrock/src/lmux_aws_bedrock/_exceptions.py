"""Map HTTP responses and transport/credential errors to the lmux exception hierarchy.

The Bedrock provider talks to the REST endpoints over httpx, so runtime errors arrive
as HTTP status codes (with an ``x-amzn-errortype`` header) or httpx transport failures.
Credentials are still resolved through boto3, so botocore's credential errors are mapped
here too.
"""

from typing import TYPE_CHECKING

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

PROVIDER = "aws-bedrock"

_BAD_REQUEST = 400
_AUTH = 401
_FORBIDDEN = 403
_NOT_FOUND = 404
_RATE_LIMIT = 429

# AWS ``x-amzn-errortype`` values that map to authentication failures (bad/expired
# credentials) regardless of the HTTP status code (some arrive as 403, some as 400).
_AUTH_ERROR_TYPES = frozenset(
    {
        "UnrecognizedClientException",
        "InvalidSignatureException",
        "ExpiredTokenException",
    }
)
# AccessDenied means valid credentials lacking access — a permission failure, not an
# authentication one.
_PERMISSION_ERROR_TYPES = frozenset({"AccessDeniedException"})


def raise_for_status(response: "httpx.Response") -> None:
    """Raise the mapped lmux error for an error response (body must be readable)."""
    if response.status_code >= _BAD_REQUEST:
        raise error_from_response(response)


def parse_body[T: BaseModel](response: "httpx.Response", model: type[T]) -> T:
    """Validate a success body into the given wire model, mapping a malformed or mis-shaped body to a ProviderError."""
    try:
        return model.model_validate_json(response.content)
    except ValidationError as e:
        msg = "malformed response body"
        raise ProviderError(msg, provider=PROVIDER, status_code=response.status_code) from e


def error_from_response(response: "httpx.Response") -> LmuxError:
    """Map an HTTP error response to an lmux exception, using the AWS error type header."""
    return _classify(_error_type(response), response.status_code, _message(response), _retry_after(response))


def error_from_stream_exception(error_type: str, message: str) -> LmuxError:
    """Map a mid-stream event-stream failure code to the lmux exception hierarchy.

    Applies to both ``exception`` frames (the ``:exception-type`` header, camelCase like
    ``throttlingException``) and unmodeled ``error`` frames (the ``:error-code`` header); the code has
    no HTTP status, so it is PascalCased and run through the same classifier as the ``x-amzn-errortype``
    response header — keeping RateLimitError/InvalidRequestError/etc. catchable for mid-stream failures.
    """
    normalized = error_type[:1].upper() + error_type[1:]
    return _classify(normalized, None, message, None)


def _classify(error_type: str, code: int | None, message: str, retry_after: float | None) -> LmuxError:
    """Map an AWS error type (and/or HTTP status) to an lmux exception."""
    if error_type in _AUTH_ERROR_TYPES or code == _AUTH:
        return AuthenticationError(message, provider=PROVIDER, status_code=code)
    if error_type in _PERMISSION_ERROR_TYPES or code == _FORBIDDEN:
        return PermissionDeniedError(message, provider=PROVIDER, status_code=code)
    if error_type == "ThrottlingException" or code == _RATE_LIMIT:
        return RateLimitError(message, provider=PROVIDER, status_code=code, retry_after=retry_after)
    if error_type == "ValidationException" or code == _BAD_REQUEST:
        return InvalidRequestError(message, provider=PROVIDER, status_code=code)
    if error_type == "ResourceNotFoundException" or code == _NOT_FOUND:
        return NotFoundError(message, provider=PROVIDER, status_code=code)
    return ProviderError(message, provider=PROVIDER, status_code=code)


def map_transport_error(error: Exception) -> LmuxError:
    """Map an httpx transport error or a credential-resolution failure to an lmux error."""
    import httpx  # noqa: PLC0415

    if isinstance(error, httpx.TimeoutException):
        return TimeoutError(str(error), provider=PROVIDER)
    return _map_credential_error(error)


def _map_credential_error(error: Exception) -> LmuxError:
    """Map botocore credential errors (raised while resolving SigV4 creds) to lmux errors."""
    import botocore.exceptions  # noqa: PLC0415

    if isinstance(error, botocore.exceptions.NoCredentialsError | botocore.exceptions.PartialCredentialsError):
        return AuthenticationError(str(error), provider=PROVIDER)
    return ProviderError(str(error), provider=PROVIDER)


def _error_type(response: "httpx.Response") -> str:
    """Extract the AWS error code from ``x-amzn-errortype``.

    The header can be a bare code, ``Code:http://...``, or ``prefix#Code``.
    """
    raw = response.headers.get("x-amzn-errortype", "")
    raw = raw.split(":", 1)[0]
    return raw.rsplit("#", 1)[-1]


def _message(response: "httpx.Response") -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text
    if isinstance(data, dict):
        message = data.get("message") or data.get("Message")
        if message:
            return str(message)
    return response.text


def _retry_after(response: "httpx.Response") -> float | None:
    header = response.headers.get("retry-after")
    if header is None:
        return None
    try:
        return float(header)
    except ValueError:
        return None
