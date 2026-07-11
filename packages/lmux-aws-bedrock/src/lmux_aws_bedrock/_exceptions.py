"""Map HTTP responses and transport/credential errors to the lmux exception hierarchy.

The Bedrock provider talks to the REST endpoints over httpx, so runtime errors arrive
as HTTP status codes (with an ``x-amzn-errortype`` header) or httpx transport failures.
Credentials are still resolved through boto3, so botocore's credential errors are mapped
here too.
"""

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

PROVIDER = "aws-bedrock"

_BAD_REQUEST = 400
_AUTH = 401
_FORBIDDEN = 403
_NOT_FOUND = 404
_RATE_LIMIT = 429

# AWS ``x-amzn-errortype`` values that map to authentication failures regardless of the
# HTTP status code (some arrive as 403, some as 400).
_AUTH_ERROR_TYPES = frozenset(
    {
        "UnrecognizedClientException",
        "InvalidSignatureException",
        "ExpiredTokenException",
        "AccessDeniedException",
    }
)


def raise_for_status(response: "httpx.Response") -> None:
    """Raise the mapped lmux error for an error response (body must be readable)."""
    if response.status_code >= _BAD_REQUEST:
        raise error_from_response(response)


def error_from_response(response: "httpx.Response") -> LmuxError:
    """Map an HTTP error response to an lmux exception, using the AWS error type header."""
    code = response.status_code
    error_type = _error_type(response)
    message = _message(response)

    if error_type in _AUTH_ERROR_TYPES or code in (_AUTH, _FORBIDDEN):
        return AuthenticationError(message, provider=PROVIDER, status_code=code)
    if error_type == "ThrottlingException" or code == _RATE_LIMIT:
        return RateLimitError(message, provider=PROVIDER, status_code=code, retry_after=_retry_after(response))
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
