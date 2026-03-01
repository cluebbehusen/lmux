"""Map botocore/boto3 exceptions to lmux exception hierarchy."""

from typing import Any

from lmux.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    LmuxError,
    NotFoundError,
    ProviderError,
    RateLimitError,
    TimeoutError,  # noqa: A004
)

PROVIDER = "aws-bedrock"

_AUTH_ERROR_CODES = frozenset(
    {
        "UnrecognizedClientException",
        "InvalidSignatureException",
        "ExpiredTokenException",
        "AccessDeniedException",
    }
)


def map_bedrock_error(error: Exception) -> LmuxError:
    """Convert a botocore/boto3 exception to the corresponding lmux exception."""
    import botocore.exceptions  # noqa: PLC0415

    if isinstance(error, botocore.exceptions.ClientError):
        return _map_client_error(error)

    if isinstance(error, botocore.exceptions.ReadTimeoutError | botocore.exceptions.ConnectTimeoutError):
        return TimeoutError(str(error), provider=PROVIDER)

    if isinstance(error, botocore.exceptions.NoCredentialsError | botocore.exceptions.PartialCredentialsError):
        return AuthenticationError(str(error), provider=PROVIDER)

    if isinstance(error, botocore.exceptions.EndpointConnectionError):
        return ProviderError(str(error), provider=PROVIDER)

    if isinstance(error, botocore.exceptions.BotoCoreError):
        return ProviderError(str(error), provider=PROVIDER)

    return ProviderError(str(error), provider=PROVIDER)


def _map_client_error(error: Exception) -> LmuxError:
    """Map a botocore ClientError based on its error code."""
    response: dict[str, Any] = getattr(error, "response", {})
    error_info: dict[str, str] = response.get("Error", {})
    code = error_info.get("Code", "")
    message = error_info.get("Message", str(error))
    http_status: int | None = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if code in _AUTH_ERROR_CODES:
        return AuthenticationError(message, provider=PROVIDER, status_code=http_status)

    if code == "ThrottlingException":
        retry_after = _extract_retry_after(response)
        return RateLimitError(message, provider=PROVIDER, status_code=http_status or 429, retry_after=retry_after)

    if code == "ValidationException":
        return InvalidRequestError(message, provider=PROVIDER, status_code=http_status or 400)

    if code == "ResourceNotFoundException":
        return NotFoundError(message, provider=PROVIDER, status_code=http_status or 404)

    return ProviderError(message, provider=PROVIDER, status_code=http_status)


def _extract_retry_after(response: dict[str, Any]) -> float | None:
    """Extract Retry-After from response headers if present."""
    headers: dict[str, str] = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
    retry_header = headers.get("retry-after")
    if retry_header is None:
        return None
    try:
        return float(retry_header)
    except (ValueError, TypeError):
        return None
