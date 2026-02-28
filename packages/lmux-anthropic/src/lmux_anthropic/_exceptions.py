"""Map Anthropic SDK exceptions to lmux exception hierarchy."""

from lmux.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    LmuxError,
    NotFoundError,
    ProviderError,
    RateLimitError,
    TimeoutError,  # noqa: A004
)

PROVIDER = "anthropic"


def map_anthropic_error(error: Exception) -> LmuxError:  # noqa: PLR0911
    """Convert an Anthropic SDK exception to the corresponding lmux exception."""
    import anthropic  # noqa: PLC0415

    if isinstance(error, anthropic.AuthenticationError):
        return AuthenticationError(str(error), provider=PROVIDER, status_code=401)

    if isinstance(error, anthropic.PermissionDeniedError):
        return AuthenticationError(str(error), provider=PROVIDER, status_code=403)

    if isinstance(error, anthropic.RateLimitError):
        retry_after = _extract_retry_after(error)
        return RateLimitError(str(error), provider=PROVIDER, status_code=429, retry_after=retry_after)

    if isinstance(error, anthropic.BadRequestError):
        return InvalidRequestError(str(error), provider=PROVIDER, status_code=400)

    if isinstance(error, anthropic.NotFoundError):
        return NotFoundError(str(error), provider=PROVIDER, status_code=404)

    if isinstance(error, anthropic.InternalServerError):
        return ProviderError(str(error), provider=PROVIDER, status_code=error.status_code)

    if isinstance(error, anthropic.APITimeoutError):
        return TimeoutError(str(error), provider=PROVIDER)

    if isinstance(error, anthropic.APIStatusError):
        return ProviderError(str(error), provider=PROVIDER, status_code=error.status_code)

    if isinstance(error, anthropic.APIError):
        return ProviderError(str(error), provider=PROVIDER)

    return ProviderError(str(error), provider=PROVIDER)


def _extract_retry_after(error: Exception) -> float | None:
    """Extract Retry-After header value from an Anthropic error response."""
    response = getattr(error, "response", None)
    if response is None:
        return None
    retry_header = response.headers.get("retry-after")
    if retry_header is None:
        return None
    try:
        return float(retry_header)
    except (ValueError, TypeError):
        return None
