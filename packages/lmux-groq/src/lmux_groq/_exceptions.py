"""Map Groq SDK exceptions to lmux exception hierarchy."""

from lmux.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    LmuxError,
    NotFoundError,
    ProviderError,
    RateLimitError,
    TimeoutError,  # noqa: A004
)

PROVIDER = "groq"


def map_groq_error(error: Exception) -> LmuxError:  # noqa: PLR0911
    """Convert a Groq SDK exception to the corresponding lmux exception."""
    import groq  # noqa: PLC0415

    if isinstance(error, groq.AuthenticationError):
        return AuthenticationError(str(error), provider=PROVIDER, status_code=401)

    if isinstance(error, groq.PermissionDeniedError):
        return AuthenticationError(str(error), provider=PROVIDER, status_code=403)

    if isinstance(error, groq.RateLimitError):
        retry_after = _extract_retry_after(error)
        return RateLimitError(str(error), provider=PROVIDER, status_code=429, retry_after=retry_after)

    if isinstance(error, groq.BadRequestError):
        return InvalidRequestError(str(error), provider=PROVIDER, status_code=400)

    if isinstance(error, groq.NotFoundError):
        return NotFoundError(str(error), provider=PROVIDER, status_code=404)

    if isinstance(error, groq.InternalServerError):
        return ProviderError(str(error), provider=PROVIDER, status_code=error.status_code)

    if isinstance(error, groq.APITimeoutError):
        return TimeoutError(str(error), provider=PROVIDER)

    if isinstance(error, groq.APIStatusError):
        return ProviderError(str(error), provider=PROVIDER, status_code=error.status_code)

    if isinstance(error, groq.APIError):
        return ProviderError(str(error), provider=PROVIDER)

    return ProviderError(str(error), provider=PROVIDER)


def _extract_retry_after(error: Exception) -> float | None:
    """Extract Retry-After header value from a Groq error response."""
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
