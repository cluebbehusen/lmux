"""Map OpenAI SDK exceptions to lmux exception hierarchy for Azure AI Foundry."""

from lmux.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    LmuxError,
    NotFoundError,
    ProviderError,
    RateLimitError,
    TimeoutError,  # noqa: A004
)

PROVIDER = "azure-foundry"


def map_azure_foundry_error(error: Exception) -> LmuxError:  # noqa: PLR0911
    """Convert an OpenAI SDK exception to the corresponding lmux exception."""
    import openai  # noqa: PLC0415

    if isinstance(error, openai.AuthenticationError):
        return AuthenticationError(str(error), provider=PROVIDER, status_code=401)

    if isinstance(error, openai.RateLimitError):
        retry_after = _extract_retry_after(error)
        return RateLimitError(str(error), provider=PROVIDER, status_code=429, retry_after=retry_after)

    if isinstance(error, openai.BadRequestError):
        return InvalidRequestError(str(error), provider=PROVIDER, status_code=400)

    if isinstance(error, openai.NotFoundError):
        return NotFoundError(str(error), provider=PROVIDER, status_code=404)

    if isinstance(error, openai.InternalServerError):
        status = error.status_code
        return ProviderError(str(error), provider=PROVIDER, status_code=status)

    if isinstance(error, openai.APITimeoutError):
        return TimeoutError(str(error), provider=PROVIDER)

    if isinstance(error, openai.APIError):
        status: int | None = getattr(error, "status_code", None)
        return ProviderError(str(error), provider=PROVIDER, status_code=status)

    return ProviderError(str(error), provider=PROVIDER)


def _extract_retry_after(error: Exception) -> float | None:
    """Extract Retry-After header value from an OpenAI error response."""
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
