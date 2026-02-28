"""Shared exception hierarchy for lmux providers."""


class LmuxError(Exception):
    """Base exception for all lmux errors."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class AuthenticationError(LmuxError):
    """Raised when authentication fails (HTTP 401)."""


class RateLimitError(LmuxError):
    """Raised when rate limit is exceeded (HTTP 429)."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, provider=provider, status_code=status_code)
        self.retry_after = retry_after


class InvalidRequestError(LmuxError):
    """Raised for invalid requests (HTTP 400)."""


class NotFoundError(LmuxError):
    """Raised when a resource is not found (HTTP 404)."""


class ProviderError(LmuxError):
    """Raised for provider-side errors (HTTP 500/502/503)."""


class TimeoutError(LmuxError):  # noqa: A001
    """Raised when a request times out."""


class UnsupportedFeatureError(LmuxError):
    """Raised when a provider does not support a requested feature."""
