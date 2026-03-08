"""Map google-genai SDK exceptions to lmux exception hierarchy."""

from lmux.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    LmuxError,
    NotFoundError,
    ProviderError,
    RateLimitError,
    TimeoutError,  # noqa: A004
)

PROVIDER = "gcp-vertex"

_STATUS_CODE_MAP: dict[int, type[LmuxError]] = {
    400: InvalidRequestError,
    401: AuthenticationError,
    403: AuthenticationError,
    404: NotFoundError,
    408: TimeoutError,
    429: RateLimitError,
}


def map_gcp_vertex_error(error: Exception) -> LmuxError:
    """Convert a google-genai SDK exception to the corresponding lmux exception."""
    from google.genai import errors as genai_errors  # noqa: PLC0415

    if isinstance(error, genai_errors.ClientError):
        return _map_client_error(error)

    if isinstance(error, genai_errors.ServerError):
        return ProviderError(str(error), provider=PROVIDER, status_code=error.code)

    if isinstance(error, genai_errors.APIError):
        return ProviderError(str(error), provider=PROVIDER, status_code=error.code)

    # google.auth errors
    auth_error = _check_auth_error(error)
    if auth_error is not None:
        return auth_error

    return ProviderError(str(error), provider=PROVIDER)


def _map_client_error(error: Exception) -> LmuxError:
    """Map a google-genai ClientError based on its HTTP status code."""
    code: int | None = getattr(error, "code", None)

    exc_cls = _STATUS_CODE_MAP.get(code) if code is not None else None
    if exc_cls is not None:
        if exc_cls is RateLimitError:
            return RateLimitError(str(error), provider=PROVIDER, status_code=code)
        return exc_cls(str(error), provider=PROVIDER, status_code=code)

    return ProviderError(str(error), provider=PROVIDER, status_code=code)


def _check_auth_error(error: Exception) -> AuthenticationError | None:
    """Check if the error is a google.auth exception and map it."""
    import google.auth.exceptions  # noqa: PLC0415

    if isinstance(error, google.auth.exceptions.DefaultCredentialsError | google.auth.exceptions.RefreshError):
        return AuthenticationError(str(error), provider=PROVIDER)

    return None
