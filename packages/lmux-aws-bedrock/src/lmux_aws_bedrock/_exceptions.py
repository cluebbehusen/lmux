"""Provider-bound wrappers over the shared AWS Bedrock error mapping.

The AWS error classification (HTTP ``x-amzn-errortype`` responses, event-stream exception
frames, and botocore credential errors) lives in :mod:`lmux_bedrock_shared.exceptions` and is
shared with the native lmux-anthropic Bedrock provider; these wrappers pre-bind the provider
name. ``parse_body`` (success-body validation) stays here — it is not AWS-error-specific.
"""

from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from lmux.exceptions import LmuxError, ProviderError
from lmux_bedrock_shared import exceptions as _shared

if TYPE_CHECKING:
    import httpx

PROVIDER = "aws-bedrock"


def raise_for_status(response: "httpx.Response") -> None:
    """Raise the mapped lmux error for an error response (body must be readable)."""
    _shared.raise_for_status(response, PROVIDER)


def error_from_response(response: "httpx.Response") -> LmuxError:
    """Map an HTTP error response to an lmux exception, using the AWS error type header."""
    return _shared.error_from_response(response, PROVIDER)


def error_from_stream_exception(error_type: str, message: str) -> LmuxError:
    """Map a mid-stream event-stream failure code to the lmux exception hierarchy."""
    return _shared.error_from_stream_exception(error_type, message, PROVIDER)


def map_transport_error(error: Exception) -> LmuxError:
    """Map an httpx transport error or a credential-resolution failure to an lmux error."""
    return _shared.map_transport_error(error, PROVIDER)


def parse_body[T: BaseModel](response: "httpx.Response", model: type[T]) -> T:
    """Validate a success body into the given wire model, mapping a malformed or mis-shaped body to a ProviderError."""
    try:
        return model.model_validate_json(response.content)
    except ValidationError as e:
        msg = "malformed response body"
        raise ProviderError(msg, provider=PROVIDER, status_code=response.status_code) from e
