"""Tests for AWS Bedrock exception mapping."""

import pytest
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    NoCredentialsError,
    PartialCredentialsError,
    ReadTimeoutError,
)

from lmux.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    LmuxError,
    NotFoundError,
    ProviderError,
    RateLimitError,
    TimeoutError,  # noqa: A004
)
from lmux_aws_bedrock._exceptions import map_bedrock_error

# MARK: Fixtures


@pytest.fixture
def throttling_error() -> ClientError:
    return ClientError(
        error_response={  # pyright: ignore[reportArgumentType]
            "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"},
            "ResponseMetadata": {"HTTPStatusCode": 429},
        },
        operation_name="Converse",
    )


@pytest.fixture
def throttling_error_with_retry_after() -> ClientError:
    return ClientError(
        error_response={  # pyright: ignore[reportArgumentType]
            "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"},
            "ResponseMetadata": {"HTTPStatusCode": 429, "HTTPHeaders": {"retry-after": "30.5"}},
        },
        operation_name="Converse",
    )


@pytest.fixture
def throttling_error_invalid_retry_after() -> ClientError:
    return ClientError(
        error_response={  # pyright: ignore[reportArgumentType]
            "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"},
            "ResponseMetadata": {"HTTPStatusCode": 429, "HTTPHeaders": {"retry-after": "not-a-number"}},
        },
        operation_name="Converse",
    )


@pytest.fixture
def throttling_error_no_retry_header() -> ClientError:
    return ClientError(
        error_response={  # pyright: ignore[reportArgumentType]
            "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"},
            "ResponseMetadata": {"HTTPStatusCode": 429, "HTTPHeaders": {}},
        },
        operation_name="Converse",
    )


@pytest.fixture
def validation_error() -> ClientError:
    return ClientError(
        error_response={  # pyright: ignore[reportArgumentType]
            "Error": {"Code": "ValidationException", "Message": "Invalid parameter"},
            "ResponseMetadata": {"HTTPStatusCode": 400},
        },
        operation_name="Converse",
    )


@pytest.fixture
def access_denied_error() -> ClientError:
    return ClientError(
        error_response={  # pyright: ignore[reportArgumentType]
            "Error": {"Code": "AccessDeniedException", "Message": "Access denied"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
        operation_name="Converse",
    )


@pytest.fixture
def unrecognized_client_error() -> ClientError:
    return ClientError(
        error_response={  # pyright: ignore[reportArgumentType]
            "Error": {"Code": "UnrecognizedClientException", "Message": "Unrecognized client"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
        operation_name="Converse",
    )


@pytest.fixture
def invalid_signature_error() -> ClientError:
    return ClientError(
        error_response={  # pyright: ignore[reportArgumentType]
            "Error": {"Code": "InvalidSignatureException", "Message": "Invalid signature"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
        operation_name="Converse",
    )


@pytest.fixture
def expired_token_error() -> ClientError:
    return ClientError(
        error_response={  # pyright: ignore[reportArgumentType]
            "Error": {"Code": "ExpiredTokenException", "Message": "Token expired"},
            "ResponseMetadata": {"HTTPStatusCode": 403},
        },
        operation_name="Converse",
    )


@pytest.fixture
def resource_not_found_error() -> ClientError:
    return ClientError(
        error_response={  # pyright: ignore[reportArgumentType]
            "Error": {"Code": "ResourceNotFoundException", "Message": "Model not found"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        },
        operation_name="Converse",
    )


@pytest.fixture
def internal_server_error() -> ClientError:
    return ClientError(
        error_response={  # pyright: ignore[reportArgumentType]
            "Error": {"Code": "InternalServerException", "Message": "Internal error"},
            "ResponseMetadata": {"HTTPStatusCode": 500},
        },
        operation_name="Converse",
    )


@pytest.fixture
def unknown_client_error() -> ClientError:
    return ClientError(
        error_response={  # pyright: ignore[reportArgumentType]
            "Error": {"Code": "SomeUnknownError", "Message": "Something unexpected"},
            "ResponseMetadata": {"HTTPStatusCode": 418},
        },
        operation_name="Converse",
    )


@pytest.fixture
def no_credentials_error() -> NoCredentialsError:
    return NoCredentialsError()


@pytest.fixture
def partial_credentials_error() -> PartialCredentialsError:
    return PartialCredentialsError(provider="env", cred_var="AWS_SECRET_ACCESS_KEY")


@pytest.fixture
def read_timeout_error() -> ReadTimeoutError:
    return ReadTimeoutError(endpoint_url="https://bedrock.us-east-1.amazonaws.com")


@pytest.fixture
def connect_timeout_error() -> ConnectTimeoutError:
    return ConnectTimeoutError(endpoint_url="https://bedrock.us-east-1.amazonaws.com")


@pytest.fixture
def endpoint_connection_error() -> EndpointConnectionError:
    return EndpointConnectionError(endpoint_url="https://bedrock.us-east-1.amazonaws.com")


@pytest.fixture
def generic_botocore_error() -> BotoCoreError:
    return BotoCoreError()


# MARK: Tests


class TestMapBedrockError:
    def test_throttling_exception(self, throttling_error: ClientError) -> None:
        result = map_bedrock_error(throttling_error)
        assert isinstance(result, RateLimitError)
        assert result.provider == "aws-bedrock"
        assert result.status_code == 429

    def test_throttling_with_retry_after(self, throttling_error_with_retry_after: ClientError) -> None:
        result = map_bedrock_error(throttling_error_with_retry_after)
        assert isinstance(result, RateLimitError)
        assert result.provider == "aws-bedrock"
        assert result.status_code == 429
        assert result.retry_after == 30.5

    def test_throttling_invalid_retry_after(self, throttling_error_invalid_retry_after: ClientError) -> None:
        result = map_bedrock_error(throttling_error_invalid_retry_after)
        assert isinstance(result, RateLimitError)
        assert result.provider == "aws-bedrock"
        assert result.retry_after is None

    def test_throttling_no_retry_header(self, throttling_error_no_retry_header: ClientError) -> None:
        result = map_bedrock_error(throttling_error_no_retry_header)
        assert isinstance(result, RateLimitError)
        assert result.provider == "aws-bedrock"
        assert result.retry_after is None

    def test_validation_exception(self, validation_error: ClientError) -> None:
        result = map_bedrock_error(validation_error)
        assert isinstance(result, InvalidRequestError)
        assert result.provider == "aws-bedrock"
        assert result.status_code == 400

    def test_access_denied(self, access_denied_error: ClientError) -> None:
        result = map_bedrock_error(access_denied_error)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "aws-bedrock"
        assert result.status_code == 403

    def test_unrecognized_client(self, unrecognized_client_error: ClientError) -> None:
        result = map_bedrock_error(unrecognized_client_error)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "aws-bedrock"
        assert result.status_code == 403

    def test_invalid_signature(self, invalid_signature_error: ClientError) -> None:
        result = map_bedrock_error(invalid_signature_error)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "aws-bedrock"
        assert result.status_code == 403

    def test_expired_token(self, expired_token_error: ClientError) -> None:
        result = map_bedrock_error(expired_token_error)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "aws-bedrock"
        assert result.status_code == 403

    def test_resource_not_found(self, resource_not_found_error: ClientError) -> None:
        result = map_bedrock_error(resource_not_found_error)
        assert isinstance(result, NotFoundError)
        assert result.provider == "aws-bedrock"
        assert result.status_code == 404

    def test_internal_server_error(self, internal_server_error: ClientError) -> None:
        result = map_bedrock_error(internal_server_error)
        assert isinstance(result, ProviderError)
        assert result.provider == "aws-bedrock"
        assert result.status_code == 500

    def test_unknown_client_error(self, unknown_client_error: ClientError) -> None:
        result = map_bedrock_error(unknown_client_error)
        assert isinstance(result, ProviderError)
        assert result.provider == "aws-bedrock"
        assert result.status_code == 418

    def test_no_credentials_error(self, no_credentials_error: NoCredentialsError) -> None:
        result = map_bedrock_error(no_credentials_error)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "aws-bedrock"

    def test_partial_credentials_error(self, partial_credentials_error: PartialCredentialsError) -> None:
        result = map_bedrock_error(partial_credentials_error)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "aws-bedrock"

    def test_read_timeout_error(self, read_timeout_error: ReadTimeoutError) -> None:
        result = map_bedrock_error(read_timeout_error)
        assert isinstance(result, TimeoutError)
        assert result.provider == "aws-bedrock"

    def test_connect_timeout_error(self, connect_timeout_error: ConnectTimeoutError) -> None:
        result = map_bedrock_error(connect_timeout_error)
        assert isinstance(result, TimeoutError)
        assert result.provider == "aws-bedrock"

    def test_endpoint_connection_error(self, endpoint_connection_error: EndpointConnectionError) -> None:
        result = map_bedrock_error(endpoint_connection_error)
        assert isinstance(result, ProviderError)
        assert result.provider == "aws-bedrock"

    def test_generic_botocore_error(self, generic_botocore_error: BotoCoreError) -> None:
        result = map_bedrock_error(generic_botocore_error)
        assert isinstance(result, ProviderError)
        assert result.provider == "aws-bedrock"

    def test_generic_exception(self) -> None:
        error = RuntimeError("something broke")
        result = map_bedrock_error(error)
        assert isinstance(result, ProviderError)
        assert result.provider == "aws-bedrock"

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(
                ClientError(
                    error_response={  # pyright: ignore[reportArgumentType]
                        "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"},
                        "ResponseMetadata": {"HTTPStatusCode": 429},
                    },
                    operation_name="Converse",
                ),
                id="throttling",
            ),
            pytest.param(
                ClientError(
                    error_response={  # pyright: ignore[reportArgumentType]
                        "Error": {"Code": "ValidationException", "Message": "Invalid"},
                        "ResponseMetadata": {"HTTPStatusCode": 400},
                    },
                    operation_name="Converse",
                ),
                id="validation",
            ),
            pytest.param(
                ClientError(
                    error_response={  # pyright: ignore[reportArgumentType]
                        "Error": {"Code": "AccessDeniedException", "Message": "Denied"},
                        "ResponseMetadata": {"HTTPStatusCode": 403},
                    },
                    operation_name="Converse",
                ),
                id="access_denied",
            ),
            pytest.param(
                ClientError(
                    error_response={  # pyright: ignore[reportArgumentType]
                        "Error": {"Code": "ResourceNotFoundException", "Message": "Not found"},
                        "ResponseMetadata": {"HTTPStatusCode": 404},
                    },
                    operation_name="Converse",
                ),
                id="resource_not_found",
            ),
            pytest.param(
                ClientError(
                    error_response={  # pyright: ignore[reportArgumentType]
                        "Error": {"Code": "InternalServerException", "Message": "Internal"},
                        "ResponseMetadata": {"HTTPStatusCode": 500},
                    },
                    operation_name="Converse",
                ),
                id="internal_server",
            ),
            pytest.param(
                ClientError(
                    error_response={  # pyright: ignore[reportArgumentType]
                        "Error": {"Code": "UnknownCode", "Message": "Unknown"},
                        "ResponseMetadata": {"HTTPStatusCode": 500},
                    },
                    operation_name="Converse",
                ),
                id="unknown_client",
            ),
            pytest.param(NoCredentialsError(), id="no_credentials"),
            pytest.param(
                PartialCredentialsError(provider="env", cred_var="AWS_SECRET_ACCESS_KEY"),
                id="partial_credentials",
            ),
            pytest.param(ReadTimeoutError(endpoint_url="https://test.com"), id="read_timeout"),
            pytest.param(ConnectTimeoutError(endpoint_url="https://test.com"), id="connect_timeout"),
            pytest.param(EndpointConnectionError(endpoint_url="https://test.com"), id="endpoint_connection"),
            pytest.param(BotoCoreError(), id="botocore"),
            pytest.param(RuntimeError("fallback"), id="runtime"),
        ],
    )
    def test_all_errors_are_lmux_errors(self, error: Exception) -> None:
        result = map_bedrock_error(error)
        assert isinstance(result, LmuxError)
        assert result.provider == "aws-bedrock"
