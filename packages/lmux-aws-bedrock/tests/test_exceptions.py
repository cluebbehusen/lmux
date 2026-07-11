"""Tests for AWS Bedrock exception mapping (HTTP responses + credential errors)."""

import httpx
import pytest
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

from lmux.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    LmuxError,
    NotFoundError,
    ProviderError,
    RateLimitError,
    TimeoutError,  # noqa: A004
)
from lmux_aws_bedrock._exceptions import error_from_response, map_transport_error, parse_json, raise_for_status


def _response(
    status: int,
    *,
    error_type: str | None = None,
    retry_after: str | None = None,
    json: object = None,
    content: bytes | None = None,
) -> httpx.Response:
    headers: dict[str, str] = {}
    if error_type is not None:
        headers["x-amzn-errortype"] = error_type
    if retry_after is not None:
        headers["retry-after"] = retry_after
    if content is not None:
        return httpx.Response(status, headers=headers, content=content)
    return httpx.Response(status, headers=headers, json=json)


# MARK: raise_for_status


class TestRaiseForStatus:
    def test_ok_does_not_raise(self) -> None:
        raise_for_status(httpx.Response(200, json={"ok": True}))

    def test_error_raises(self) -> None:
        with pytest.raises(InvalidRequestError):
            raise_for_status(_response(400, json={"message": "bad"}))


# MARK: error_from_response


class TestErrorFromResponse:
    def test_auth_by_error_type(self) -> None:
        result = error_from_response(_response(400, error_type="AccessDeniedException", json={"message": "denied"}))
        assert isinstance(result, AuthenticationError)
        assert result.status_code == 400

    def test_auth_by_status_401(self) -> None:
        assert isinstance(error_from_response(_response(401, json={"message": "no"})), AuthenticationError)

    def test_auth_by_status_403(self) -> None:
        assert isinstance(error_from_response(_response(403, json={"message": "no"})), AuthenticationError)

    def test_throttling_by_error_type_with_retry_after(self) -> None:
        result = error_from_response(
            _response(400, error_type="ThrottlingException", retry_after="30.5", json={"message": "slow"})
        )
        assert isinstance(result, RateLimitError)
        assert result.retry_after == 30.5

    def test_rate_limit_by_status(self) -> None:
        result = error_from_response(_response(429, json={"message": "slow"}))
        assert isinstance(result, RateLimitError)
        assert result.retry_after is None

    def test_validation_by_error_type(self) -> None:
        result = error_from_response(_response(500, error_type="ValidationException", json={"message": "bad"}))
        assert isinstance(result, InvalidRequestError)

    def test_invalid_request_by_status(self) -> None:
        assert isinstance(error_from_response(_response(400, json={"message": "bad"})), InvalidRequestError)

    def test_not_found_by_error_type(self) -> None:
        result = error_from_response(_response(500, error_type="ResourceNotFoundException", json={"message": "gone"}))
        assert isinstance(result, NotFoundError)

    def test_not_found_by_status(self) -> None:
        assert isinstance(error_from_response(_response(404, json={"message": "gone"})), NotFoundError)

    def test_fallback_provider_error(self) -> None:
        result = error_from_response(_response(503, json={"message": "boom"}))
        assert isinstance(result, ProviderError)
        assert result.status_code == 503


# MARK: Error-type header parsing


class TestErrorTypeParsing:
    def test_type_with_colon_suffix(self) -> None:
        result = error_from_response(_response(400, error_type="AccessDeniedException:http://internal", json={}))
        assert isinstance(result, AuthenticationError)

    def test_type_with_hash_prefix(self) -> None:
        result = error_from_response(_response(500, error_type="coral#ValidationException", json={}))
        assert isinstance(result, InvalidRequestError)


# MARK: Message extraction


class TestMessage:
    def test_lowercase_message_key(self) -> None:
        result = error_from_response(_response(400, json={"message": "lower"}))
        assert "lower" in str(result)

    def test_uppercase_message_key(self) -> None:
        result = error_from_response(_response(400, json={"Message": "upper"}))
        assert "upper" in str(result)

    def test_json_without_message_falls_back_to_text(self) -> None:
        result = error_from_response(_response(400, json={"foo": "bar"}))
        assert isinstance(result, InvalidRequestError)

    def test_non_json_body_uses_text(self) -> None:
        result = error_from_response(_response(400, content=b"plain text error"))
        assert "plain text error" in str(result)

    def test_non_dict_json_body_falls_back_to_text(self) -> None:
        result = error_from_response(_response(400, json=["not", "a", "dict"]))
        assert isinstance(result, InvalidRequestError)
        assert "not" in str(result)


# MARK: retry-after parsing


class TestRetryAfter:
    def test_invalid_retry_after_is_none(self) -> None:
        result = error_from_response(_response(429, retry_after="not-a-number", json={"message": "slow"}))
        assert isinstance(result, RateLimitError)
        assert result.retry_after is None


# MARK: map_transport_error


class TestMapTransportError:
    def test_timeout(self) -> None:
        result = map_transport_error(httpx.ConnectTimeout("slow"))
        assert isinstance(result, TimeoutError)
        assert result.provider == "aws-bedrock"

    def test_no_credentials(self) -> None:
        result = map_transport_error(NoCredentialsError())
        assert isinstance(result, AuthenticationError)

    def test_partial_credentials(self) -> None:
        result = map_transport_error(PartialCredentialsError(provider="env", cred_var="AWS_SECRET_ACCESS_KEY"))
        assert isinstance(result, AuthenticationError)

    def test_generic_error(self) -> None:
        result = map_transport_error(RuntimeError("boom"))
        assert isinstance(result, ProviderError)
        assert result.provider == "aws-bedrock"

    def test_connect_error(self) -> None:
        result = map_transport_error(httpx.ConnectError("refused"))
        assert isinstance(result, ProviderError)

    def test_all_map_to_lmux_error(self) -> None:
        for error in (
            httpx.ConnectTimeout("slow"),
            NoCredentialsError(),
            RuntimeError("boom"),
        ):
            assert isinstance(map_transport_error(error), LmuxError)


class TestParseJson:
    def test_valid_body(self) -> None:
        assert parse_json(_response(200, json={"ok": 1})) == {"ok": 1}

    def test_malformed_body_maps_to_provider_error(self) -> None:
        with pytest.raises(ProviderError):
            parse_json(_response(200, content=b"not json"))

    def test_non_object_body_maps_to_provider_error(self) -> None:
        with pytest.raises(ProviderError):
            parse_json(_response(200, json=["not", "an", "object"]))
