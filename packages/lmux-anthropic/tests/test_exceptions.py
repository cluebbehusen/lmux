"""Tests for Anthropic HTTP exception mapping."""

import httpx
import pytest

from lmux.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    LmuxError,
    NotFoundError,
    ProviderError,
    RateLimitError,
    TimeoutError,  # noqa: A004
)
from lmux_anthropic._exceptions import (
    error_from_response,
    error_from_stream,
    map_transport_error,
    parse_json,
    raise_for_status,
)


def _response(status: int, *, message: str = "boom", headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, json={"type": "error", "error": {"message": message}}, headers=headers)


# MARK: error_from_response


class TestErrorFromResponse:
    def test_authentication_401(self) -> None:
        result = error_from_response(_response(401))
        assert isinstance(result, AuthenticationError)
        assert result.provider == "anthropic"
        assert result.status_code == 401

    def test_authentication_403(self) -> None:
        result = error_from_response(_response(403))
        assert isinstance(result, AuthenticationError)
        assert result.status_code == 403

    def test_rate_limit_without_retry_after(self) -> None:
        result = error_from_response(_response(429))
        assert isinstance(result, RateLimitError)
        assert result.status_code == 429
        assert result.retry_after is None

    def test_rate_limit_with_retry_after(self) -> None:
        result = error_from_response(_response(429, headers={"retry-after": "30.5"}))
        assert isinstance(result, RateLimitError)
        assert result.retry_after == 30.5

    def test_rate_limit_invalid_retry_after(self) -> None:
        result = error_from_response(_response(429, headers={"retry-after": "soon"}))
        assert isinstance(result, RateLimitError)
        assert result.retry_after is None

    def test_bad_request_400(self) -> None:
        result = error_from_response(_response(400))
        assert isinstance(result, InvalidRequestError)
        assert result.status_code == 400

    def test_not_found_404(self) -> None:
        result = error_from_response(_response(404))
        assert isinstance(result, NotFoundError)
        assert result.status_code == 404

    def test_server_error_500(self) -> None:
        result = error_from_response(_response(500))
        assert isinstance(result, ProviderError)
        assert result.status_code == 500

    def test_custom_provider_propagated(self) -> None:
        result = error_from_response(_response(401), "anthropic-vertex")
        assert result.provider == "anthropic-vertex"

    def test_message_extracted_from_error_body(self) -> None:
        result = error_from_response(_response(400, message="bad model"))
        assert "bad model" in str(result)

    def test_non_json_body_uses_text(self) -> None:
        result = error_from_response(httpx.Response(500, text="plain failure"))
        assert "plain failure" in str(result)

    def test_json_without_error_dict_uses_text(self) -> None:
        result = error_from_response(httpx.Response(400, json={"detail": "nope"}))
        assert isinstance(result, InvalidRequestError)

    def test_error_dict_without_message_falls_back_to_text(self) -> None:
        result = error_from_response(httpx.Response(400, json={"error": {"type": "x"}}))
        assert isinstance(result, InvalidRequestError)


# MARK: raise_for_status


class TestRaiseForStatus:
    def test_ok_does_not_raise(self) -> None:
        raise_for_status(httpx.Response(200, json={}))

    def test_error_raises(self) -> None:
        with pytest.raises(InvalidRequestError):
            raise_for_status(_response(400))


# MARK: map_transport_error


class TestMapTransportError:
    def test_timeout(self) -> None:
        result = map_transport_error(httpx.ConnectTimeout("slow"))
        assert isinstance(result, TimeoutError)
        assert result.provider == "anthropic"

    def test_generic_transport_error(self) -> None:
        result = map_transport_error(httpx.ConnectError("refused"))
        assert isinstance(result, ProviderError)
        assert "refused" in str(result)

    def test_custom_provider(self) -> None:
        result = map_transport_error(httpx.ConnectError("refused"), "anthropic-foundry")
        assert result.provider == "anthropic-foundry"

    def test_lmux_error_passed_through(self) -> None:
        original = ProviderError("already mapped", provider="anthropic-vertex")
        assert map_transport_error(original) is original

    def test_all_mapped_are_lmux_errors(self) -> None:
        assert isinstance(map_transport_error(RuntimeError("x")), LmuxError)


class TestParseJson:
    def test_valid_body(self) -> None:
        assert parse_json(httpx.Response(200, json={"ok": 1})) == {"ok": 1}

    def test_malformed_body_maps_to_provider_error(self) -> None:
        with pytest.raises(ProviderError):
            parse_json(httpx.Response(200, text="not json"))

    def test_non_object_body_maps_to_provider_error(self) -> None:
        with pytest.raises(ProviderError):
            parse_json(httpx.Response(200, json=["not", "an", "object"]))


class TestErrorFromStream:
    def test_error_object_with_message(self) -> None:
        err = error_from_stream({"type": "error", "error": {"message": "stream boom"}})
        assert isinstance(err, ProviderError)
        assert "stream boom" in str(err)

    def test_error_not_a_dict(self) -> None:
        err = error_from_stream({"type": "error", "error": "raw string error"})
        assert isinstance(err, ProviderError)
        assert "raw string error" in str(err)
