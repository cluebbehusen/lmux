"""Tests for Google HTTP error mapping."""

import httpx
import pytest
from google.auth.exceptions import DefaultCredentialsError, RefreshError

from lmux.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    NotFoundError,
    PermissionDeniedError,
    ProviderError,
    RateLimitError,
    TimeoutError,  # noqa: A004
)
from lmux_google._exceptions import (
    error_from_response,
    error_from_stream,
    map_transport_error,
    parse_json,
    raise_for_status,
)


def _resp(status: int, *, json: object = None, text: str = "", headers: dict[str, str] | None = None) -> httpx.Response:
    if json is not None:
        return httpx.Response(status, json=json, headers=headers or {})
    return httpx.Response(status, text=text, headers=headers or {})


class TestRaiseForStatus:
    def test_ok_does_not_raise(self) -> None:
        raise_for_status(_resp(200, json={}))

    def test_error_raises(self) -> None:
        with pytest.raises(InvalidRequestError):
            raise_for_status(_resp(400, json={"error": {"message": "bad"}}))


class TestErrorFromResponse:
    @pytest.mark.parametrize(
        ("status", "exc"),
        [
            (400, InvalidRequestError),
            (401, AuthenticationError),
            (403, PermissionDeniedError),
            (404, NotFoundError),
            (408, TimeoutError),
            (429, RateLimitError),
            (500, ProviderError),
            (418, ProviderError),
        ],
    )
    def test_status_mapping(self, status: int, exc: type[Exception]) -> None:
        err = error_from_response(_resp(status, json={"error": {"message": "m"}}))
        assert isinstance(err, exc)
        assert getattr(err, "status_code", None) == status
        assert getattr(err, "provider", None) == "google"


class TestErrorMessage:
    def test_error_dict_with_message(self) -> None:
        assert "the message" in str(error_from_response(_resp(400, json={"error": {"message": "the message"}})))

    def test_error_dict_without_message_falls_back(self) -> None:
        assert isinstance(error_from_response(_resp(400, json={"error": {"code": "x"}})), InvalidRequestError)

    def test_json_but_not_error_dict(self) -> None:
        assert isinstance(error_from_response(_resp(400, json=["not", "a", "dict"])), InvalidRequestError)

    def test_non_json_body(self) -> None:
        assert "plain text error" in str(error_from_response(_resp(400, text="plain text error")))


class TestMapTransportError:
    def test_timeout(self) -> None:
        assert isinstance(map_transport_error(httpx.ReadTimeout("timed out")), TimeoutError)

    def test_connect_error(self) -> None:
        assert isinstance(map_transport_error(httpx.ConnectError("refused")), ProviderError)

    def test_default_credentials_error(self) -> None:
        err = map_transport_error(DefaultCredentialsError("no creds"))
        assert isinstance(err, AuthenticationError)
        assert err.provider == "google"

    def test_refresh_error(self) -> None:
        err = map_transport_error(RefreshError("refresh failed"))
        assert isinstance(err, AuthenticationError)

    def test_generic_exception(self) -> None:
        assert isinstance(map_transport_error(Exception("boom")), ProviderError)


class TestParseJson:
    def test_valid_body(self) -> None:
        assert parse_json(_resp(200, json={"ok": 1})) == {"ok": 1}

    def test_malformed_body_maps_to_provider_error(self) -> None:
        with pytest.raises(ProviderError):
            parse_json(_resp(200, text="not json"))

    def test_non_object_body_maps_to_provider_error(self) -> None:
        with pytest.raises(ProviderError):
            parse_json(_resp(200, json=["not", "an", "object"]))


class TestErrorFromStream:
    def test_error_object_with_message(self) -> None:
        err = error_from_stream({"error": {"message": "stream boom"}})
        assert isinstance(err, ProviderError)
        assert "stream boom" in str(err)

    def test_error_not_a_dict(self) -> None:
        err = error_from_stream({"error": "raw string error"})
        assert isinstance(err, ProviderError)
        assert "raw string error" in str(err)
