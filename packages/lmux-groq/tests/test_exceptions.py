"""Tests for Groq HTTP error mapping."""

import httpx
import pytest

from lmux.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    NotFoundError,
    ProviderError,
    RateLimitError,
    TimeoutError,  # noqa: A004
)
from lmux_groq._exceptions import error_from_response, map_transport_error, raise_for_status


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
            (401, AuthenticationError),
            (403, AuthenticationError),
            (429, RateLimitError),
            (400, InvalidRequestError),
            (404, NotFoundError),
            (500, ProviderError),
            (418, ProviderError),
        ],
    )
    def test_status_mapping(self, status: int, exc: type[Exception]) -> None:
        err = error_from_response(_resp(status, json={"error": {"message": "m"}}))
        assert isinstance(err, exc)
        assert getattr(err, "status_code", None) == status

    def test_retry_after_valid(self) -> None:
        err = error_from_response(_resp(429, json={"error": {"message": "m"}}, headers={"retry-after": "1.5"}))
        assert isinstance(err, RateLimitError)
        assert err.retry_after == 1.5

    def test_retry_after_absent(self) -> None:
        err = error_from_response(_resp(429, json={"error": {"message": "m"}}))
        assert isinstance(err, RateLimitError)
        assert err.retry_after is None

    def test_retry_after_invalid(self) -> None:
        err = error_from_response(_resp(429, json={"error": {"message": "m"}}, headers={"retry-after": "soon"}))
        assert isinstance(err, RateLimitError)
        assert err.retry_after is None


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

    def test_generic_exception(self) -> None:
        assert isinstance(map_transport_error(Exception("boom")), ProviderError)
