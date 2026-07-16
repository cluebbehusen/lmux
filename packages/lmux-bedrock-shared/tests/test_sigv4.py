"""Regression tests for the vendored SigV4 signer.

The golden signatures below were validated for byte-parity against
``botocore.auth.SigV4Auth`` during the SDK-lite PoC; they guard against
regressions without pulling botocore into the test path.
"""

import datetime

from lmux_bedrock_shared.sigv4 import sign

_NOW = datetime.datetime(2026, 7, 11, 12, 0, 0, tzinfo=datetime.UTC)


def _sign(url: str, body: bytes, token: str | None) -> dict[str, str]:
    return sign(
        method="POST", url=url, headers={"Content-Type": "application/json"}, body=body,
        access_key="AKIDEXAMPLE", secret_key="SECRETKEY", region="us-east-1",  # noqa: S106
        service="bedrock", now=_NOW, session_token=token,
    )  # fmt: skip


def _signature(signed: dict[str, str]) -> str:
    return signed["Authorization"].split("Signature=")[1]


class TestSigV4Signature:
    def test_with_session_token(self) -> None:
        url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/invoke"
        assert (
            _signature(_sign(url, b'{"x":1}', "TOKEN"))
            == "d29aa3fe44b845ce1c56a8a0c15fe8f395e4478c7aca3fd8aa9352b9073c0564"
        )

    def test_without_session_token(self) -> None:
        url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse"
        assert (
            _signature(_sign(url, b'{"a":2}', None))
            == "ab3e75faeaa140ce4b34217725698bf637ee92c316cc89ea4b17b668044ce409"
        )


class TestSigV4Output:
    def test_headers_added_no_token(self) -> None:
        signed = _sign("https://host.example.com/path", b"{}", None)
        assert signed["X-Amz-Date"] == "20260711T120000Z"
        assert signed["Authorization"].startswith(
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260711/us-east-1/bedrock/aws4_request"
        )
        assert "X-Amz-Security-Token" not in signed

    def test_session_token_header(self) -> None:
        assert _sign("https://host.example.com/path", b"{}", "TT")["X-Amz-Security-Token"] == "TT"

    def test_root_path_and_query_branches(self) -> None:
        # empty path -> "/" canonical uri; query present -> non-empty canonical query
        assert "Authorization" in _sign("https://host.example.com", b"", None)
        assert "Authorization" in _sign("https://host.example.com/p?b=2&a=1", b"", None)
