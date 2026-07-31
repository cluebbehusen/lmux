"""Minimal AWS Signature Version 4 signer (stdlib-only).

Bedrock's SDK-lite transport signs requests with SigV4 without pulling botocore
into the request path. This implements header-based SigV4 (AWS4-HMAC-SHA256).
It is unit-tested for byte-parity against ``botocore.auth.SigV4Auth`` (a dev-only
dependency), so the hand-rolled signer is safe to ship.
"""

import hashlib
import hmac
from collections.abc import Mapping
from datetime import datetime
from urllib.parse import quote, urlsplit

_ALGORITHM = "AWS4-HMAC-SHA256"


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def _signing_key(secret: str, date: str, region: str, service: str) -> bytes:
    k_date = _hmac(("AWS4" + secret).encode(), date)
    k_region = _hmac(k_date, region)
    k_service = _hmac(k_region, service)
    return _hmac(k_service, "aws4_request")


def _canonical_uri(path: str) -> str:
    return quote(path, safe="/~") if path else "/"


def _canonical_query(query: str) -> str:
    if not query:
        return ""
    # Match botocore: sort the raw (already-encoded) key/value pairs and rejoin.
    pairs = [(item.partition("=")[0], item.partition("=")[2]) for item in query.split("&")]
    return "&".join(f"{k}={v}" for k, v in sorted(pairs))


def _canonical_header_value(value: str) -> str:
    return " ".join(value.split())


def sign(  # noqa: PLR0913
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    access_key: str,
    secret_key: str,
    region: str,
    service: str,
    now: datetime,
    session_token: str | None = None,
) -> dict[str, str]:
    """Return ``headers`` with the SigV4 ``Authorization`` (and X-Amz-*) headers added."""
    parts = urlsplit(url)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date = now.strftime("%Y%m%d")
    payload_hash = _sha256_hex(body)

    signed: dict[str, str] = {k.lower(): _canonical_header_value(v) for k, v in headers.items()}
    signed["host"] = parts.netloc
    signed["x-amz-date"] = amz_date
    if session_token:
        signed["x-amz-security-token"] = session_token

    names = sorted(signed)
    canonical_headers = "".join(f"{n}:{signed[n]}\n" for n in names)
    signed_headers = ";".join(names)
    canonical_request = "\n".join(
        [
            method,
            _canonical_uri(parts.path),
            _canonical_query(parts.query),
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )
    scope = f"{date}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join([_ALGORITHM, amz_date, scope, _sha256_hex(canonical_request.encode())])
    signature = hmac.new(
        _signing_key(secret_key, date, region, service), string_to_sign.encode(), hashlib.sha256
    ).hexdigest()

    result = dict(headers)
    result["Authorization"] = (
        f"{_ALGORITHM} Credential={access_key}/{scope}, SignedHeaders={signed_headers}, Signature={signature}"
    )
    result["X-Amz-Date"] = amz_date
    if session_token:
        result["X-Amz-Security-Token"] = session_token
    return result
