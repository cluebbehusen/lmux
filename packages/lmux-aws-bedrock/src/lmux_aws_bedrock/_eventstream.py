"""Minimal AWS event-stream (``vnd.amazon.eventstream``) decoder.

Bedrock's Converse streaming responses use AWS's binary event-stream framing.
This decodes the frames enough to recover each event's ``:event-type`` header and
JSON payload, which is all the Converse stream needs. Unit-tested for parity with
``botocore.eventstream`` (a dev-only dependency).
"""

import struct
from collections.abc import Iterator

_PRELUDE_LEN = 12  # total_length(4) + headers_length(4) + prelude_crc(4)
_MESSAGE_CRC_LEN = 4
_HEADER_STRING_TYPE = 7


def _parse_headers(raw: bytes) -> dict[str, str]:
    headers: dict[str, str] = {}
    i = 0
    while i < len(raw):
        name_len = raw[i]
        i += 1
        name = raw[i : i + name_len].decode()
        i += name_len
        value_type = raw[i]
        i += 1
        if value_type != _HEADER_STRING_TYPE:  # pragma: no cover - Bedrock events only use string headers
            msg = f"unsupported event-stream header type {value_type}"
            raise ValueError(msg)
        value_len = struct.unpack(">H", raw[i : i + 2])[0]
        i += 2
        headers[name] = raw[i : i + value_len].decode()
        i += value_len
    return headers


def decode_messages(data: bytes) -> Iterator[tuple[dict[str, str], bytes]]:
    """Yield ``(headers, payload)`` for each complete event-stream message in ``data``."""
    offset = 0
    while offset < len(data):
        total_length, headers_length = struct.unpack(">II", data[offset : offset + 8])
        headers_start = offset + _PRELUDE_LEN
        payload_start = headers_start + headers_length
        payload_end = offset + total_length - _MESSAGE_CRC_LEN
        yield _parse_headers(data[headers_start:payload_start]), data[payload_start:payload_end]
        offset += total_length
