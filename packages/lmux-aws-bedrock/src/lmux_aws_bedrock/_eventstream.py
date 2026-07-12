"""Minimal AWS event-stream (``vnd.amazon.eventstream``) decoder.

Bedrock's Converse streaming responses use AWS's binary event-stream framing.
This decodes the frames enough to recover each event's ``:event-type`` header and
JSON payload, which is all the Converse stream needs. Unit-tested for parity with
``botocore.eventstream`` (a dev-only dependency).
"""

import struct
import zlib
from collections.abc import Iterator

_PRELUDE_LEN = 12  # total_length(4) + headers_length(4) + prelude_crc(4)
_PRELUDE_DATA_LEN = 8  # total_length(4) + headers_length(4), the bytes the prelude CRC covers
_MESSAGE_CRC_LEN = 4
_HEADER_STRING_TYPE = 7
_TOTAL_LENGTH_LEN = 4
_MAX_FRAME_LEN = 32 * 1024 * 1024  # anti-DoS bound; above botocore's max message and any real (KB-scale) frame


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


def _decode_frame(frame: bytes) -> tuple[dict[str, str], bytes]:
    """Decode one complete event-stream message frame into ``(headers, payload)``.

    Both CRC32 checksums are verified first (matching botocore); a corrupted frame that would
    otherwise still decode is rejected rather than silently yielding altered headers or payload.
    """
    total_length, headers_length = struct.unpack(">II", frame[:_PRELUDE_DATA_LEN])
    if struct.unpack(">I", frame[_PRELUDE_DATA_LEN:_PRELUDE_LEN])[0] != zlib.crc32(frame[:_PRELUDE_DATA_LEN]):
        msg = "event-stream prelude checksum mismatch"
        raise ValueError(msg)
    if struct.unpack(">I", frame[-_MESSAGE_CRC_LEN:])[0] != zlib.crc32(frame[:-_MESSAGE_CRC_LEN]):
        msg = "event-stream message checksum mismatch"
        raise ValueError(msg)
    payload_start = _PRELUDE_LEN + headers_length
    payload_end = total_length - _MESSAGE_CRC_LEN
    return _parse_headers(frame[_PRELUDE_LEN:payload_start]), frame[payload_start:payload_end]


def decode_messages(data: bytes) -> Iterator[tuple[dict[str, str], bytes]]:
    """Yield ``(headers, payload)`` for each complete event-stream message in ``data``."""
    offset = 0
    while offset < len(data):
        total_length = struct.unpack(">I", data[offset : offset + _TOTAL_LENGTH_LEN])[0]
        yield _decode_frame(data[offset : offset + total_length])
        offset += total_length


class EventStreamDecoder:
    """Incremental event-stream decoder.

    Feed it response byte chunks as they arrive; each ``feed`` yields every complete
    ``(headers, payload)`` message now buffered, holding any trailing partial frame until
    the rest of its bytes arrive.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> Iterator[tuple[dict[str, str], bytes]]:
        self._buffer.extend(chunk)
        while len(self._buffer) >= _PRELUDE_LEN:
            total_length = struct.unpack(">I", self._buffer[:_TOTAL_LENGTH_LEN])[0]
            # Validate the prelude (its own CRC covers the first 8 bytes) and bound the advertised
            # frame size BEFORE buffering total_length bytes, so a corrupt/huge length cannot drive
            # the buffer toward gigabytes before the frame's own checksum would catch it.
            if struct.unpack(">I", self._buffer[_PRELUDE_DATA_LEN:_PRELUDE_LEN])[0] != zlib.crc32(
                self._buffer[:_PRELUDE_DATA_LEN]
            ):
                msg = "event-stream prelude checksum mismatch"
                raise ValueError(msg)
            if total_length > _MAX_FRAME_LEN:
                msg = f"event-stream frame length {total_length} exceeds the {_MAX_FRAME_LEN}-byte limit"
                raise ValueError(msg)
            if len(self._buffer) < total_length:
                break
            frame = bytes(self._buffer[:total_length])
            del self._buffer[:total_length]
            yield _decode_frame(frame)
