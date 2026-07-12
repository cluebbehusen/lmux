"""Parity tests for the event-stream decoder against botocore."""

import json
import struct
import zlib

import pytest
from botocore.eventstream import EventStreamBuffer

from lmux_aws_bedrock._eventstream import EventStreamDecoder, decode_messages


def _encode(headers: dict[str, str], payload: bytes) -> bytes:
    hb = b""
    for name, value in headers.items():
        nb, vb = name.encode(), value.encode()
        hb += bytes([len(nb)]) + nb + bytes([7]) + struct.pack(">H", len(vb)) + vb
    total = 16 + len(hb) + len(payload)
    prelude = struct.pack(">II", total, len(hb))
    body = prelude + struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF) + hb + payload
    return body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


_FRAMES = _encode(
    {":event-type": "messageStart", ":content-type": "application/json"}, b'{"role":"assistant"}'
) + _encode({":event-type": "contentBlockDelta"}, b'{"delta":{"text":"Hi"},"contentBlockIndex":0}')


class TestDecodeMessages:
    def test_parity_with_botocore(self) -> None:
        mine = [(h[":event-type"], json.loads(p)) for h, p in decode_messages(_FRAMES)]
        buf = EventStreamBuffer()
        buf.add_data(_FRAMES)
        theirs = [(m.headers[":event-type"], json.loads(m.payload)) for m in buf]  # ty: ignore[not-subscriptable]
        assert mine == theirs

    def test_recovers_events(self) -> None:
        events = list(decode_messages(_FRAMES))
        assert events[0][0][":event-type"] == "messageStart"
        assert json.loads(events[1][1]) == {"delta": {"text": "Hi"}, "contentBlockIndex": 0}

    def test_empty(self) -> None:
        assert list(decode_messages(b"")) == []


class TestEventStreamDecoder:
    def test_yields_complete_frames(self) -> None:
        decoder = EventStreamDecoder()
        messages = list(decoder.feed(_FRAMES))
        assert [h[":event-type"] for h, _ in messages] == ["messageStart", "contentBlockDelta"]

    def test_holds_partial_frame_until_complete(self) -> None:
        decoder = EventStreamDecoder()
        # A few bytes in (>= 4, but short of a full frame) nothing decodes yet.
        assert list(decoder.feed(_FRAMES[:6])) == []
        messages = list(decoder.feed(_FRAMES[6:]))
        assert [h[":event-type"] for h, _ in messages] == ["messageStart", "contentBlockDelta"]


class TestCrcValidation:
    def test_bad_prelude_crc_raises(self) -> None:
        frame = bytearray(_encode({":event-type": "contentBlockDelta"}, b'{"x":1}'))
        frame[8] ^= 0xFF  # corrupt the prelude CRC (bytes 8:12)
        with pytest.raises(ValueError, match="prelude checksum"):
            list(decode_messages(bytes(frame)))

    def test_bad_message_crc_raises(self) -> None:
        frame = bytearray(_encode({":event-type": "contentBlockDelta"}, b'{"x":1}'))
        frame[-5] ^= 0xFF  # corrupt the last payload byte -> message CRC no longer matches
        with pytest.raises(ValueError, match="message checksum"):
            list(decode_messages(bytes(frame)))
