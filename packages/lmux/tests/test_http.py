"""Tests for lmux._http (httpx client factories + SSE parser)."""

import httpx

from lmux._http import aiter_sse, create_async_client, create_sync_client, iter_sse


class TestCreateSyncClient:
    def test_defaults(self) -> None:
        client = create_sync_client(base_url="https://api.example.com", headers={"Authorization": "Bearer x"})
        assert str(client.base_url) == "https://api.example.com"
        assert client.headers["Authorization"] == "Bearer x"
        assert client.timeout.read == 600.0
        client.close()

    def test_timeout_and_retries(self) -> None:
        client = create_sync_client(base_url="https://x", headers={}, timeout=12.0, max_retries=3)
        assert client.timeout.read == 12.0
        client.close()


class TestCreateAsyncClient:
    async def test_defaults(self) -> None:
        client = create_async_client(base_url="https://x", headers={"a": "b"})
        assert client.headers["a"] == "b"
        assert client.timeout.read == 600.0
        await client.aclose()

    async def test_retries(self) -> None:
        client = create_async_client(base_url="https://x", headers={}, timeout=5.0, max_retries=2)
        assert client.timeout.read == 5.0
        await client.aclose()


def _sse(text: str) -> httpx.Response:
    return httpx.Response(200, content=text.encode())


class TestIterSse:
    def test_data_events(self) -> None:
        assert list(iter_sse(_sse("data: a\n\ndata: b\n\n"))) == [(None, "a"), (None, "b")]

    def test_event_and_data(self) -> None:
        assert list(iter_sse(_sse("event: ping\ndata: {}\n\n"))) == [("ping", "{}")]

    def test_comment_ignored(self) -> None:
        assert list(iter_sse(_sse(": heartbeat\ndata: x\n\n"))) == [(None, "x")]

    def test_unknown_field_ignored(self) -> None:
        assert list(iter_sse(_sse("id: 1\ndata: x\n\n"))) == [(None, "x")]

    def test_multiline_data(self) -> None:
        assert list(iter_sse(_sse("data: line1\ndata: line2\n\n"))) == [(None, "line1\nline2")]

    def test_trailing_without_blank(self) -> None:
        assert list(iter_sse(_sse("data: last"))) == [(None, "last")]

    def test_blank_without_data(self) -> None:
        assert list(iter_sse(_sse("\n\ndata: x\n\n"))) == [(None, "x")]

    def test_carriage_returns(self) -> None:
        assert list(iter_sse(_sse("data: x\r\n\r\n"))) == [(None, "x")]

    def test_empty(self) -> None:
        assert list(iter_sse(_sse(""))) == []


class TestAiterSse:
    async def test_data_events(self) -> None:
        assert [e async for e in aiter_sse(_sse("data: a\n\ndata: b\n\n"))] == [(None, "a"), (None, "b")]

    async def test_trailing_flush(self) -> None:
        assert [e async for e in aiter_sse(_sse("data: last"))] == [(None, "last")]

    async def test_empty(self) -> None:
        assert [e async for e in aiter_sse(_sse(""))] == []
