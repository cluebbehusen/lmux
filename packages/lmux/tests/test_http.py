"""Tests for lmux._http (httpx client factories, opt-in retries, SSE parser)."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx
from pytest_mock import MockerFixture

from lmux._http import aiter_sse, create_async_client, create_sync_client, iter_sse

_BASE = "https://api.example.com"
_URL = f"{_BASE}/v1/thing"


@pytest.fixture
def sync_sleep(mocker: MockerFixture) -> MagicMock:
    return mocker.patch("lmux._http.time.sleep")


@pytest.fixture
def async_sleep(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch("lmux._http.asyncio.sleep", new_callable=AsyncMock)


# MARK: Client construction


class TestCreateSyncClient:
    def test_default_is_a_plain_client(self) -> None:
        client = create_sync_client(base_url=_BASE, headers={"Authorization": "Bearer x"})
        assert type(client) is httpx.Client  # no retry subclass by default (opt-in)
        assert client.headers["Authorization"] == "Bearer x"
        assert client.timeout.read == 600.0
        assert client.timeout.connect == 10.0
        assert client.follow_redirects is True  # gateways that redirect a custom base_url are followed
        client.close()

    def test_explicit_timeout_applies_to_connect(self) -> None:
        client = create_sync_client(base_url=_BASE, headers={}, timeout=12.0)
        assert client.timeout.read == 12.0
        assert client.timeout.connect == 12.0  # not pinned to the 10s default
        client.close()

    def test_max_retries_opts_into_retry_client(self) -> None:
        client = create_sync_client(base_url=_BASE, headers={}, max_retries=3)
        assert type(client) is not httpx.Client  # retry subclass
        assert isinstance(client, httpx.Client)
        client.close()


class TestCreateAsyncClient:
    async def test_default_is_a_plain_client(self) -> None:
        client = create_async_client(base_url=_BASE, headers={"a": "b"})
        assert type(client) is httpx.AsyncClient
        assert client.timeout.read == 600.0
        await client.aclose()

    async def test_max_retries_opts_into_retry_client(self) -> None:
        client = create_async_client(base_url=_BASE, headers={}, timeout=5.0, max_retries=2)
        assert type(client) is not httpx.AsyncClient
        assert client.timeout.connect == 5.0
        await client.aclose()


# MARK: Sync retries


class TestSyncRetry:
    def test_no_retry_by_default(self, respx_mock: respx.MockRouter, sync_sleep: MagicMock) -> None:
        route = respx_mock.post(_URL).mock(side_effect=[httpx.Response(429), httpx.Response(200, json={})])
        client = create_sync_client(base_url=_BASE, headers={})
        response = client.post("/v1/thing")
        assert response.status_code == 429  # first response returned, unretried
        assert route.call_count == 1
        sync_sleep.assert_not_called()
        client.close()

    def test_retries_retryable_status_then_succeeds(self, respx_mock: respx.MockRouter, sync_sleep: MagicMock) -> None:
        route = respx_mock.post(_URL).mock(
            side_effect=[httpx.Response(429), httpx.Response(503), httpx.Response(200, json={"ok": 1})]
        )
        client = create_sync_client(base_url=_BASE, headers={}, max_retries=3)
        response = client.post("/v1/thing")
        assert response.status_code == 200
        assert route.call_count == 3
        assert sync_sleep.call_count == 2
        client.close()

    def test_retries_non_enumerated_5xx(self, respx_mock: respx.MockRouter, sync_sleep: MagicMock) -> None:
        # 520 (Cloudflare) / 529 (Anthropic overloaded) are outside the 4xx set but still 5xx.
        route = respx_mock.post(_URL).mock(side_effect=[httpx.Response(520), httpx.Response(200, json={})])
        client = create_sync_client(base_url=_BASE, headers={}, max_retries=2)
        response = client.post("/v1/thing")
        assert response.status_code == 200
        assert route.call_count == 2
        sync_sleep.assert_called_once()
        client.close()

    def test_exhausts_retries_and_returns_last_response(
        self, respx_mock: respx.MockRouter, sync_sleep: MagicMock
    ) -> None:
        route = respx_mock.post(_URL).mock(return_value=httpx.Response(503))
        client = create_sync_client(base_url=_BASE, headers={}, max_retries=2)
        response = client.post("/v1/thing")
        assert response.status_code == 503
        assert route.call_count == 3  # 1 initial + 2 retries
        assert sync_sleep.call_count == 2
        client.close()

    def test_no_retry_on_non_retryable_status(self, respx_mock: respx.MockRouter, sync_sleep: MagicMock) -> None:
        route = respx_mock.post(_URL).mock(return_value=httpx.Response(400))
        client = create_sync_client(base_url=_BASE, headers={}, max_retries=2)
        response = client.post("/v1/thing")
        assert response.status_code == 400
        assert route.call_count == 1
        sync_sleep.assert_not_called()
        client.close()

    def test_retries_transport_error_then_succeeds(self, respx_mock: respx.MockRouter, sync_sleep: MagicMock) -> None:
        route = respx_mock.post(_URL).mock(side_effect=[httpx.ConnectError("boom"), httpx.Response(200, json={})])
        client = create_sync_client(base_url=_BASE, headers={}, max_retries=2)
        response = client.post("/v1/thing")
        assert response.status_code == 200
        assert route.call_count == 2
        sync_sleep.assert_called_once()
        client.close()

    def test_transport_error_exhausted_raises(self, respx_mock: respx.MockRouter, sync_sleep: MagicMock) -> None:
        respx_mock.post(_URL).mock(side_effect=httpx.ConnectError("boom"))
        client = create_sync_client(base_url=_BASE, headers={}, max_retries=1)
        with pytest.raises(httpx.ConnectError):
            client.post("/v1/thing")
        sync_sleep.assert_called_once()
        client.close()

    def test_honors_numeric_retry_after(self, respx_mock: respx.MockRouter, sync_sleep: MagicMock) -> None:
        respx_mock.post(_URL).mock(
            side_effect=[httpx.Response(429, headers={"retry-after": "2"}), httpx.Response(200, json={})]
        )
        client = create_sync_client(base_url=_BASE, headers={}, max_retries=2)
        client.post("/v1/thing")
        sync_sleep.assert_called_once_with(2.0)
        client.close()

    def test_invalid_retry_after_falls_back_to_backoff(
        self, respx_mock: respx.MockRouter, sync_sleep: MagicMock
    ) -> None:
        respx_mock.post(_URL).mock(
            side_effect=[httpx.Response(429, headers={"retry-after": "soon"}), httpx.Response(200, json={})]
        )
        client = create_sync_client(base_url=_BASE, headers={}, max_retries=2)
        client.post("/v1/thing")
        sync_sleep.assert_called_once_with(0.5)  # _backoff(1)
        client.close()


# MARK: Async retries


class TestAsyncRetry:
    async def test_no_retry_by_default(self, respx_mock: respx.MockRouter, async_sleep: AsyncMock) -> None:
        route = respx_mock.post(_URL).mock(side_effect=[httpx.Response(429), httpx.Response(200, json={})])
        client = create_async_client(base_url=_BASE, headers={})
        response = await client.post("/v1/thing")
        assert response.status_code == 429
        assert route.call_count == 1
        async_sleep.assert_not_awaited()
        await client.aclose()

    async def test_retries_retryable_status_then_succeeds(
        self, respx_mock: respx.MockRouter, async_sleep: AsyncMock
    ) -> None:
        route = respx_mock.post(_URL).mock(side_effect=[httpx.Response(500), httpx.Response(200, json={})])
        client = create_async_client(base_url=_BASE, headers={}, max_retries=2)
        response = await client.post("/v1/thing")
        assert response.status_code == 200
        assert route.call_count == 2
        async_sleep.assert_awaited_once()
        await client.aclose()

    async def test_exhausts_retries_and_returns_last_response(
        self, respx_mock: respx.MockRouter, async_sleep: AsyncMock
    ) -> None:
        route = respx_mock.post(_URL).mock(return_value=httpx.Response(429, headers={"retry-after": "1"}))
        client = create_async_client(base_url=_BASE, headers={}, max_retries=1)
        response = await client.post("/v1/thing")
        assert response.status_code == 429
        assert route.call_count == 2
        async_sleep.assert_awaited_once_with(1.0)
        await client.aclose()

    async def test_retries_transport_error_then_succeeds(
        self, respx_mock: respx.MockRouter, async_sleep: AsyncMock
    ) -> None:
        route = respx_mock.post(_URL).mock(side_effect=[httpx.ConnectError("boom"), httpx.Response(200, json={})])
        client = create_async_client(base_url=_BASE, headers={}, max_retries=2)
        response = await client.post("/v1/thing")
        assert response.status_code == 200
        assert route.call_count == 2
        async_sleep.assert_awaited_once()
        await client.aclose()

    async def test_transport_error_exhausted_raises(self, respx_mock: respx.MockRouter, async_sleep: AsyncMock) -> None:
        respx_mock.post(_URL).mock(side_effect=httpx.ConnectError("boom"))
        client = create_async_client(base_url=_BASE, headers={}, max_retries=1)
        with pytest.raises(httpx.ConnectError):
            await client.post("/v1/thing")
        async_sleep.assert_awaited_once()
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
