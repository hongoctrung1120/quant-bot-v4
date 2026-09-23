"""Integration tests for WebSocket stream reconnect."""

import asyncio

import pytest

from core.config import DataConfig
from core.data.ingestion import TradeIngestionEngine
from core.data.ingestion.stream import StreamConnectionState, WebSocketTradeStream


class TestWebSocketTradeStream:
    @pytest.mark.asyncio
    async def test_stream_processes_messages(self):
        config = DataConfig(reconnect_max_retries=1)
        engine = TradeIngestionEngine(config=config)
        stream = WebSocketTradeStream(
            config=config,
            ingestion_engine=engine,
            url="wss://mock",
        )

        messages = [
            {
                "timestamp": "2024-01-15T10:00:00Z",
                "symbol": "BTC/USDT",
                "price": 50000.0,
                "quantity": 0.1,
                "side": "buy",
                "trade_id": "ws-001",
                "exchange": "binance",
            },
        ]

        async def source():
            for msg in messages:
                yield msg

        await asyncio.wait_for(stream.run(source), timeout=5.0)
        assert stream.state == StreamConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_reconnect_on_failure(self):
        config = DataConfig(
            reconnect_max_retries=2,
            reconnect_backoff_seconds=0,
        )
        engine = TradeIngestionEngine(config=config)
        stream = WebSocketTradeStream(
            config=config,
            ingestion_engine=engine,
            url="wss://mock",
        )

        call_count = 0

        async def failing_source():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Disconnected")
            yield  # pragma: no cover — marks async generator

        await asyncio.wait_for(stream.run(failing_source), timeout=5.0)
        assert call_count == 3
        assert stream.state == StreamConnectionState.FAILED
