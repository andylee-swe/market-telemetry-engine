"""Smoke tests for dashboard rendering (no TTY required)."""

from __future__ import annotations

from rich.console import Console

from src.analytics import BUY, SELL, MarketAnalytics, Quote, Trade
from src.dashboard import Dashboard, format_summary
from src.engine import FeedState, FeedStatus, TelemetrySnapshot

T0 = 1_760_000_000.0


def _snap() -> TelemetrySnapshot:
    ma = MarketAnalytics(["BTC-USD", "ETH-USD"], clock=lambda: T0)
    ma.on_trade(
        Trade(
            product_id="BTC-USD",
            trade_id=1,
            price=67_421.15,
            size=2.15,
            side=BUY,
            exchange_time=T0,
            receipt_time=T0 + 0.045,
        )
    )
    ma.on_trade(
        Trade(
            product_id="ETH-USD",
            trade_id=2,
            price=3_412.61,
            size=0.5,
            side=SELL,
            exchange_time=T0 + 0.2,
            receipt_time=T0 + 0.25,
        )
    )
    ma.on_quote(Quote("BTC-USD", 67_420.50, 67_421.20, T0, T0 + 0.01))
    market = ma.snapshot(now=T0 + 1.0)
    feed = FeedStatus(
        url="wss://ws-feed.exchange.coinbase.com",
        products=("BTC-USD", "ETH-USD"),
        channels=("matches", "ticker", "heartbeat"),
        state=FeedState.LIVE,
        last_message_at=T0 + 0.9,
    )
    return TelemetrySnapshot(timestamp=T0 + 1.0, market=market, feed=feed.snapshot(T0 + 1.0))


def test_format_summary_contains_required_fields():
    text = format_summary(_snap())
    assert "feed=LIVE" in text
    assert "BTC-USD" in text and "ETH-USD" in text
    assert "vwap=" in text
    assert "ofi10=" in text
    assert "latency avg=" in text
    assert "ticks/s=" in text


def test_live_layout_renders_all_panels():
    console = Console(record=True, width=140, height=40, force_terminal=True, color_system="truecolor")
    dash = Dashboard(lambda: _snap(), console=console, fps=8.0)
    console.print(dash.render(_snap()))
    html = console.export_text()
    assert "Live Ticker" in html
    assert "Recent Large Block Trades" in html
    assert "System Telemetry" in html
    assert "Alerts" in html
    assert "BTC-USD" in html
    assert "VWAP" in html
    assert "OFI 10s" in html
    assert "Ticks/sec" in html
    assert "Latency avg" in html
    assert "LARGE_TRADE" in html  # 2.15 BTC exceeds 1.5
