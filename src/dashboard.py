"""Live terminal dashboard built on ``rich.live.Live``.

The dashboard is a pure *view*: on a fixed timer it pulls an immutable
``TelemetrySnapshot`` from the engine and renders four regions -

    +----------------------------------------------------------------+
    | header: feed state, products, uptime, wall clock               |
    +----------------------------------------------------------------+
    | Live Ticker: price, VWAP, OFI (10s/30s/60s), bid/ask, spread   |
    +---------------------------------+------------------------------+
    | Recent Large Block Trades       | System Telemetry             |
    +---------------------------------+------------------------------+
    | Alerts (timestamped warnings)                                  |
    +----------------------------------------------------------------+

Rendering is decoupled from ingestion, so UI cost never adds latency to the
tick-processing path.
"""

from __future__ import annotations

import asyncio
import math
from typing import Callable, Iterable, Mapping

from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .analytics import (
    ALERT_LARGE_TRADE,
    ALERT_SPREAD_WIDENING,
    Alert,
    MarketSnapshot,
    ProductSnapshot,
    Trade,
    base_currency,
    format_utc,
)
from .engine import FeedState, FeedStatus, TelemetrySnapshot

__all__ = ["Dashboard", "format_summary"]

_STATE_STYLES = {
    FeedState.IDLE: "bright_black",
    FeedState.CONNECTING: "yellow",
    FeedState.CONNECTED: "yellow",
    FeedState.LIVE: "bold green",
    FeedState.RECONNECTING: "bold red",
    FeedState.STOPPED: "bright_black",
}
_ALERT_STYLES = {ALERT_LARGE_TRADE: "bold magenta", ALERT_SPREAD_WIDENING: "bold yellow"}
_DASH = "\u2014"


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def fmt_price(x: float) -> str:
    return f"{x:,.2f}" if _finite(x) else _DASH


def fmt_qty(x: float, digits: int = 4) -> str:
    return f"{x:,.{digits}f}" if _finite(x) else _DASH


def fmt_signed(x: float, digits: int = 4) -> str:
    return f"{x:+,.{digits}f}" if _finite(x) else _DASH


def fmt_bps(x: float, digits: int = 1) -> str:
    return f"{x:+.{digits}f}bp" if _finite(x) else _DASH


def fmt_ms(x: float) -> str:
    return f"{x:.1f}" if _finite(x) else _DASH


def fmt_usd(x: float) -> str:
    """Compact USD notional: ``$1,234`` / ``$45.6K`` / ``$1.23M`` / ``$4.5B``."""
    if not _finite(x):
        return _DASH
    a = abs(x)
    sign = "-" if x < 0 else ""
    if a >= 1e9:
        return f"{sign}${a / 1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}${a / 1e6:.2f}M"
    if a >= 1e4:
        return f"{sign}${a / 1e3:.1f}K"
    return f"{sign}${a:,.0f}"


def fmt_duration(seconds: float) -> str:
    if not _finite(seconds) or seconds < 0:
        return _DASH
    s = int(seconds)
    days, s = divmod(s, 86_400)
    h, s = divmod(s, 3_600)
    m, s = divmod(s, 60)
    return f"{days}d {h:02d}:{m:02d}:{s:02d}" if days else f"{h:02d}:{m:02d}:{s:02d}"


def fmt_clock(ts: float) -> str:
    return format_utc(ts, date=False) if _finite(ts) else _DASH


def _sign_style(x: float) -> str:
    if not _finite(x) or x == 0.0:
        return "white"
    return "green" if x > 0 else "red"


def _latency_style(ms: float) -> str:
    if not _finite(ms):
        return "bright_black"
    if ms < 100.0:
        return "green"
    if ms < 300.0:
        return "yellow"
    return "red"


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
class Dashboard:
    """Real-time terminal UI driven by ``rich.live.Live``."""

    def __init__(
        self,
        snapshot_provider: Callable[[], TelemetrySnapshot],
        *,
        console: Console | None = None,
        fps: float = 8.0,
        title: str = "Market Telemetry Engine",
        block_rows: int | None = None,
        alert_rows: int | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self._provider = snapshot_provider
        self.console = console or Console()
        self.fps = float(fps)
        self.title = title
        self._block_rows = block_rows
        self._alert_rows = alert_rows
        self.frames = 0

    # -- lifecycle ---------------------------------------------------------- #
    async def run(self, stop: asyncio.Event) -> None:
        """Render frames at ``fps`` until ``stop`` is set."""
        interval = 1.0 / self.fps
        with Live(
            self.render(self._provider()),
            console=self.console,
            auto_refresh=False,
            screen=False,
            vertical_overflow="ellipsis",
        ) as live:
            while not stop.is_set():
                await asyncio.sleep(interval)
                live.update(self.render(self._provider()), refresh=True)
                self.frames += 1

    # -- layout ------------------------------------------------------------- #
    def render(self, snap: TelemetrySnapshot) -> RenderableType:
        height = self.console.size.height
        block_rows = self._block_rows if self._block_rows is not None else max(3, min(10, height - 26))
        alert_rows = self._alert_rows if self._alert_rows is not None else max(3, min(8, height - 28))

        market = snap.market
        lower = Table.grid(expand=True, padding=(0, 1))
        lower.add_column(ratio=3)
        lower.add_column(ratio=2)
        lower.add_row(
            self._block_trades_panel(market.block_trades, market.products, block_rows),
            self._telemetry_panel(snap),
        )
        return Group(
            self._header(snap),
            self._ticker_panel(market),
            lower,
            self._alerts_panel(market.alerts, alert_rows),
        )

    def _header(self, snap: TelemetrySnapshot) -> RenderableType:
        feed = snap.feed
        state_style = _STATE_STYLES.get(feed.state, "white")
        text = Text.assemble(
            ("\u25cf ", state_style),
            (feed.state.value, state_style),
            ("   ", ""),
            (feed.url, "bright_black"),
            ("   ", ""),
            (", ".join(feed.products), "cyan"),
            ("   up ", "bright_black"),
            (fmt_duration(snap.market.uptime), "white"),
            ("   ", ""),
            (format_utc(snap.timestamp).replace("T", " ").replace("Z", " UTC"), "bright_black"),
        )
        return Panel(text, title=f"[bold]{self.title}[/bold]", border_style="blue", box=box.ROUNDED)

    def _ticker_panel(self, market: MarketSnapshot) -> RenderableType:
        windows = _all_windows(market)
        table = Table(box=box.SIMPLE_HEAD, expand=True, show_edge=False, pad_edge=False, header_style="bold")
        table.add_column("Asset", style="bold cyan", no_wrap=True)
        table.add_column("Price", justify="right", no_wrap=True)
        table.add_column("VWAP", justify="right", no_wrap=True)
        table.add_column("vs VWAP", justify="right", no_wrap=True)
        for w in windows:
            table.add_column(f"OFI {int(w)}s", justify="right", no_wrap=True)
        table.add_column("Bid / Ask", justify="right", no_wrap=True)
        table.add_column("Spread", justify="right", no_wrap=True)
        table.add_column("\u03c3 1m", justify="right", no_wrap=True)

        for pid in sorted(market.products):
            p = market.products[pid]
            table.add_row(
                pid,
                _price_cell(p),
                fmt_price(p.vwap),
                Text(fmt_bps(p.vwap_deviation_bps), style=_sign_style(p.vwap_deviation_bps)),
                *(_ofi_cell(p, w) for w in windows),
                _quote_cell(p),
                _spread_cell(p),
                Text(fmt_qty(p.realized_vol_bps, 1) if _finite(p.realized_vol_bps) else _DASH, style="bright_black"),
            )
        if not market.products:
            table.add_row("waiting for data\u2026", *([_DASH] * (6 + len(windows))))
        return Panel(table, title="[bold]Live Ticker[/bold]", border_style="cyan", box=box.ROUNDED)

    def _block_trades_panel(
        self, trades: Iterable[Trade], products: Mapping[str, ProductSnapshot], rows: int
    ) -> RenderableType:
        table = Table(box=box.SIMPLE_HEAD, expand=True, show_edge=False, pad_edge=False, header_style="bold")
        table.add_column("Time (UTC)", no_wrap=True, style="bright_black")
        table.add_column("Asset", no_wrap=True, style="cyan")
        table.add_column("Side", no_wrap=True)
        table.add_column("Size", justify="right", no_wrap=True)
        table.add_column("Price", justify="right", no_wrap=True)
        table.add_column("Notional", justify="right", no_wrap=True)

        recent = list(trades)[-rows:]
        for t in reversed(recent):
            side_style = "bold green" if t.is_buy else "bold red"
            table.add_row(
                fmt_clock(t.exchange_time),
                t.product_id,
                Text(t.side.upper(), style=side_style),
                Text(fmt_qty(t.size), style="bold"),
                fmt_price(t.price),
                Text(fmt_usd(t.notional), style="magenta"),
            )
        if not recent:
            thresholds = _DASH
            if products:
                thresholds = ", ".join(
                    f"{pid} > {p.block_threshold:g} {base_currency(pid)}" for pid, p in sorted(products.items())
                )
            table.add_row(Text(f"no block trades yet ({thresholds})", style="bright_black"), "", "", "", "", "")
        return Panel(table, title="[bold]Recent Large Block Trades[/bold]", border_style="magenta", box=box.ROUNDED)

    def _telemetry_panel(self, snap: TelemetrySnapshot) -> RenderableType:
        market, feed = snap.market, snap.feed
        lat = market.latency
        grid = Table.grid(padding=(0, 1), expand=True)
        grid.add_column(style="bright_black", no_wrap=True)
        grid.add_column(no_wrap=True)

        state_style = _STATE_STYLES.get(feed.state, "white")
        feed_text = Text.assemble(("\u25cf ", state_style), (feed.state.value, state_style))
        if _finite(feed.last_message_age):
            feed_text.append(f"  last frame {feed.last_message_age:.1f}s ago", style="bright_black")
        grid.add_row("Feed", feed_text)
        grid.add_row("Uptime", fmt_duration(market.uptime))
        grid.add_row(
            "Ticks/sec",
            Text.assemble(
                (f"{market.ticks_per_sec:.1f}", "bold"),
                (f"  (10s avg {market.ticks_per_sec_avg:.1f})", "bright_black"),
            ),
        )
        grid.add_row(
            "Latency avg",
            Text.assemble(
                (f"{fmt_ms(lat.mean_ms)} ms", f"bold {_latency_style(lat.mean_ms)}"),
                (f"  p50 {fmt_ms(lat.p50_ms)}  p95 {fmt_ms(lat.p95_ms)}  p99 {fmt_ms(lat.p99_ms)}", "bright_black"),
            ),
        )
        grid.add_row(
            "Latency range",
            Text(f"min {fmt_ms(lat.min_ms)}  max {fmt_ms(lat.max_ms)}  last {fmt_ms(lat.last_ms)} ms", style="bright_black"),
        )
        volume = "  ".join(
            f"{base_currency(pid)} {fmt_qty(v)}" for pid, v in sorted(market.volume_by_product.items())
        )
        grid.add_row("Volume", Text(volume or _DASH, style="bold"))
        grid.add_row("Notional", Text(fmt_usd(market.notional_total), style="bold magenta"))
        grid.add_row(
            "Processed",
            f"{market.trades_total:,} trades  {market.quotes_total:,} quotes  {feed.messages:,} frames",
        )
        buffers = "  ".join(
            f"{base_currency(pid)} {p.buffer_len:,}/{p.buffer_capacity:,}" for pid, p in sorted(market.products.items())
        )
        grid.add_row("Ring buffer", buffers or _DASH)
        integrity = Text.assemble(
            (f"{feed.reconnects} reconnects", "red" if feed.reconnects else "white"),
            ("  ", ""),
            (f"{feed.missed_trades} missed", "yellow" if feed.missed_trades else "white"),
            ("  ", ""),
            (f"{feed.parse_errors} parse err", "yellow" if feed.parse_errors else "white"),
        )
        grid.add_row("Integrity", integrity)
        grid.add_row(
            "Alerts",
            Text.assemble(
                (f"{market.alerts_total:,} total", "bold"),
                (f"  {market.block_trades_total:,} block trades", "bright_black"),
            ),
        )
        if feed.last_error:
            grid.add_row("Last error", Text(feed.last_error, style="red", overflow="ellipsis", no_wrap=True))
        return Panel(grid, title="[bold]System Telemetry[/bold]", border_style="green", box=box.ROUNDED)

    def _alerts_panel(self, alerts: Iterable[Alert], rows: int) -> RenderableType:
        recent = list(alerts)[-rows:]
        lines = Text()
        for i, a in enumerate(reversed(recent)):
            if i:
                lines.append("\n")
            lines.append(format_utc(a.timestamp, date=False), style="bright_black")
            lines.append("  ")
            lines.append(f"{a.product_id:<8}", style="cyan")
            lines.append(f"{a.kind:<16}", style=_ALERT_STYLES.get(a.kind, "bold"))
            lines.append(a.message)
        if not recent:
            lines.append("no anomalies detected", style="bright_black")
        return Panel(lines, title="[bold]Alerts[/bold]", border_style="yellow", box=box.ROUNDED)


# --------------------------------------------------------------------------- #
# Cell renderers
# --------------------------------------------------------------------------- #
def _all_windows(market: MarketSnapshot) -> list[float]:
    windows: set[float] = set()
    for p in market.products.values():
        windows.update(p.ofi.keys())
    return sorted(windows)


def _price_cell(p: ProductSnapshot) -> Text:
    if not _finite(p.last_price):
        return Text(_DASH)
    arrow, style = {1: ("\u25b2 ", "green"), -1: ("\u25bc ", "red")}.get(p.tick_direction, ("  ", "white"))
    return Text.assemble((arrow, style), (fmt_price(p.last_price), f"bold {style}"))


def _ofi_cell(p: ProductSnapshot, window: float) -> Text:
    reading = p.ofi.get(window)
    if reading is None:
        return Text(_DASH)
    style = _sign_style(reading.delta)
    return Text.assemble(
        (fmt_signed(reading.delta), style),
        (f" {reading.ratio:+.0%}", "bright_black"),
    )


def _quote_cell(p: ProductSnapshot) -> Text:
    if not (_finite(p.bid) and _finite(p.ask)):
        return Text(_DASH)
    return Text.assemble((fmt_price(p.bid), "green"), (" / ", "bright_black"), (fmt_price(p.ask), "red"))


def _spread_cell(p: ProductSnapshot) -> Text:
    if not _finite(p.spread):
        return Text(_DASH)
    style = "bold red" if p.spread_is_wide else "white"
    text = Text.assemble((fmt_price(p.spread), style), (f" ({p.spread_bps:.2f}bp)", "bright_black"))
    if p.spread_is_wide:
        text.append(" WIDE", style="bold red")
    return text


# --------------------------------------------------------------------------- #
# Headless output
# --------------------------------------------------------------------------- #
def format_summary(snap: TelemetrySnapshot) -> str:
    """Multi-line plain-text summary used by ``--headless`` mode and logs."""
    market, feed = snap.market, snap.feed
    lat = market.latency
    lines = [
        (
            f"feed={feed.state.value} up={fmt_duration(market.uptime)} "
            f"ticks/s={market.ticks_per_sec:.1f} (10s {market.ticks_per_sec_avg:.1f}) "
            f"latency avg={fmt_ms(lat.mean_ms)}ms p50={fmt_ms(lat.p50_ms)} p95={fmt_ms(lat.p95_ms)} p99={fmt_ms(lat.p99_ms)} "
            f"trades={market.trades_total:,} notional={fmt_usd(market.notional_total)} "
            f"reconnects={feed.reconnects} missed={feed.missed_trades} alerts={market.alerts_total}"
        )
    ]
    for pid in sorted(market.products):
        p = market.products[pid]
        ofi = " ".join(f"ofi{int(w)}={r.delta:+.4f}" for w, r in sorted(p.ofi.items()))
        if _finite(p.spread_bps):
            spread = f"{fmt_price(p.spread)} ({p.spread_bps:.2f}bp{' WIDE' if p.spread_is_wide else ''})"
        else:
            spread = _DASH
        lines.append(
            f"  {pid}: px={fmt_price(p.last_price)} vwap={fmt_price(p.vwap)} ({fmt_bps(p.vwap_deviation_bps)}) "
            f"{ofi} spread={spread} vol={fmt_qty(p.total_volume)} {base_currency(pid)} "
            f"buf={p.buffer_len:,}/{p.buffer_capacity:,}"
        )
    return "\n".join(lines)
