#!/usr/bin/env python3
"""Market Telemetry Engine - command-line entry point.

Examples
--------
    python main.py                                   # BTC-USD + ETH-USD dashboard
    python main.py --pair BTC-USD                    # single product
    python main.py --pair BTC-USD --pair SOL-USD --block-size 1.5 --block-size SOL-USD=500
    python main.py --headless --summary-interval 10 --log-file telemetry.log
    python main.py --duration 60 --export ticks.csv  # smoke test + data dump
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import signal
import sys
import time
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.logging import RichHandler

from src import __version__
from src.analytics import (
    DEFAULT_BLOCK_THRESHOLD,
    DEFAULT_BUFFER_CAPACITY,
    DEFAULT_WINDOWS,
    Alert,
    MarketAnalytics,
)
from src.dashboard import Dashboard, format_summary
from src.engine import DEFAULT_URL, FeedError, TelemetryEngine

DEFAULT_PAIRS: tuple[str, ...] = ("BTC-USD", "ETH-USD")
_PRODUCT_RE = re.compile(r"[A-Z0-9]+-[A-Z0-9]+")

log = logging.getLogger("mte.main")
alert_log = logging.getLogger("mte.alerts")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-telemetry-engine",
        description="Real-time market telemetry & analytics engine for the Coinbase Exchange WebSocket feed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    feed = parser.add_argument_group("market data")
    feed.add_argument(
        "-p",
        "--pair",
        dest="pairs",
        action="append",
        metavar="PAIR",
        help="Product to subscribe to, e.g. BTC-USD. Repeat the flag or pass a comma-separated list. "
        f"Default: {', '.join(DEFAULT_PAIRS)}",
    )
    feed.add_argument("--url", default=DEFAULT_URL, help="WebSocket feed URL.")
    feed.add_argument(
        "--stale-timeout",
        type=float,
        default=15.0,
        metavar="SEC",
        help="Force a reconnect when no frame (incl. heartbeats) arrives for this many seconds.",
    )

    quant = parser.add_argument_group("analytics")
    quant.add_argument(
        "--windows",
        default=",".join(str(int(w)) for w in DEFAULT_WINDOWS),
        metavar="SECS",
        help="Comma-separated OFI window lengths in seconds.",
    )
    quant.add_argument(
        "--buffer-size",
        type=int,
        default=DEFAULT_BUFFER_CAPACITY,
        metavar="N",
        help="Ring buffer capacity (ticks retained per product).",
    )
    quant.add_argument(
        "--block-size",
        action="append",
        metavar="[PAIR=]SIZE",
        help="Large-trade threshold in base units (strictly greater than). A bare number sets the default "
        f"for every pair; PAIR=SIZE overrides one pair, e.g. --block-size ETH-USD=25. Default: {DEFAULT_BLOCK_THRESHOLD}",
    )
    quant.add_argument(
        "--spread-factor",
        type=float,
        default=3.0,
        metavar="X",
        help="Flag spread widening when spread >= X times its time-decayed baseline.",
    )
    quant.add_argument(
        "--spread-min-bps",
        type=float,
        default=1.0,
        metavar="BPS",
        help="Absolute floor (basis points) a spread must reach before it can be flagged.",
    )
    quant.add_argument(
        "--spread-half-life",
        type=float,
        default=30.0,
        metavar="SEC",
        help="Half-life of the EWMA spread baseline.",
    )
    quant.add_argument(
        "--spread-cooldown",
        type=float,
        default=10.0,
        metavar="SEC",
        help="Minimum seconds between spread alerts for the same product.",
    )

    ui = parser.add_argument_group("output")
    ui.add_argument("--headless", action="store_true", help="Disable the dashboard; log periodic summaries instead.")
    ui.add_argument("--fps", type=float, default=8.0, help="Dashboard refresh rate (frames per second).")
    ui.add_argument(
        "--summary-interval",
        type=float,
        default=5.0,
        metavar="SEC",
        help="Seconds between summaries in --headless mode.",
    )
    ui.add_argument(
        "--quiet-console",
        action="store_true",
        help="In dashboard mode, keep warnings in the Alerts panel / log file only (do not print above the UI).",
    )
    ui.add_argument("--log-file", metavar="PATH", help="Append timestamped logs (incl. alerts) to this file.")
    ui.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Log verbosity for the log file and headless console.",
    )

    run = parser.add_argument_group("run control")
    run.add_argument("--duration", type=float, metavar="SEC", help="Stop automatically after this many seconds.")
    run.add_argument("--export", metavar="CSV", help="On exit, dump the ring buffers to this CSV file.")
    return parser


def parse_pairs(values: Sequence[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_PAIRS)
    pairs: list[str] = []
    for raw in values:
        for token in raw.split(","):
            token = token.strip().upper()
            if not token:
                continue
            if not _PRODUCT_RE.fullmatch(token):
                raise ValueError(f"invalid product id {token!r} (expected BASE-QUOTE, e.g. BTC-USD)")
            if token not in pairs:
                pairs.append(token)
    if not pairs:
        raise ValueError("at least one --pair is required")
    return pairs


def parse_block_sizes(values: Sequence[str] | None) -> tuple[float, dict[str, float]]:
    """Return ``(default_threshold, {pair: threshold})`` from ``[PAIR=]SIZE`` tokens."""
    default = DEFAULT_BLOCK_THRESHOLD
    overrides: dict[str, float] = {}
    for raw in values or ():
        pair, sep, number = raw.partition("=")
        text = number if sep else pair
        try:
            size = float(text)
        except ValueError:
            raise ValueError(f"invalid block size {raw!r}") from None
        if not (size >= 0.0):
            raise ValueError(f"block size must be non-negative, got {raw!r}")
        if sep:
            key = pair.strip().upper()
            if not _PRODUCT_RE.fullmatch(key):
                raise ValueError(f"invalid product id in block size {raw!r}")
            overrides[key] = size
        else:
            default = size
    return default, overrides


def parse_windows(raw: str) -> tuple[float, ...]:
    try:
        windows = tuple(sorted({float(x) for x in raw.split(",") if x.strip()}))
    except ValueError:
        raise ValueError(f"invalid --windows value {raw!r}") from None
    if not windows or any(w <= 0.0 for w in windows):
        raise ValueError("--windows must list one or more positive numbers of seconds")
    return windows


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class _UTCFormatter(logging.Formatter):
    converter = staticmethod(time.gmtime)  # type: ignore[assignment]


def configure_logging(console: Console, *, level: str, log_file: str | None, dashboard: bool, quiet_console: bool) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    numeric = getattr(logging, level)
    if not (dashboard and quiet_console):
        console_handler = RichHandler(
            console=console,
            show_path=False,
            markup=False,
            rich_tracebacks=True,
            log_time_format="[%H:%M:%S]",
        )
        # In dashboard mode only warnings (alerts, connectivity) surface above the UI.
        console_handler.setLevel(logging.WARNING if dashboard else numeric)
        root.addHandler(console_handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(numeric)
        file_handler.setFormatter(
            _UTCFormatter("%(asctime)s.%(msecs)03dZ %(levelname)-8s %(name)s: %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
        )
        root.addHandler(file_handler)

    logging.getLogger("websockets").setLevel(logging.DEBUG if level == "DEBUG" else logging.WARNING)


def _log_alert(alert: Alert) -> None:
    """Alert handler: timestamped console/log warning."""
    alert_log.warning("%s", alert.console_line())


# --------------------------------------------------------------------------- #
# Async application
# --------------------------------------------------------------------------- #
def _install_signal_handlers(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError, ValueError, AttributeError):
            # Windows: fall back to asyncio.run's KeyboardInterrupt handling.
            pass


async def _stop_after(seconds: float, stop: asyncio.Event) -> None:
    await asyncio.sleep(seconds)
    log.info("Configured duration of %.0fs elapsed; shutting down", seconds)
    stop.set()


async def _headless_loop(engine: TelemetryEngine, stop: asyncio.Event, interval: float) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return
        except TimeoutError:
            pass
        log.info("telemetry\n%s", format_summary(engine.snapshot()))


def _export_csv(analytics: MarketAnalytics, path: str) -> None:
    df = analytics.to_dataframe()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    log.info("Exported %d ticks to %s", len(df), target)


async def run_async(
    args: argparse.Namespace,
    console: Console,
    pairs: Sequence[str],
    block_default: float,
    block_overrides: dict[str, float],
    windows: tuple[float, ...],
) -> int:
    analytics = MarketAnalytics(
        pairs,
        windows=windows,
        buffer_capacity=args.buffer_size,
        block_thresholds=block_overrides,
        default_block_threshold=block_default,
        spread_config={
            "half_life": args.spread_half_life,
            "widen_factor": args.spread_factor,
            "min_spread_bps": args.spread_min_bps,
            "cooldown": args.spread_cooldown,
        },
    )
    engine = TelemetryEngine(
        analytics,
        products=pairs,
        url=args.url,
        alert_handlers=[_log_alert],
        stale_timeout=args.stale_timeout,
    )

    stop = asyncio.Event()
    _install_signal_handlers(stop)

    tasks: set[asyncio.Task[object]] = {
        asyncio.create_task(engine.run(), name="feed"),
        asyncio.create_task(stop.wait(), name="stop"),
    }
    if args.headless:
        tasks.add(asyncio.create_task(_headless_loop(engine, stop, args.summary_interval), name="headless"))
    else:
        dashboard = Dashboard(engine.snapshot, console=console, fps=args.fps)
        tasks.add(asyncio.create_task(dashboard.run(stop), name="dashboard"))
    if args.duration:
        tasks.add(asyncio.create_task(_stop_after(args.duration, stop), name="timer"))

    exit_code = 0
    log.info("Market Telemetry Engine %s starting for %s", __version__, ", ".join(pairs))
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task.cancelled():
                continue
            exc = task.exception()
            if exc is not None:
                exit_code = 1
                log.error("%s task failed: %s", task.get_name(), exc, exc_info=not isinstance(exc, FeedError))
    finally:
        stop.set()
        engine.stop()
        remaining = [t for t in tasks if not t.done()]
        if remaining:
            # Give the UI a moment to draw its final frame and restore the cursor.
            _, still_pending = await asyncio.wait(remaining, timeout=3.0)
            for task in still_pending:
                task.cancel()
            if still_pending:
                await asyncio.gather(*still_pending, return_exceptions=True)

        if args.export:
            try:
                _export_csv(analytics, args.export)
            except Exception:  # noqa: BLE001
                log.exception("CSV export failed")
                exit_code = exit_code or 1

        console.print()
        console.rule("[bold]Session summary[/bold]")
        console.print(format_summary(engine.snapshot()), highlight=False, markup=False)
    return exit_code


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        pairs = parse_pairs(args.pairs)
        block_default, block_overrides = parse_block_sizes(args.block_size)
        windows = parse_windows(args.windows)
    except ValueError as exc:
        parser.error(str(exc))
    unknown = sorted(p for p in block_overrides if p not in pairs)
    if unknown:
        parser.error(f"--block-size override for unsubscribed pair(s): {', '.join(unknown)}")
    if args.buffer_size <= 0:
        parser.error("--buffer-size must be positive")
    if not (0.5 <= args.fps <= 30.0):
        parser.error("--fps must be between 0.5 and 30")
    if args.spread_factor <= 1.0:
        parser.error("--spread-factor must be greater than 1")
    if args.spread_half_life <= 0.0 or args.stale_timeout <= 0.0 or args.summary_interval <= 0.0:
        parser.error("--spread-half-life, --stale-timeout and --summary-interval must be positive")
    if args.duration is not None and args.duration <= 0.0:
        parser.error("--duration must be positive")

    console = Console()
    configure_logging(
        console,
        level=args.log_level,
        log_file=args.log_file,
        dashboard=not args.headless,
        quiet_console=args.quiet_console,
    )

    try:
        return asyncio.run(run_async(args, console, pairs, block_default, block_overrides, windows))
    except KeyboardInterrupt:
        console.print("[bold yellow]Interrupted.[/bold yellow]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
