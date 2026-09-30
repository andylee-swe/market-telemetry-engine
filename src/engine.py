"""WebSocket ingestion and asynchronous stream handling.

``CoinbaseFeed`` maintains a resilient connection to the Coinbase Exchange
public WebSocket feed and turns raw JSON frames into ``Trade`` / ``Quote``
objects, which it hands to synchronous callbacks on the event loop thread
(no queues or cross-thread hops on the hot path).

Channels
--------
* ``matches``   - executed trades (the primary telemetry stream).
* ``ticker``    - top-of-book best bid / ask after every trade, used for
                  bid-ask spread monitoring.
* ``heartbeat`` - 1 Hz liveness signal per product; drives the stale-feed
                  watchdog so silent disconnects are detected quickly.

Resilience
----------
* Exponential backoff with equal jitter between reconnect attempts.
* Watchdog task that force-closes the socket when no frame (of any type) has
  arrived for ``stale_timeout`` seconds.
* Per-product ``trade_id`` continuity tracking to surface dropped trades.
* Malformed frames are counted and skipped, never fatal.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import WebSocketException

from . import __version__
from .analytics import BUY, SELL, Alert, MarketAnalytics, MarketSnapshot, Quote, Trade

__all__ = [
    "DEFAULT_CHANNELS",
    "DEFAULT_URL",
    "CoinbaseFeed",
    "FeedError",
    "FeedState",
    "FeedStatus",
    "TelemetryEngine",
    "TelemetrySnapshot",
    "build_subscribe_message",
    "parse_exchange_time",
    "parse_match",
    "parse_ticker",
]

log = logging.getLogger("mte.engine")

DEFAULT_URL = "wss://ws-feed.exchange.coinbase.com"
DEFAULT_CHANNELS: tuple[str, ...] = ("matches", "ticker", "heartbeat")
USER_AGENT = f"market-telemetry-engine/{__version__}"

try:  # Optional accelerator; falls back to the stdlib parser when absent.
    import orjson as _orjson  # type: ignore[import-not-found]

    _loads: Callable[[str | bytes], Any] = _orjson.loads
except ImportError:  # pragma: no cover - depends on environment
    _loads = json.loads


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
_EPOCH_BY_DATE: dict[str, float] = {}
_UTC = timezone.utc


def parse_exchange_time(value: str) -> float:
    """Parse a Coinbase ISO-8601 UTC timestamp into epoch seconds.

    Fast path for the canonical ``YYYY-MM-DDTHH:MM:SS[.ffffff]Z`` layout: the
    date component's epoch is cached and the time-of-day is parsed with plain
    integer arithmetic (~4x faster than ``datetime.fromisoformat``).  Anything
    unexpected falls back to the stdlib parser.
    """
    try:
        if value[4] != "-" or value[7] != "-" or value[10] != "T" or value[13] != ":" or value[16] != ":":
            raise ValueError
        date_part = value[:10]
        day_epoch = _EPOCH_BY_DATE.get(date_part)
        if day_epoch is None:
            day_epoch = datetime(
                int(value[0:4]), int(value[5:7]), int(value[8:10]), tzinfo=_UTC
            ).timestamp()
            if len(_EPOCH_BY_DATE) >= 64:
                _EPOCH_BY_DATE.clear()
            _EPOCH_BY_DATE[date_part] = day_epoch
        hours = int(value[11:13])
        minutes = int(value[14:16])
        seconds = int(value[17:19])
        if hours > 23 or minutes > 59 or seconds > 60:
            raise ValueError
        tail = value[19:]
        if tail.endswith("Z"):
            tail = tail[:-1]
        elif tail.endswith("+00:00"):
            tail = tail[:-6]
        frac = 0.0
        if tail:
            if tail[0] != ".":
                raise ValueError
            frac = float(tail)  # float(".123456") == 0.123456
        return day_epoch + hours * 3600 + minutes * 60 + seconds + frac
    except (ValueError, IndexError, TypeError):
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_UTC)
        return dt.timestamp()


def parse_match(msg: Any, receipt_time: float) -> Trade:
    """Convert a ``match`` / ``last_match`` frame into a ``Trade``.

    Coinbase reports ``side`` as the *maker* side.  A ``sell`` maker means the
    aggressor bought (up-tick), so the aggressor side is the inverse.
    """
    maker_side = msg["side"]
    if maker_side == SELL:
        taker_side = BUY
    elif maker_side == BUY:
        taker_side = SELL
    else:
        raise ValueError(f"unknown side {maker_side!r}")
    return Trade(
        product_id=msg["product_id"],
        trade_id=int(msg["trade_id"]),
        price=float(msg["price"]),
        size=float(msg["size"]),
        side=taker_side,
        exchange_time=parse_exchange_time(msg["time"]),
        receipt_time=receipt_time,
        sequence=int(msg.get("sequence", 0)),
    )


def parse_ticker(msg: Any, receipt_time: float) -> Quote | None:
    """Convert a ``ticker`` frame into a ``Quote`` (``None`` if no book is present)."""
    bid = msg.get("best_bid")
    ask = msg.get("best_ask")
    if not bid or not ask:
        return None
    ts = msg.get("time")
    return Quote(
        product_id=msg["product_id"],
        bid=float(bid),
        ask=float(ask),
        exchange_time=parse_exchange_time(ts) if ts else receipt_time,
        receipt_time=receipt_time,
    )


def build_subscribe_message(products: Sequence[str], channels: Sequence[str] = DEFAULT_CHANNELS) -> str:
    """JSON subscription request for the given products and channels."""
    return json.dumps(
        {"type": "subscribe", "product_ids": list(products), "channels": list(channels)},
        separators=(",", ":"),
    )


# --------------------------------------------------------------------------- #
# Feed state
# --------------------------------------------------------------------------- #
class FeedError(Exception):
    """Protocol-level error reported by the exchange."""

    def __init__(self, message: str, reason: str | None = None, *, retryable: bool = True) -> None:
        detail = f"{message} ({reason})" if reason else message
        super().__init__(detail)
        self.message = message
        self.reason = reason
        self.retryable = retryable


class FeedState(str, Enum):
    IDLE = "IDLE"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"  # socket open, subscription sent, awaiting ack
    LIVE = "LIVE"  # subscription acknowledged, data flowing
    RECONNECTING = "RECONNECTING"
    STOPPED = "STOPPED"


@dataclass(slots=True)
class FeedStatus:
    """Mutable connection telemetry owned by ``CoinbaseFeed``."""

    url: str
    products: tuple[str, ...]
    channels: tuple[str, ...]
    state: FeedState = FeedState.IDLE
    connections: int = 0
    reconnects: int = 0
    stale_reconnects: int = 0
    connected_since: float | None = None
    last_message_at: float | None = None
    messages: int = 0
    bytes: int = 0
    trades: int = 0
    quotes: int = 0
    heartbeats: int = 0
    parse_errors: int = 0
    handler_errors: int = 0
    unknown_messages: int = 0
    missed_trades: int = 0
    last_error: str | None = None
    last_message_age: float = field(default=float("nan"))

    def snapshot(self, now: float) -> "FeedStatus":
        copy = replace(self)
        copy.last_message_age = (now - self.last_message_at) if self.last_message_at is not None else float("nan")
        return copy


# --------------------------------------------------------------------------- #
# WebSocket client
# --------------------------------------------------------------------------- #
class CoinbaseFeed:
    """Resilient client for the Coinbase Exchange public WebSocket feed."""

    def __init__(
        self,
        products: Iterable[str],
        *,
        on_trade: Callable[[Trade], None],
        on_quote: Callable[[Quote], None] | None = None,
        url: str = DEFAULT_URL,
        channels: Sequence[str] = DEFAULT_CHANNELS,
        stale_timeout: float = 15.0,
        connect_timeout: float = 10.0,
        ping_interval: float = 20.0,
        backoff_initial: float = 1.0,
        backoff_max: float = 30.0,
        max_queue: int = 1024,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.products = tuple(dict.fromkeys(p.upper() for p in products))
        if not self.products:
            raise ValueError("at least one product is required")
        self.url = url
        self.channels = tuple(channels)
        self.stale_timeout = float(stale_timeout)
        self.connect_timeout = float(connect_timeout)
        self.ping_interval = float(ping_interval)
        self.backoff_initial = float(backoff_initial)
        self.backoff_max = float(backoff_max)
        self.max_queue = int(max_queue)
        self._on_trade = on_trade
        self._on_quote = on_quote
        self._clock = clock
        self.status = FeedStatus(url=url, products=self.products, channels=self.channels)
        self._stop_event = asyncio.Event()
        self._ws: ClientConnection | None = None
        self._last_message_at = 0.0
        self._session_messages = 0
        self._last_trade_ids: dict[str, int] = {}

    # -- lifecycle ---------------------------------------------------------- #
    def subscribe_message(self) -> str:
        return build_subscribe_message(self.products, self.channels)

    def status_snapshot(self, now: float | None = None) -> FeedStatus:
        return self.status.snapshot(self._clock() if now is None else now)

    def stop(self) -> None:
        """Request shutdown; interrupts backoff sleeps and closes the socket."""
        self._stop_event.set()
        ws = self._ws
        if ws is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            loop.create_task(ws.close(code=1000, reason="client shutdown"))

    async def run(self) -> None:
        """Connect, consume and reconnect until ``stop()`` is called.

        Raises ``FeedError`` for non-retryable protocol errors (e.g. an invalid
        product id rejected by the exchange).
        """
        attempt = 0
        try:
            while not self._stop_event.is_set():
                self._session_messages = 0
                try:
                    await self._session()
                except asyncio.CancelledError:
                    raise
                except FeedError as exc:
                    self.status.last_error = str(exc)
                    if not exc.retryable:
                        log.error("Fatal feed error: %s", exc)
                        raise
                    log.error("Feed error: %s", exc)
                except (OSError, WebSocketException, TimeoutError) as exc:
                    self.status.last_error = f"{type(exc).__name__}: {exc}"
                    log.warning("Connection lost: %s", self.status.last_error)
                else:
                    if self._stop_event.is_set():
                        break
                    log.warning("Connection closed by peer or watchdog")

                if self._stop_event.is_set():
                    break
                # A session that delivered data resets the backoff schedule.
                attempt = 1 if self._session_messages > 0 else attempt + 1
                delay = self._backoff_delay(attempt)
                self.status.reconnects += 1
                self._set_state(FeedState.RECONNECTING)
                log.warning("Reconnecting in %.1fs (attempt %d)", delay, attempt)
                await self._interruptible_sleep(delay)
        finally:
            self._set_state(FeedState.STOPPED)

    # -- internals ---------------------------------------------------------- #
    def _set_state(self, state: FeedState) -> None:
        if self.status.state is not state:
            log.debug("Feed state %s -> %s", self.status.state.value, state.value)
            self.status.state = state

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with equal jitter: ``[cap/2, cap]`` where ``cap = min(max, initial * 2**(n-1))``."""
        cap = min(self.backoff_max, self.backoff_initial * (2.0 ** max(0, attempt - 1)))
        return cap * 0.5 + random.uniform(0.0, cap * 0.5)

    async def _interruptible_sleep(self, delay: float) -> None:
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
        except TimeoutError:
            pass

    async def _session(self) -> None:
        self._set_state(FeedState.CONNECTING)
        log.info("Connecting to %s for %s", self.url, ", ".join(self.products))
        async with connect(
            self.url,
            open_timeout=self.connect_timeout,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_interval,
            close_timeout=5.0,
            max_queue=self.max_queue,
            compression=None,
            user_agent_header=USER_AGENT,
        ) as ws:
            self._ws = ws
            now = self._clock()
            self._last_message_at = now
            self._last_trade_ids.clear()
            st = self.status
            st.connections += 1
            st.connected_since = now
            st.last_message_at = now
            await ws.send(self.subscribe_message())
            self._set_state(FeedState.CONNECTED)
            watchdog = asyncio.create_task(self._watchdog(ws), name="feed-watchdog")
            try:
                async for raw in ws:
                    self._handle(raw, self._clock())
            finally:
                self._ws = None
                watchdog.cancel()
                await asyncio.gather(watchdog, return_exceptions=True)

    async def _watchdog(self, ws: ClientConnection) -> None:
        interval = max(0.5, self.stale_timeout / 4.0)
        while True:
            await asyncio.sleep(interval)
            idle = self._clock() - self._last_message_at
            if idle > self.stale_timeout:
                self.status.stale_reconnects += 1
                log.warning("No frames for %.1fs (limit %.1fs); forcing reconnect", idle, self.stale_timeout)
                await ws.close(code=1000, reason="stale feed watchdog")
                return

    def _handle(self, raw: str | bytes, now: float) -> None:
        st = self.status
        st.messages += 1
        st.bytes += len(raw)
        st.last_message_at = now
        self._last_message_at = now
        self._session_messages += 1
        try:
            msg = _loads(raw)
            mtype = msg.get("type")
        except (ValueError, AttributeError):
            st.parse_errors += 1
            log.debug("Undecodable frame: %r", raw[:200])
            return

        if mtype == "match":
            try:
                trade = parse_match(msg, now)
            except (KeyError, ValueError, TypeError) as exc:
                st.parse_errors += 1
                log.debug("Bad match frame (%s): %r", exc, msg)
                return
            self._track_trade_gap(trade.product_id, trade.trade_id)
            st.trades += 1
            try:
                self._on_trade(trade)
            except Exception:  # noqa: BLE001 - handler bugs must not kill the feed
                st.handler_errors += 1
                log.exception("on_trade handler failed")
        elif mtype == "ticker":
            if self._on_quote is None:
                return
            try:
                quote = parse_ticker(msg, now)
            except (KeyError, ValueError, TypeError) as exc:
                st.parse_errors += 1
                log.debug("Bad ticker frame (%s): %r", exc, msg)
                return
            if quote is None:
                return
            st.quotes += 1
            try:
                self._on_quote(quote)
            except Exception:  # noqa: BLE001
                st.handler_errors += 1
                log.exception("on_quote handler failed")
        elif mtype == "heartbeat":
            st.heartbeats += 1
        elif mtype == "last_match":
            # Reference trade at subscription time.  Its exchange timestamp
            # predates the connection, so it is used only to seed trade-id
            # continuity tracking and is excluded from analytics/latency.
            try:
                self._last_trade_ids[msg["product_id"]] = int(msg["trade_id"])
            except (KeyError, ValueError, TypeError):
                st.parse_errors += 1
        elif mtype == "subscriptions":
            self._set_state(FeedState.LIVE)
            summary = ", ".join(
                f"{c.get('name')}[{','.join(c.get('product_ids') or [])}]" for c in msg.get("channels", [])
            )
            log.info("Subscribed: %s", summary or "(no channels)")
        elif mtype == "error":
            message = str(msg.get("message", "unknown error"))
            reason = msg.get("reason")
            retryable = "subscribe" not in message.lower()
            raise FeedError(message, str(reason) if reason else None, retryable=retryable)
        else:
            st.unknown_messages += 1
            log.debug("Ignoring frame type %r", mtype)

    def _track_trade_gap(self, product_id: str, trade_id: int) -> int:
        """Count skipped trade ids (Coinbase trade ids are contiguous per product)."""
        last = self._last_trade_ids.get(product_id)
        missed = 0
        if last is not None and trade_id > last + 1:
            missed = trade_id - last - 1
            self.status.missed_trades += missed
            log.debug("%s: %d trade id(s) skipped between %d and %d", product_id, missed, last, trade_id)
        if last is None or trade_id > last:
            self._last_trade_ids[product_id] = trade_id
        return missed


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class TelemetrySnapshot:
    """Everything the dashboard needs for one frame."""

    timestamp: float
    market: MarketSnapshot
    feed: FeedStatus


class TelemetryEngine:
    """Wires ``CoinbaseFeed`` -> ``MarketAnalytics`` -> alert handlers."""

    def __init__(
        self,
        analytics: MarketAnalytics,
        *,
        products: Iterable[str] | None = None,
        url: str = DEFAULT_URL,
        channels: Sequence[str] = DEFAULT_CHANNELS,
        alert_handlers: Iterable[Callable[[Alert], None]] = (),
        clock: Callable[[], float] = time.time,
        **feed_options: Any,
    ) -> None:
        self.analytics = analytics
        self._alert_handlers: list[Callable[[Alert], None]] = list(alert_handlers)
        self._clock = clock
        self.feed = CoinbaseFeed(
            tuple(products) if products is not None else tuple(analytics.products),
            on_trade=self._on_trade,
            on_quote=self._on_quote,
            url=url,
            channels=channels,
            clock=clock,
            **feed_options,
        )

    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        self._alert_handlers.append(handler)

    def _on_trade(self, trade: Trade) -> None:
        alert = self.analytics.on_trade(trade)
        if alert is not None:
            self._emit(alert)

    def _on_quote(self, quote: Quote) -> None:
        alert = self.analytics.on_quote(quote)
        if alert is not None:
            self._emit(alert)

    def _emit(self, alert: Alert) -> None:
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception:  # noqa: BLE001
                log.exception("alert handler %r failed", handler)

    async def run(self) -> None:
        await self.feed.run()

    def stop(self) -> None:
        self.feed.stop()

    def snapshot(self) -> TelemetrySnapshot:
        now = self._clock()
        return TelemetrySnapshot(
            timestamp=now,
            market=self.analytics.snapshot(now),
            feed=self.feed.status_snapshot(now),
        )
