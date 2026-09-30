"""Quantitative analytics for the Market Telemetry Engine.

Everything in this module is pure computation with no I/O, which keeps it
deterministic and unit-testable.  The hot path (``MarketAnalytics.on_trade``)
is O(1) amortised per tick:

* ``TickBuffer``            - ring buffer of the most recent trades
                              (``collections.deque(maxlen=N)``).
* ``RollingWindow``         - time-based sliding window of signed volume with
                              running sums (append right / evict left).
* ``OrderFlowImbalance``    - a set of ``RollingWindow`` instances (10s/30s/60s).
* ``VWAP``                  - running sum(price * size) / sum(size), reset at
                              the UTC session boundary.
* ``SpreadMonitor``         - time-decayed EWMA baseline of the bid-ask spread
                              used to flag sudden widening.
* ``LargeTradeDetector``    - block-trade threshold check.
* ``LatencyTracker``        - exchange -> client one-way latency statistics.
* ``RateMeter``             - sliding-window event rate (ticks/sec).

Formulas
--------
Order Flow Imbalance over a window ``w``::

    OFI_w  = sum(size_i * 1[taker_i = buy]) - sum(size_i * 1[taker_i = sell])
    nOFI_w = OFI_w / (buy_volume_w + sell_volume_w)          in [-1, +1]

Volume-Weighted Average Price (intraday, resets at 00:00 UTC)::

    VWAP_t = sum(price_i * size_i) / sum(size_i)     for all i in the session

Bid-ask spread in basis points and its time-decayed baseline::

    spread_bps = (ask - bid) / mid * 10_000,  mid = (ask + bid) / 2
    decay      = 0.5 ** (dt / half_life)
    baseline   = decay * baseline + (1 - decay) * spread_bps

Ingestion latency::

    latency_ms = (client_receipt_time - exchange_time) * 1_000
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import islice
from typing import Any, Deque, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "ALERT_LARGE_TRADE",
    "ALERT_SPREAD_WIDENING",
    "BUY",
    "DEFAULT_BLOCK_THRESHOLD",
    "DEFAULT_BUFFER_CAPACITY",
    "DEFAULT_WINDOWS",
    "SELL",
    "Alert",
    "LargeTradeDetector",
    "LatencySnapshot",
    "LatencyTracker",
    "MarketAnalytics",
    "MarketSnapshot",
    "OFIReading",
    "OrderFlowImbalance",
    "ProductAnalytics",
    "ProductSnapshot",
    "Quote",
    "RateMeter",
    "RollingWindow",
    "SpreadMonitor",
    "TickBuffer",
    "Trade",
    "VWAP",
    "base_currency",
    "format_utc",
]

log = logging.getLogger("mte.analytics")

BUY = "buy"
SELL = "sell"

SECONDS_PER_DAY = 86_400
DEFAULT_WINDOWS: tuple[float, ...] = (10.0, 30.0, 60.0)
DEFAULT_BUFFER_CAPACITY = 10_000
DEFAULT_BLOCK_THRESHOLD = 1.5

ALERT_LARGE_TRADE = "LARGE_TRADE"
ALERT_SPREAD_WIDENING = "SPREAD_WIDENING"

_NAN = math.nan


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def base_currency(product_id: str) -> str:
    """Return the base asset of a product id (``"BTC-USD" -> "BTC"``)."""
    return product_id.split("-", 1)[0]


def format_utc(ts: float, *, date: bool = True) -> str:
    """Format an epoch timestamp as ISO-8601 UTC with millisecond precision."""
    if not math.isfinite(ts):
        return "-"
    whole = math.floor(ts)
    millis = int((ts - whole) * 1000.0)
    dt = datetime.fromtimestamp(whole, tz=timezone.utc)
    fmt = "%Y-%m-%dT%H:%M:%S" if date else "%H:%M:%S"
    suffix = "Z" if date else ""
    return f"{dt.strftime(fmt)}.{millis:03d}{suffix}"


# --------------------------------------------------------------------------- #
# Market data model
# --------------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class Trade:
    """A single executed trade (one tick).

    ``side`` is the *aggressor* (taker) side.  Coinbase reports the maker side
    on the ``matches`` channel, so the ingestion layer inverts it before
    constructing a ``Trade``.  Timestamps are epoch seconds (UTC).
    """

    product_id: str
    trade_id: int
    price: float
    size: float
    side: str
    exchange_time: float
    receipt_time: float
    sequence: int = 0

    @property
    def is_buy(self) -> bool:
        return self.side == BUY

    @property
    def notional(self) -> float:
        return self.price * self.size

    @property
    def latency_ms(self) -> float:
        """One-way ingestion latency (exchange timestamp -> client receipt)."""
        return (self.receipt_time - self.exchange_time) * 1000.0


@dataclass(slots=True, frozen=True)
class Quote:
    """Top-of-book snapshot (best bid / best ask)."""

    product_id: str
    bid: float
    ask: float
    exchange_time: float
    receipt_time: float

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.ask + self.bid) * 0.5

    @property
    def spread_bps(self) -> float:
        mid = self.mid
        return (self.ask - self.bid) / mid * 10_000.0 if mid > 0.0 else _NAN


@dataclass(slots=True, frozen=True)
class Alert:
    """A timestamped anomaly notification."""

    kind: str
    product_id: str
    timestamp: float
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def console_line(self) -> str:
        """Single-line representation suitable for logs / console output."""
        return f"{format_utc(self.timestamp)} [{self.product_id}] {self.kind}: {self.message}"


# --------------------------------------------------------------------------- #
# Ring buffer
# --------------------------------------------------------------------------- #
class TickBuffer:
    """Fixed-capacity in-memory ring buffer of ``Trade`` objects.

    Backed by ``collections.deque(maxlen=capacity)``: appends are O(1) and the
    oldest tick is discarded automatically once the buffer is full.
    """

    __slots__ = ("_buf", "total_appended")

    def __init__(self, capacity: int = DEFAULT_BUFFER_CAPACITY) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._buf: Deque[Trade] = deque(maxlen=capacity)
        self.total_appended = 0

    def append(self, trade: Trade) -> None:
        self._buf.append(trade)
        self.total_appended += 1

    def __len__(self) -> int:
        return len(self._buf)

    def __iter__(self) -> Iterator[Trade]:
        return iter(self._buf)

    @property
    def capacity(self) -> int:
        return self._buf.maxlen or 0

    @property
    def evicted(self) -> int:
        """Number of ticks pushed out of the buffer since start-up."""
        return self.total_appended - len(self._buf)

    @property
    def utilisation(self) -> float:
        return len(self._buf) / self.capacity if self.capacity else 0.0

    @property
    def latest(self) -> Trade | None:
        return self._buf[-1] if self._buf else None

    def recent(self, n: int) -> list[Trade]:
        """Return the newest ``n`` ticks in chronological order."""
        buf = self._buf
        if n <= 0:
            return []
        if n >= len(buf):
            return list(buf)
        out = list(islice(reversed(buf), n))
        out.reverse()
        return out

    def prices(self, n: int | None = None) -> np.ndarray:
        """Prices of the newest ``n`` ticks (all ticks when ``n`` is None)."""
        trades = self.recent(n) if n is not None else list(self._buf)
        return np.fromiter((t.price for t in trades), dtype=np.float64, count=len(trades))

    def to_dataframe(self) -> pd.DataFrame:
        """Materialise the buffer as a pandas DataFrame for offline analysis."""
        columns = [
            "product_id",
            "trade_id",
            "sequence",
            "exchange_time",
            "receipt_time",
            "price",
            "size",
            "side",
            "notional",
            "latency_ms",
        ]
        records = [
            (
                t.product_id,
                t.trade_id,
                t.sequence,
                t.exchange_time,
                t.receipt_time,
                t.price,
                t.size,
                t.side,
                t.notional,
                t.latency_ms,
            )
            for t in self._buf
        ]
        df = pd.DataFrame.from_records(records, columns=columns)
        if not df.empty:
            df["exchange_time"] = pd.to_datetime(df["exchange_time"], unit="s", utc=True)
            df["receipt_time"] = pd.to_datetime(df["receipt_time"], unit="s", utc=True)
        return df


# --------------------------------------------------------------------------- #
# Order Flow Imbalance
# --------------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class OFIReading:
    """Order-flow statistics for one window."""

    window: float
    buy_volume: float
    sell_volume: float
    delta: float
    ratio: float
    trades: int


class RollingWindow:
    """Sliding time window of signed volume with O(1) amortised updates.

    Membership rule (half-open interval): an observation stamped ``ts`` is
    inside the window at time ``now`` iff ``ts > now - window``.  Observations
    exactly ``window`` seconds old are evicted.

    Running sums are maintained incrementally.  Because floating-point
    add/subtract cycles drift over millions of operations, the sums are
    re-synchronised from the underlying entries every ``resync_every``
    insertions (``math.fsum``), and hard-reset to exactly ``0.0`` whenever the
    window becomes empty.
    """

    __slots__ = (
        "window",
        "buy_volume",
        "sell_volume",
        "buy_count",
        "sell_count",
        "_entries",
        "_ops",
        "_resync_every",
    )

    def __init__(self, window: float, *, resync_every: int = 4096) -> None:
        if not (window > 0.0) or math.isinf(window):
            raise ValueError(f"window must be a positive finite number of seconds, got {window!r}")
        if resync_every <= 0:
            raise ValueError("resync_every must be positive")
        self.window = float(window)
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.buy_count = 0
        self.sell_count = 0
        self._entries: Deque[tuple[float, float, bool]] = deque()
        self._ops = 0
        self._resync_every = resync_every

    # -- mutation ---------------------------------------------------------- #
    def add(self, ts: float, size: float, is_buy: bool) -> None:
        """Insert an observation and evict everything that is now too old."""
        self.push((ts, size, is_buy))

    def push(self, entry: tuple[float, float, bool]) -> None:
        """Insert a pre-built ``(ts, size, is_buy)`` entry (shared across windows)."""
        self._entries.append(entry)
        if entry[2]:
            self.buy_volume += entry[1]
            self.buy_count += 1
        else:
            self.sell_volume += entry[1]
            self.sell_count += 1
        self._ops += 1
        if self._ops >= self._resync_every:
            self._resync()
        self.evict(entry[0])

    def evict(self, now: float) -> int:
        """Drop observations with ``ts <= now - window``; return how many."""
        cutoff = now - self.window
        entries = self._entries
        removed = 0
        while entries and entries[0][0] <= cutoff:
            _, size, is_buy = entries.popleft()
            if is_buy:
                self.buy_volume -= size
                self.buy_count -= 1
            else:
                self.sell_volume -= size
                self.sell_count -= 1
            removed += 1
        if not entries:
            # Exact reset: eliminates any accumulated floating-point drift.
            self.buy_volume = 0.0
            self.sell_volume = 0.0
            self.buy_count = 0
            self.sell_count = 0
        return removed

    def _resync(self) -> None:
        self._ops = 0
        entries = self._entries
        self.buy_volume = math.fsum(e[1] for e in entries if e[2])
        self.sell_volume = math.fsum(e[1] for e in entries if not e[2])

    def clear(self) -> None:
        self._entries.clear()
        self.buy_volume = self.sell_volume = 0.0
        self.buy_count = self.sell_count = 0
        self._ops = 0

    # -- read -------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._entries)

    @property
    def total_volume(self) -> float:
        return self.buy_volume + self.sell_volume

    @property
    def delta(self) -> float:
        """Order flow imbalance: buy volume minus sell volume."""
        return self.buy_volume - self.sell_volume

    @property
    def ratio(self) -> float:
        """Normalised imbalance in ``[-1, 1]`` (``0.0`` for an empty window)."""
        total = self.buy_volume + self.sell_volume
        return (self.buy_volume - self.sell_volume) / total if total > 0.0 else 0.0

    def reading(self) -> OFIReading:
        return OFIReading(
            window=self.window,
            buy_volume=self.buy_volume,
            sell_volume=self.sell_volume,
            delta=self.delta,
            ratio=self.ratio,
            trades=self.buy_count + self.sell_count,
        )


class OrderFlowImbalance:
    """Rolling buy-vs-sell volume delta over several windows simultaneously."""

    __slots__ = ("_windows",)

    def __init__(self, windows: Iterable[float] = DEFAULT_WINDOWS, *, resync_every: int = 4096) -> None:
        lengths = tuple(sorted({float(w) for w in windows}))
        if not lengths:
            raise ValueError("at least one window is required")
        self._windows = tuple(RollingWindow(w, resync_every=resync_every) for w in lengths)

    @property
    def windows(self) -> tuple[float, ...]:
        return tuple(w.window for w in self._windows)

    def window(self, length: float) -> RollingWindow:
        for w in self._windows:
            if w.window == float(length):
                return w
        raise KeyError(f"no {length}s window configured (have {self.windows})")

    def update(self, ts: float, size: float, is_buy: bool) -> None:
        entry = (ts, size, is_buy)
        for w in self._windows:
            w.push(entry)

    def evict(self, now: float) -> None:
        for w in self._windows:
            w.evict(now)

    def delta(self, length: float, now: float | None = None) -> float:
        w = self.window(length)
        if now is not None:
            w.evict(now)
        return w.delta

    def ratio(self, length: float, now: float | None = None) -> float:
        w = self.window(length)
        if now is not None:
            w.evict(now)
        return w.ratio

    def snapshot(self, now: float) -> dict[float, OFIReading]:
        """Evict stale observations as of ``now`` and return every window's reading."""
        out: dict[float, OFIReading] = {}
        for w in self._windows:
            w.evict(now)
            out[w.window] = w.reading()
        return out

    def clear(self) -> None:
        for w in self._windows:
            w.clear()


# --------------------------------------------------------------------------- #
# VWAP
# --------------------------------------------------------------------------- #
class VWAP:
    """Real-time intraday volume-weighted average price.

    ``VWAP = sum(price_i * size_i) / sum(size_i)``.  With ``reset_daily=True``
    the accumulators reset when the trade timestamp crosses a UTC day boundary
    (the conventional intraday session for 24/7 crypto venues).

    Numerical note: sums are plain doubles.  For ~1e6 trades/day the relative
    accumulation error is on the order of 1e-10, far below quote precision.
    """

    __slots__ = ("_pv", "_volume", "_count", "_session_day", "reset_daily")

    def __init__(self, *, reset_daily: bool = True) -> None:
        self._pv = 0.0
        self._volume = 0.0
        self._count = 0
        self._session_day: int | None = None
        self.reset_daily = reset_daily

    def update(self, price: float, size: float, ts: float) -> float:
        """Fold one trade into the running VWAP and return the new value."""
        if self.reset_daily:
            day = int(ts // SECONDS_PER_DAY)
            if day != self._session_day:
                if self._session_day is not None:
                    self.reset()
                self._session_day = day
        if size > 0.0:
            self._pv += price * size
            self._volume += size
            self._count += 1
        return self.value

    def reset(self) -> None:
        self._pv = 0.0
        self._volume = 0.0
        self._count = 0
        self._session_day = None

    @property
    def value(self) -> float:
        """Current VWAP, or ``nan`` before any volume has traded."""
        return self._pv / self._volume if self._volume > 0.0 else _NAN

    @property
    def volume(self) -> float:
        return self._volume

    @property
    def notional(self) -> float:
        return self._pv

    @property
    def count(self) -> int:
        return self._count

    @property
    def session_day(self) -> int | None:
        return self._session_day

    def deviation_bps(self, price: float) -> float:
        """Distance of ``price`` from VWAP in basis points (positive = above)."""
        v = self.value
        return (price - v) / v * 10_000.0 if v > 0.0 else _NAN


# --------------------------------------------------------------------------- #
# Anomaly detection
# --------------------------------------------------------------------------- #
class LargeTradeDetector:
    """Flags trades whose size is strictly greater than ``threshold`` base units."""

    __slots__ = ("threshold",)

    def __init__(self, threshold: float = DEFAULT_BLOCK_THRESHOLD) -> None:
        if not (threshold >= 0.0):
            raise ValueError("threshold must be non-negative")
        self.threshold = float(threshold)

    def check(self, trade: Trade) -> Alert | None:
        if trade.size <= self.threshold:
            return None
        base = base_currency(trade.product_id)
        return Alert(
            kind=ALERT_LARGE_TRADE,
            product_id=trade.product_id,
            timestamp=trade.exchange_time,
            message=(
                f"{trade.side.upper()} {trade.size:,.4f} {base} @ {trade.price:,.2f} "
                f"(${trade.notional:,.0f}) exceeds {self.threshold:g} {base} block threshold"
            ),
            details={
                "size": trade.size,
                "price": trade.price,
                "notional": trade.notional,
                "threshold": self.threshold,
                "side": trade.side,
                "trade_id": trade.trade_id,
            },
        )


class SpreadMonitor:
    """Detects sudden bid-ask spread widening against a time-decayed baseline.

    The baseline is an exponentially weighted moving average of the spread in
    basis points whose weight decays with *elapsed time* rather than message
    count, so it is robust to bursts and lulls in quote frequency::

        decay    = 0.5 ** (dt / half_life)
        baseline = decay * baseline + (1 - decay) * spread_bps

    A quote is flagged when *all* hold:

    * at least ``min_observations`` quotes have been seen (warm-up),
    * ``spread_bps >= min_spread_bps`` (absolute floor, filters noise on
      ultra-tight books),
    * ``spread_bps >= widen_factor * baseline`` (relative jump),
    * at least ``cooldown`` seconds have elapsed since the previous alert.
    """

    __slots__ = (
        "half_life",
        "widen_factor",
        "min_spread_bps",
        "min_observations",
        "cooldown",
        "baseline_bps",
        "last_spread_bps",
        "last_quote",
        "observations",
        "is_wide",
        "alerts",
        "_last_alert_at",
        "_last_ts",
    )

    def __init__(
        self,
        *,
        half_life: float = 30.0,
        widen_factor: float = 3.0,
        min_spread_bps: float = 1.0,
        min_observations: int = 10,
        cooldown: float = 10.0,
    ) -> None:
        if half_life <= 0.0:
            raise ValueError("half_life must be positive")
        if widen_factor <= 1.0:
            raise ValueError("widen_factor must be greater than 1")
        if min_spread_bps < 0.0 or cooldown < 0.0 or min_observations < 1:
            raise ValueError("invalid SpreadMonitor configuration")
        self.half_life = float(half_life)
        self.widen_factor = float(widen_factor)
        self.min_spread_bps = float(min_spread_bps)
        self.min_observations = int(min_observations)
        self.cooldown = float(cooldown)
        self.baseline_bps = _NAN
        self.last_spread_bps = _NAN
        self.last_quote: Quote | None = None
        self.observations = 0
        self.is_wide = False
        self.alerts = 0
        self._last_alert_at: float | None = None
        self._last_ts = 0.0

    def update(self, quote: Quote) -> Alert | None:
        spread_bps = quote.spread_bps
        self.last_quote = quote
        self.last_spread_bps = spread_bps
        if not math.isfinite(spread_bps) or spread_bps < 0.0:
            # Crossed / locked / malformed book: do not pollute the baseline.
            self.is_wide = False
            return None

        now = quote.receipt_time
        alert: Alert | None = None
        baseline = self.baseline_bps

        if self.observations >= self.min_observations:
            if baseline > 0.0:
                ratio = spread_bps / baseline
            else:
                ratio = math.inf if spread_bps > 0.0 else 0.0
            self.is_wide = spread_bps >= self.min_spread_bps and ratio >= self.widen_factor
            if self.is_wide and (
                self._last_alert_at is None or now - self._last_alert_at >= self.cooldown
            ):
                self._last_alert_at = now
                self.alerts += 1
                spread = quote.spread
                alert = Alert(
                    kind=ALERT_SPREAD_WIDENING,
                    product_id=quote.product_id,
                    timestamp=quote.exchange_time,
                    message=(
                        f"spread {spread:,.2f} ({spread_bps:.2f} bps) is "
                        f"{ratio:.1f}x the {baseline:.3f} bps baseline "
                        f"[bid {quote.bid:,.2f} / ask {quote.ask:,.2f}]"
                    ),
                    details={
                        "spread": spread,
                        "spread_bps": spread_bps,
                        "baseline_bps": baseline,
                        "ratio": ratio,
                        "bid": quote.bid,
                        "ask": quote.ask,
                    },
                )
        else:
            self.is_wide = False

        # Time-decayed EWMA baseline update.
        if self.observations == 0:
            self.baseline_bps = spread_bps
        else:
            dt = now - self._last_ts
            decay = 0.5 ** (dt / self.half_life) if dt > 0.0 else 1.0
            self.baseline_bps = decay * baseline + (1.0 - decay) * spread_bps
        self._last_ts = now
        self.observations += 1
        return alert


# --------------------------------------------------------------------------- #
# Telemetry primitives
# --------------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class LatencySnapshot:
    count: int
    window_size: int
    mean_ms: float
    lifetime_mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    last_ms: float

    @classmethod
    def empty(cls) -> "LatencySnapshot":
        return cls(0, 0, _NAN, _NAN, _NAN, _NAN, _NAN, _NAN, _NAN, _NAN)


class LatencyTracker:
    """Ingestion latency statistics over a bounded sample window.

    Keeps the most recent ``window`` samples for percentile statistics and a
    Welford running mean over the whole lifetime of the process.
    """

    __slots__ = ("_samples", "count", "lifetime_mean_ms", "last_ms")

    def __init__(self, window: int = 5_000) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self._samples: Deque[float] = deque(maxlen=window)
        self.count = 0
        self.lifetime_mean_ms = 0.0
        self.last_ms = _NAN

    def record(self, latency_ms: float) -> None:
        self._samples.append(latency_ms)
        self.count += 1
        self.lifetime_mean_ms += (latency_ms - self.lifetime_mean_ms) / self.count
        self.last_ms = latency_ms

    def __len__(self) -> int:
        return len(self._samples)

    def snapshot(self) -> LatencySnapshot:
        n = len(self._samples)
        if n == 0:
            return LatencySnapshot.empty()
        arr = np.fromiter(self._samples, dtype=np.float64, count=n)
        p50, p95, p99 = np.percentile(arr, (50.0, 95.0, 99.0))
        return LatencySnapshot(
            count=self.count,
            window_size=n,
            mean_ms=float(arr.mean()),
            lifetime_mean_ms=self.lifetime_mean_ms,
            p50_ms=float(p50),
            p95_ms=float(p95),
            p99_ms=float(p99),
            min_ms=float(arr.min()),
            max_ms=float(arr.max()),
            last_ms=self.last_ms,
        )


class RateMeter:
    """Sliding-window event rate (events per second).

    Timestamps are kept for the longest window; shorter windows are answered
    by scanning back from the newest event.
    """

    __slots__ = ("window", "_ts", "total")

    def __init__(self, window: float = 10.0) -> None:
        if window <= 0.0:
            raise ValueError("window must be positive")
        self.window = float(window)
        self._ts: Deque[float] = deque()
        self.total = 0

    def tick(self, now: float) -> None:
        self._ts.append(now)
        self.total += 1
        self._evict(now)

    def _evict(self, now: float) -> None:
        cutoff = now - self.window
        ts = self._ts
        while ts and ts[0] <= cutoff:
            ts.popleft()

    def rate(self, now: float, window: float | None = None) -> float:
        """Events per second over the trailing ``window`` seconds (default: full window)."""
        self._evict(now)
        if window is None or window >= self.window:
            return len(self._ts) / self.window
        if window <= 0.0:
            raise ValueError("window must be positive")
        cutoff = now - window
        n = 0
        for t in reversed(self._ts):
            if t <= cutoff:
                break
            n += 1
        return n / window


# --------------------------------------------------------------------------- #
# Per-product and aggregate state
# --------------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class ProductSnapshot:
    product_id: str
    last_price: float
    tick_direction: int
    last_trade_time: float
    vwap: float
    vwap_deviation_bps: float
    ofi: Mapping[float, OFIReading]
    bid: float
    ask: float
    spread: float
    spread_bps: float
    spread_baseline_bps: float
    spread_is_wide: bool
    quote_time: float
    total_volume: float
    total_notional: float
    buy_volume: float
    sell_volume: float
    trade_count: int
    realized_vol_bps: float
    buffer_len: int
    buffer_capacity: int
    block_threshold: float


class ProductAnalytics:
    """All per-instrument state: ring buffer, VWAP, OFI windows, spread monitor."""

    __slots__ = (
        "product_id",
        "buffer",
        "vwap",
        "ofi",
        "spread",
        "block_detector",
        "last_trade",
        "last_quote",
        "tick_direction",
        "total_volume",
        "total_notional",
        "buy_volume",
        "sell_volume",
        "trade_count",
        "quote_count",
        "vol_lookback",
    )

    def __init__(
        self,
        product_id: str,
        *,
        windows: Iterable[float] = DEFAULT_WINDOWS,
        buffer_capacity: int = DEFAULT_BUFFER_CAPACITY,
        block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
        spread_monitor: SpreadMonitor | None = None,
        vol_lookback: int = 200,
    ) -> None:
        self.product_id = product_id
        self.buffer = TickBuffer(buffer_capacity)
        self.vwap = VWAP()
        self.ofi = OrderFlowImbalance(windows)
        self.spread = spread_monitor if spread_monitor is not None else SpreadMonitor()
        self.block_detector = LargeTradeDetector(block_threshold)
        self.last_trade: Trade | None = None
        self.last_quote: Quote | None = None
        self.tick_direction = 0
        self.total_volume = 0.0
        self.total_notional = 0.0
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.trade_count = 0
        self.quote_count = 0
        self.vol_lookback = vol_lookback

    def on_trade(self, trade: Trade) -> Alert | None:
        """Fold a trade into every metric. O(1) amortised."""
        prev = self.last_trade
        if prev is not None:
            if trade.price > prev.price:
                self.tick_direction = 1
            elif trade.price < prev.price:
                self.tick_direction = -1
        self.last_trade = trade
        self.buffer.append(trade)
        self.vwap.update(trade.price, trade.size, trade.exchange_time)
        # Windows are keyed on the client receipt clock so they keep decaying
        # during feed stalls and are immune to exchange clock skew.
        self.ofi.update(trade.receipt_time, trade.size, trade.is_buy)
        self.total_volume += trade.size
        self.total_notional += trade.notional
        if trade.is_buy:
            self.buy_volume += trade.size
        else:
            self.sell_volume += trade.size
        self.trade_count += 1
        return self.block_detector.check(trade)

    def on_quote(self, quote: Quote) -> Alert | None:
        self.last_quote = quote
        self.quote_count += 1
        return self.spread.update(quote)

    def realized_volatility_bps(self, lookback: int | None = None, horizon_seconds: float = 60.0) -> float:
        """Tick-return realized volatility scaled to ``horizon_seconds`` (in bps).

        ``sigma_h = std(diff(log p)) * sqrt(ticks_per_horizon) * 1e4`` using the
        newest ``lookback`` ticks of the ring buffer.  ``nan`` with < 3 ticks.
        """
        trades = self.buffer.recent(lookback or self.vol_lookback)
        n = len(trades)
        if n < 3:
            return _NAN
        span = trades[-1].exchange_time - trades[0].exchange_time
        if span <= 0.0:
            return _NAN
        prices = np.fromiter((t.price for t in trades), dtype=np.float64, count=n)
        if not np.all(prices > 0.0):
            return _NAN
        log_returns = np.diff(np.log(prices))
        sigma_tick = float(log_returns.std(ddof=1))
        ticks_per_horizon = (n - 1) / span * horizon_seconds
        return sigma_tick * math.sqrt(ticks_per_horizon) * 10_000.0

    def snapshot(self, now: float) -> ProductSnapshot:
        last = self.last_trade
        quote = self.last_quote
        spread_seen = self.spread.observations > 0
        return ProductSnapshot(
            product_id=self.product_id,
            last_price=last.price if last else _NAN,
            tick_direction=self.tick_direction,
            last_trade_time=last.exchange_time if last else _NAN,
            vwap=self.vwap.value,
            vwap_deviation_bps=self.vwap.deviation_bps(last.price) if last else _NAN,
            ofi=self.ofi.snapshot(now),
            bid=quote.bid if quote else _NAN,
            ask=quote.ask if quote else _NAN,
            spread=quote.spread if quote else _NAN,
            spread_bps=quote.spread_bps if quote else _NAN,
            spread_baseline_bps=self.spread.baseline_bps if spread_seen else _NAN,
            spread_is_wide=self.spread.is_wide,
            quote_time=quote.exchange_time if quote else _NAN,
            total_volume=self.total_volume,
            total_notional=self.total_notional,
            buy_volume=self.buy_volume,
            sell_volume=self.sell_volume,
            trade_count=self.trade_count,
            realized_vol_bps=self.realized_volatility_bps(),
            buffer_len=len(self.buffer),
            buffer_capacity=self.buffer.capacity,
            block_threshold=self.block_detector.threshold,
        )


@dataclass(slots=True, frozen=True)
class MarketSnapshot:
    """Immutable, render-ready view of the whole analytics state."""

    timestamp: float
    uptime: float
    products: Mapping[str, ProductSnapshot]
    latency: LatencySnapshot
    ticks_per_sec: float
    ticks_per_sec_avg: float
    trades_total: int
    quotes_total: int
    volume_by_product: Mapping[str, float]
    notional_total: float
    block_trades: Sequence[Trade]
    block_trades_total: int
    alerts: Sequence[Alert]
    alerts_total: int


class MarketAnalytics:
    """Aggregate analytics across products plus process-wide telemetry.

    The ingestion layer calls ``on_trade`` / ``on_quote``; the dashboard calls
    ``snapshot`` on its own timer.  Both run on the same event loop, so no
    locking is required.
    """

    def __init__(
        self,
        products: Iterable[str],
        *,
        windows: Iterable[float] = DEFAULT_WINDOWS,
        buffer_capacity: int = DEFAULT_BUFFER_CAPACITY,
        block_thresholds: Mapping[str, float] | float = DEFAULT_BLOCK_THRESHOLD,
        default_block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
        spread_config: Mapping[str, Any] | None = None,
        latency_window: int = 5_000,
        rate_window: float = 10.0,
        block_history: int = 100,
        alert_history: int = 200,
        clock=time.time,
    ) -> None:
        self._windows = tuple(windows)
        self._buffer_capacity = buffer_capacity
        self._spread_config = dict(spread_config or {})
        if isinstance(block_thresholds, Mapping):
            self._block_thresholds = dict(block_thresholds)
            self._default_block_threshold = default_block_threshold
        else:
            self._block_thresholds = {}
            self._default_block_threshold = float(block_thresholds)
        self._clock = clock
        self.started_at = clock()
        self.products: dict[str, ProductAnalytics] = {}
        for pid in products:
            self.register(pid)
        self.latency = LatencyTracker(latency_window)
        self.tick_rate = RateMeter(rate_window)
        self.block_trades: Deque[Trade] = deque(maxlen=block_history)
        self.alerts: Deque[Alert] = deque(maxlen=alert_history)
        self.trades_total = 0
        self.quotes_total = 0
        self.block_trades_total = 0
        self.alerts_total = 0

    # -- configuration ------------------------------------------------------ #
    def register(self, product_id: str) -> ProductAnalytics:
        """Create (or return the existing) per-product state."""
        pa = self.products.get(product_id)
        if pa is None:
            pa = ProductAnalytics(
                product_id,
                windows=self._windows,
                buffer_capacity=self._buffer_capacity,
                block_threshold=self._block_thresholds.get(product_id, self._default_block_threshold),
                spread_monitor=SpreadMonitor(**self._spread_config),
            )
            self.products[product_id] = pa
        return pa

    @property
    def windows(self) -> tuple[float, ...]:
        return self._windows

    # -- hot path ----------------------------------------------------------- #
    def on_trade(self, trade: Trade) -> Alert | None:
        pa = self.products.get(trade.product_id)
        if pa is None:
            log.info("Auto-registering unexpected product %s", trade.product_id)
            pa = self.register(trade.product_id)
        alert = pa.on_trade(trade)
        self.latency.record(trade.latency_ms)
        self.tick_rate.tick(trade.receipt_time)
        self.trades_total += 1
        if alert is not None:
            self.block_trades.append(trade)
            self.block_trades_total += 1
            self._record_alert(alert)
        return alert

    def on_quote(self, quote: Quote) -> Alert | None:
        pa = self.products.get(quote.product_id)
        if pa is None:
            pa = self.register(quote.product_id)
        alert = pa.on_quote(quote)
        self.quotes_total += 1
        if alert is not None:
            self._record_alert(alert)
        return alert

    def _record_alert(self, alert: Alert) -> None:
        self.alerts.append(alert)
        self.alerts_total += 1

    # -- read --------------------------------------------------------------- #
    def snapshot(self, now: float | None = None) -> MarketSnapshot:
        if now is None:
            now = self._clock()
        return MarketSnapshot(
            timestamp=now,
            uptime=max(0.0, now - self.started_at),
            products={pid: pa.snapshot(now) for pid, pa in self.products.items()},
            latency=self.latency.snapshot(),
            ticks_per_sec=self.tick_rate.rate(now, 1.0),
            ticks_per_sec_avg=self.tick_rate.rate(now),
            trades_total=self.trades_total,
            quotes_total=self.quotes_total,
            volume_by_product={pid: pa.total_volume for pid, pa in self.products.items()},
            notional_total=sum(pa.total_notional for pa in self.products.values()),
            block_trades=tuple(self.block_trades),
            block_trades_total=self.block_trades_total,
            alerts=tuple(self.alerts),
            alerts_total=self.alerts_total,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Concatenate every product's ring buffer into one DataFrame."""
        frames = [pa.buffer.to_dataframe() for pa in self.products.values()]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return TickBuffer(1).to_dataframe()
        df = pd.concat(frames, ignore_index=True)
        return df.sort_values(["exchange_time", "product_id"], kind="stable").reset_index(drop=True)
