"""Unit tests for the quantitative analytics core (OFI, VWAP, spread, latency, buffers)."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from src.analytics import (
    ALERT_LARGE_TRADE,
    ALERT_SPREAD_WIDENING,
    BUY,
    SELL,
    VWAP,
    LargeTradeDetector,
    LatencyTracker,
    MarketAnalytics,
    OrderFlowImbalance,
    Quote,
    RateMeter,
    RollingWindow,
    SpreadMonitor,
    TickBuffer,
    Trade,
    format_utc,
)

T0 = 1_760_000_000.0  # 2025-10-09T08:53:20Z - an arbitrary fixed epoch anchor
_ids = iter(range(1, 10_000_000))


def make_trade(
    price: float,
    size: float,
    side: str = BUY,
    t: float = 0.0,
    *,
    product: str = "BTC-USD",
    latency: float = 0.050,
) -> Trade:
    """Build a trade ``t`` seconds after ``T0`` with the given ingestion latency."""
    exchange_time = T0 + t
    return Trade(
        product_id=product,
        trade_id=next(_ids),
        price=price,
        size=size,
        side=side,
        exchange_time=exchange_time,
        receipt_time=exchange_time + latency,
    )


def make_quote(bid: float, ask: float, t: float = 0.0, *, product: str = "BTC-USD") -> Quote:
    return Quote(product_id=product, bid=bid, ask=ask, exchange_time=T0 + t, receipt_time=T0 + t)


# --------------------------------------------------------------------------- #
# Trade / Quote model
# --------------------------------------------------------------------------- #
class TestModel:
    def test_trade_derived_fields(self):
        t = make_trade(100.0, 2.5, BUY, latency=0.0425)
        assert t.notional == pytest.approx(250.0)
        assert t.is_buy is True
        assert t.latency_ms == pytest.approx(42.5)

    def test_quote_spread_metrics(self):
        q = make_quote(99.0, 101.0)
        assert q.spread == pytest.approx(2.0)
        assert q.mid == pytest.approx(100.0)
        assert q.spread_bps == pytest.approx(200.0)

    def test_format_utc(self):
        assert format_utc(0.0) == "1970-01-01T00:00:00.000Z"
        assert format_utc(1_700_000_000.1234) == "2023-11-14T22:13:20.123Z"
        assert format_utc(1_700_000_000.1234, date=False) == "22:13:20.123"
        assert format_utc(math.nan) == "-"


# --------------------------------------------------------------------------- #
# VWAP
# --------------------------------------------------------------------------- #
class TestVWAP:
    def test_empty_is_nan(self):
        assert math.isnan(VWAP().value)
        assert math.isnan(VWAP().deviation_bps(100.0))

    def test_single_trade_equals_price(self):
        v = VWAP()
        assert v.update(101.5, 0.3, T0) == pytest.approx(101.5)

    def test_matches_closed_form(self):
        # sum(p*q) = 100*1 + 110*3 + 90*2 = 610 ; sum(q) = 6 ; VWAP = 101.666...
        v = VWAP()
        for price, size in [(100.0, 1.0), (110.0, 3.0), (90.0, 2.0)]:
            v.update(price, size, T0)
        assert v.value == pytest.approx(610.0 / 6.0)
        assert v.volume == pytest.approx(6.0)
        assert v.notional == pytest.approx(610.0)
        assert v.count == 3

    def test_is_volume_weighted_not_simple_mean(self):
        v = VWAP()
        v.update(100.0, 9.0, T0)
        v.update(200.0, 1.0, T0)
        assert v.value == pytest.approx(110.0)  # simple mean would be 150

    def test_matches_numpy_on_random_data(self):
        rng = np.random.default_rng(42)
        prices = rng.uniform(50_000, 70_000, size=5_000)
        sizes = rng.exponential(0.05, size=5_000)
        v = VWAP()
        for p, s in zip(prices, sizes):
            v.update(float(p), float(s), T0)
        expected = float(np.sum(prices * sizes) / np.sum(sizes))
        assert v.value == pytest.approx(expected, rel=1e-12)

    def test_zero_size_is_ignored(self):
        v = VWAP()
        v.update(100.0, 0.0, T0)
        assert math.isnan(v.value)
        v.update(100.0, 1.0, T0)
        v.update(999.0, 0.0, T0)
        assert v.value == pytest.approx(100.0)

    def test_resets_at_utc_day_boundary(self):
        day_start = (T0 // 86_400) * 86_400
        v = VWAP()
        v.update(100.0, 1.0, day_start + 10)
        v.update(120.0, 1.0, day_start + 86_399)
        assert v.value == pytest.approx(110.0)
        v.update(300.0, 2.0, day_start + 86_400)  # first tick of the next session
        assert v.value == pytest.approx(300.0)
        assert v.count == 1

    def test_no_reset_when_disabled(self):
        day_start = (T0 // 86_400) * 86_400
        v = VWAP(reset_daily=False)
        v.update(100.0, 1.0, day_start + 10)
        v.update(300.0, 1.0, day_start + 86_400)
        assert v.value == pytest.approx(200.0)

    def test_manual_reset(self):
        v = VWAP()
        v.update(100.0, 1.0, T0)
        v.reset()
        assert math.isnan(v.value) and v.volume == 0.0 and v.count == 0

    def test_deviation_bps(self):
        v = VWAP()
        v.update(100.0, 1.0, T0)
        assert v.deviation_bps(101.0) == pytest.approx(100.0)
        assert v.deviation_bps(99.0) == pytest.approx(-100.0)


# --------------------------------------------------------------------------- #
# Rolling window / OFI
# --------------------------------------------------------------------------- #
class TestRollingWindow:
    def test_rejects_bad_windows(self):
        for bad in (0, -1.0, math.inf, math.nan):
            with pytest.raises(ValueError):
                RollingWindow(bad)

    def test_running_sums(self):
        w = RollingWindow(10.0)
        w.add(0.0, 2.0, True)
        w.add(1.0, 0.5, False)
        w.add(2.0, 1.0, True)
        assert w.buy_volume == pytest.approx(3.0)
        assert w.sell_volume == pytest.approx(0.5)
        assert w.delta == pytest.approx(2.5)
        assert w.ratio == pytest.approx(2.5 / 3.5)
        assert w.buy_count == 2 and w.sell_count == 1 and len(w) == 3

    def test_half_open_eviction_boundary(self):
        w = RollingWindow(10.0)
        w.add(0.0, 1.0, True)
        assert w.evict(9.999) == 0 and w.delta == pytest.approx(1.0)
        assert w.evict(10.0) == 1  # exactly `window` seconds old -> evicted
        assert w.delta == 0.0 and len(w) == 0

    def test_add_evicts_relative_to_newest_timestamp(self):
        w = RollingWindow(10.0)
        w.add(0.0, 1.0, True)
        w.add(5.0, 1.0, False)
        w.add(12.0, 0.25, True)  # evicts the t=0 buy (12 - 0 >= 10)
        assert len(w) == 2
        assert w.delta == pytest.approx(0.25 - 1.0)

    def test_empty_window_resets_to_exact_zero(self):
        w = RollingWindow(1.0)
        for i in range(1000):
            w.add(i * 0.001, 0.1, i % 2 == 0)
        w.evict(1_000.0)
        assert w.buy_volume == 0.0 and w.sell_volume == 0.0  # exact, not approx
        assert w.buy_count == 0 and w.sell_count == 0

    def test_running_sums_track_brute_force_over_long_run(self):
        rng = random.Random(7)
        w = RollingWindow(5.0, resync_every=257)
        entries: list[tuple[float, float, bool]] = []
        t = 0.0
        for _ in range(20_000):
            t += rng.expovariate(50.0)
            size = rng.lognormvariate(-3.0, 1.2)
            is_buy = rng.random() < 0.55
            w.add(t, size, is_buy)
            entries.append((t, size, is_buy))
        live = [e for e in entries if e[0] > t - 5.0]
        assert w.buy_volume == pytest.approx(math.fsum(e[1] for e in live if e[2]), rel=1e-9)
        assert w.sell_volume == pytest.approx(math.fsum(e[1] for e in live if not e[2]), rel=1e-9)
        assert len(w) == len(live)

    def test_ratio_is_bounded_and_zero_when_empty(self):
        w = RollingWindow(10.0)
        assert w.ratio == 0.0
        w.add(0.0, 3.0, True)
        assert w.ratio == pytest.approx(1.0)
        w.add(0.0, 3.0, False)
        assert w.ratio == pytest.approx(0.0)
        w.add(0.0, 6.0, False)
        assert w.ratio == pytest.approx(-0.5)


class TestOrderFlowImbalance:
    def test_windows_sorted_and_deduplicated(self):
        ofi = OrderFlowImbalance([60, 10, 30, 10])
        assert ofi.windows == (10.0, 30.0, 60.0)
        with pytest.raises(ValueError):
            OrderFlowImbalance([])
        with pytest.raises(KeyError):
            ofi.window(15)

    def test_delta_formula_across_windows(self):
        """OFI_w = sum(buy sizes) - sum(sell sizes) over the trailing w seconds."""
        ofi = OrderFlowImbalance((10, 30, 60))
        ofi.update(0.0, 2.0, True)  # buy  2.0 @ t=0
        ofi.update(5.0, 0.5, False)  # sell 0.5 @ t=5
        ofi.update(25.0, 1.0, True)  # buy  1.0 @ t=25
        snap = ofi.snapshot(now=30.0)
        # 10s window: (20, 30]  -> only the t=25 buy
        assert snap[10.0].delta == pytest.approx(1.0)
        assert snap[10.0].trades == 1
        # 30s window: (0, 30]   -> t=5 sell and t=25 buy (t=0 is exactly 30s old -> evicted)
        assert snap[30.0].delta == pytest.approx(0.5)
        assert snap[30.0].buy_volume == pytest.approx(1.0)
        assert snap[30.0].sell_volume == pytest.approx(0.5)
        assert snap[30.0].ratio == pytest.approx(0.5 / 1.5)
        # 60s window: (-30, 30] -> everything
        assert snap[60.0].delta == pytest.approx(2.5)
        assert snap[60.0].trades == 3

    def test_nested_windows_are_monotonic(self):
        rng = random.Random(3)
        ofi = OrderFlowImbalance((10, 30, 60))
        t = 0.0
        for _ in range(5_000):
            t += rng.expovariate(20.0)
            ofi.update(t, rng.lognormvariate(-3, 1), rng.random() < 0.5)
        snap = ofi.snapshot(t)
        assert snap[10.0].buy_volume <= snap[30.0].buy_volume <= snap[60.0].buy_volume
        assert snap[10.0].sell_volume <= snap[30.0].sell_volume <= snap[60.0].sell_volume
        assert snap[10.0].trades <= snap[30.0].trades <= snap[60.0].trades

    def test_delta_matches_brute_force(self):
        rng = random.Random(11)
        ofi = OrderFlowImbalance((10, 30, 60))
        events: list[tuple[float, float, bool]] = []
        t = 0.0
        for _ in range(8_000):
            t += rng.expovariate(30.0)
            size = rng.lognormvariate(-4, 1.5)
            is_buy = rng.random() < 0.5
            ofi.update(t, size, is_buy)
            events.append((t, size, is_buy))
        now = t + 2.0
        snap = ofi.snapshot(now)
        for w in (10.0, 30.0, 60.0):
            live = [e for e in events if e[0] > now - w]
            expected = math.fsum(s if b else -s for _, s, b in live)
            assert snap[w].delta == pytest.approx(expected, abs=1e-9)
            assert snap[w].trades == len(live)

    def test_delta_and_ratio_helpers_evict(self):
        ofi = OrderFlowImbalance((10,))
        ofi.update(0.0, 1.0, True)
        assert ofi.delta(10, now=5.0) == pytest.approx(1.0)
        assert ofi.delta(10, now=10.0) == 0.0
        assert ofi.ratio(10) == 0.0

    def test_stale_flow_decays_to_zero(self):
        ofi = OrderFlowImbalance((10, 30, 60))
        ofi.update(0.0, 5.0, True)
        snap = ofi.snapshot(now=61.0)
        assert all(r.delta == 0.0 and r.trades == 0 for r in snap.values())


# --------------------------------------------------------------------------- #
# Anomaly detection
# --------------------------------------------------------------------------- #
class TestLargeTradeDetector:
    def test_strictly_greater_than_threshold(self):
        det = LargeTradeDetector(1.5)
        assert det.check(make_trade(100.0, 1.5)) is None
        assert det.check(make_trade(100.0, 1.4999)) is None
        alert = det.check(make_trade(100.0, 1.5001, SELL))
        assert alert is not None
        assert alert.kind == ALERT_LARGE_TRADE
        assert alert.product_id == "BTC-USD"
        assert alert.timestamp == pytest.approx(T0)
        assert alert.details["size"] == pytest.approx(1.5001)
        assert alert.details["side"] == SELL
        assert "SELL" in alert.message and "BTC" in alert.message
        assert alert.console_line().startswith(format_utc(T0))

    def test_rejects_negative_threshold(self):
        with pytest.raises(ValueError):
            LargeTradeDetector(-1.0)


class TestSpreadMonitor:
    @staticmethod
    def _warm(monitor: SpreadMonitor, n: int = 20, spread: float = 1.0, mid: float = 50_000.0, dt: float = 0.5):
        for i in range(n):
            assert monitor.update(make_quote(mid - spread / 2, mid + spread / 2, t=i * dt)) is None
        return n * dt

    def test_no_alert_during_warmup(self):
        m = SpreadMonitor(min_observations=10)
        for i in range(9):
            assert m.update(make_quote(100.0, 100.01 if i % 2 else 110.0, t=i)) is None

    def test_steady_spread_is_not_flagged(self):
        m = SpreadMonitor()
        t = self._warm(m, n=50)
        assert m.update(make_quote(49_999.5, 50_000.5, t=t)) is None
        assert m.is_wide is False
        assert m.baseline_bps == pytest.approx(0.2, rel=1e-6)

    def test_sudden_widening_is_flagged_with_details(self):
        m = SpreadMonitor(widen_factor=3.0, min_spread_bps=1.0)
        t = self._warm(m, n=50, spread=1.0)  # baseline 0.2 bps
        alert = m.update(make_quote(49_995.0, 50_005.0, t=t))  # 10.0 spread = 2.0 bps = 10x baseline
        assert alert is not None
        assert alert.kind == ALERT_SPREAD_WIDENING
        assert alert.details["spread"] == pytest.approx(10.0)
        assert alert.details["spread_bps"] == pytest.approx(2.0)
        assert alert.details["baseline_bps"] == pytest.approx(0.2, rel=1e-6)
        assert alert.details["ratio"] == pytest.approx(10.0, rel=1e-6)
        assert m.is_wide is True

    def test_absolute_floor_suppresses_tiny_spreads(self):
        m = SpreadMonitor(widen_factor=3.0, min_spread_bps=1.0)
        t = self._warm(m, n=50, spread=0.01)  # 0.002 bps baseline
        # 20x wider but still only 0.04 bps -> below the 1 bps floor.
        assert m.update(make_quote(49_999.9, 50_000.1, t=t)) is None
        assert m.is_wide is False

    def test_cooldown_limits_alert_rate(self):
        m = SpreadMonitor(cooldown=10.0)
        t = self._warm(m, n=50)
        assert m.update(make_quote(49_995.0, 50_005.0, t=t)) is not None
        assert m.update(make_quote(49_995.0, 50_005.0, t=t + 1.0)) is None  # still wide, in cooldown
        assert m.is_wide is True
        assert m.update(make_quote(49_995.0, 50_005.0, t=t + 10.0)) is not None
        assert m.alerts == 2

    def test_baseline_decays_by_half_life(self):
        m = SpreadMonitor(half_life=30.0, min_observations=1)
        m.update(make_quote(99.0, 101.0, t=0.0))  # 200 bps
        assert m.baseline_bps == pytest.approx(200.0)
        m.update(make_quote(99.5, 100.5, t=30.0))  # 100 bps one half-life later
        assert m.baseline_bps == pytest.approx(150.0)
        m.update(make_quote(99.5, 100.5, t=60.0))
        assert m.baseline_bps == pytest.approx(125.0)

    def test_crossed_book_does_not_pollute_baseline(self):
        m = SpreadMonitor(min_observations=1)
        m.update(make_quote(99.0, 101.0, t=0.0))
        assert m.update(make_quote(101.0, 99.0, t=1.0)) is None
        assert m.baseline_bps == pytest.approx(200.0)
        assert m.observations == 1

    def test_configuration_validation(self):
        with pytest.raises(ValueError):
            SpreadMonitor(widen_factor=1.0)
        with pytest.raises(ValueError):
            SpreadMonitor(half_life=0.0)


# --------------------------------------------------------------------------- #
# Telemetry primitives
# --------------------------------------------------------------------------- #
class TestLatencyTracker:
    def test_empty_snapshot(self):
        snap = LatencyTracker().snapshot()
        assert snap.count == 0 and math.isnan(snap.mean_ms) and math.isnan(snap.p99_ms)

    def test_statistics(self):
        tracker = LatencyTracker(window=1_000)
        samples = list(range(1, 101))  # 1..100 ms
        for s in samples:
            tracker.record(float(s))
        snap = tracker.snapshot()
        assert snap.count == 100 and snap.window_size == 100
        assert snap.mean_ms == pytest.approx(50.5)
        assert snap.lifetime_mean_ms == pytest.approx(50.5)
        assert snap.p50_ms == pytest.approx(np.percentile(samples, 50))
        assert snap.p95_ms == pytest.approx(np.percentile(samples, 95))
        assert snap.p99_ms == pytest.approx(np.percentile(samples, 99))
        assert snap.min_ms == 1.0 and snap.max_ms == 100.0 and snap.last_ms == 100.0

    def test_window_bounds_percentiles_but_not_lifetime_mean(self):
        tracker = LatencyTracker(window=10)
        for s in [1_000.0] * 10 + [10.0] * 10:
            tracker.record(s)
        snap = tracker.snapshot()
        assert snap.window_size == 10 and snap.mean_ms == pytest.approx(10.0)
        assert snap.lifetime_mean_ms == pytest.approx(505.0)
        assert snap.count == 20


class TestRateMeter:
    def test_rates_over_nested_windows(self):
        meter = RateMeter(window=10.0)
        for i in range(100):  # 100 events over 10 s -> 10/s ; last second has 10 events
            meter.tick(i * 0.1)
        now = 9.95
        assert meter.rate(now) == pytest.approx(10.0)
        assert meter.rate(now, window=1.0) == pytest.approx(10.0)
        assert meter.rate(now + 100.0) == 0.0
        assert meter.total == 100

    def test_rejects_non_positive_window(self):
        with pytest.raises(ValueError):
            RateMeter(0.0)


class TestTickBuffer:
    def test_ring_semantics(self):
        buf = TickBuffer(capacity=3)
        for i in range(5):
            buf.append(make_trade(100.0 + i, 1.0))
        assert len(buf) == 3 and buf.capacity == 3
        assert buf.total_appended == 5 and buf.evicted == 2
        assert [t.price for t in buf] == [102.0, 103.0, 104.0]
        assert buf.latest is not None and buf.latest.price == 104.0
        assert buf.utilisation == pytest.approx(1.0)

    def test_recent_preserves_chronology(self):
        buf = TickBuffer(capacity=10)
        for i in range(6):
            buf.append(make_trade(float(i), 1.0))
        assert [t.price for t in buf.recent(3)] == [3.0, 4.0, 5.0]
        assert [t.price for t in buf.recent(100)] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        assert buf.recent(0) == []
        np.testing.assert_allclose(buf.prices(2), [4.0, 5.0])

    def test_dataframe_export(self):
        buf = TickBuffer(capacity=10)
        buf.append(make_trade(100.0, 0.5, BUY, t=0.0))
        buf.append(make_trade(101.0, 0.25, SELL, t=1.0))
        df = buf.to_dataframe()
        assert list(df.columns) == [
            "product_id", "trade_id", "sequence", "exchange_time", "receipt_time",
            "price", "size", "side", "notional", "latency_ms",
        ]
        assert len(df) == 2
        assert df["notional"].tolist() == pytest.approx([50.0, 25.25])
        assert str(df["exchange_time"].dtype).startswith("datetime64")
        assert TickBuffer(5).to_dataframe().empty

    def test_rejects_zero_capacity(self):
        with pytest.raises(ValueError):
            TickBuffer(0)


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
class TestMarketAnalytics:
    def test_routes_trades_and_tracks_totals(self):
        ma = MarketAnalytics(["BTC-USD", "ETH-USD"], block_thresholds={"ETH-USD": 10.0}, clock=lambda: T0)
        assert ma.on_trade(make_trade(60_000.0, 0.5, BUY, t=0.0)) is None
        assert ma.on_trade(make_trade(60_100.0, 0.25, SELL, t=1.0)) is None
        alert = ma.on_trade(make_trade(3_000.0, 12.0, BUY, t=1.5, product="ETH-USD"))
        assert alert is not None and alert.product_id == "ETH-USD"
        assert ma.on_trade(make_trade(3_000.0, 5.0, BUY, t=1.6, product="ETH-USD")) is None  # below ETH override

        snap = ma.snapshot(now=T0 + 2.0)
        btc = snap.products["BTC-USD"]
        assert btc.last_price == 60_100.0 and btc.tick_direction == 1
        assert btc.vwap == pytest.approx((60_000 * 0.5 + 60_100 * 0.25) / 0.75)
        assert btc.ofi[10.0].delta == pytest.approx(0.25)
        assert btc.total_volume == pytest.approx(0.75)
        assert btc.trade_count == 2 and btc.buffer_len == 2
        assert snap.products["ETH-USD"].block_threshold == 10.0
        assert snap.trades_total == 4
        assert snap.block_trades_total == 1 and snap.block_trades[0].size == 12.0
        assert snap.alerts_total == 1 and snap.alerts[0].kind == ALERT_LARGE_TRADE
        assert snap.volume_by_product == {"BTC-USD": pytest.approx(0.75), "ETH-USD": pytest.approx(17.0)}
        assert snap.notional_total == pytest.approx(60_000 * 0.5 + 60_100 * 0.25 + 3_000 * 17.0)
        assert snap.latency.count == 4 and snap.latency.mean_ms == pytest.approx(50.0)
        assert snap.uptime == pytest.approx(2.0)

    def test_quotes_feed_spread_monitor(self):
        ma = MarketAnalytics(["BTC-USD"], spread_config={"min_observations": 2, "cooldown": 0.0})
        assert ma.on_quote(make_quote(49_999.5, 50_000.5, t=0.0)) is None
        assert ma.on_quote(make_quote(49_999.5, 50_000.5, t=0.1)) is None
        alert = ma.on_quote(make_quote(49_990.0, 50_010.0, t=0.2))
        assert alert is not None and alert.kind == ALERT_SPREAD_WIDENING
        snap = ma.snapshot()
        p = snap.products["BTC-USD"]
        assert p.spread == pytest.approx(20.0) and p.spread_is_wide is True
        assert snap.quotes_total == 3 and snap.alerts_total == 1

    def test_unknown_product_is_auto_registered(self):
        ma = MarketAnalytics(["BTC-USD"])
        ma.on_trade(make_trade(10.0, 1.0, product="SOL-USD"))
        assert "SOL-USD" in ma.products

    def test_realized_volatility_scales_with_price_noise(self):
        rng = np.random.default_rng(0)
        calm = MarketAnalytics(["BTC-USD"]).products["BTC-USD"]
        wild = MarketAnalytics(["BTC-USD"]).products["BTC-USD"]
        price_c = price_w = 50_000.0
        for i in range(300):
            price_c *= math.exp(rng.normal(0, 1e-5))
            price_w *= math.exp(rng.normal(0, 1e-3))
            calm.on_trade(make_trade(price_c, 0.1, t=i * 0.5))
            wild.on_trade(make_trade(price_w, 0.1, t=i * 0.5))
        assert 0 < calm.realized_volatility_bps() < wild.realized_volatility_bps()
        assert math.isnan(MarketAnalytics(["BTC-USD"]).products["BTC-USD"].realized_volatility_bps())

    def test_ticks_per_second(self):
        ma = MarketAnalytics(["BTC-USD"])
        for i in range(20):
            ma.on_trade(make_trade(100.0, 0.01, t=i * 0.1, latency=0.0))  # 20 ticks over 2 s
        snap = ma.snapshot(now=T0 + 1.95)
        assert snap.ticks_per_sec == pytest.approx(10.0)
        assert snap.ticks_per_sec_avg == pytest.approx(2.0)  # 20 ticks / 10 s window

    def test_dataframe_across_products_sorted_by_time(self):
        ma = MarketAnalytics(["BTC-USD", "ETH-USD"])
        ma.on_trade(make_trade(3_000.0, 1.0, t=1.0, product="ETH-USD"))
        ma.on_trade(make_trade(60_000.0, 1.0, t=0.0))
        df = ma.to_dataframe()
        assert df["product_id"].tolist() == ["BTC-USD", "ETH-USD"]
        assert MarketAnalytics(["BTC-USD"]).to_dataframe().empty
