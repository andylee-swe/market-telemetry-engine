"""Unit tests for WebSocket frame parsing, subscribe payloads, and the feed handler."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.analytics import BUY, SELL, Quote, Trade
from src.engine import (
    DEFAULT_CHANNELS,
    CoinbaseFeed,
    FeedError,
    FeedState,
    build_subscribe_message,
    parse_exchange_time,
    parse_match,
    parse_ticker,
)

MATCH_SELL_MAKER = {
    "type": "match",
    "trade_id": 10,
    "sequence": 50,
    "maker_order_id": "ac928c66-ca31-4811-a182-57b32a2edf1a",
    "taker_order_id": "132db6ae-c6d4-4d20-8be6-98a21d9b947c",
    "time": "2014-11-07T08:19:27.028459Z",
    "product_id": "BTC-USD",
    "size": "5.23512",
    "price": "400.23",
    "side": "sell",
}

TICKER = {
    "type": "ticker",
    "sequence": 1,
    "product_id": "ETH-USD",
    "price": "3412.50",
    "best_bid": "3412.40",
    "best_ask": "3412.60",
    "time": "2026-09-30T06:46:03.412000Z",
    "trade_id": 99,
    "last_size": "0.10",
}


def _feed(*, on_trade=lambda t: None, on_quote=None, products=("BTC-USD", "ETH-USD")) -> CoinbaseFeed:
    return CoinbaseFeed(products, on_trade=on_trade, on_quote=on_quote, clock=lambda: 1_700_000_000.0)


# --------------------------------------------------------------------------- #
# Timestamp parsing
# --------------------------------------------------------------------------- #
class TestParseExchangeTime:
    def test_canonical_zulu_with_fraction(self):
        ts = parse_exchange_time("2014-11-07T08:19:27.028459Z")
        expected = datetime(2014, 11, 7, 8, 19, 27, 28459, tzinfo=timezone.utc).timestamp()
        assert ts == pytest.approx(expected)

    def test_whole_seconds(self):
        ts = parse_exchange_time("2024-01-01T00:00:00Z")
        assert ts == datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()

    def test_offset_utc(self):
        z = parse_exchange_time("2024-06-15T12:30:00.5Z")
        offset = parse_exchange_time("2024-06-15T12:30:00.5+00:00")
        assert z == pytest.approx(offset)

    def test_date_epoch_is_cached(self):
        a = parse_exchange_time("2025-03-01T00:00:01.000001Z")
        b = parse_exchange_time("2025-03-01T00:00:01.000001Z")
        assert a == b

    def test_fallback_fromisoformat_naive_assumes_utc(self):
        # Missing separators on purpose so the fast path rejects, then fromisoformat
        # still parses a well-formed ISO string with a space instead of T.
        ts = parse_exchange_time("2024-01-01 00:00:00+00:00")
        assert ts == datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()


# --------------------------------------------------------------------------- #
# Match / ticker
# --------------------------------------------------------------------------- #
class TestParseMatch:
    def test_maker_sell_is_taker_buy(self):
        trade = parse_match(MATCH_SELL_MAKER, receipt_time=1_417_000_000.5)
        assert trade.product_id == "BTC-USD"
        assert trade.trade_id == 10
        assert trade.sequence == 50
        assert trade.price == pytest.approx(400.23)
        assert trade.size == pytest.approx(5.23512)
        assert trade.side == BUY
        assert trade.is_buy is True
        assert trade.exchange_time == pytest.approx(parse_exchange_time(MATCH_SELL_MAKER["time"]))
        assert trade.receipt_time == 1_417_000_000.5
        assert trade.latency_ms == pytest.approx((trade.receipt_time - trade.exchange_time) * 1000.0)

    def test_maker_buy_is_taker_sell(self):
        msg = dict(MATCH_SELL_MAKER, side="buy")
        trade = parse_match(msg, receipt_time=0.0)
        assert trade.side == SELL
        assert trade.is_buy is False

    def test_rejects_unknown_side(self):
        with pytest.raises(ValueError, match="unknown side"):
            parse_match(dict(MATCH_SELL_MAKER, side="crossed"), 0.0)

    def test_rejects_missing_fields(self):
        with pytest.raises(KeyError):
            parse_match({"type": "match", "side": "sell"}, 0.0)


class TestParseTicker:
    def test_best_bid_ask(self):
        quote = parse_ticker(TICKER, receipt_time=1.0)
        assert quote is not None
        assert quote.product_id == "ETH-USD"
        assert quote.bid == pytest.approx(3412.40)
        assert quote.ask == pytest.approx(3412.60)
        assert quote.spread == pytest.approx(0.20)
        assert quote.receipt_time == 1.0
        assert quote.exchange_time == pytest.approx(parse_exchange_time(TICKER["time"]))

    def test_missing_book_returns_none(self):
        assert parse_ticker({"product_id": "BTC-USD", "best_bid": "", "best_ask": "1"}, 0.0) is None
        assert parse_ticker({"product_id": "BTC-USD"}, 0.0) is None

    def test_missing_exchange_time_falls_back_to_receipt(self):
        quote = parse_ticker({"product_id": "BTC-USD", "best_bid": "1", "best_ask": "2"}, receipt_time=42.0)
        assert quote is not None
        assert quote.exchange_time == 42.0


class TestSubscribeMessage:
    def test_compact_json_payload(self):
        payload = json.loads(build_subscribe_message(("BTC-USD", "ETH-USD")))
        assert payload == {
            "type": "subscribe",
            "product_ids": ["BTC-USD", "ETH-USD"],
            "channels": list(DEFAULT_CHANNELS),
        }

    def test_custom_channels(self):
        payload = json.loads(build_subscribe_message(["SOL-USD"], channels=["matches"]))
        assert payload["product_ids"] == ["SOL-USD"]
        assert payload["channels"] == ["matches"]


# --------------------------------------------------------------------------- #
# Feed handler (no network)
# --------------------------------------------------------------------------- #
class TestCoinbaseFeedHandler:
    def test_requires_at_least_one_product(self):
        with pytest.raises(ValueError):
            CoinbaseFeed([], on_trade=lambda t: None)

    def test_normalises_and_deduplicates_products(self):
        feed = _feed(products=("btc-usd", "BTC-USD", "eth-usd"))
        assert feed.products == ("BTC-USD", "ETH-USD")

    def test_match_frame_invokes_on_trade(self):
        trades: list[Trade] = []
        feed = _feed(on_trade=trades.append)
        feed._handle(json.dumps(MATCH_SELL_MAKER), now=1_417_000_000.5)
        assert len(trades) == 1
        assert trades[0].side == BUY
        assert feed.status.trades == 1
        assert feed.status.messages == 1
        assert feed.status.parse_errors == 0

    def test_ticker_frame_invokes_on_quote(self):
        quotes: list[Quote] = []
        feed = _feed(on_quote=quotes.append)
        feed._handle(json.dumps(TICKER), now=1.0)
        assert len(quotes) == 1
        assert quotes[0].spread == pytest.approx(0.20)
        assert feed.status.quotes == 1

    def test_ticker_without_handler_is_ignored(self):
        feed = _feed(on_quote=None)
        feed._handle(json.dumps(TICKER), now=1.0)
        assert feed.status.quotes == 0

    def test_last_match_seeds_trade_ids_but_is_not_a_trade(self):
        trades: list[Trade] = []
        feed = _feed(on_trade=trades.append)
        last = dict(MATCH_SELL_MAKER, type="last_match", trade_id=9)
        feed._handle(json.dumps(last), now=1.0)
        assert trades == []
        assert feed.status.trades == 0
        # The next match at id 11 implies one skipped print.
        nxt = dict(MATCH_SELL_MAKER, trade_id=11)
        feed._handle(json.dumps(nxt), now=1.1)
        assert len(trades) == 1
        assert feed.status.missed_trades == 1

    def test_contiguous_trade_ids_are_not_missed(self):
        feed = _feed()
        feed._handle(json.dumps(dict(MATCH_SELL_MAKER, trade_id=1)), now=1.0)
        feed._handle(json.dumps(dict(MATCH_SELL_MAKER, trade_id=2)), now=1.1)
        assert feed.status.missed_trades == 0
        assert feed.status.trades == 2

    def test_subscriptions_ack_transitions_to_live(self):
        feed = _feed()
        assert feed.status.state is FeedState.IDLE
        feed._handle(
            json.dumps(
                {
                    "type": "subscriptions",
                    "channels": [{"name": "matches", "product_ids": ["BTC-USD"]}],
                }
            ),
            now=1.0,
        )
        assert feed.status.state is FeedState.LIVE

    def test_heartbeat_is_counted(self):
        feed = _feed()
        feed._handle(json.dumps({"type": "heartbeat", "product_id": "BTC-USD"}), now=1.0)
        assert feed.status.heartbeats == 1

    def test_undecodable_frame_is_counted(self):
        feed = _feed()
        feed._handle(b"not-json{", now=1.0)
        assert feed.status.parse_errors == 1
        assert feed.status.trades == 0

    def test_bad_match_is_counted(self):
        feed = _feed()
        feed._handle(json.dumps({"type": "match", "side": "sell"}), now=1.0)
        assert feed.status.parse_errors == 1

    def test_handler_exception_does_not_propagate(self):
        def boom(_trade: Trade) -> None:
            raise RuntimeError("handler blew up")

        feed = _feed(on_trade=boom)
        feed._handle(json.dumps(MATCH_SELL_MAKER), now=1.0)
        assert feed.status.handler_errors == 1
        assert feed.status.trades == 1

    def test_unknown_type_is_counted(self):
        feed = _feed()
        feed._handle(json.dumps({"type": "l2update"}), now=1.0)
        assert feed.status.unknown_messages == 1

    def test_subscribe_error_is_fatal(self):
        feed = _feed()
        with pytest.raises(FeedError) as exc:
            feed._handle(
                json.dumps({"type": "error", "message": "Failed to subscribe", "reason": "BTC-USD is not a valid product"}),
                now=1.0,
            )
        assert exc.value.retryable is False
        assert "Failed to subscribe" in str(exc.value)

    def test_generic_error_is_retryable(self):
        feed = _feed()
        with pytest.raises(FeedError) as exc:
            feed._handle(json.dumps({"type": "error", "message": "internal error"}), now=1.0)
        assert exc.value.retryable is True

    def test_bytes_counter_and_status_snapshot(self):
        feed = _feed()
        raw = json.dumps(MATCH_SELL_MAKER)
        feed._handle(raw, now=1_700_000_010.0)
        snap = feed.status_snapshot(now=1_700_000_012.0)
        assert snap.bytes == len(raw)
        assert snap.last_message_age == pytest.approx(2.0)

    def test_subscribe_message_matches_builder(self):
        feed = _feed()
        assert json.loads(feed.subscribe_message())["product_ids"] == ["BTC-USD", "ETH-USD"]
