"""Unit tests for CLI argument parsing helpers."""

from __future__ import annotations

import pytest

from main import (
    DEFAULT_PAIRS,
    build_parser,
    parse_block_sizes,
    parse_pairs,
    parse_windows,
)
from src.analytics import DEFAULT_BLOCK_THRESHOLD, DEFAULT_WINDOWS


class TestParsePairs:
    def test_default_when_omitted(self):
        assert parse_pairs(None) == list(DEFAULT_PAIRS)
        assert parse_pairs([]) == list(DEFAULT_PAIRS)

    def test_single_and_repeated(self):
        assert parse_pairs(["btc-usd"]) == ["BTC-USD"]
        assert parse_pairs(["BTC-USD", "ETH-USD"]) == ["BTC-USD", "ETH-USD"]

    def test_comma_separated_and_deduplicated(self):
        assert parse_pairs(["btc-usd, eth-usd, BTC-USD"]) == ["BTC-USD", "ETH-USD"]

    def test_rejects_junk(self):
        with pytest.raises(ValueError, match="invalid product id"):
            parse_pairs(["bitcoin"])
        with pytest.raises(ValueError, match="at least one"):
            parse_pairs([",,"])


class TestParseBlockSizes:
    def test_default_when_omitted(self):
        default, overrides = parse_block_sizes(None)
        assert default == DEFAULT_BLOCK_THRESHOLD
        assert overrides == {}

    def test_bare_number_sets_default(self):
        default, overrides = parse_block_sizes(["2.5"])
        assert default == pytest.approx(2.5)
        assert overrides == {}

    def test_pair_override(self):
        default, overrides = parse_block_sizes(["1.5", "ETH-USD=15", "sol-usd=500"])
        assert default == pytest.approx(1.5)
        assert overrides == {"ETH-USD": 15.0, "SOL-USD": 500.0}

    def test_rejects_negative_and_malformed(self):
        with pytest.raises(ValueError, match="non-negative"):
            parse_block_sizes(["-1"])
        with pytest.raises(ValueError, match="invalid block size"):
            parse_block_sizes(["abc"])
        with pytest.raises(ValueError, match="invalid product id"):
            parse_block_sizes(["DOGE=1"])


class TestParseWindows:
    def test_default_shape(self):
        assert parse_windows("10,30,60") == DEFAULT_WINDOWS

    def test_sorted_unique(self):
        assert parse_windows("60, 10, 30, 10") == (10.0, 30.0, 60.0)

    def test_rejects_empty_and_non_positive(self):
        with pytest.raises(ValueError):
            parse_windows("")
        with pytest.raises(ValueError):
            parse_windows("10,0")
        with pytest.raises(ValueError):
            parse_windows("nope")


class TestBuildParser:
    def test_defaults(self):
        args = build_parser().parse_args([])
        assert args.pairs is None
        assert args.headless is False
        assert args.buffer_size == 10_000
        assert args.fps == 8.0
        assert args.windows == "10,30,60"

    def test_pair_flags_round_trip(self):
        args = build_parser().parse_args(["--pair", "BTC-USD", "--pair", "ETH-USD", "--headless"])
        assert args.pairs == ["BTC-USD", "ETH-USD"]
        assert args.headless is True
