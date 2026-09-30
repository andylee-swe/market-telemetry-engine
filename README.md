# Market Telemetry Engine

Production-grade real-time market telemetry and analytics for the Coinbase Exchange public WebSocket feed. The engine ingests every match (trade) tick, maintains a sub-millisecond in-memory ring buffer, and streams quantitative metrics — order-flow imbalance, intraday VWAP, block-trade / spread-widening alerts, and socket latency — into a live terminal dashboard.

```
                       Coinbase Exchange
              wss://ws-feed.exchange.coinbase.com
                                 │
                    matches · ticker · heartbeat
                                 │
     ┌───────────────────────────▼────────────────────────────┐
     │  src/engine.py                                         │
     │  CoinbaseFeed                                          │
     │    • asyncio + websockets                              │
     │    • exponential backoff + equal jitter                │
     │    • stale-feed watchdog (heartbeat-aware)             │
     │    • maker→taker side inversion                        │
     │    • trade-id gap detection                            │
     └───────────────────────────┬────────────────────────────┘
                                 │  Trade / Quote
                                 ▼
     ┌────────────────────────────────────────────────────────┐
     │  src/analytics.py                                      │
     │  MarketAnalytics                                       │
     │    TickBuffer     deque(maxlen=10_000)  per product    │
     │    OFI            rolling Δ(buy, sell)  10s / 30s / 60s│
     │    VWAP           Σ(p·q) / Σq           UTC session    │
     │    SpreadMonitor  time-decayed EWMA     widen alerts   │
     │    LatencyTracker exchange_ts → receipt  p50/p95/p99   │
     └───────────────────────────┬────────────────────────────┘
                                 │  TelemetrySnapshot
                                 ▼
     ┌────────────────────────────────────────────────────────┐
     │  src/dashboard.py          rich.live.Live  (8 fps)     │
     │    Live Ticker  │  Block Trades  │  System Telemetry   │
     └────────────────────────────────────────────────────────┘
                                 ▲
                                 │
                            main.py  (--pair, --headless, …)
```

Ingestion and analytics run on the same asyncio event loop. The dashboard samples an immutable snapshot on its own timer, so UI rendering never sits on the tick hot path.

---

## Quantitative formulas

### Order Flow Imbalance (OFI)

Coinbase `matches.side` is the **maker** side. The engine inverts it so every tick is labelled by the **aggressor (taker)**:

```
taker = buy   if maker = sell     (buy-initiated / uptick)
taker = sell  if maker = buy      (sell-initiated / downtick)
```

Over a rolling window of `w` seconds (`w ∈ {10, 30, 60}`):

```
OFI_w  =  Σ size_i · 1[taker_i = buy]  −  Σ size_i · 1[taker_i = sell]
nOFI_w =  OFI_w  /  (buy_volume_w + sell_volume_w)          ∈ [−1, +1]
```

Membership is half-open: a tick stamped `t` is inside the window at time `now` iff `t > now − w`. Windows are keyed on the **client receipt clock**, so they keep decaying during feed stalls and are immune to exchange-clock skew. Running sums are incrementally maintained (O(1) amortised per tick) and re-synchronised with `math.fsum` every 4096 insertions to bound floating-point drift.

### Volume-Weighted Average Price (VWAP)

Intraday VWAP, updated on every incoming trade tick:

```
VWAP_t  =  Σ (price_i · size_i)  /  Σ size_i
```

Accumulators reset when the trade timestamp crosses a **UTC midnight** boundary (the conventional session for 24/7 crypto venues). Zero-size prints are ignored. Deviation of the last print from VWAP is reported in basis points:

```
dev_bps  =  (price − VWAP) / VWAP  ·  10_000
```

### Bid–ask spread and widening

Top-of-book comes from the `ticker` channel (matches do not carry a book):

```
mid         =  (bid + ask) / 2
spread      =  ask − bid
spread_bps  =  spread / mid · 10_000
```

The baseline is a **time-decayed** EWMA of `spread_bps` (half-life 30 s by default), so it is robust to bursts and lulls in quote frequency:

```
decay     =  0.5 ^ (Δt / half_life)
baseline  =  decay · baseline  +  (1 − decay) · spread_bps
```

A quote is flagged as a sudden widening when all of the following hold:

1. at least `min_observations` quotes have been seen (warm-up),
2. `spread_bps ≥ min_spread_bps` (absolute floor; default 1 bp),
3. `spread_bps ≥ widen_factor · baseline` (relative jump; default 3×),
4. at least `cooldown` seconds have elapsed since the previous alert.

### Block trades

Any print whose size is **strictly greater than 1.5** base units is flagged (1.5 BTC on `BTC-USD`; the same 1.5-unit default applies to other pairs, override per product with `--block-size ETH-USD=15`).

### Ingestion latency

```
latency_ms  =  (client_receipt_time − exchange_time) · 1_000
```

`exchange_time` is the ISO-8601 timestamp on the match frame; `client_receipt_time` is taken with `time.time()` the instant the frame is decoded. The dashboard reports windowed mean / p50 / p95 / p99 plus a lifetime Welford mean.

### Realized volatility (ticker column `σ 1m`)

Tick-return realized volatility scaled to a 60-second horizon, in basis points:

```
σ_1m  =  std(Δ log p)  ·  √(ticks_per_minute)  ·  10_000
```

computed from the newest 200 prints in the ring buffer.

---

## Requirements

- Python **3.11+**
- Network access to `wss://ws-feed.exchange.coinbase.com` (public, unauthenticated)

---

## Installation

### Windows (PowerShell)

```powershell
cd C:\projects\market-telemetry-engine
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Linux / macOS

```bash
cd /path/to/market-telemetry-engine
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

---

## Running

Default pairs are **BTC-USD** and **ETH-USD**, subscribed to the `matches`, `ticker`, and `heartbeat` channels.

### Windows

```powershell
python main.py
python main.py --pair BTC-USD
python main.py --pair BTC-USD --pair ETH-USD --block-size ETH-USD=15
python main.py --headless --summary-interval 5
python main.py --duration 60 --export ticks.csv
```

### Linux / macOS

```bash
python main.py
python main.py --pair BTC-USD
python main.py --pair BTC-USD --pair ETH-USD --block-size ETH-USD=15
python main.py --headless --summary-interval 5
python main.py --duration 60 --export ticks.csv
```

Stop with `Ctrl+C`. On exit the engine prints a session summary; `--export` dumps the ring buffers to CSV.

### Useful flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--pair PAIR` | BTC-USD, ETH-USD | Repeat or comma-separate (`--pair BTC-USD,ETH-USD`) |
| `--windows` | `10,30,60` | OFI window lengths in seconds |
| `--buffer-size` | `10000` | Ring-buffer capacity (ticks retained per product) |
| `--block-size [PAIR=]SIZE` | `1.5` | Large-trade threshold in base units |
| `--spread-factor` | `3.0` | Flag when spread ≥ this × the EWMA baseline |
| `--headless` | off | Log periodic summaries instead of the live UI |
| `--fps` | `8` | Dashboard refresh rate |
| `--stale-timeout` | `15` | Force reconnect if no frame (incl. heartbeats) arrives |
| `--duration SEC` | none | Auto-stop (useful for smoke tests) |
| `--export CSV` | none | Write the ring buffers to CSV on shutdown |
| `--log-file PATH` | none | Append timestamped logs (including alerts) |

`python main.py --help` lists every option.

---

## Tests

```powershell
python -m pytest -q
```

```bash
python -m pytest -q
```

`tests/test_analytics.py` verifies OFI and VWAP against closed-form and brute-force references (including a 20 000-tick drift check). Additional tests cover spread widening, latency percentiles, the ring buffer, WebSocket frame parsing, and CLI argument handling.

---

## Sample output

```
╭────────────────────────── Market Telemetry Engine ──────────────────────────╮
│ ● LIVE   wss://ws-feed.exchange.coinbase.com   BTC-USD, ETH-USD             │
│   up 00:04:17   2026-09-30 06:46:03.412 UTC                                 │
╰─────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────── Live Ticker ────────────────────────────────╮
│ Asset    Price        VWAP     vs VWAP    OFI 10s       OFI 30s       OFI 60s       Bid / Ask              Spread         σ 1m │
│ BTC-USD  ▲ 67,421.15  67,388.40  +4.9bp   +2.1400 +41%  +5.8820 +18%  +11.2041 +9%  67,420.50 / 67,421.20  0.70 (0.10bp)  4.2 │
│ ETH-USD  ▼  3,412.61   3,409.88  +8.0bp   -18.550 -22%  +40.120 +6%   +88.441 +4%   3,412.55 / 3,412.70    0.15 (0.44bp)  6.8 │
╰─────────────────────────────────────────────────────────────────────────────╯
╭──────────── Recent Large Block Trades ────────────╮ ╭──────── System Telemetry ─────────╮
│ Time (UTC)    Asset    Side  Size     Price       │ │ Feed         ● LIVE  last frame 0.1s ago │
│ 06:46:02.881  BTC-USD  BUY   2.1500   67,421.15   │ │ Ticks/sec    14.0  (10s avg 11.3)  │
│ 06:45:41.102  ETH-USD  SELL  22.0000   3,413.20   │ │ Latency avg  48.2 ms  p50 41.0  p95 92.1 │
│                                                   │ │ Volume       BTC 38.4412  ETH 412.8800 │
│                                                   │ │ Notional     $3.98M                   │
│                                                   │ │ Processed    2,841 trades  2,790 quotes │
│                                                   │ │ Ring buffer  BTC 2,841/10,000  ETH 1,902/10,000 │
╰───────────────────────────────────────────────────╯ ╰───────────────────────────────────╯
╭──────────────────────────────────── Alerts ─────────────────────────────────╮
│ 06:46:02.881  BTC-USD   LARGE_TRADE      BUY 2.1500 BTC @ 67,421.15 ($144,956) exceeds 1.5 BTC block threshold │
│ 06:45:41.102  ETH-USD   LARGE_TRADE      SELL 22.0000 ETH @ 3,413.20 ($75,090) exceeds 1.5 ETH block threshold │
╰─────────────────────────────────────────────────────────────────────────────╯
```

Headless mode (`--headless`) prints the same numbers as a compact log line every `--summary-interval` seconds:

```
2026-09-30T06:46:03.412Z INFO mte.main: telemetry
feed=LIVE up=00:04:17 ticks/s=14.0 (10s 11.3) latency avg=48.2ms p50=41.0 p95=92.1 p99=140.4 trades=2,841 notional=$3.98M reconnects=0 missed=0 alerts=2
  BTC-USD: px=67,421.15 vwap=67,388.40 (+4.9bp) ofi10=+2.1400 ofi30=+5.8820 ofi60=+11.2041 spread=0.70 (0.10bp) vol=38.4412 BTC buf=2,841/10,000
  ETH-USD: px=3,412.61 vwap=3,409.88 (+8.0bp) ofi10=-18.5500 ofi30=+40.1200 ofi60=+88.4410 spread=0.15 (0.44bp) vol=412.8800 ETH buf=1,902/10,000
```

Timestamped warnings for block trades and spread widening also go to stderr (and to `--log-file` when set):

```
2026-09-30T06:46:02.881Z [BTC-USD] LARGE_TRADE: BUY 2.1500 BTC @ 67,421.15 ($144,956) exceeds 1.5 BTC block threshold
```

---

## Project layout

```
market-telemetry-engine/
├── main.py                  CLI entry point (argparse)
├── requirements.txt         websockets, rich, pandas, numpy, pytest
├── pytest.ini
├── src/
│   ├── __init__.py
│   ├── engine.py            WebSocket ingestion & async stream handling
│   ├── analytics.py         OFI, VWAP, anomalies, latency, ring buffer
│   └── dashboard.py         Rich live terminal visualizer
└── tests/
    ├── test_analytics.py    OFI / VWAP / spread / latency / buffer
    ├── test_engine.py       frame parsing, subscribe payload, feed handler
    └── test_cli.py          --pair / --block-size / --windows parsing
```

---

## Resilience

- **Reconnect** with exponential backoff and equal jitter (`[cap/2, cap]`, cap doubles each failed attempt, resets after a session that delivered data, ceiling 30 s).
- **Watchdog** force-closes the socket when no frame of any type has arrived for `--stale-timeout` seconds. The heartbeat channel (1 Hz per product) keeps the watchdog quiet during genuine trade lulls.
- **Trade-id continuity**: Coinbase match ids are contiguous per product; gaps are counted as `missed` in System Telemetry.
- **Malformed frames** are counted and skipped. Handler exceptions are isolated so a dashboard bug cannot kill the feed.
- **Non-retryable protocol errors** (for example an invalid product id rejected at subscribe time) abort the process with a non-zero exit code.

The feed is public and unauthenticated. This engine is for telemetry and research; it does not place orders.
