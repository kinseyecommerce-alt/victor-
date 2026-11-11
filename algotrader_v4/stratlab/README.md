# stratlab — a screening lab for candidate strategies

Research code only. It never touches a live agent, a broker or a production API; it reads bar files and
writes reports to a scratch directory (it refuses to write inside the repository).

The point: with hundreds of candidate strategies, some will look good by luck. So a strategy is a small
function, every strategy goes through the **same simulator**, and every strategy is judged by the **same
pre-registered protocol** whose hold-out period it cannot see until it has passed a screen.

```
cd algotrader_v4
python -m stratlab.run --kind all --list
python -m stratlab.run --data <bars dir> --kind intraday --out <scratch dir>
python -m stratlab.run --data <bars dir> --kind scalping --ids scalp_orb5,scalp_vwap_bounce --symbols 20 --out <scratch dir>
```

`--out` gets `summary.txt` (the table), `results.csv`, `data_quality.json` and `trades/<id>.csv`.

| file | role |
|------|------|
| `data.py` | loads 1-minute bars (both on-disk layouts), caches them as `.npz` outside the repo, derives daily bars |
| `indicators.py` | per-session indicator arrays (`Session`), the truncated `View`, plain `ema/sma/rsi/atr` helpers |
| `api.py` | what a strategy imports: `@strategy`, `Entry`, `Swing`, helpers |
| `sim.py` | the simulator (intraday and positional) and the look-ahead audit |
| `protocol.py` | the pre-registered pass/fail protocol — **do not tune it** |
| `catalog/` | the strategies; every `.py` file in it is imported automatically |
| `run.py` | the CLI |

## Adding a strategy

Create a new file in `catalog/` (any name not starting with `_`), or add to an existing one. Nothing else
needs editing. Use a new unique `id`. Keep parameters few and at textbook values, and **do not tune them on
results**: every variant you try is one more candidate that can pass by luck.

### Intraday, scalping, futures-intraday — a worked example

```python
from stratlab.api import strategy, Entry, LONG, SHORT, hhmm, first_true     # inside catalog/: from ..api import ...

@strategy("orb15_break",                      # unique id
          "intraday",                         # kind: intraday | scalping | swing | futures
          "orb",                              # family (free text, used for grouping)
          "Break of the first 15-minute range, stop at the other side, 1.5R",   # one line of plain English
          {"minutes": 15, "rr": 1.5, "last": "12:00"})                           # parameters -> p.minutes, p.rr
def orb15_break(s, p):
    hi, lo, k = s.orb(p.minutes)              # opening range; it is complete from bar index k
    if k >= len(s) or not hi > lo:
        return []
    end = s.idx(hhmm(p.last))                 # index of the first bar at/after 12:00
    out = []
    i = first_true(s.c > hi, k, end)          # first 1-minute close above the range
    if i >= 0:
        out.append(Entry(i, LONG, stop=lo, rr=p.rr))
    i = first_true(s.c < lo, k, end)
    if i >= 0:
        out.append(Entry(i, SHORT, stop=hi, rr=p.rr))
    return out                                # candidates; the simulator takes the first it can
```

`fn(s, p)` receives one symbol's session `s` and returns a list of `Entry`. An entry at bar `i` means
"decided at the close of bar i"; it is filled at the open of bar `i+1`.

`Entry(i, side, stop, target=None, rr=None, time_stop=None, trail=None)`

* `side` `LONG` (+1) or `SHORT` (-1); `stop` is a price and is required.
* `target` is a price, or give `rr` (target = fill ± rr × (fill − stop)), or neither (hold to square-off).
* `time_stop` minutes after the fill; `trail` is `("pct", x)`, `("atr", m)` or `("be", r)` (see `api.py`).

What `s` offers (all arrays have one element per 1-minute bar of the session):

| | |
|---|---|
| bars | `s.t` (minute of day, 09:15 = 555), `s.o s.h s.l s.c s.v` |
| session | `s.vwap`, `s.vsd` (σ around VWAP, so bands are `s.vwap ± 2*s.vsd`), `s.day_hi`, `s.day_lo` (running) |
| 1-minute | `s.ema9 s.ema21 s.ema50 s.rsi s.atr`, `s.vr` (volume ÷ mean volume of the previous 20 bars) |
| 5-minute | `s.e9_5 s.e21_5 s.e50_5 s.rsi5 s.atr5` (value of the last *completed* 5-minute bar), `s.bars5()` → `o h l c v end` of completed 5-minute bars |
| opening range | `s.orb(15)` → `(high, low, k)`; `s.idx(hhmm("10:00"))` → first bar index at/after a time |
| yesterday | `s.pdo s.pdh s.pdl s.pdc`, `s.gap` (today's open ÷ `pdc` − 1), `s.datr` (14-day ATR), `s.dsma20`, `s.dvol20` |
| history | `s.prev.o/.h/.l/.c/.v` daily arrays up to yesterday (`s.prev.c[-1]` is yesterday's close) |
| index | `s.nifty` (NIFTY 50 close at the same minute, or `None`), `s.nifty_pdc`. Declare `needs_index=True` if you use it |
| identity | `s.symbol`, `s.date` (yyyymmdd) |

Helpers in `api.py`: `first_true(mask, start, stop)`, `cross_up(a, b)`, `cross_dn(a, b)`,
`between(s, "09:45", "14:30")`, `runlen(mask)`, `shift(x, k)`, `prev_high(h, n)`, `prev_low(l, n)`, `hhmm("10:00")`.

Decorator options: `max_per_day=` (default 1 for intraday, 3 for scalping; a strategy that wants several
trades returns several entries — those that fall while a position is open are ignored), `style="bar"`,
`needs_index=True`, and for `kind="futures"` the mandatory `horizon="intraday"` or `"positional"`.

### No look-ahead

* `style="vector"` (default, fast): `fn` gets the whole session. Your signal at bar `i` must use bars `0..i`
  only. This is **checked**: on sampled sessions every run re-executes the strategy on views truncated at
  several times of day, and if any earlier signal changes the strategy is reported `INVALID: LOOK-AHEAD`.
  Typical mistakes it catches: `s.h.max()`, `s.c[-1]`, using `s.orb(15)` before bar `k`, `np.argmax` over the
  day. Time-triggered signals use `i = s.idx(hhmm("14:29")); if i >= len(s): return []`.
* `style="bar"` (slower, safe by construction): `fn(s, p)` is called once per bar while flat, with `s`
  truncated to end at the current bar (`s.c[-1]` is the bar just closed); return one `Entry` or `None`.

### Swing and futures-positional

```python
from stratlab.api import strategy, Swing, LONG

@strategy("swing_20d_breakout", "swing", "breakout",
          "Close above the highest high of the previous 20 sessions: buy next open, 5 % stop, 10 % target, 10 sessions",
          {"n": 20, "stop_pct": 5.0, "target_pct": 10.0, "max_hold": 10})
def swing_20d_breakout(d, p):
    if len(d) < p.n + 1:
        return None
    if d.c[-1] > d.h[-p.n - 1:-1].max():
        return Swing(LONG, stop_pct=p.stop_pct, target_pct=p.target_pct, max_hold=p.max_hold)
    return None
```

`fn(d, p)` is called once per day while flat. `d.date d.o d.h d.l d.c d.v` are daily arrays ending at the
decision day (`d.c[-1]`); `d.nifty` and `d.vix` are daily closes aligned to the same days (or `None`). It
cannot see later days: the arrays are cut. Use `stratlab.indicators.ema/sma/rsi/atr/highest/lowest` on them.
Return `Swing(side, stop_pct=, target_pct=, max_hold=)` (or `stop=` / `target=` prices) or `None`.
By default the fill is the next session's open. With `decision_time="15:20"` in the decorator the last daily
bar is the day *so far* at that minute and the fill is that minute's bar open.
`kind="swing"` is delivery and long-only; shorts go under `kind="futures", horizon="positional"`.

## Candlestick patterns (`candles.py`, `catalog/candlestick.py`)

`candles.resample()` turns a session's 1-minute bars into 5m / 15m / 60m candles aligned at 09:15 (the 60m day
ends with a short 15:15-15:30 candle); daily candles are the lab's daily bars. A candle is usable only once it is
complete — at the close of its last minute (the 09:44 bar for the 09:30-09:45 candle), so the earliest fill is
the open of the 09:45 bar. `session_candles(s, tf)` prepends up to 100 candles of earlier sessions (read through
`s._sd`, never at or after today's first bar) for EMA20 / ATR / average body. The detectors are pure functions
with their textbook definitions (Nison / Bulkowski) and fixed ratios in docstrings and module constants; one
prior-trend rule (close vs EMA20 and vs the close 5 candles earlier, on the candle before the pattern) gates every
reversal. The catalogue registers 152 strategies `cs_<pattern>_<tf>[_conf]` with one fixed plan (next-open
entry, stop beyond the pattern extreme and >= 0.3 ATR, 2R, square-off 15:15 / 10 sessions) — count all 152 when
judging how many screen passes luck alone would give. `Swing(rr=...)` sets a positional target at rr × the risk
from the fill.

What the candlestick results do and do not test (read before quoting a pass):

* "Long" body means B > 1.0 × the average body of the previous 10 candles (TA-Lib's default, pre-registered), i.e.
  "above average" — about 39 % of 15m candles — not Nison's "distinctly long". "Short" is below average.
* Ratio comparisons use a 1e-6 rupee tolerance (`candles.EPS`) so exact ties (frequent with 0.05 ticks) resolve
  the same way whatever the floating-point noise: a tie is neither long nor short and satisfies "<=" / ">=".
* Intraday piercing line, dark cloud cover, morning star and evening star need all their candles in today's
  session, so their "gap" is only the tick-level jump between consecutive intraday candles, not Nison's
  overnight gap. A pass by these intraday variants is not evidence for the textbook pattern.
* `_conf` = the next candle closes beyond the pattern candle's REAL BODY in the trade direction (Nison's
  hanging-man rule, applied to every confirmation variant).
* Daily candles are built from 1-minute bars: open = the 09:15 minute's open, close = the 15:29 minute's close,
  not NSE's official close (a 30-minute VWAP); borderline gap / engulfing conditions can differ from an official
  chart. Daily EMA20 / ATR use the whole history up to the decision day.
* 60m: the short 15:15-15:30 candle of earlier sessions is left out of the history (averages, EMA, ATR).
* Some daily patterns fire only a handful of times a year (piercing line, three black crows, stars, ...): they
  fail the minimum-trades gate on count alone — report them as "insufficient trades", not "no edge".

## Simulator assumptions (`sim.py`)

* **Data**: 1-minute bars, 09:15–15:29 IST. Sessions with fewer than 200 bars (the Diwali muhurat hour, a
  half-downloaded day) are dropped entirely, so "yesterday" is always the last full session.
* **Slippage** is a per-run setting: `--slippage-bps` (default **2 bps per side** for NIFTY-100 stocks) and
  `--index-slippage-bps` (default **1 bps** for NIFTY / BANKNIFTY futures). 2 bps comes from the cost audit: the
  effective half-spread of NIFTY-100 stocks for ₹1 lakh orders measured on 1-minute bars was about 2.4x below the
  old 5 bps default. Every report also shows each strategy's DISCOVERY net at `--sens-bps` (default 5 bps; index
  futures scaled in proportion, 2.5 bps) and whether it would still pass the screen there (`net@5`, `scr`).
* **Entry**: signal at the close of bar `i` → filled at the **open of bar i+1**, `slippage` worse. An entry whose
  stop is already through at that open, or whose target is on the wrong side, is skipped (counted in
  `skipped_entries`). No fills at or after 15:15.
* **Exit**: stop and target are checked on each later bar's low/high, the entry bar included. **If one bar
  touches both, the stop is assumed.** A bar that opens beyond the stop fills at its **open**, not the stop.
  Stop, time-stop, square-off and data-end exits are market orders and pay `slippage`. A target is a resting
  limit order: it fills only when a bar trades **at least one tick through it** (a bare touch is not a fill);
  if the bar opens a tick or more beyond it, it fills at that open. Tick = ₹0.05, or ₹0.01 for stocks under
  ₹250 from 16 June 2025 (`sim.equity_tick`); index futures 0.05.
* **Square-off** at the open of the 15:15 bar. If the day's data ends earlier, at the last available close.
* **Size**: ₹1,00,000 notional per trade in whole shares. One open position per symbol; `max_per_day`
  trades per symbol per day. Strategies are simulated independently: no portfolio cap, no capital limit.
* **Costs**: `costs.round_trip_cost` (2026 Zerodha/NSE rates, `costs.COST_MODEL_VERSION`) — `equity_mis` for
  intraday and scalping, `equity_cnc` for swing (incl. the ₹15.34 DP charge per sell), `future` for futures
  (0.05 % STT on the sell side). On ₹1 lakh that is about ₹83 for an intraday round trip, ₹238 for delivery and
  ₹104 for a futures round trip, plus about ₹40 of slippage at 2 bps (₹100 at 5 bps): an intraday trade starts
  roughly 0.12 % behind (0.18 % at 5 bps).
* **Stock futures** (`kind="futures"` without `underlying`) are simulated on the cash price with futures costs:
  no basis, lot size, margin or rollover.
* **Index futures** (`kind="futures", underlying="NIFTY"` or `"BANKNIFTY"`, optional `lots=`) run only on
  `_NIFTY50_min.json` / `_NIFTYBANK_min.json`, with futures costs, `lots` × the 2026 lot size (NIFTY 65,
  BANKNIFTY 30 — NSE circular FAOP70616, from the January 2026 expiries; used for every date) and index
  slippage. Positional holds are allowed. The spot index stands in for the future: basis and expiry roll are
  ignored (a hold across expiry would really pay a roll spread and another round trip). The screen's
  "no symbol > 35 % of net" gate cannot apply to a one-instrument strategy and is skipped for them; every other
  gate is unchanged.
* **Positional**: stop/target checked on daily high/low (stop first); a gap through the stop fills at the
  open; the position is closed at the close of its `max_hold`-th session, or at the close of the last session of
  the period being simulated (`period_end`), so DISCOVERY trades never use FINAL-period prices. Delivery is long-only.
* **Not modelled**: order-book depth, partial fills, circuit limits, short-selling bans, corporate actions
  (the bars are used as given — check `data_quality.json` for unadjusted splits), options.

## The protocol (`protocol.py`) — pre-registered, do not tune

Sessions are split chronologically: **DISCOVERY** = first 70 % (halved into **H1** and **H2**), **FINAL** = last
30 %. A trade belongs to the period of its entry date.

A strategy **passes the screen** only if, on DISCOVERY, all of these hold:

1. net after costs > 0 in H1 **and** in H2,
2. profit factor ≥ 1.15,
3. at least 150 trades (intraday, scalping, futures-intraday) or 60 (swing, futures-positional),
4. no single symbol contributes more than 35 % of the net.

Only strategies that pass are run on FINAL. **CONFIRMED** = net > 0 and profit factor ≥ 1.05 on FINAL;
otherwise `REJECTED ON FINAL`.

* FINAL for screen failures is not hidden by formatting — it is **never simulated**, and their trade CSVs
  contain DISCOVERY trades only. The one way to see it is `--peek-final`, which labels the result `PEEK`
  and records it, so that strategy can never be reported as CONFIRMED on that data.
* FINAL is reported once per strategy AND dataset: the first result is kept in `final_ledger.json` in the
  data cache (one cache per data directory; the key also holds the FINAL dates, session and symbol counts, and
  the entry records the data directory), so a 12-month and a 3-year run each have their own single FINAL. If
  the strategy's code or parameters change after its FINAL was seen, a later pass is reported as
  `RETEST (FINAL already used)`, not as a confirmation.
* The ledger key also holds the **cost/fill model** (`sim.COST_MODEL` = `costs.COST_MODEL_VERSION` /
  `sim.FILL_MODEL_VERSION`; entries from before versioning have no suffix and count as "legacy"). Re-running the
  same, unchanged strategy under corrected costs is therefore not a second look: it gets its own FINAL, whose note
  lists every earlier-model FINAL result. A FINAL that was peeked, or seen with different code, under any cost
  model still makes a later pass a RETEST. Per-run slippage is deliberately NOT part of the key, so a slippage
  chosen after seeing FINAL never buys a fresh look.
* `--screen-only` never simulates FINAL, not even for passers (verdict `PASSED SCREEN`, FINAL hidden, nothing
  written to the ledger): use it to compare cost models without spending the hold-out.
* Below 100 sessions there is **no verdict** ("too few sessions"); the discovery numbers are shown for
  debugging only.
* A strategy that raises, or fails the look-ahead audit, is `INVALID`.
* Every report ends with how many passed out of how many were tested and the reminder that about 5 % of
  useless strategies pass each coin-flip-like gate. With 250 candidates, a dozen lucky screen passes and a few
  lucky confirmations are expected from strategies with no edge. A confirmation earns a paper-trading trial,
  nothing more.

## Index options: weekly NIFTY short strangles (`options_vrp.py`)

`sim.py` cannot hold option legs, so `python -m stratlab.options_vrp --data <bars dir> --out <scratch dir>` runs the
two pre-registered volatility-risk-premium strategies (VRP-STRANGLE-016-NAKED, 1 lot; VRP-STRANGLE-016-WINGED,
+/-400-point wings, 2 lots) through `options_replay.ShortStrangle` (Black-Scholes on the NIFTY 50 minute bars, India
VIX previous close as one flat vol) and judges them with `protocol.evaluate`: same DISCOVERY / FINAL split, H1 / H2
screen, PF >= 1.15, >= 60 trades, FINAL once through the data directory's `final_ledger.json` (keyed by cost model
`2026-11/o1`), no `--peek-final`. Entry 10:20 on the first session after each weekly expiry (week skipped if fewer
than 3 sessions remain to expiry), 0.16-delta strikes, held to expiry settlement, no stop or target; lot 75, 65 from
2025-12-31. Reports rupees, rupees per lot, bps of margin (naked 12 % of notional; winged max loss + 2 %), worst week
and max drawdown, plus DISCOVERY-only sensitivities: the haircut x2.5 ("net@5") and approximate historical lot sizes.
The credit is a model value (no skew); treat a pass as a reason to paper-trade, not as a fill record.

## Tests

`cd algotrader_v4 && python test_pipeline.py` — the `STRATLAB` section covers: the per-bar view cannot read
bar i+1; changing future bars does not change earlier fills; the audit catches a cheating vector strategy;
indicators are causal; next-open fills and slippage; stop-before-target; square-off and time stop; costs;
the positional simulator; the protocol gates and that FINAL is never simulated for screen failures; both data
layouts; every catalogue strategy runs and passes the audit; the CLI end to end. The `COSTS 2026 AND STRATLAB
FILL MODEL` section covers the 2026 cost rates, per-run slippage and the 5 bps sensitivity column, stops gapped
through filling at the open, market exits paying slippage, the one-tick target rule, index futures, the
cost-model-keyed FINAL ledger and `--screen-only`.
