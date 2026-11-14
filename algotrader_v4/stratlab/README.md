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

## Opening-auction fade (`auction.py`)

`python -m stratlab.auction --data <bars dir> --out <scratch dir> [--screen-only]` runs the two pre-registered
strategies AUC-FADE-2PCT and AUC-FADE-3PCT: fade the 09:15 call-auction print when |auction / previous close - 1|
exceeds 2 % (3 %), up to 20 %; enter AT the 09:15 open (live: a pre-open market order at 09:07), exit at the 09:20
open; Rs 1 lakh per trade, at most the 10 largest |gap| per day. The intraday simulator fills at the bar after a
signal and cannot express an auction fill, so this module has its own fill model (`FILL_MODEL`, part of the FINAL
ledger key) and goes through `protocol.evaluate` unchanged (same split, screen and ledger; no `--peek-final`).
Slippage defaults to the family's pre-registered 2.1 bps per side; every report adds a 5 bps stress column on
DISCOVERY (and on FINAL, re-costing the same FINAL trades, only for screen passers), the net without the best five
dates, a date-clustered t on the gross and quarterly nets. Caveat: the backtest uses the final auction price, a
mild look-ahead against the 09:07 indicative price used live. 3 % is a nested subset of 2 %: one effect.

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

## Attention overnight long (`attention.py`)

`python -m stratlab.attention --data <bars3y> --out <scratch dir> [--screen-only]` runs SWING-ATTN-ON-3X (swing
round 2, pre-registered): on day t, if cumulative volume 09:15-15:20 > 3.0 x the mean full-day volume of the previous
20 sessions and the 15:20 close > the previous close, buy the stock future (simulated on the cash price, futures
costs, 2 bps/side) at the 15:21 open and sell at the next session's 09:15 auction print; top 10 volume ratios per
day, Rs 1 lakh each, one night, no stop. Special sessions (median bar count < 300) on t / t+1, CA_GAPS days and a
next session beyond the simulated period are excluded. Own fill model (`attn1`, in the ledger key), judged by the
unchanged `protocol.evaluate` (positional: >= 60 trades). Sensitivities: 5 bps slippage, a 09:16-open exit and CNC
costs. Tests: `test_pipeline.py` section "STRATLAB ATTENTION OVERNIGHT LONG".

## Tests

`cd algotrader_v4 && python test_pipeline.py` — the `STRATLAB` section covers: the per-bar view cannot read
bar i+1; changing future bars does not change earlier fills; the audit catches a cheating vector strategy;
indicators are causal; next-open fills and slippage; stop-before-target; square-off and time stop; costs;
the positional simulator; the protocol gates and that FINAL is never simulated for screen failures; both data
layouts; every catalogue strategy runs and passes the audit; the CLI end to end. The `COSTS 2026 AND STRATLAB
FILL MODEL` section covers the 2026 cost rates, per-run slippage and the 5 bps sensitivity column, stops gapped
through filling at the open, market exits paying slippage, the one-tick target rule, index futures, the
cost-model-keyed FINAL ledger and `--screen-only`.

## Cross-sectional baskets (`ltr.py`)

`ltr.py` runs a portfolio strategy that the per-symbol simulator cannot express: LTR-12-1-LOSERS-CNC, the 10 NIFTY-100
stocks with the lowest 12-1 month return (C[t-21]/C[t-252]-1), long-only CNC, Rs 1 lakh each from Rs 10 lakh,
rebalanced every 21 sessions at the 15:14 bar, kept names not topped up, pre-registered CNC costs (DP 15.34 + GST)
and 2.1 bps slippage (5 bps sensitivity). Known unadjusted corporate-action gaps (`CA_GAPS`: SIEMENS 2025-04-07,
VEDL 2026-04-30) are removed. It goes through the SAME `protocol.evaluate` (70/30 split, H1/H2 screen, PF >= 1.15,
>= 60 trades, 35 % symbol cap, FINAL once, same `final_ledger.json`) with one "trade" = one name over one period and
its net = EXCESS over the equal-weight universe; the per-period table (portfolio level, idle cash included) and its
t statistic are printed next to the verdict. `--hist` adds read-only history-bars day files for lookback only
(days before each stock's first minute day). Tests: `tests_ltr.py`, run by `test_pipeline.py`.

```
python -m stratlab.ltr --data <bars3y> --hist <day files> --out <scratch dir> [--screen-only]
```

Result on the 3-year data (DISCOVERY only; FINAL never simulated): FAILED SCREEN - PAYTM is 52 % of the excess net.

## Round-2 scalping: SCALP-AUC-FADE-1TO3-0925 (`auction_r2.py`)

A pre-registered parameter variant of the opening-auction fade, using `auction.py`'s fill model (`auction1`) and
`protocol.evaluate` unchanged: fade the 09:15 auction print when 1.0 % <= |gap| <= 3.0 % (inclusive), exit at the
09:25 open, top 10 |gap| per day, Rs 1 lakh each, no stop; skip a symbol-day whose previous session was short
(< 200 bars, dropped by the loader). Slippage 2 bps per side (5 bps stress). Reports long / short split, top-symbol
share, net without the best 5 dates and a date-clustered t on DISCOVERY and (for a screen passer) FINAL.

    python -m stratlab.auction_r2 --data <bars3y> --out <scratch dir> [--screen-only]

## NSE end-of-day archive data (`nsedata.py`)

Delivery %, F&O bhavcopy (futures OI / basis, per-strike NIFTY / BANKNIFTY option prices and OI, stock option OI
and per-strike prices of the two nearest expiries), participant-wise OI and the F&O ban list, normalised in
`<scratchpad>/nse` (env `STRATLAB_NSE_DIR`) and cached as `.npz` in `<nse>/_cache` (rebuilt when the source files
change). **Availability rule:** NSE publishes these files in the evening of trade date D; a value dated D is usable
from the open of the next session. Every accessor takes `before` and returns only rows dated strictly before it;
use `asof_session(s)` in an intraday `@strategy`, `asof_swing(d)` in a next-open Swing (sees the decision day's
EOD) and `asof_swing_intraday(d)` with `decision_time` (sees only the previous session). The ban file dated D lists
the securities banned FOR trade date D and is published the evening before. Option strategies that hold across
days use `OptionSpec` + `run_option_specs` (EOD-close fills, haircut max(₹0.05, 1 %) per leg per fill,
`options_replay.leg_cost` fees, expiry settlement at the expiring future's final settlement price, the same
`protocol.evaluate` split and FINAL ledger as `options_vrp.py`; callbacks run under `cutoff(today)`, so reading
today's EOD raises `LookAheadError`). Tests: `<scratchpad>/nse/tests_nsedata.py`. Full API: the module docstring.

    python -m stratlab.nsedata --check
    python -m stratlab.nsedata --data <bars3y> --out <scratch> --specs stratlab.my_module:SPECS [--screen-only]

## Old era 2012-01-01 .. 2023-11-09: daily-only out-of-sample round (`data_daily.py`, `oldera_fo.py`, `protocol_oos.py`)

The decade before the 3-year data was never used by any strategy. It is judged as ONE out-of-sample block, once per
strategy, with gates frozen in `protocol_oos.GATES` (fingerprint recorded with every judgement and in the
pre-registration file): net > 0 in H1 (2012-01-01..2017-12-31) AND H2 (2018-01-01..2023-11-09), PF >= 1.15 over the
era, >= 60 trades (positional) in EACH half, top symbol <= 35 % of net (skipped for one-instrument strategies); the
5 bps run (index 2.5 bps, options haircut x2.5) is reported, not a gate. protocol.py is not used for the verdict and
not changed; its constants are imported.

* Data: NSE bhavcopy (`<scratch>/nse_old`, read-only): official open / high / low / volume, close = NSE LAST traded
  price (the minute lab's daily close is the 15:29 minute close; `STRATLAB_DAILY_CLOSE=close` for the official close),
  back-adjusted for splits / bonuses / large demergers, renames merged (MUNDRAPORT -> ADANIPORTS ...), delisted names
  kept, special sessions (Muhurat, DR-test Saturdays, 2017-07-10) dropped with a causal turnover rule. Index futures run
  on NIFTY / BANKNIFTY official daily bars; `d.nifty` / `d.vix` are official closes.
* Universe: point-in-time monthly top 100 by 60-session turnover (`universe_monthly.csv`); a stock strategy may open a
  position only on a member decision day (`data_daily.gate`, the strategy function is wrapped, not edited).
* nsedata: `oldera_fo.use()` points it at `<scratch>/nse_old_view` (delivery, F&O bhavcopy with filled lots,
  participant OI, ban list, index spot); the availability rule (EOD of D usable from D+1's open) is unchanged.
* Eligible: positional @strategy specs without `decision_time` (minus three hard-coded-calendar specs) and the monthly /
  expiry-week real-price option specs of options_r4. `python -m stratlab.protocol_oos --list` prints both lists with
  reasons. Tests: `python -m stratlab.tests_oldera` (no catalogue strategy is run on the old era by the tests).

```
python -m stratlab.data_daily --build && python -m stratlab.oldera_fo --build-view
python -m stratlab.protocol_oos --preregister <scratch>/oldera_oos/PREREGISTERED_OOS.json
python -m stratlab.protocol_oos --out <scratch>/oldera_oos/run1 --prereg <scratch>/oldera_oos/PREREGISTERED_OOS.json --jobs 4
```

## Trials ledger and Deflated Sharpe Ratio (`trials.py`)

Every evaluation is a trial, and the best of N trials looks good by luck. `trials.py` keeps an **append-only** CSV of
every evaluation (default `<scratchpad>/stratlab_cache/trials_v2.csv`; `--ledger PATH` or `$STRATLAB_TRIALS` to move it;
refused inside the repository) and deflates Sharpe ratios by the number of trials.

    python -m stratlab.trials backfill [--scratch DIR] [--ledger PATH]   # rebuild from everything on disk (idempotent)
    python -m stratlab.trials summary                                    # rows / variants per era and stage
    python -m stratlab.trials report --out <scratch dir>                 # DSR of deployed + FINAL-confirmed strategies
    python -m stratlab.run ... --trials <ledger>                         # record a new run (or export STRATLAB_TRIALS)

Recording from `stratlab.run` is off unless `--trials` or `$STRATLAB_TRIALS` is set (so tests never write to the
real ledger): **export STRATLAB_TRIALS before research runs**, and run `backfill` after module runs (auction,
attention, options, ltr) so they are counted too.

* **Row** = one evaluation: strategy id, family (catalogue/module prefix), parameter fingerprint (`p:` + hash of the
  results.csv `params`, else the code fingerprint), era (`2023-26` / `2012-23`), protocol, stage (`screen`,
  `screen-smoke`, `final`, `oos`, `explore`), dataset, period, cost model, trades, net, PF, per-trade mean / std /
  Sharpe of the net, mean gross and mean cost per trade, and, when the trade list exists, the daily-P&L Sharpe,
  skewness and kurtosis over the period (DISCOVERY for screens, FINAL for FINAL looks, the era for the OOS), verdict,
  source file and trade file. Rows are never rewritten; re-appending the same evaluation is a no-op.
* **Backfill sources**: every `results.csv` of a protocol run under the scratchpad (stratlab.run, r3/r4 drivers,
  options modules, candlestick runs, reviewer neighbour runs; smoke runs marked `screen-smoke`), the module
  `report.json` files (auction, auction_r2, attention), the LTR basket, the one-look FINAL ledgers
  (`stratlab_cache/final_ledger.json`, `stratlab_cache/*/final_ledger.json`: authoritative FINAL verdicts), the
  2012-23 OOS ledger with its trade files, `pm_robust_edges`, `dd/screen_old.json`, and the exploratory effect
  measurements (round-1 facts, r2 intraday / swing / F&O effects, cross-sectional series, lead-lag, volatility
  measurements, event studies). Live-agent replays under `edge/` are not strategy trials and are not counted.
* **Counting N honestly**: every screened variant counts, passers or not. A variant is a strategy id plus each
  distinct parameter set run under it; a re-run of the same variant (smoke, another cost model or slippage, another
  dataset of the same era, a reviewer rebuild) is not a new trial. Exploratory effect tests count (they shaped what
  was pre-registered). As backfilled on 2026-11-14: **N = 1,115 variants on 2023-26 data** (640 protocol screens /
  FINALs + 475 exploratory effects), 38 FINAL looks, 178 variants judged on 2012-23.
* **PSR / DSR** (Bailey & Lopez de Prado 2012, 2014), non-annualised daily Sharpe, skewness, NON-excess kurtosis:
  `PSR(SR*) = Phi((SR - SR*) sqrt(T-1) / sqrt(1 - skew SR + (kurt-1)/4 SR^2))`;
  `SR0 = sqrt(V[SR]) ((1-g) Z^-1(1-1/N) + g Z^-1(1-1/(N e)))`, g = Euler-Mascheroni; `DSR = PSR(SR0)`.
  The unit tests reproduce the paper's worked example (SR 2.5 annual, T 1250, N 100, V 0.5, skew -3, kurt 10:
  SR0 = 0.1132 daily, DSR = 0.9004).
* **V[SR]** is the cross-trial variance of the DISCOVERY daily Sharpe over the 631 trials with a 3-year trade list.
  The naive sample variance (sd 0.59 per day) is dominated by high-turnover strategies whose cost drag gives SR of -1
  to -5 per day (not luck), so the headline uses the robust (1.4826 x MAD)^2 (sd 0.174 per day); the report also
  shows the naive value and the most lenient floor 1/(T-1) (noise only). FINAL rows use N = FINAL looks and the
  variance of the FINAL Sharpes.
* **Gate** (`deflation_gate`), replacing any "PF haircut": **0.5 x mean gross per trade - mean cost per trade > 0**
  (half the edge must still pay the full costs; gross already includes the modelled slippage) and/or **DSR > 0.95**.
  It is an additional report: protocol verdicts and ledgers are never changed by it.

Result of the 2026-11-14 report (`report --out`; DISCOVERY + FINAL trade lists on the 726-session calendar):

| strategy | SR ann. disc / final | DSR disc (robust V, N 1,115) | DSR disc (floor V) | DSR final (N 38) | haircut/trade disc / final (Rs) |
|---|---|---|---|---|---|
| AUC-FADE-3PCT (deployed) | 2.39 / 1.13 | 0.00 | 0.56 | 0.10 | +189 / +57 |
| SWING-ATTN-ON-3X (deployed) | 2.49 / 0.71 | 0.00 | 0.64 | 0.06 | +85 / **-24** |
| VRP-STRANGLE-016-WINGED (deployed) | 0.59 / 0.63 | 0.00 | 0.01 | 0.07 | +571 / +615 |
| AUC-FADE-2PCT | 2.00 / 0.44 | 0.00 | 0.30 | 0.03 | +76 / **-20** |
| best other FINAL-confirmed (r3_options_wc_fri) | 1.40 / 3.16 | 0.00 | 0.15 | 0.71 | +626 / +915 |

No deployed or confirmed strategy reaches DSR 0.95 under any variance estimate, even the noise-only floor: with
about 1,100 variants tried, a 2-2.5 annual Sharpe on two years of DISCOVERY is about what the luckiest zero-edge
variant would show. SWING-ATTN-ON-3X, AUC-FADE-2PCT and r4_delivery_brk_fake_short fail the haircut on their FINAL
trades. `cs_shooting_star_d` was confirmed only on the 12-month dataset; the 3-year run never opened its FINAL, so it
has no 3-year DSR. Full table: `<scratchpad>/stratlab_cache/dsr_v2/dsr_report.txt`.

**Corrected re-backfill (`trials_v2.csv`, `dsr_v2/`, 2026-11-14).** The first ledger (`trials.csv`, report in `dsr/`)
matched a FINAL ledger entry to a trade file by strategy id alone. Its bars3y FINAL row for `cs_gravestone_doji_d`
(ledger: 32 trades, net 1,146.67) took Sharpe and moments from the 12-month run's file
(`candles/out_12m`, 22 trades, net -1,585.39 in the window): daily SR -0.0078 instead of 0.0052, and that wrong value
entered V[SR] of the FINAL looks. A FINAL row now borrows statistics only from a same-dataset, same-cost-model file
that reproduces the ledger's FINAL trades and net (else the Sharpe columns stay blank), and `report` applies the same
check to every file it uses. No backtest was re-run and no verdict changed; `trials.csv` is kept unchanged as the
superseded first version. Effect: V[SR] FINAL 0.004631 -> 0.004629 (34 looks); N, the DISCOVERY V[SR] and every
DSR / PSR in the table above are unchanged at the precision shown. Tests: `tests_trials.py`, run by
`test_pipeline.py` (section "STRATLAB: trials ledger and Deflated Sharpe Ratio").

## Slow factor momentum — pre-registered portfolio test (`factor_momentum.py`, `prereg/momentum_v1.json`)

A portfolio backtest (not the per-trade protocol): cross-sectional momentum is a slow rebalance strategy, so the
trade-level PF / trade-count gates do not fit it. Its own low-frequency gates were fixed in `prereg/momentum_v1.json`
(sha256 `2a7cf8bf595cd9f7289aa55d92be0d904b03e1897bf4063f3c62bf2fc372a202`, checked by the CLI) BEFORE any result was
computed. One look only: the ledger (`momentum_oos/ledger.json` in the scratchpad) refuses to re-run a variant.

    python -m stratlab.factor_momentum --prereg <PREREGISTERED_MOMENTUM.json> --old <nse_old> --bars <bars3y> \
        --nse <nse> --ledger <ledger.json> --out <scratch dir>

* Signal at the close of each month's last session: mean of cross-sectional z-scores (clipped ±3) of
  vol-adjusted 12-1 and 6-1 momentum (`C[t-21]/C[t-252]-1`, `C[t-21]/C[t-126]-1`, ÷ annualised std of 252 daily
  log returns). Execution at the **next session's OPEN**; the signal never sees that session.
* Top 20 equal weight (cap 10 %), 0.5 %-of-equity no-trade band, CNC costs from `costs.py` + 10 bps per side,
  ₹10 L, cash at 0 %. Trend filter = NIFTY < SMA200 → cash, checked monthly. A buy or re-weight of a stock with no
  trade on the execution day is skipped; a full exit of one is done at its last close (counted as a stale exit).
* Era 1: nse_old 2013-01-31 → 2023-11-09, point-in-time top-100 by turnover, delisted names kept (H1 starts 2013,
  not 2012-02: the 12-1 signal needs 253 sessions of data that begins 2012-01-02; stated in the prereg).
  Era 2: bars3y 2023-11-30 → 2026-11-09 on the 97 current NIFTY-100 names — **survivorship-biased**; lookback history
  spliced from nse_old through the 2023-11-10 bhavcopy return.
* Gates: each half CAGR > 0 and CAGR > NIFTY (price index; the strategy is price-only too), era-1 Sharpe ≥ 0.5;
  era 2 CAGR > 0 and > NIFTY. PASS needs all.

Results (run once, 2026-11-14; all net of costs; % per year; excess = CAGR − NIFTY CAGR):

| variant | era1 H1 CAGR / NIFTY / excess | H2 CAGR / NIFTY / excess | era1 Sharpe | era1 maxDD | turnover / cost drag (era1) | era2 CAGR / NIFTY / excess (biased) | verdict |
|---|---|---|---|---|---|---|---|
| monthly, no filter | 16.7 / 12.0 / **+4.7** | 10.3 / 11.0 / **−0.7** | 0.68 | −46 % | 407 % / 1.80 % | 4.8 / 7.7 / −2.9 | FAIL (H2, era2 excess) |
| monthly, filter | 11.1 / 12.0 / −0.9 | 5.8 / 11.0 / −5.2 | 0.54 | −34 % | 388 % / 1.73 % | 12.2 / 7.7 / +4.5 | FAIL (H1, H2 excess) |
| semi-annual, no filter | 11.6 / 12.0 / −0.4 | 9.7 / 11.0 / −1.3 | 0.58 | −42 % | 155 % / 0.69 % | 9.3 / 7.7 / +1.6 | FAIL (H1, H2 excess) |
| semi-annual, filter | 6.4 / 12.0 / −5.6 | 4.3 / 11.0 / −6.7 | 0.39 | −33 % | 214 % / 0.96 % | 14.3 / 7.7 / +6.6 | FAIL (H1, H2 excess, Sharpe) |

**0 of 4 pass.** Monthly momentum beat NIFTY only in 2013-2017 and lost to it after costs in 2018-2023 and on the
(favourably biased) 2023-26 names; the 200-day filter cost more in whipsaws (13 cash switches) than it saved in
2013-2023. Do not re-run with other lookbacks / holdings / filters on these data: that would be tuning on the result.
The live `momentum_rotation.py` (top 5, raw 12-1/6-1, 200-DMA) was never pre-registered or tested out of sample, and
its built-in `backtest()` charges no costs (it sets `held = pick` before computing the cost of the change).

## Round 5 — Zerodha / Kite ecosystem strategies, tested as published (`catalog/r5_kite.py`, `kite_*.py`)

69 published strategies (Zerodha Varsity, Streak examples, AlgoTest Pine templates, Tradetron blog builds,
AlgoBulls pyalgostrategypool, Sensibull templates; source list `<scratchpad>/kite/strategies_69.json`) were classified,
pre-registered and run once, with no tuning. Pre-registration `prereg/kite_ecosystem_v1.json` (sha256
`6f2c38290cc43860bfb5d65008b754ae20b0e4d733dbb2d767e461002b88d4d6`, byte-identical to `<scratchpad>/PREREGISTERED_KITE.json`,
written before any result) lists every id, its published rule, the ONE plain default for each unspecified field, the eras
and the code fingerprint. `prereg/kite_ecosystem_v1_amendment1.json` (sha256 `d4ae96fd…dcb842`) adds
`r5_kite_vars_calendar_v2`: v1 crashed (missing contract row) on its first 2012-23 run and is recorded INVALID with no
P&L; the fix runs under a new id, so the one-look ledger keeps v1's INVALID entry.

Code: `kite_ind.py` (platform formulas: Pine RMA RSI/ATR/DMI/SAR/stoch, TA-Lib AROON/STOCH, AlgoTest's Ichimoku and
non-ratcheting Supertrend as coded, Tradetron double Supertrend), `kite_tf.py` (continuous Kite-aligned candles with
the short last candle kept, and the entry/exit "machine"), `kite_stats.py` (ADF = statsmodels adfuller defaults,
checked against statsmodels), `kite_spreads.py` (Varsity PTM2 / PTM1 / calendar spread), `kite_options.py`
(Tradetron iron condor and double-Supertrend debit spreads on real EOD option prices), `kite_momentum.py` (Varsity
momentum on factor_momentum's data, costs and gates). `sim.py` gained OPTIONAL fields that change nothing for
existing strategies: `Entry.exit_sig` (exit by rule at the next bar's open; an opposite entry on the same bar fills at
that open = stop-and-reverse; audited for look-ahead like the entry), `Entry.bracket` (stop / target measured from
the fill, points or %), `Swing.exit_fn` (positional exit by rule at the next open, same-open reversal) and
`Swing.lots`. Tests: `tests_kite.py` (hand-computed Aroon, Williams %R, stochastic, Ichimoku, Supertrend / double
Supertrend; ADF vs statsmodels; the simulator extensions; dense look-ahead audit of all 29 intraday specs).

Classification (a = as published 23, b = one plain default needed 26, c = not testable 20):
(c) the 14 Sensibull templates (legs only: no entry time, no exit), AlgoTest 9:20 straddle, Tradetron Rs150 strangle
and 09:30 straddle, AlgoBulls strangle (all need INTRADAY option prices; the lab has EOD bhavcopy only and a flat-vol
Black-Scholes model would test the model), AlgoTest VWAP (as coded `ta.vwap(volume, 14)` makes the line the bar's
volume) and VWAP+EMA+volume (both need NIFTY futures volume; the lab's NIFTY bars are the spot index, volume 0).
Dedup: no strategy in `trials_v2.csv` has identical rules; same-family neighbours with other parameters were tested
anew. Eras: intraday on 2023-26 only; daily rules on 2023-26 (protocol.py) and once on 2012-23 (protocol_oos);
Varsity candlesticks enter at the 15:20 bar on 2023-26 and, as `*_nxo` twins (next-open fill), on 2012-23.

Results (2023-26: DISCOVERY trades / PF / net Rs, verdict; 2012-23: era trades / PF / net Rs, verdict;
"gross" = before costs, after the modelled slippage):

| source | id (r5_kite_…) | cls | 2023-26 disc tr / PF / net | verdict | 2012-23 tr / PF / net | verdict | main reason |
|---|---|---|---|---|---|---|---|
| Streak | streak_rsi30m_short | b | 8,958 / 0.71 / -9.7 L | FAIL | – | – | gross < 0 (-26/trade) and costs 82/trade |
| Streak | streak_orb5_long | b | 31,224 / 0.78 / -32.4 L | FAIL | – | – | gross < 0 |
| Streak | streak_4th30m_long | b | 26,326 / 0.74 / -23.0 L | FAIL | – | – | gross ~0 (-5/trade), killed by costs |
| Streak | streak_st7_long | b | 344,457 / 0.40 / -449 L | FAIL | – | – | gross < 0, ~680 trades/day over 97 stocks |
| Varsity | vars_momentum12 | b | era2 CAGR 7.9 % vs NIFTY 7.7 % | PASS (gates) | 2013-23 CAGR 13.7 % vs 11.5 %, Sharpe 0.64 | PASS (gates) | margins: H2 excess +0.09 %/yr, era2 +0.22 %/yr (biased), maxDD -57 % |
| Varsity | vars_ptm2 | b | 635 / 0.66 / -5.7 L | FAIL | 3,450 / 0.86 / -14.8 L | FAIL | gross < 0 both eras |
| Varsity | vars_ptm1 | b | 2,576 / 0.85 / -6.8 L | FAIL | 14,946 / 0.89 / -39.5 L | FAIL | gross ~0 (PF 0.97-0.98), killed by costs |
| Varsity | vars_calendar_v2 | b | 1,827 / 0.05 / -5.1 L | FAIL | 11,583 / 0.24 / -31.6 L | FAIL | gross < 0 and costs 3x gross |
| Varsity | vars_px_ema50 | a | 1,776 / 1.02 / +0.8 L | FAIL | 11,376 / 1.26 / +59.6 L | CONFIRMED | 2023-26: H2 < 0, PAYTM 175 % of net; 2012-23: = NIFTY drift (excess vs NIFTY < 0 both halves) |
| Varsity | vars_ema9_21 | b | 1,003 / 1.07 / +1.9 L | FAIL | 6,014 / 1.53 / +99.6 L | CONFIRMED | 2023-26: H2 < 0, PAYTM 52 %; 2012-23 excess vs NIFTY +17 L H1, -0.4 L H2 |
| Varsity | vars_ema25_50 | b | 344 / 1.33 / +4.0 L | FAIL | 2,225 / 1.70 / +78.6 L | CONFIRMED | 2023-26: PAYTM 38 % of net; 2012-23 excess vs NIFTY +13 L H1, -10 L H2 |
| Varsity | vars_ema50_100 | b | 155 / 1.19 / +1.3 L | FAIL | 1,024 / 1.97 / +70.3 L | CONFIRMED | 2023-26: H1 < 0, PAYTM 62 %; 2012-23 excess vs NIFTY +5 L H1, -3 L H2 |
| Varsity | vars_ema100_200 | b | 57 / 1.56 / +1.5 L | FAIL | 454 / 2.86 / +66.1 L | CONFIRMED | 2023-26: 57 trades, PAYTM 58 %; 2012-23 excess vs NIFTY -6.7 L H1, +11.8 L H2 |
| Varsity | vars_ema15x30_1m | b | 267,610 / 0.41 / -339 L | FAIL | – | – | gross < 0, ~530 trades/day over 97 stocks |
| Varsity | vars_ema5x10_1m | b | 836,887 / 0.24 / -1,090 L | FAIL | – | – | gross < 0 |
| Varsity | vars_macd_d | b | 3,798 / 0.64 / -37.2 L | FAIL | 22,171 / 1.07 / +45.2 L | FAIL | 2023-26 gross < 0; 2012-23 H1 < 0, PF 1.07 |
| Varsity | vars_bb20_d | b | 2,687 / 1.20 / +10.5 L | FAIL | 15,706 / 0.84 / -74.7 L | FAIL | failed H1 (2023-26); gross < 0 over 2012-23 |
| Varsity | vars_cs_bull_marubozu (/_nxo) | b | 765 / 1.11 / +1.2 L | FAIL | 2,819 / 1.05 / +2.7 L | FAIL | PF < 1.15; small gross edge mostly eaten by costs |
| Varsity | vars_cs_bear_marubozu | b | 1,160 / 0.73 / -5.0 L | FAIL | 3,465 / 0.93 / -4.0 L | FAIL | gross < 0 |
| Varsity | vars_cs_hammer | b | 643 / 1.04 / +0.3 L | FAIL | 2,299 / 1.07 / +2.1 L | FAIL | H2 < 0, GAIL 78 % (2023-26); PF 1.07 (2012-23) |
| Varsity | vars_cs_hanging_man | b | 855 / 0.38 / -2.0 L | FAIL | 1,881 / 0.47 / -5.8 L | FAIL | gross < 0 |
| Varsity | vars_cs_shooting_star | b | 710 / 0.75 / -2.0 L | FAIL | 3,070 / 0.92 / -2.7 L | FAIL | gross < 0 (2023-26) / ~0 (2012-23) |
| Varsity | vars_cs_bull_engulfing | b | 703 / 1.24 / +2.6 L; FINAL 264 / 1.02 / +0.1 L | REJECTED ON FINAL | 4,244 / 1.05 / +5.9 L | FAIL | the only screen pass of the round; FINAL PF 1.02 < 1.05; 2012-23 PF 1.05 |
| Varsity | vars_cs_bear_engulfing | b | 1,330 / 0.72 / -7.3 L | FAIL | 7,765 / 0.95 / -7.8 L | FAIL | gross < 0 / ~0 |
| Varsity | vars_cs_piercing | b | 518 / 1.04 / +0.2 L | FAIL | 2,753 / 0.98 / -1.1 L | FAIL | VEDL > 100 % of net; killed by costs (2012-23) |
| Varsity | vars_cs_dark_cloud | b | 889 / 0.78 / -2.6 L | FAIL | 5,189 / 0.92 / -6.4 L | FAIL | gross < 0 |
| Varsity | vars_cs_morning_star | b | 30 / 0.92 / -0.1 L | FAIL | 136 / 1.08 / +0.4 L | FAIL | too few trades; AUROPHARMA 59 % |
| Varsity | vars_cs_evening_star | b | 20 / 1.63 / +0.2 L | FAIL | 140 / 0.71 / -1.6 L | FAIL | too few trades (2023-26); gross < 0 (2012-23) |
| AlgoTest | at_rsi / at_supertrend / at_sma9_21 / at_macd / at_bollinger / at_adx / at_psar / at_stoch / at_ema_macd_rsi / at_williams / at_ichimoku | a | 90-3,350 trades each, PF 0.35-0.57, -0.9 L to -43 L each (11 total -255 L) | FAIL (all 11) | – | – | gross < 0 for all 11 (gross PF 0.72-0.95) before ~Rs 920 cost per NIFTY lot round trip |
| Tradetron | tt_ic_wed | b | 105 / 0.83 / -0.5 L | FAIL | not run (no weeklies before 2019) | – | gross < 0 (EOD-only stop, EOD fills) |
| Tradetron | tt_dst_nifty | a | 28 / 0.47 / -5.2 L | FAIL | 132 / 1.76 / +11.2 L | CONFIRMED | 2023-26 gross < 0, 28 trades; 2012-23 = NIFTY exposure (holding 2 lots of NIFTY made Rs 19 L) |
| Tradetron | tt_dst_spreads | b | 41 / 0.62 / -1.3 L | FAIL | 199 / 1.08 / +0.6 L | FAIL | gross < 0 (2023-26); H1 < 0, PF 1.08 (2012-23) |
| Tradetron | tt_orb15_stock (stock proxy) | b | 58,903 / 0.78 / -53.4 L | FAIL | – | – | gross ~0 (-8.5/trade), killed by costs |
| Tradetron | tt_orb30_list (US template mapped) | b | 45,098 / 0.81 / -38.2 L | FAIL | – | – | gross ~0 (-2.5/trade; +H1, -H2), killed by costs |
| AlgoBulls | ab_ema12_20 / ab_macd_5m / ab_aroon_15m / ab_bollinger_60m / ab_reverse_rsi_60m / ab_stoch_60m / ab_vwap_60m / ab_atr_trend_60m / ab_inverse_ema_60m / ab_rsi_macd_60m | a | 10,517-268,536 trades each, PF 0.53-0.76 (10 total -830 L) | FAIL (all 10) | – | – | gross < 0 for all 10 (-6 to -74 Rs/trade) before 82 Rs/trade costs |

(L = lakh = Rs 1,00,000; every stock trade is Rs 1 lakh notional, so -Rs 9.7 L over 8,958 trades is -0.11 % per trade.)

Verdict: **nothing tested here is a deployable edge.** On 2023-26, 53 strategies: 1 screen pass (Varsity bullish
engulfing at 15:20), rejected on FINAL; 0 confirmed. The 25 intraday platform templates (AlgoTest, AlgoBulls,
Streak) lose money BEFORE costs. The opening-range and pair / calendar trades are about flat before costs and lose
to them. On 2012-23, 24 judgements: 6 OOS-CONFIRMED, all long-only daily trend rules (Varsity price > EMA50 and the
four EMA crossovers) or the long-biased double Supertrend. Each one earned about what holding NIFTY earned over the same
holding periods (net minus the same-period NIFTY move is negative in at least one half for every one), and all six
failed the 2023-26 screen. These confirmations are exposure to a decade in which NIFTY rose 4x. They are not an
edge: the protocol_oos gates test net > 0, not net > the index. Varsity momentum
passes its factor_momentum gates, but it is the 5th correlated momentum variant on these data (4 failed), and its
2018-23 and 2023-26 margins over NIFTY are +0.09 % and +0.22 % a year (the latter on today's NIFTY-100 names, biased in
its favour), with a 57 % drawdown. At most it is a paper-trading candidate, not evidence of an edge. New trials:
54 variants on 2023-26 (N 1,115 -> 1,169) and 25 on 2012-23 (N 178 -> 203), 66 new ids in all (including the INVALID
calendar v1). Outputs: `<scratchpad>/kite/` (run3y, spreads3y, opts3y, momentum) and `<scratchpad>/oldera_oos/run1/bkite_*`.

## Round 6 — GitHub-sourced strategies, re-implemented from source with known bugs fixed (`catalog/r6_github.py`, `r6_portfolio.py`, `r6_ind.py`)

A 20-entry shortlist taken from public GitHub code (QuantConnect Tutorials strategy library, letianzj QuantResearch,
freqtrade-strategies, pysystemtrade, Screeni-py, openalgo, marketcalls, quantrocket qmom, rkohli3 TSMOM,
NiftyOnTheMove, the 101 Formulaic Alphas paper) was classified, pre-registered and run once, with no tuning. The
downloaded repositories (`<scratchpad>/dl/github`) were only read, never executed. Pre-registration
`prereg/github_v1.json` (sha256 `b283cda39dd84f6afd65175bc34f7901d9de542bc4bda079848297ec2ecfaf09`, byte-identical
to `<scratchpad>/PREREGISTERED_GITHUB.json`, committed before any result) lists for every entry the source files with
their sha256, the rule as coded, the bug fixes, the ONE plain default for each unspecified field, the eras and the
code fingerprint of every id.

Code: `r6_ind.py` (TD Sequential counts / perfection, Williams VIX Fix, Dynamic Breakout II adaptive lookback,
pysystemtrade EWMAC / robust vol / mr_wings, Meeus ch. 49 moon phases, ghost-trader replay, R-Breaker pivots, Clenow
regression momentum, residual-momentum score, Alpha101 #9 #12 #33 #54 #101), `catalog/r6_github.py` (14 per-symbol
specs through the unchanged `sim.py` / `protocol.py` / `protocol_oos.py`), `r6_portfolio.py` (15 portfolio specs:
`factor_momentum.backtest` generalised to negative weights = stock futures, and to weekly / quarterly / Q-NOV / daily
schedules; same data, 10 bps slippage, costs and `factor_momentum.judge` gates; one-look ledger
`<scratchpad>/github/portfolio_ledger.json`; `--january` writes the January-barometer report). `sim.py` was not changed.
Tests: `tests_r6.py` (hand-computed values for every formula, Meeus' worked examples 49.a / 49.b and the USNO January
2024 phases, the portfolio engine's next-open / long-short / cost / stale-name arithmetic, a noise-after-t causality
check of every weights function, the dense look-ahead audit of the intraday specs), run by `test_pipeline.py`.

Bugs fixed (all listed in the pre-registration): marketcalls' two-quarter lag (`winner.shift(1)`); QC 66's inverted
turnover, its `int(len x 0.01)` empty tail and 2-month / overwriting tranches; QC 125's absolute-price ranking (269 is
ranked by % return); freqtrade TD perfection "below either" bar-6/7 low (DeMark: below both); ghost_trader's
Donchian window that includes the current bar; letianzj Dynamic Breakout II's `(v_t/v_y)/v_t`, its short branch that
buys and its commented-out breakout test (QC's vol window indexing too); NiftyOnTheMove's up-only gap filter, EMAs
named SMA and buys outside the top 20 %; rkohli3's sum of a long mean and a short mean; yli188's Alpha101 code is not
used (its `rank()` ranks along time over the whole sample and it back-fills prices).

Classification (a = as coded after the fixes 5, b = one plain default 13, c = not testable 2): (c) QC 24 liquidity
premium (needs shares outstanding and market cap; the lab's top-100-by-turnover universe is the opposite end of the
population the effect lives in, so a proxy would be a different strategy) and the NIFTY-BANKNIFTY IV-spread z-score
(the source never trades options: its P&L is position x the IV-spread LEVEL x tte^0.7 at 1-minute frequency with a
full-sample grid search; a faithful EOD test needs a new two-underlying straddle simulator with vega matching and
expiry rolls). Defaults that matter: residual momentum on a NIFTY-only regression (no Fama-French data for India);
momentum x turnover uses a LABELLED proxy (decision-day traded value / 60-day mean) because shares outstanding are
missing; mr_wings is the binary +-1 lot version; WVF exits after a fixed 5-session hold; the 5-EMA trigger is valid
only in the candle after the alert candle. Dedup: no identical rule in `trials_v2.csv` or round 5; residual / QMOM /
Clenow / turnover momentum and per-stock TSMOM are CORRELATED variants of momentum families tested before, counted as
new trials anyway. Eras: intraday on 2023-26 only; daily per-symbol rules on 2023-26 (protocol.py) and once on 2012-23
(protocol_oos); portfolios on era 1 = 2013-23 (point-in-time top-100, delisted kept) and era 2 = 2023-26
(survivorship-biased), judged by the factor_momentum gates.

Results (per-symbol: trades / PF / net Rs, "gross" = before costs after modelled slippage; portfolios: CAGR / excess
over NIFTY per year; "excess vs NIFTY" for trade-level rules = net minus the same notional's NIFTY move over each
trade, per half; L = Rs 1 lakh):

| source | id (r6_gh_…) | cls | 2023-26 | verdict | 2012-23 | verdict | main reason |
|---|---|---|---|---|---|---|---|
| letianzj r_breaker.py | rbreaker_nifty | b | 146 / 0.88 / -0.5 L (gross +0.9 L) | FAIL | – | – | killed by costs (gross PF 1.28, cost 1.3 L); H2 < 0 |
| letianzj r_breaker.py | rbreaker_banknifty | b | 87 / 1.08 / +0.2 L (gross +1.0 L) | FAIL | – | – | one half (H2 < 0), PF < 1.15, costs take 80 % of gross |
| Screeni-py + original | ema5_nifty | b | 1,921 / 0.38 / -25.4 L | FAIL | – | – | no edge before costs (gross -7.7 L, gross PF 0.72) |
| Screeni-py + original | ema5_banknifty | b | 1,750 / 0.46 / -22.7 L | FAIL | – | – | no edge before costs (gross -6.5 L, gross PF 0.78) |
| freqtrade TDSequential | td9_stock | a | 429 / 0.95 / -0.7 L | FAIL | 4,179 / 1.17 / +27.3 L (H1 +15.2, H2 +12.1) | OOS-CONFIRMED | 2023-26 gross ~0 killed by costs; 2012-23 pass trails NIFTY (excess -3.4 L H1, -4.4 L H2) |
| freqtrade TDSequential | td9_nifty | a | 1 / 0.00 / -0.8 L | FAIL | 22 / 1.97 / +2.3 L | OOS-FAILED | too few trades (13 / 9 per half) |
| freqtrade TDSequential | td9_banknifty | a | 1 / 0.00 / -0.1 L | FAIL | 27 / 0.93 / -0.4 L | OOS-FAILED | too few trades; gross < 0 |
| pysystemtrade mr_wings | mrwings_nifty | b | 3 / 1.69 / +0.1 L | FAIL | 4 / 0.14 / -0.3 L | OOS-FAILED | too few trades (3-sd wings fire ~once a year) |
| pysystemtrade mr_wings | mrwings_banknifty | b | 1 / inf / +0.03 L | FAIL | 4 / 0.21 / -0.7 L | OOS-FAILED | too few trades |
| openalgo WVF | wvf_stock | b | 2,255 / 1.30 / +9.0 L; FINAL 1,293 / 0.88 / -2.3 L | REJECTED ON FINAL | 15,379 / 0.91 / -29.6 L (H1 +2.2, H2 -31.8) | OOS-FAILED | the round's only screen pass: its DISCOVERY profit was index drift (excess vs NIFTY +0.6 L H1, -2.2 L H2), FINAL < 0; 2012-23 gross PF 1.02 killed by costs, trails NIFTY |
| QC 61 lunar | lunar_nifty | a | 53 / 0.45 / -6.3 L | FAIL | 294 / 0.98 / -0.4 L | OOS-FAILED | no edge before costs (2023-26 gross < 0; 2012-23 gross PF 1.04, H2 < 0) |
| letianzj ghost_trader | ghost_nifty | a | 8 / 0.16 / -1.9 L | FAIL | 31 / 0.81 / -0.8 L | OOS-FAILED | too few trades; gross < 0 |
| letianzj ghost_trader | ghost_banknifty | a | 6 / 2.40 / +1.8 L | FAIL | 31 / 0.71 / -2.0 L | OOS-FAILED | too few trades; gross < 0 over 2012-23 |
| QC 04 / letianzj DBO II | dbo2_stock | a | 1,202 / 0.92 / -3.3 L | FAIL | 7,567 / 1.08 / +24.4 L (H1 +0.7, H2 +23.7) | OOS-FAILED | no edge before costs on 2023-26 (H2 < 0); PF 1.08 over 2012-23 and trails NIFTY (excess -38.9 L / -17.7 L) |
| QC 136 | resid_mom_long | b | CAGR 4.2 % / excess -3.5 | FAIL | H1 (2015-17) 15.3 % / +9.0, H2 21.8 % / +10.8, Sharpe 0.86 | (era1 gates met) | fails only era 2: trails NIFTY 2023-26 on the (favourably biased) current names |
| QC 136 | resid_mom_ls | b | -6.6 % / -14.3 | FAIL | 3.1 % / -3.2, 11.9 % / +1.0, Sharpe 0.94 | FAIL | one half (H1 excess < 0), era 2 < 0; market-neutral book cannot beat NIFTY's CAGR |
| QC 269 | season_1y | b | -3.3 % / -11.0 | FAIL | -5.3 % / -17.3, 0.2 % / -10.8 | FAIL | no edge before costs (gross -2.0 % / +3.8 % per half) |
| Heston-Sadka (QC 269 ext.) | season_5y | b | 6.7 % / -1.0 | FAIL | -4.8 % / -16.8, -6.7 % / -17.7 | FAIL | no edge before costs over 2013-23 |
| QC 66 | mom_turnover | b | -13.2 % / -20.9 | FAIL | 0.8 % / -11.2, 7.9 % / -3.1 | FAIL | gross 3.1 % / 10.4 %, era 2 gross -11 %: no stable edge, trails NIFTY |
| marketcalls / QC 33 | etf_switch | a | 8.7 % / +3.3 | FAIL | 4.5 % / -9.4, 8.9 % / -2.1, Sharpe 0.67 | FAIL | trails NIFTY in both 2013-23 halves (75 % invested as coded) |
| 101 Formulaic Alphas | alpha101_a101 | b | -51.9 % / -59.6 | FAIL | -55.5 % / -67.5, -48.5 % / -59.5 | FAIL | no edge before costs (gross -15 % / -1 % per half) and ~64 %/yr cost drag |
| 101 Formulaic Alphas | alpha101_a33 | b | -40.6 % / -48.3 | FAIL | -36.7 % / -48.7, -46.3 % / -57.3 | FAIL | killed by costs (gross +20 % / +1 % per half, +12 % era 2; cost drag 64 %/yr) |
| 101 Formulaic Alphas | alpha101_a12 | b | -40.1 % / -47.8 | FAIL | -39.0 % / -51.0, -39.2 % / -50.1 | FAIL | killed by costs (gross +5.8 % / +4.8 %; drag 55 %/yr) |
| 101 Formulaic Alphas | alpha101_a54 | b | -47.8 % / -55.5 | FAIL | -46.7 % / -58.7, -46.4 % / -57.4 | FAIL | no edge before costs (gross ~0) |
| 101 Formulaic Alphas | alpha101_a9 | b | -41.3 % / -49.0 | FAIL | -41.0 % / -53.1, -42.8 % / -53.7 | FAIL | killed by costs (gross +4.5 % / +1.1 %; drag 57 %/yr) |
| shortlist (no source file) | ibs_xs | b | -47.1 % / -54.8 | FAIL | -43.4 % / -55.4, -46.7 % / -57.6 | FAIL | killed by costs (gross +6.9 % / +1.6 %; drag 64 %/yr) |
| NiftyOnTheMove (Clenow) | clenow | b | 6.3 % / -1.2 | FAIL | 2.5 % / -9.7, 3.3 % / -7.6, Sharpe 0.48 | FAIL | trails NIFTY (only ~25 % invested: top 20 % of a 100-name ranking leaves few buys) |
| quantrocket qmom | qmom | b | 11.5 % / +3.8 | FAIL | 12.6 % / -0.9, 6.5 % / -4.5, Sharpe 0.46 | FAIL | trails NIFTY in both 2013-23 halves (gross excess +0.5 / -3.3) |
| rkohli3 TSMOM | tsmom_stock | b | 0.7 % / -7.0 | FAIL | 8.2 % / -3.8, 0.8 % / -10.2, Sharpe 0.36 | FAIL | trails NIFTY; gross 9.2 % / 1.9 % per half |
| QC 24 | (liquidity premium) | c | – | not run | – | – | needs shares outstanding / market cap; wrong population |
| beastytitan18 | (IV-spread z-score) | c | – | not run | – | – | source P&L is not a traded instrument; needs a new 2-underlying option simulator |
| paperswithbacktest | january_barometer | b | report only | – | NIFTY 2013-2026: 1.12x vs 4.24x buy-and-hold, out of the market 10 of 14 years | no verdict | ~14 observations: no power; January was negative in 10 of 14 years and Feb-Dec rose in 9 of those 10 |

Gross figures for the portfolios come from a zero-cost re-run made AFTER the one-look verdicts were written
(`<scratchpad>/github/portfolio_gross_diag.json`, diagnostic only: not a trial, ledger untouched).

Verdict: **nothing here is a deployable edge.** 2023-26: 14 per-symbol strategies, 1 screen pass (WVF on stocks),
REJECTED ON FINAL; 0 confirmed; 15 portfolios, 0 pass the factor_momentum gates. 2012-23: 10 daily per-symbol
judgements, 1 OOS-CONFIRMED (TD Sequential 9 on stocks, PF 1.17) - and it earned less than holding NIFTY over the same
trades in both halves, after failing the 2023-26 screen (gross ~0 there): exposure, not an edge. The closest portfolio
is residual momentum, long-only: +9 / +11 %/yr over NIFTY in 2015-17 / 2018-23 with Sharpe 0.86, but -3.5 %/yr on
2023-26 (even with today's names); it is one of four momentum variants in this round on top of five tested before, and its long-short version (the
actual factor) fails, so the long-only result is mostly a momentum-tilted long book. Every daily cross-sectional
book (5 alphas, IBS) turns over ~190x a year: a few % of gross edge against ~55-65 %/yr of costs. The intraday
templates (5-EMA, R-Breaker) have no edge before costs or lose it to costs. New trials: 29 variants on 2023-26
(N 1,169 -> 1,198; 1 new FINAL look) and 26 on 2012-23 (N 203 -> 229, incl. the January-barometer report row); 30 new
ids in the pre-registration. Outputs: `<scratchpad>/github/` (run3y_swing, run3y_futures, portfolio, january,
aggregate.json, scripts/) and `<scratchpad>/oldera_oos/run1/github_daily`.

## Round 7 — second GitHub shortlist, OHLCV-only items (`catalog/r7_github2.py`, `r7_portfolio.py`, `r7_tps.py`, `r7_ind.py`)

The second GitHub shortlist (`<scratchpad>/github_round2_shortlist.md`; sources in `<scratchpad>/dl/github2`, only read,
never executed). Each strategy was rebuilt from the source code or the book rules with the known bugs fixed,
pre-registered, and run once with no tuning. The pre-registration is `prereg/github_v2.json` (sha256
`1745b3e24092cae2d9ea7009ee6380c322a5fcc67a3f35617a8e6e840e2b7b92`). It is byte-identical to
`<scratchpad>/PREREGISTERED_GITHUB2.json` and was committed before any result existed. For every entry it lists the
source files with their sha256, the rule as coded, the bug fixes, the one plain default for each unspecified field, the
eras, the fingerprint of every id and the sha256 of the round's code files. 30 ids in all.

Not in this round (each needs a data build or an ML pipeline):
1. Promoter stake increase ≥ 0.5 pp: needs point-in-time NSE shareholding-pattern filings with their filing dates.
2. Promoter open-market buys (PIT / SAST): needs an archive of the disclosures with acquirer category, mode and
   dissemination time.
3. Earnings-day gap momentum: needs real results-announcement dates and times; the lab only has the gap + volume proxy
   it already tested.
6. Qlib Alpha158 + LightGBM: needs NIFTY-500 point-in-time membership and a walk-forward retraining pipeline with its
   own trial accounting.

Code:
* `r7_ind.py` holds the formulas: Clenow counter-trend, Turtle Soup / +1, Smash Day with MT5's SMA ATR, Holy Grail
  gate, Double 7s, Bensdorp ATR% / MRL / MRS, the TPS state machine, the 80-20 setup and the intraday stop re-entry.
* `catalog/r7_github2.py` has 25 per-symbol specs, run through `sim.py` / `protocol.py` / `protocol_oos.py`.
* `r7_tps.py` is Connors TPS. Its scale-in (10/20/30/40 %) gets its own trade simulator. One trade = one scale-in
  sequence, judged by the unchanged protocols.
* `r7_portfolio.py` holds:
  * volatility-managed NIFTY;
  * weekly stock-futures carry, with P&L on the futures contracts and rolls;
  * sector rotation, on `r6_portfolio.backtest`, unchanged.
  It uses its own one-look ledger, `<scratchpad>/github2_r7/portfolio_ledger.json`.

`sim.py` gained OPTIONAL `Swing` fields `entry` / `valid` / `entry_stop_ticks`, used by `sim._order_trade`. This is a
resting stop or limit entry for the next session(s), on daily bars:
* An armed stop (Turtle Soup) that reaches both its arm and its level in one session counts as filled. Daily bars
  cannot tell whether the high or the low came first, and assuming a fill is the pessimistic choice.
* A limit order needs a trade one tick through its level.
* On the fill session only the stop is checked, assuming the low came after the fill. Targets start the next session.

No existing strategy sets these fields, so `FILL_MODEL_VERSION` is unchanged. The controls were re-run after the change
and are unchanged: RANDOM_0 PF 0.77, PLANTED_20pct 1.15, CHEAT 19.57. Era-2 sector-index data was built for this round
from 994 NSE `ind_close_all` files in `<scratchpad>/github2_r7/ind_close_all`; their sha256 is in the
pre-registration. Tests are in `tests_r7.py`, run by `test_pipeline.py`. They cover:
* hand-computed values for every formula;
* the resting-order fills;
* the TPS sequence arithmetic, and the carry and vol-managed engines by hand;
* look-ahead: noise after t leaves the vol-managed and sector-rotation decisions at t unchanged, and the intraday specs
  pass a dense audit.

Bugs fixed (all in the pre-registration):
* Volatility-managed portfolio (coolgan): `c` came from full-sample standard deviations, and the monthly weight used the
  22-day variance ending on the month's first day, which includes that day's own return. Replaced by an expanding `c`
  and the previous month's realised variance.
* Turtle Soup: the centred `argrelextrema` port (which reads future bars) and the 3-bar close-back-inside port are
  replaced by the book rules.
* Smash Day: the README paragraph contradicts its rule list; the rule list is used, with a 48-hour (2-session) order
  life.
* Holy Grail: the ports fill one bar after the breakout; replaced by a resting stop one tick above the touch bar.
* Double 7s: pinkfish fills at the signal close; replaced by the next open. The `main()` overrides (SMA70, 15 % stop,
  2x margin) are dropped.
* Bensdorp (MRL / MRS) and Connors (TPS): the TuringTrader ports use EMA-smoothed RSI / ADX; replaced by Wilder.
* 80-20 and Oops: the 1-hour crypto adaptations are replaced by the book rules on 1-minute sessions.
* Clenow: the notebook already uses the std of price differences, not % changes, and that is kept.

Carry vs the tested `r4_oi_basis_*` rules: those are per-stock time-series z-scores of the near future's premium over
cash, held 5 days. Carry here is a cross-sectional weekly rank of the front-vs-second calendar slope, with P&L on the
contracts. The two are related through the basis but are different rules. Sector rotation: the porter warns that his
top-2 code does not reproduce Alvarez's published results and that picking the 2 worst sectors would. Only the
described momentum version was tested.

Results, under the same conventions as Round 6 ("excess" = net minus the same notional's index move over each trade,
sign-adjusted for shorts, per half; stocks are measured vs NIFTY, index futures vs their own index; L = Rs 1 lakh):

| # | id (r7_gh2_…) | 2023-26 (DISC trades / PF / net) | verdict | 2012-23 (trades / PF / net) | verdict | main reason |
|---|---|---|---|---|---|---|
| 4 | clenow_ct_nifty | 10 / 1.81 / +1.2 L | FAIL | 70 / 1.36 / +2.3 L | OOS-FAILED | too few trades (33 / 37 per half); trails holding NIFTY (excess -0.3 / -0.7 L) |
| 4 | clenow_ct_banknifty | 11 / 4.25 / +3.5 L | FAIL | 60 / 0.85 / -1.7 L | OOS-FAILED | too few trades; 2012-23 H2 < 0, gross PF 0.88 |
| 4 | clenow_ct_stock | 866 / 1.55 / +9.8 L; FINAL 347 / 0.74 / -2.8 L | REJECTED ON FINAL | 5,896 / 1.19 / +37.0 L | OOS-CONFIRMED | the round's only screen pass, lost on FINAL; the 2012-23 pass trails NIFTY in both halves (excess -4.4 / -4.3 L): long-only dip buying = index exposure |
| 5 | volmanaged_nifty | Sharpe 0.52 vs buy-and-hold 0.62; maxDD -21 % vs -16 %; CAGR 6.9 vs 7.6 % | FAIL | Sharpe H1 0.84 vs 0.86, H2 0.68 vs 0.67; maxDD -20 % vs -38 %; CAGR 9.4 vs 11.4 % | FAIL | no Sharpe gain (H1 and era 2 lower); shallower drawdown in 2012-23 only, paid for with 2 %/yr of CAGR |
| 7 | ts_stock / ts_nifty / ts_banknifty | 2,990 / 0.76 / -8.1 L; 38 / 0.47; 28 / 0.70 | FAIL (3) | 17,515 / 0.67 / -92 L; 170 / 0.58; 174 / 0.66 | OOS-FAILED (3) | no edge before costs (gross PF 0.72-0.84) |
| 7 | tsp1_stock | 1,089 / 1.10 / +1.3 L | FAIL | 6,749 / 0.89 / -13.1 L | OOS-FAILED | PF < 1.15 (costs take half the gross); 2012-23 gross PF 0.95 |
| 7 | tsp1_nifty / tsp1_banknifty | 11 / 0.98; 11 / 0.09 | FAIL (2) | 58 / 0.77; 73 / 0.55 | OOS-FAILED (2) | too few trades; gross < 0 |
| 8 | smash_stock | 3,003 / 1.10 / +7.0 L | FAIL | 17,457 / 0.90 / -60.5 L | OOS-FAILED | PF < 1.15 and trails NIFTY (excess -7.7 / -2.4 L); 2012-23 gross PF 0.93 |
| 8 | smash_nifty / smash_banknifty | 38 / 0.86; 34 / 1.30 | FAIL (2) | 254 / 0.77; 230 / 0.76 | OOS-FAILED (2) | gross < 0 over 2012-23; too few trades on 2023-26 |
| 9 | hg_stock | 75 / 1.39 / +0.4 L | FAIL | 397 / 0.89 / -1.1 L | OOS-FAILED | HAL is 51 % of net; 2012-23 gross < 0 |
| 9 | hg_nifty / hg_banknifty | 0 / – ; 0 / – | FAIL (2) | 2 / 1.56; 2 / 1.88 | OOS-FAILED (2) | almost never fires: ADX > 30 and rising on the bar before the touch rarely meets a pullback to EMA20 on an index |
| 10 | carry_xs | CAGR -12.1 % / excess -19.6 | FAIL | H1 -10.0 % / -22.1, H2 -15.8 % / -26.8, Sharpe -2.09 | FAIL | no stable edge before costs (gross +3.0 % / -4.4 % / -0.5 %/yr) and a 13 %/yr cost drag (weekly quintile churn + monthly rolls) |
| 11 | sector_rot | CAGR 12.7 % / excess +5.2 | (era-2 gates met) | H1 17.8 % / +5.3, H2 1.7 % / -9.2, Sharpe 0.52 | FAIL | trails NIFTY by 9 %/yr in 2018-23 (maxDD -53 %). Era 2 uses the real sector indices, so it is not survivorship-biased |
| 12 | d7_nifty / d7_banknifty | 13 / 2.13; 13 / 1.84 | FAIL (2) | 114 / 0.97; 105 / 0.85 | OOS-FAILED (2) | too few trades; 2012-23 H2 < 0, PF < 1 |
| 13 | mrl_stock | 3 / 3.02 / +0.03 L | FAIL | 217 / 1.06 / +0.4 L | OOS-FAILED | filter rarely met on NIFTY-100 names (ATR% > 4 and ADX7 > 45); ADANIENT is 78 % of the 2012-23 net |
| 13 | mrs_stock | 17 / 0.81 / -0.07 L | FAIL | 485 / 1.25 / +2.5 L (H1 +2.45, H2 +0.10) | OOS-CONFIRMED | 2012-23 pass carried by H1 (excess +2.9 / +0.3 L); H2 net ≈ 0 and 2023-26 negative |
| 14 | 8020_stock / 8020_nifty / 8020_banknifty | 5,942 / 0.87 / -2.8 L; 71 / 0.78; 74 / 0.70 | FAIL (3) | – | – | gross ≈ 0 (gross PF 1.04-1.21) killed by costs; index versions too few trades |
| 14 | oops_stock / oops_nifty / oops_banknifty | 7,424 / 0.90 / -3.2 L; 133 / 0.82; 134 / 0.77 | FAIL (3) | – | – | gross ≈ 0 (gross PF 1.08-1.20) killed by costs (stocks: gross +2.9 L, costs 6.1 L) |
| 15 | tps_nifty | 19 sequences / 6.20 / +1.0 L | FAIL | 143 / 1.31 / +1.0 L (H1 +0.59, H2 +0.44) | OOS-CONFIRMED | too few sequences on 2023-26. The 2012-23 pass is small: about 23 % of sessions in the market, gross excess per tranche vs NIFTY -0.1 / +1.0 L, and 1 lot held throughout earned +3.8 / +5.8 L |
| 15 | tps_banknifty | 15 / 2.65 / +0.4 L | FAIL | 143 / 1.10 / +0.5 L | OOS-FAILED | too few sequences (2023-26); 2012-23 H2 < 0, PF 1.10 |

The carry gross figures and the TPS per-tranche excess come from diagnostics run after the one-look verdicts. They are
in `<scratchpad>/github2_r7/diag_after_verdicts.json`, are not trials, and do not touch any ledger.

Verdict: **nothing here is a deployable edge.**
* 2023-26: 27 per-symbol / TPS strategies gave 1 screen pass (Clenow counter-trend on stock futures), REJECTED ON FINAL.
  The 3 portfolios fail their gates.
* 2012-23: 21 trade-level judgements gave 3 OOS-CONFIRMED. With 21 judged, about one per gate is expected by luck.
  * Clenow on stocks earned less than NIFTY over the same trades in both halves.
  * MRS short is carried by 2012-17, with ≈ 0 in 2018-23 and a loss on 2023-26.
  * TPS on NIFTY is a small, rarely-in-the-market pass with mixed excess. It is the fourth short-term RSI(2)
    index-reversion variant, after r3_futures_nifty_rsi2_ls, r3_swing_nifty_rsi2 and r3_swing_rsi2_sma200.
* The book reversal patterns (Turtle Soup, Smash Day, Holy Grail, 80-20, Oops) lose before costs on daily data, or have
  only a gross edge that costs remove intraday.
* Volatility management halves NIFTY's 2013-23 drawdown but does not raise its Sharpe, and costs 2 %/yr of CAGR.
* Sector rotation beat NIFTY in 2013-17 and 2023-26 and lost 9 %/yr in 2018-23.

New trials: 30 variants on 2023-26 (N 1,198 -> 1,228; 1 new FINAL look) and 24 on 2012-23 (N 229 -> 253). Outputs are
in `<scratchpad>/github2_r7/`: run3y_{swing,futures,intraday}, tps3y, oos_daily, tps_oos, portfolio, aggregate.json,
scripts/.

## Old-era minute OOS (2016-2023) (`protocol_min_oos.py`, `prereg/oldera_min_v1.json`)

**Plain language.** The three strategies trading in PAPER (AUC-FADE-3PCT, SWING-ATTN-ON-3X, VRP-STRANGLE-016-WINGED)
were run once, with no tuning, on 2016-01-01..2023-11-09 one-minute bars that none of them was developed on. All three
pass the same gates the 2012-23 daily round used. That is real evidence, and for the auction fade and the attention
trade it is strong: of 373 non-option minute strategies that failed the 2023-26 screen, only 2 (0.5 %) pass here. But
each pass has a weak spot: the 3 % auction fade made 84 % of its 8-year profit in 2020 and barely broke even in
2016-2019 (loses there at 5 bps); the attention trade's 2016-2019 profit is less than holding NIFTY over the same
nights would have made; the strangle is a Black-Scholes MODEL on VIX, not option prices, and that model passes a third
of the option structures that failed on 2023-26, so its pass says more about the model than about the trade. Keep all
three in paper trading; none of this is a reason to switch to LIVE.

Pre-registered before any 2016-2023 strategy result was computed: `prereg/oldera_min_v1.json` (sha256
`8e18deeccf2506fecfc75a5e444adbd7a80fe09cd73e9f99553b1dbebecf6203`, byte-identical to
`<scratchpad>/oldera_min/PREREGISTERED_MIN_OOS.json`, committed before the run). Every judgement is in the one-look
ledger `<scratchpad>/oldera_min/ledger.json` (a second run returns the first result, never recomputes); 420 rows
(era `2016-23`, stage `oos`) were appended to `trials_v2.csv`.

```
python -m stratlab.protocol_min_oos --build-cache && python -m stratlab.protocol_min_oos --data-check
python -m stratlab.protocol_min_oos --run a|b|c --prereg <scratch>/oldera_min/PREREGISTERED_MIN_OOS.json --trials <ledger>
```

* **Gates** = protocol_oos's (constants imported): net > 0 in H1 (2016-2019) and H2 (2020-01-01..2023-11-09), PF >= 1.15
  over the era, >= 150 (intraday) / 60 (positional) trades in EACH half, top symbol <= 35 % of net (not for one
  instrument). Report-only: 5 bps (index 2.5, auction family 5 vs its 2.1, options haircut x2.5), would an intraday id pass
  with 60 trades per half, net minus the same notional's NIFTY move over each trade ("excess"), NIFTY weeklies restricted
  to listed contracts (from 2019-02-11). "Proven in both eras" = FINAL-CONFIRMED on 2023-26 AND OOS-CONFIRMED here.
* **Costs** costs.py 2026 rates on every year; each runner's own fills (sim.py, auction.py 2.1 bps, attention.py,
  options_vrp / options_r3 Black-Scholes on the index minute bars with the previous official India VIX close as a flat vol,
  r3_crosssec, ltr); 2026 lot sizes on every date.
* **Data** (`--data-check`, `<scratchpad>/oldera_min/data_check.json`): 97 of today's NIFTY-100 stocks plus NIFTY 50, NIFTY
  BANK and INDIA VIX minute files, all present (nothing blocked: strangle, NIFTY-relative and needs_index rules all ran).
  31 symbol-year files are empty (listed later: ADANIGREEN, DMART, ETERNAL, HAL, HDFCLIFE, ICICIGI, IRCTC, JIOFIN, LODHA,
  PAYTM, SBILIFE). Split / bonus adjusted on all 51 checkable NSE split / bonus events (2 more fell on missing sessions; RELIANCE 1:1
  bonus 2017-09-07: NSE gap 0.500, Kite 1.001); demergers are not (ADANIENT 2018-09-06, GRASIM 2017-07-19) or wrongly (NMDC 2022-10-27, ABB 2019-12-20)
  adjusted; volume is adjusted on most events but not on 9 (BRITANNIA 2018, CHOLAFIN, HCLTECH, HDFCBANK, NTPC, WIPRO, GAIL,
  VBL 2019, EICHERMOT 2020), which inflates 20-session volume ratios for a few weeks after them. **28 regular sessions are
  missing from every file** (download chunk boundaries, e.g. 2016-03-01, 2019-05-02, 2023-11-01) plus RELIANCE
  2018-03-05..04-30 and NAUKRI 2018-09-03..10-31. IRCTC before its 2019-10-14 listing and PIIND before 2020-02 hold another
  instrument's prints and were dropped. Pre-registered **dirty rule**: a trade is excluded if its entry..exit (or the
  session after its exit) touches a session after a hole, or an overnight link where Kite's open/previous close differs from
  NSE's adjusted ratio by > 5 % (2,837 symbol-dates, ~1.5 % of symbol-sessions; 12 % of auction gap candidates, because
  two-day gaps after a hole are large). Kite's last-minute NIFTY close vs the official close: median 0.05 %. No dates
  overlap bars3y; the 2023-11-09 close -> 2023-11-10 open seam has a median |gap| of 0.2 % (largest ONGC +3.6 %), so the two
  sets are on the same adjustment basis. No daily history is spliced before 2016: lookbacks warm up in 2016.
* **Survivorship**: today's NIFTY-100 names only. This favours every long strategy; read every long result with it.

**(a) The live PAPER strategies** (L = Rs 1 lakh; "xs" = excess vs NIFTY):

| | AUC-FADE-3PCT | SWING-ATTN-ON-3X | VRP-STRANGLE-016-WINGED (modelled) |
|---|---|---|---|
| 2023-26 | FINAL 252 tr, PF 1.32, +0.49 L | FINAL 194 tr, PF 1.18, +0.11 L | FINAL 45 tr, PF 1.27, +0.61 L |
| H1 2016-19 trades / PF / net | 702 / 1.05 / +0.23 L | 1,164 / 1.22 / +0.81 L | 187 / 1.60 / +2.31 L |
| H2 2020-23 trades / PF / net | 1,269 / 1.53 / +3.93 L | 1,324 / 1.97 / +3.69 L | 178 / 1.36 / +2.77 L |
| era PF / net | 1.35 / +4.16 L | 1.60 / +4.50 L | 1.44 / +5.08 L |
| top symbol | PIIND 6.6 % | ADANIGREEN 10.6 % | one instrument |
| sensitivity | 5 bps: +3.02 L but H1 -0.18 L -> would FAIL | 5 bps: +3.02 L, H1 +0.11 L -> passes; 09:16 exit +2.88 L; CNC costs +1.19 L but H1 -0.74 L | haircut x2.5: +1.79 L, PF 1.147 -> would FAIL by 0.003 |
| excess vs NIFTY H1 / H2 | -0.25 L / +3.51 L (both sides; not long-biased) | **-0.90 L** / +1.86 L (long-only) | n/a |
| other (after the verdict, `diag_after_verdicts.json`) | 2020 alone +3.48 L (84 %); without 2020: 1,375 tr, PF 1.08, +0.68 L; best 5 days +2.01 L | without 2020: PF 1.50, +3.04 L | listed weeklies only (2019-02-11..): 218 tr, PF 1.39, +3.47 L; 2016 -0.35 L; worst week -0.56 L (2019-09-20); max DD 1.33 L |
| verdict | OOS-CONFIRMED | OOS-CONFIRMED | OOS-CONFIRMED (model) |
| proven in both eras | yes, fragile | yes, but H1 is index exposure | yes, on modelled premiums |

What it means for paper trading: keep AUC-FADE-3PCT, but expect lumpy results: it was positive in only 4 of the 8 calendar
years, a 2020-like crash year dominates it, and at 5 bps 2016-2019 is a loss, so live fill quality decides it. Keep SWING-ATTN-ON-3X but judge it
against NIFTY's overnight move, not against zero: in 2016-2019 holding NIFTY on the same nights would have earned more,
and on CNC (delivery) costs 2016-2019 loses. Keep VRP-STRANGLE-016-WINGED as a paper test of the model: the paper fills are
the only real option prices it will see; before 2019-02 the weekly contracts did not even exist.

**(b) Other 2023-26 FINAL-confirmed minute-bar strategies** (r4_realopt_nf_wk_strangle_3pct is a daily real-price spec
with no weeklies before 2019 and was not run):

| id | H1 tr / net | H2 tr / net | PF | net@5 (pass?) | verdict |
|---|---|---|---|---|---|
| AUC-FADE-2PCT | 1,751 / +1.50 L | 2,580 / +4.94 L | 1.29 | +3.94 L (Y) | OOS-CONFIRMED (without 2020 PF 1.15) |
| VRP-STRANGLE-016-NAKED (model) | 187 / +1.31 L | 178 / +1.52 L | 1.43 | +2.01 L (Y) | OOS-CONFIRMED |
| r3_options_ic_25_10 (model) | 187 / +0.57 L | 178 / +1.75 L | 1.36 | +0.62 L (N) | OOS-CONFIRMED |
| r3_options_wc_fri (model) | 181 / +1.41 L | 167 / +1.66 L | 1.63 | +1.50 L (Y) | OOS-CONFIRMED |
| r3_tfweekly_wk_loser_rev | 159 / +1.34 L | 344 / +0.99 L | 1.11 | +2.03 L (N) | OOS-FAILED (PF 1.11) |

**(c) The luck rate.** 412 other minute-bar ids (8 calendar-bound r4_mechanics ids ineligible): 394 judged, 18 INVALID
(every r3_ai_crosssec id: its feature audit finds a decision-time feature that reads a later bar when a session lacks its
09:15 bar, which happens in the old data; the module then invalidates all its ids). 12 OOS-CONFIRMED:

| 2023-26 status | judged | OOS-CONFIRMED |
|---|---|---|
| FAILED SCREEN (all) | 385 | **6 (1.6 %)**: r3_tfweekly_volshock_flat, r3_tfweekly_wk_excess_loser_rev, and 4 option models (ibf_400, ibf_400_tp25, cs_trend20, ls_highvix) |
| FAILED SCREEN, not options | 373 | **2 (0.5 %)** |
| FAILED SCREEN, option models | 12 | 4 (33 %) |
| REJECTED ON FINAL | 8 | 5: r3_futures_stk_tom_long (+42.8 L, long-only, positive excess both halves, passes at 5 bps), SCALP-AUC-FADE-1TO3-0925 (+13.2 L, fails at 5 bps), 3 r3_options_ss16_stop2x* models |
| INVALID on 2023-26 | 1 | 1 (r3_options_cs_trend50, model) |
| **FINAL-CONFIRMED (a + b)** | 8 | **7 (87.5 %)** |

Read: for stock rules the 2023-26 screen is informative (0.5 % of its failures pass here against 87.5 % of its
confirmations), so the auction and attention confirmations are unlikely to be luck. For options it is not: the flat-VIX
Black-Scholes model makes short premium look profitable in most years (VIX above realised volatility, no skew), so a third
of the screen-failed option structures pass too; an option "pass" here is a property of the model. Several 2023-26 FINAL
rejections pass 2016-23 (the turn-of-month stock-futures long, the 1-3 % auction fade to 09:25): their edge, if any, did
not survive into 2023-26, which is exactly what a one-era test cannot see.

**Proven in both eras: 7** (AUC-FADE-3PCT, AUC-FADE-2PCT, SWING-ATTN-ON-3X, VRP-STRANGLE-016-WINGED,
VRP-STRANGLE-016-NAKED, r3_options_ic_25_10, r3_options_wc_fri); 3 on stock data (one effect counted twice: 3 % is a
subset of 2 %), 4 on modelled option premiums. Caveats: survivorship (today's names); modelled premiums, not prices, and
NIFTY weeklies only from 2019-02; 2026 costs and lot sizes on 2016 trades (2016 STT / charges differed); 28 missing
sessions handled by exclusion; unadjusted volume on 9 events; DSR still unaddressed (N grows by 420 trials on this era).
Outputs: `<scratchpad>/oldera_min/` (run_a, run_b, run_c: summary.txt, results.csv / .json, trades/*.csv.gz for a, b and
the module runners; aggregate.json, diag_after_verdicts.json, scripts/). Tests: `tests_min_oos.py` (synthetic), run by
`test_pipeline.py`.
