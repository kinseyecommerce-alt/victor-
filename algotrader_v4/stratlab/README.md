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

## Simulator assumptions (`sim.py`)

* **Data**: 1-minute bars, 09:15–15:29 IST. Sessions with fewer than 200 bars (the Diwali muhurat hour, a
  half-downloaded day) are dropped entirely, so "yesterday" is always the last full session.
* **Entry**: signal at the close of bar `i` → filled at the **open of bar i+1**, 5 bps worse. An entry whose
  stop is already through at that open, or whose target is on the wrong side, is skipped (counted in
  `skipped_entries`). No fills at or after 15:15.
* **Exit**: stop and target are checked on each later bar's low/high, the entry bar included. **If one bar
  touches both, the stop is assumed.** A bar that opens beyond the stop fills at its open. Stop, time-stop and
  square-off exits pay 5 bps slippage; a target is a resting limit order and fills at its price.
* **Square-off** at the open of the 15:15 bar. If the day's data ends earlier, at the last available close.
* **Size**: ₹1,00,000 notional per trade in whole shares. One open position per symbol; `max_per_day`
  trades per symbol per day. Strategies are simulated independently: no portfolio cap, no capital limit.
* **Costs**: `costs.round_trip_cost` — `equity_mis` for intraday and scalping, `equity_cnc` for swing,
  `future` for futures. On ₹1 lakh that is about ₹80 for an intraday round trip and about ₹230 for delivery,
  plus about ₹100 of slippage: an intraday trade starts roughly 0.18 % behind.
* **Futures** are simulated on the cash price with futures costs: no basis, lot size, margin or rollover.
* **Positional**: stop/target checked on daily high/low (stop first); a gap through the stop fills at the
  open; the position is closed at the close of its `max_hold`-th session. Delivery is long-only.
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
* FINAL is reported once per strategy: the first result is kept in `final_ledger.json` in the data cache. If
  the strategy's code or parameters change after its FINAL was seen, a later pass is reported as
  `RETEST (FINAL already used)`, not as a confirmation.
* Below 100 sessions there is **no verdict** ("too few sessions"); the discovery numbers are shown for
  debugging only.
* A strategy that raises, or fails the look-ahead audit, is `INVALID`.
* Every report ends with how many passed out of how many were tested and the reminder that about 5 % of
  useless strategies pass each coin-flip-like gate. With 250 candidates, a dozen lucky screen passes and a few
  lucky confirmations are expected from strategies with no edge. A confirmation earns a paper-trading trial,
  nothing more.

## Tests

`cd algotrader_v4 && python test_pipeline.py` — the `STRATLAB` section covers: the per-bar view cannot read
bar i+1; changing future bars does not change earlier fills; the audit catches a cheating vector strategy;
indicators are causal; next-open fills and slippage; stop-before-target; square-off and time stop; costs;
the positional simulator; the protocol gates and that FINAL is never simulated for screen failures; both data
layouts; every catalogue strategy runs and passes the audit; the CLI end to end.
