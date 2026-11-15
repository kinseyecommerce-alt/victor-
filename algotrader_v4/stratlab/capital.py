"""
stratlab/capital.py — a human-like Rs 10 lakh account per agent, for the timeframe grid (tfgrid.py). Research only.

The lab's simulators (sim.py, the module runners) produce CANDIDATE trades per symbol at a nominal size. A person
with one account cannot take them all: money, open-position slots, the broker's margin and the day's loss limit
decide which signals become trades and how big they are. `simulate()` walks the candidates in time order and takes
or skips each one exactly as such an account would, using only what is known at the candidate's fill time.

Rules (frozen in the tfgrid pre-registration; AgentConfig holds the numbers):
  capital        Rs 10,00,000 per agent, NOT compounded (risk and buying power measured on the fixed base);
                 compound=True (report only) measures them on base + realised P&L so far
  size           risk RISK_PCT (1 %) of capital per trade from the entry-to-stop distance:
                 equity qty = floor(risk / |fill - stop|); futures lots = floor(risk / (|fill - stop| x lot))
                 candidates without a stop (the lab strategies that define none): `fallback_notional` (Rs 1 lakh, the
                 lab's notional) for stocks, the strategy's own lots for index futures, 1 lot for stock futures
  buying power   equity MIS (intraday, scalping): free capital x MIS_LEVERAGE (5); equity CNC (swing longs): free
                 capital x 1; futures: lots x lot size x price x FUT_MARGIN_PCT (15 %, config.ai_agent_future_margin_pct,
                 an approximate SPAN + exposure margin) must fit in free capital; index-futures orders are also capped at
                 config.max_futures_order_value (Rs 25 lakh contract value). Free capital = capital - margin blocked by
                 open positions (P&L is realised at exit)
  slots          at most `max_pos` open positions (config.max_*_positions), one per symbol, at most `max_trades_day`
                 entries per day (config.max_trades_*), equity positions below config.min_position_value_abs
                 (Rs 25,000) are skipped
  daily loss     no new entries on a day whose realised net P&L (exits so far that day, after costs) is <= -daily_loss
                 (config.max_daily_loss / config.total_capital = 1 % of capital -> Rs 10,000)
  priority       candidates are taken in time order of their fill; ties (same minute) by the higher PREVIOUS-session
                 traded value of the symbol (known before the open) - no hindsight. Exits at a minute are processed
                 before entries at the same minute (a stop-and-reverse or a freed slot can be used at that open)
  costs          costs.round_trip_cost on the actual buy / sell values: equity_mis, equity_cnc (incl. DP), future
Every skipped candidate is counted by reason. Results are trades in sim.TRADE_FIELDS order (plus margin), so
protocol.screen / protocol_min_oos.judge judge them unchanged.
"""
from __future__ import annotations

import heapq
import math
import os
import sys
from dataclasses import dataclass, field, asdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from costs import round_trip_cost  # noqa: E402

CAPITAL = 1_000_000.0
RISK_PCT = 1.0
MIS_LEVERAGE = 5.0
SKIP_REASONS = ("symbol_busy", "slots", "max_trades_day", "daily_loss", "capital", "below_one_lot", "too_small")
# a candidate tuple (see Cand): fields by index
C_SYM, C_SIDE, C_EKEY, C_FILL, C_STOP, C_XKEY, C_XPX, C_REASON, C_PRIO, C_INST, C_LOT, C_FBLOTS, C_ID = range(13)
C_OPT = 13          # option structures only: (qty, gross, cost, margin) as modelled by the module


def _cfg():
    try:
        import config
        return config.settings
    except Exception:          # pragma: no cover - config needs pydantic; the lab must still import
        return None


def config_values() -> dict:
    """The live config numbers the account rules use (recorded in the pre-registration)."""
    s = _cfg()
    g = (lambda k, d: getattr(s, k, d)) if s is not None else (lambda k, d: d)
    return {
        "max_intraday_positions": g("max_intraday_positions", 5), "max_scalping_positions": g("max_scalping_positions", 3),
        "max_swing_positions": g("max_swing_positions", 8), "fno_max_open_positions": g("fno_max_open_positions", 2),
        "max_trades_intraday": g("max_trades_intraday", None), "max_trades_scalping": g("max_trades_scalping", None),
        "max_trades_swing": g("max_trades_swing", None), "max_trades_fno": g("max_trades_fno", None),
        "max_daily_loss": g("max_daily_loss", None), "total_capital": g("total_capital", None),
        "min_position_value_abs": g("min_position_value_abs", 0.0),
        "ai_agent_future_margin_pct": g("ai_agent_future_margin_pct", 15.0),
        "max_futures_order_value": g("max_futures_order_value", 2_500_000.0),
        "risk_per_trade_pct (config; NOT used, the task fixes 1 %)": g("risk_per_trade_pct", None),
        "intraday_leverage (config; NOT used, the task fixes MIS 5x)": g("intraday_leverage", None),
        "validated_auction_daily_loss_stop": g("validated_auction_daily_loss_stop", 16000.0),
        "validated_attention_daily_loss_stop": g("validated_attention_daily_loss_stop", 10000.0),
        "validated_auction_max_gross": g("validated_auction_max_gross", 1_000_000.0),
        "validated_attention_max_gross": g("validated_attention_max_gross", 1_000_000.0),
        "validated_vrp_lots": g("validated_vrp_lots", 2), "validated_vrp_max_margin": g("validated_vrp_max_margin", 250000.0),
        "validated_vrp_max_loss": g("validated_vrp_max_loss", 75000.0),
        "validated_vrp_dd_kill": g("validated_vrp_dd_kill", 195000.0),
    }


@dataclass
class AgentConfig:
    name: str
    max_pos: int
    max_trades_day: int | None
    daily_loss: float | None
    capital: float = CAPITAL
    risk_pct: float = RISK_PCT
    mis_leverage: float = MIS_LEVERAGE
    fut_margin_pct: float = 15.0
    min_value: float = 25000.0
    max_fut_order_value: float = 2_500_000.0
    fallback_notional: float = 100_000.0
    fixed_notional: float | None = None      # validated agents: every trade at this notional (as deployed)
    compound: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def agents(cv: dict | None = None) -> dict:
    """The five grid agents with the config numbers (fallbacks of the task where config has none)."""
    cv = cv or config_values()
    dl = None
    if cv.get("max_daily_loss") and cv.get("total_capital"):
        dl = CAPITAL * float(cv["max_daily_loss"]) / float(cv["total_capital"])      # 1 % of capital
    fm = float(cv.get("ai_agent_future_margin_pct") or 15.0)
    mv = float(cv.get("min_position_value_abs") or 0.0)
    fo = float(cv.get("max_futures_order_value") or 2_500_000.0)
    mk = lambda name, mp, mt: AgentConfig(name, int(mp), (int(mt) if mt else None), dl, fut_margin_pct=fm,
                                          min_value=mv, max_fut_order_value=fo)
    return {
        "intraday": mk("intraday", cv.get("max_intraday_positions") or 5, cv.get("max_trades_intraday")),
        "scalping": mk("scalping", cv.get("max_scalping_positions") or 3, cv.get("max_trades_scalping")),
        "swing": mk("swing", cv.get("max_swing_positions") or 8, cv.get("max_trades_swing")),
        "options": mk("options", cv.get("fno_max_open_positions") or 2, cv.get("max_trades_fno")),
        "futures": mk("futures", 3, None),          # no futures-agent caps in config.py: the task's fallback 3
    }


# instruments: "mis" equity intraday, "cnc" equity delivery, "fut" futures (stock or index), "opt" option structure
COST_KIND = {"mis": "equity_mis", "cnc": "equity_cnc", "fut": "future"}


def cand(sym, side, ekey, fill, stop, xkey, xpx, reason, prio=0.0, inst="mis", lot=1, fb_lots=1, cid="", opt=None):
    """One candidate trade: ekey / xkey = yyyymmdd * 10000 + minute of day of the fill / the exit (a close at minute
    m is keyed m + 1), fill / xpx include slippage, stop = the initial stop price (NaN = none)."""
    return (str(sym), int(side), int(ekey), float(fill), float(stop) if stop is not None else float("nan"),
            int(xkey), float(xpx), str(reason), float(prio), str(inst), int(lot), int(fb_lots), str(cid)) + \
        ((tuple(opt),) if opt is not None else ())


def size(c, ag: AgentConfig, equity: float, free: float):
    """(qty, margin, skip_reason). qty in shares / units (lots x lot size for futures)."""
    fill, stop, inst = c[C_FILL], c[C_STOP], c[C_INST]
    risk = equity * ag.risk_pct / 100.0
    dist = abs(fill - stop) if math.isfinite(stop) else float("nan")
    if inst in ("mis", "cnc"):
        lev = ag.mis_leverage if inst == "mis" else 1.0
        if ag.fixed_notional is not None:
            q_risk = int(ag.fixed_notional // fill)
        elif math.isfinite(dist) and dist > 0:
            q_risk = int(risk // dist)
        else:
            q_risk = int(ag.fallback_notional // fill)
        q_bp = int(max(free, 0.0) * lev // fill)
        q = min(q_risk, q_bp)
        if q <= 0 or q * fill < ag.min_value:
            return 0, 0.0, ("capital" if q_bp < max(q_risk, 1) or q_bp * fill < ag.min_value else "too_small")
        return q, q * fill / lev, None
    if inst == "opt":                              # a modelled option structure: its own lots and margin
        q, _, _, m = c[C_OPT]
        return (int(q), float(m), None) if m <= free else (0, 0.0, "capital")
    if inst == "fut":
        lot = max(1, int(c[C_LOT]))
        if ag.fixed_notional is not None:          # validated stock-future strategy simulated on cash at a notional
            q = int(ag.fixed_notional // fill)
            m = q * fill * ag.fut_margin_pct / 100.0
            if q <= 0:
                return 0, 0.0, "too_small"
            if m > free:
                return 0, 0.0, "capital"
            return q, m, None
        per_lot = lot * fill * ag.fut_margin_pct / 100.0
        if math.isfinite(dist) and dist > 0:
            l_risk = int(risk // (dist * lot))
        else:
            l_risk = max(1, int(c[C_FBLOTS]))
        l_bp = int(max(free, 0.0) // per_lot) if per_lot > 0 else 0
        l_cap = int(ag.max_fut_order_value // (lot * fill)) if ag.max_fut_order_value else l_risk
        lots = min(l_risk, l_bp, max(l_cap, 1))
        if l_risk < 1:
            return 0, 0.0, "below_one_lot"
        if lots < 1:
            return 0, 0.0, "capital"
        return lots * lot, lots * per_lot, None
    raise ValueError(f"unknown instrument {inst!r}")


def _cost(inst, side, fill, xpx, q):
    bv, sv = (fill * q, xpx * q) if side > 0 else (xpx * q, fill * q)
    return round_trip_cost(bv, sv, COST_KIND[inst])


def order(cands) -> list:
    """Time order of the fills; ties: higher previous-session traded value first, then symbol (deterministic)."""
    return sorted(cands, key=lambda c: (c[C_EKEY], -c[C_PRIO], c[C_SYM], c[C_SIDE]))


def simulate(cands, ag: AgentConfig, sid: str = "", presorted: bool = False) -> dict:
    """Run one account over candidate trades. Returns {"trades": [TRADE_FIELDS + (margin,)], "skipped": {reason: n},
    "taken": n, "candidates": n, "daily": {yyyymmdd: realised net}}."""
    cs = cands if presorted else order(cands)
    open_ = []                   # heap (xkey, seq, sym, margin, net, xdate)
    busy: dict = {}
    blocked = 0.0
    realised = 0.0
    day_net: dict = {}
    day_n: dict = {}
    skipped = {k: 0 for k in SKIP_REASONS}
    trades = []
    seq = 0

    def release(upto):
        nonlocal blocked, realised
        while open_ and open_[0][0] <= upto:
            xk, _, sym, m, net, xd = heapq.heappop(open_)
            blocked -= m
            realised += net
            day_net[xd] = day_net.get(xd, 0.0) + net
            busy.pop(sym, None)

    for c in cs:
        ek = c[C_EKEY]
        release(ek)
        d = ek // 10000
        if c[C_SYM] in busy:
            skipped["symbol_busy"] += 1
            continue
        if len(open_) >= ag.max_pos:
            skipped["slots"] += 1
            continue
        if ag.max_trades_day is not None and day_n.get(d, 0) >= ag.max_trades_day:
            skipped["max_trades_day"] += 1
            continue
        if ag.daily_loss is not None and day_net.get(d, 0.0) <= -ag.daily_loss:
            skipped["daily_loss"] += 1
            continue
        equity = ag.capital + (realised if ag.compound else 0.0)
        free = equity - blocked
        q, m, why = size(c, ag, equity, free)
        if why:
            skipped[why] += 1
            continue
        side, fill, xpx, inst = c[C_SIDE], c[C_FILL], c[C_XPX], c[C_INST]
        if inst == "opt":
            gross, cost = float(c[C_OPT][1]), float(c[C_OPT][2])
        else:
            gross = side * (xpx - fill) * q
            cost = _cost(inst, side, fill, xpx, q)
        net = gross - cost
        xk = c[C_XKEY]
        xd = xk // 10000
        heapq.heappush(open_, (xk, seq, c[C_SYM], m, net, xd))
        seq += 1
        busy[c[C_SYM]] = xk
        blocked += m
        day_n[d] = day_n.get(d, 0) + 1
        et, xt = ek % 10000, xk % 10000
        trades.append((sid or c[C_ID], c[C_SYM], int(d), int(side), f"{et // 60:02d}:{et % 60:02d}", round(fill, 4),
                       int(xd), f"{min(xt, 929) // 60:02d}:{min(xt, 929) % 60:02d}" if xt >= 900 else
                       f"{xt // 60:02d}:{xt % 60:02d}", round(xpx, 4), int(q), round(gross, 2), cost,
                       round(net, 2), c[C_REASON], round(m, 2)))
    release(10 ** 13)
    return {"trades": trades, "skipped": skipped, "taken": len(trades), "candidates": len(cs), "daily": day_net,
            "agent": ag.name, "compound": ag.compound}


# ── account metrics ───────────────────────────────────────────────────────
def metrics(res: dict, calendar, lo: int, hi: int, capital: float = CAPITAL) -> dict:
    """Plain account numbers over [lo, hi] (calendar = session dates): final P&L, return % a year, max drawdown (Rs
    and % of the peak account value), worst month, win rate, PF, Sharpe (daily realised P&L incl. zero days,
    x sqrt(250)), trades, skipped signals. Daily P&L is realised on the exit date."""
    cal = [int(x) for x in calendar if lo <= int(x) <= hi]
    tr = [t for t in res["trades"] if lo <= t[2] <= hi]
    nets = np.array([t[12] for t in tr], float)
    daily = np.zeros(len(cal))
    idx = {d: i for i, d in enumerate(cal)}
    for t in tr:
        x = int(t[6])
        i = idx.get(x)
        if i is None:
            j = int(np.searchsorted(cal, x))
            i = min(j, len(cal) - 1)
        daily[i] += t[12]
    eq = capital + np.cumsum(daily) if len(daily) else np.array([capital])
    peak = np.maximum.accumulate(np.concatenate(([capital], eq)))[1:] if len(eq) else np.array([capital])
    dd = eq - peak
    k = int(np.argmin(dd)) if len(dd) else 0
    years = max(len(cal) / 250.0, 1e-9)
    net = float(nets.sum()) if len(nets) else 0.0
    gp, gl = float(nets[nets > 0].sum()), float(-nets[nets < 0].sum())
    months: dict = {}
    for d, v in zip(cal, daily):
        months[d // 100] = months.get(d // 100, 0.0) + v
    wm = min(months.items(), key=lambda kv: kv[1]) if months else (0, 0.0)
    sd = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    final = capital + net
    return {
        "trades": int(len(nets)), "net": round(net, 2), "final_value": round(final, 2),
        "return_pct_per_year": round(100.0 * net / capital / years, 2),
        "cagr_pct": round(100.0 * ((final / capital) ** (1.0 / years) - 1.0), 2) if final > 0 else -100.0,
        "max_dd_rs": round(float(-dd.min()) if len(dd) else 0.0, 2),
        "max_dd_pct": round(float(-dd[k] / peak[k] * 100.0) if len(dd) and peak[k] > 0 else 0.0, 2),
        "worst_month": str(wm[0]), "worst_month_rs": round(float(wm[1]), 2),
        "win_pct": round(100.0 * float((nets > 0).mean()), 1) if len(nets) else 0.0,
        "pf": round(gp / gl, 3) if gl > 0 else (float("inf") if gp > 0 else 0.0),
        "sharpe": round(float(daily.mean() / sd * math.sqrt(250.0)), 3) if sd > 0 else 0.0,
        "sr_daily": float(daily.mean() / sd) if sd > 0 else float("nan"),
        "skipped": dict(res.get("skipped", {})), "candidates": res.get("candidates", 0),
        "gross": round(float(sum(t[10] for t in tr)), 2), "costs": round(float(sum(t[11] for t in tr)), 2),
        "sessions": len(cal),
    }


def daily_series(res: dict, calendar, lo: int, hi: int) -> list:
    cal = [int(x) for x in calendar if lo <= int(x) <= hi]
    idx = {d: i for i, d in enumerate(cal)}
    out = [0.0] * len(cal)
    for t in res["trades"]:
        if lo <= t[2] <= hi:
            i = idx.get(int(t[6]))
            if i is None:
                i = min(int(np.searchsorted(cal, int(t[6]))), len(cal) - 1)
            out[i] += t[12]
    return out
