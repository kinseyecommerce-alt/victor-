"""
optimizer.py — walk-forward optimisation of the live agents' parameters.

Why walk-forward: picking the parameters that did best on the past guarantees nothing;
most of that "best" is noise fitted to one period. So the data is split into folds in time
order. Every candidate parameter set is scored on the TRAIN part of each fold, the best few
are then scored on the TEST part they never saw, and a candidate is adopted only if its
out-of-sample result beats the current parameters on net P&L after costs, with enough trades
and no worse drawdown. If nothing qualifies, nothing changes.

Everything runs through replay_backtest (the same code that trades live). Chosen parameters
are saved to data/strategy_params.json and applied to the live agent instances; they can be
reset from the API.
"""
from __future__ import annotations

import itertools
import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from config import settings
from ist_clock import now_ist

PARAMS_FILE = Path(os.getenv("STRATEGY_PARAMS_FILE") or Path(__file__).resolve().parent / "data" / "strategy_params.json")

# What may be tuned, and the values tried. Small grids on purpose: 27 candidates × a few
# symbols replay in well under a minute; large grids only find noise.
# The agents use max(ATR × multiple, price × min %): at normal 1-minute volatility the min-%
# floor is what binds, so that is what gets tuned (the ATR multiples barely change anything).
GRIDS: dict[str, dict[str, list]] = {
    "intraday": {"MIN_SCORE": [4, 5, 6], "SL_MIN_PCT": [0.4, 0.5, 0.7], "TGT_MIN_PCT": [0.8, 1.0, 1.4]},
    "scalping": {"MIN_SCORE": [3, 4, 5], "SL_PCT": [0.2, 0.25, 0.35], "TGT_PCT": [0.4, 0.5, 0.7]},
}
MIN_TEST_TRADES = 10
MAX_DD_WORSENING = 1.2          # candidate's test drawdown may be at most 20% worse than current
TOP_K = 3                       # train winners that get a test run

_state: dict = {"running": False, "started_at": None, "finished_at": None, "report": None, "error": None}
_lock = threading.Lock()


# ── persistence / application ─────────────────────────────────────────────

def load_params() -> dict:
    try:
        return json.loads(PARAMS_FILE.read_text())
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.error("[optimizer] params unreadable: {}", exc)
        return {}


def save_params(data: dict) -> None:
    PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=PARAMS_FILE.parent, prefix=".params.")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, PARAMS_FILE)


def apply_params(agents: Optional[dict] = None) -> dict[str, dict]:
    """Set the saved parameters on the live agent instances. Returns what was applied."""
    if agents is None:
        from agents.strategy_agents import ALL_AGENTS
        agents = ALL_AGENTS
    applied = {}
    for name, entry in load_params().items():
        a = agents.get(name)
        params = (entry or {}).get("params") or {}
        if not a or not params:
            continue
        for k, v in params.items():
            if k in GRIDS.get(name, {}):
                setattr(a, k, v)
        applied[name] = params
    if applied:
        logger.info("[optimizer] applied parameters: {}", applied)
    return applied


def current_params(name: str) -> dict:
    """Defaults (class attributes), overridden by anything saved."""
    from agents.strategy_agents import IntradayAgent, ScalpingAgent
    cls = {"intraday": IntradayAgent, "scalping": ScalpingAgent}[name]
    base = {k: getattr(cls, k) for k in GRIDS[name]}
    base.update((load_params().get(name) or {}).get("params") or {})
    return base


def reset(name: str) -> None:
    from agents.strategy_agents import IntradayAgent, ScalpingAgent, ALL_AGENTS
    data = load_params()
    data.pop(name, None)
    save_params(data)
    cls = {"intraday": IntradayAgent, "scalping": ScalpingAgent}[name]
    a = ALL_AGENTS.get(name)
    if a:
        for k in GRIDS[name]:
            if k in vars(a):
                delattr(a, k)                     # back to the class default


# ── walk-forward ──────────────────────────────────────────────────────────

def _folds(bars: list[dict], n_folds: int = 3, train_frac: float = 0.7) -> list[tuple[list, list]]:
    """Time-ordered folds; each is (train_bars, test_bars), test after train, no overlap."""
    days = sorted({b["ts"].date() for b in bars})
    if len(days) < 4:
        return []
    per = max(len(days) // n_folds, 2)
    out = []
    for k in range(n_folds):
        chunk = days[k * per: (k + 1) * per] if k < n_folds - 1 else days[k * per:]
        if len(chunk) < 2:
            continue
        split = max(1, int(len(chunk) * train_frac))
        tr_days, te_days = set(chunk[:split]), set(chunk[split:])
        if not te_days:
            continue
        out.append(([b for b in bars if b["ts"].date() in tr_days], [b for b in bars if b["ts"].date() in te_days]))
    return out


async def _score(agent: str, params: dict, bars_by_symbol: dict[str, list], days: int) -> dict:
    import replay_backtest as rb
    res = await rb.replay_async(agent, list(bars_by_symbol), days, params=params, bars_by_symbol=bars_by_symbol)
    m = res.metrics()
    return {"net": m["net_pnl"], "trades": m["trades"], "win_rate": m["win_rate"], "pf": m["profit_factor"],
            "max_dd": m["max_drawdown"]}


async def optimise_async(agent: str, symbols: list[str], days: int = 45,
                         bars_by_symbol: Optional[dict[str, list]] = None, apply: bool = True) -> dict:
    """Walk-forward search for one agent. Returns a report; adopts a candidate only if it
    beats the current parameters out of sample."""
    import replay_backtest as rb
    if agent not in GRIDS:
        raise ValueError(f"no parameter grid for {agent}")
    provided = bool(bars_by_symbol)
    bars_by_symbol = bars_by_symbol or {}
    if not bars_by_symbol:
        import asyncio
        for s in symbols:
            bars, src = await asyncio.to_thread(rb.load_bars, s, days)
            if bars:
                bars_by_symbol[s] = bars
    if not bars_by_symbol:
        return {"agent": agent, "error": "no bar data"}
    folds_by_symbol = {s: _folds(b) for s, b in bars_by_symbol.items()}
    n_folds = min((len(f) for f in folds_by_symbol.values() if f), default=0)
    if n_folds == 0:
        return {"agent": agent, "error": "not enough days for walk-forward folds (need ≥ 4 trading days)"}

    def part(k: int, which: int) -> dict[str, list]:
        return {s: f[k][which] for s, f in folds_by_symbol.items() if len(f) > k}

    cur = current_params(agent)
    grid = GRIDS[agent]
    candidates = [dict(zip(grid, vals)) for vals in itertools.product(*grid.values())]
    # 1. train: every candidate on every fold's train part
    train_scores = []
    for c in candidates:
        total = 0.0
        for k in range(n_folds):
            total += (await _score(agent, c, part(k, 0), days))["net"]
        train_scores.append((total, c))
    train_scores.sort(key=lambda x: -x[0])
    top = [c for _, c in train_scores[:TOP_K]]
    # 2. test: the current parameters and the top candidates on the unseen parts
    async def test_of(c: dict) -> dict:
        agg = {"net": 0.0, "trades": 0, "max_dd": 0.0}
        for k in range(n_folds):
            r = await _score(agent, c, part(k, 1), days)
            agg["net"] += r["net"]; agg["trades"] += r["trades"]; agg["max_dd"] = max(agg["max_dd"], r["max_dd"])
        agg["net"] = round(agg["net"], 2)
        return agg
    cur_test = await test_of(cur)
    results = []
    for c in top:
        if c == cur:
            results.append({"params": c, "test": cur_test, "is_current": True})
            continue
        results.append({"params": c, "test": await test_of(c), "is_current": False})
    # 3. decide
    best, best_reason = None, ""
    for r in sorted(results, key=lambda x: -x["test"]["net"]):
        if r["is_current"]:
            continue
        t = r["test"]
        if t["trades"] < MIN_TEST_TRADES:
            r["verdict"] = f"too few out-of-sample trades ({t['trades']} < {MIN_TEST_TRADES})"; continue
        if t["net"] <= cur_test["net"]:
            r["verdict"] = f"does not beat current out of sample (₹{t['net']:,.0f} vs ₹{cur_test['net']:,.0f})"; continue
        if t["net"] <= 0:
            r["verdict"] = "loses money out of sample"; continue
        if cur_test["max_dd"] > 0 and t["max_dd"] > cur_test["max_dd"] * MAX_DD_WORSENING:
            r["verdict"] = f"drawdown {t['max_dd']:,.0f} worse than current {cur_test['max_dd']:,.0f} × {MAX_DD_WORSENING}"; continue
        r["verdict"] = "adopted"; best = r; break
    report = {"agent": agent, "symbols": list(bars_by_symbol), "days": days, "folds": n_folds,
              "candidates": len(candidates), "current": {"params": cur, "test": cur_test},
              "top": results, "adopted": best["params"] if best else None,
              "data": "provided" if provided else ("synthetic (paper simulator)" if not settings.uses_kite_market_data else "real"),
              "at": now_ist().isoformat(timespec="seconds")}
    if best and apply:
        data = load_params()
        data[agent] = {"params": best["params"], "chosen_at": report["at"], "test_net": best["test"]["net"],
                       "current_test_net": cur_test["net"], "symbols": list(bars_by_symbol), "days": days}
        save_params(data)
        apply_params()
        logger.info("[optimizer] {} → {} (test ₹{:,.0f} vs ₹{:,.0f})", agent, best["params"], best["test"]["net"], cur_test["net"])
    return report


def optimise(agent: str, symbols: list[str], days: int = 45, bars_by_symbol=None, apply: bool = True) -> dict:
    import asyncio
    return asyncio.run(optimise_async(agent, symbols, days, bars_by_symbol, apply))


async def run_all_async(symbols: list[str], days: int = 45) -> dict:
    """Both tunable agents, one after the other; state visible through status()."""
    with _lock:
        if _state["running"]:
            return {"error": "already running", **_state}
        _state.update(running=True, started_at=now_ist().isoformat(timespec="seconds"), finished_at=None, report=None, error=None)
    try:
        report = {}
        for agent in GRIDS:
            report[agent] = await optimise_async(agent, symbols, days)
        _state["report"] = report
        return report
    except Exception as exc:
        _state["error"] = str(exc)
        logger.exception("[optimizer] failed")
        return {"error": str(exc)}
    finally:
        _state.update(running=False, finished_at=now_ist().isoformat(timespec="seconds"))


def status() -> dict:
    return {**_state, "saved": load_params(), "grids": GRIDS}
