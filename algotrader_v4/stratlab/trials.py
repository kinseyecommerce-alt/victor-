"""
stratlab/trials.py — the trials ledger and the Deflated Sharpe Ratio.

Every strategy evaluation is a trial. With hundreds of trials the best backtest looks good by luck, so the
honest question is not "is this Sharpe positive?" but "is it higher than the best Sharpe that N useless trials
would produce?". This module

  1. keeps an APPEND-ONLY CSV of every evaluation (`TrialsLedger`, default
     <scratchpad>/stratlab_cache/trials_v2.csv, override with --ledger or $STRATLAB_TRIALS), and rebuilds it from the
     existing ledgers, results.csv files, report.json files and exploratory effect files (`backfill`);
  2. implements the Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012) and the Deflated Sharpe Ratio
     (Bailey & Lopez de Prado 2014) with skewness and kurtosis (`psr`, `expected_max_sr`, `dsr`);
  3. reports the DSR of the deployed / FINAL-confirmed strategies from their trade lists (`report`);
  4. adds a gate that replaces any "PF haircut" idea: the mean GROSS edge per trade, cut by half, must still beat
     the mean cost per trade, and/or DSR > 0.95 (`deflation_gate`). It is an extra report only: it never changes a
     verdict of protocol.py.

Research code only: reads files, writes outside the repository.

    python -m stratlab.trials backfill [--scratch DIR] [--ledger PATH]
    python -m stratlab.trials summary  [--ledger PATH]
    python -m stratlab.trials report   [--ledger PATH] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
from statistics import NormalDist

_SCRATCH_DEFAULT = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"
DEFAULT_LEDGER = os.environ.get("STRATLAB_TRIALS") or os.path.join(_SCRATCH_DEFAULT, "stratlab_cache", "trials_v2.csv")
# trials_v2.csv: corrected re-backfill (2026-11-14). The first ledger, trials.csv, is kept unchanged (append-only)
# but superseded: its FINAL row for cs_gravestone_doji_d took Sharpe / moments from a 12-month run's trade file.

EULER_GAMMA = 0.5772156649015329
DSR_MIN = 0.95              # gate: probability that the true Sharpe beats the expected best-of-N luck
HAIRCUT = 0.5               # gate: half the mean gross edge per trade must still pay the full mean cost

FIELDS = ("recorded_at", "trial_key", "strategy_id", "family", "group", "params_fp", "era", "protocol", "stage",
          "dataset", "period", "cost_model", "trades", "net", "pf", "mean_net", "std_net", "sr_trade",
          "mean_gross", "mean_cost", "n_days", "sr_daily", "skew_daily", "kurt_daily", "verdict", "source",
          "trades_file", "note")

_N = NormalDist()


# ───────────────────────────── statistics ─────────────────────────────

def moments(x) -> dict:
    """n, mean, sample std (ddof 1), Sharpe = mean/std, skewness and NON-excess kurtosis (normal = 3), the
    latter two from population moments as in Bailey & Lopez de Prado."""
    x = [float(v) for v in x]
    n = len(x)
    out = {"n": n, "mean": float("nan"), "std": float("nan"), "sr": float("nan"),
           "skew": float("nan"), "kurt": float("nan")}
    if n == 0:
        return out
    m = sum(x) / n
    out["mean"] = m
    if n < 2:
        return out
    var1 = sum((v - m) ** 2 for v in x) / (n - 1)
    m2 = sum((v - m) ** 2 for v in x) / n
    out["std"] = math.sqrt(var1)
    if var1 > 0:
        out["sr"] = m / math.sqrt(var1)
    if m2 > 0:
        out["skew"] = (sum((v - m) ** 3 for v in x) / n) / m2 ** 1.5
        out["kurt"] = (sum((v - m) ** 4 for v in x) / n) / m2 ** 2
    return out


def psr(sr: float, sr_star: float, t: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > sr_star) given an observed, NON-annualised `sr` over `t`
    observations with skewness `skew` and kurtosis `kurt` (normal = 3).
        PSR = Phi( (sr - sr_star) * sqrt(t - 1) / sqrt(1 - skew * sr + (kurt - 1) / 4 * sr^2) )"""
    if t is None or t < 2 or any(map(_bad, (sr, sr_star, skew, kurt))):
        return float("nan")
    den = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if den <= 0:
        return float("nan")
    return _N.cdf((sr - sr_star) * math.sqrt(t - 1) / math.sqrt(den))


def expected_max_sr(n_trials: int, var_sr: float) -> float:
    """Expected maximum Sharpe of `n_trials` independent trials whose true SR is 0 and whose estimated SRs have
    variance `var_sr` (same units as the SR being deflated):
        SR0 = sqrt(V) * ((1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e))),  g = Euler-Mascheroni.
    N <= 1 gives 0 (no selection)."""
    if n_trials is None or n_trials <= 1 or _bad(var_sr) or var_sr <= 0:
        return 0.0
    n = float(n_trials)
    return math.sqrt(var_sr) * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n)
                                + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n * math.e)))


def dsr(sr: float, t: int, skew: float, kurt: float, n_trials: int, var_sr: float) -> dict:
    """Deflated Sharpe Ratio = PSR(SR0) with SR0 = expected_max_sr(n_trials, var_sr)."""
    sr0 = expected_max_sr(n_trials, var_sr)
    return {"sr": sr, "t": t, "skew": skew, "kurt": kurt, "n_trials": n_trials, "var_sr": var_sr, "sr0": sr0,
            "psr0": psr(sr, 0.0, t, skew, kurt), "dsr": psr(sr, sr0, t, skew, kurt)}


def deflation_gate(trade_net, trade_gross=None, trade_cost=None, dsr_value=None,
                   haircut: float = HAIRCUT, dsr_min: float = DSR_MIN) -> dict:
    """The replacement for a "PF haircut": (a) haircut on the mean edge per trade — `haircut` x mean GROSS per
    trade minus the full mean cost per trade must be > 0 (gross and cost from the trade list; when only nets exist
    the haircut falls back to the net, which is weaker and flagged); (b) DSR > dsr_min. `passed` = both,
    `either` = at least one. An additional report only: protocol verdicts are never changed by it."""
    net = [float(x) for x in trade_net]
    n = len(net)
    mean_net = sum(net) / n if n else float("nan")
    basis = "gross-cost"
    if trade_gross is not None and len(trade_gross) == n and n:
        g = sum(float(x) for x in trade_gross) / n
        c = (sum(float(x) for x in trade_cost) / n) if trade_cost is not None and len(trade_cost) == n else g - mean_net
        hc = haircut * g - c
    else:
        g, c, hc, basis = float("nan"), float("nan"), haircut * mean_net, "net (no gross/cost split)"
    hc_ok = bool(n and hc > 0)
    d_ok = bool(dsr_value is not None and not _bad(dsr_value) and dsr_value > dsr_min)
    return {"trades": n, "mean_net": mean_net, "mean_gross": g, "mean_cost": c, "haircut": haircut,
            "haircut_mean": hc, "haircut_basis": basis, "haircut_ok": hc_ok, "dsr": dsr_value, "dsr_min": dsr_min,
            "dsr_ok": d_ok, "passed": hc_ok and d_ok, "either": hc_ok or d_ok}


def _bad(v) -> bool:
    return v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))


# ───────────────────────────── trade files ─────────────────────────────

def _d8(v) -> int | None:
    """yyyymmdd from 20231110, '20231110', '2023-11-10', '20231116 15:29', '2023-11-10T00:00:00'."""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return int(m.group(1) + m.group(2) + m.group(3))
    m = re.match(r"^(\d{8})", s)
    return int(m.group(1)) if m else None


def _f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def load_trades(path: str) -> list:
    """Trade CSV of any lab layout -> [{"entry": yyyymmdd, "exit": yyyymmdd, "net", "gross", "cost"}].
    Entry = `date`; exit = `exit_date`, else a date-time `exit` (options replay), else `expiry`, else entry.
    cost = the `cost` column if present, else gross - net (option fees + slippage haircut)."""
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            e = _d8(r.get("date"))
            net = _f(r.get("net"))
            if e is None or net is None:
                continue
            x = _d8(r.get("exit_date"))
            if x is None and " " in str(r.get("exit", "")):
                x = _d8(r.get("exit"))
            if x is None:
                x = _d8(r.get("expiry"))
            gross = _f(r.get("gross"))
            cost = _f(r.get("cost"))
            if cost is None and gross is not None:
                cost = gross - net
            out.append({"entry": e, "exit": x if x and x >= e else e, "net": net, "gross": gross, "cost": cost})
    return out


def daily_pnl(trades: list, calendar: list, lo: int, hi: int) -> list:
    """Daily P&L over the sessions of `calendar` in [lo, hi]: trades whose ENTRY is in [lo, hi] (the protocol's
    rule), each net booked on its exit session (or the next session after it, if the exit day is missing),
    zero on days without an exit. Exits after `hi` are booked on the last session (positional holds)."""
    days = [d for d in calendar if lo <= d <= hi]
    if not days:
        return []
    pos = {d: i for i, d in enumerate(days)}
    pnl = [0.0] * len(days)
    import bisect
    for t in trades:
        if not lo <= t["entry"] <= hi:
            continue
        x = t["exit"]
        i = pos.get(x)
        if i is None:
            i = min(bisect.bisect_left(days, x), len(days) - 1)
        pnl[i] += t["net"]
    return pnl


def series_stats(trades: list, calendar: list, lo: int, hi: int) -> dict:
    """Per-trade and daily statistics for the trades whose entry is in [lo, hi]."""
    sel = [t for t in trades if lo <= t["entry"] <= hi]
    nets = [t["net"] for t in sel]
    tm = moments(nets)
    gp = sum(x for x in nets if x > 0)
    gl = -sum(x for x in nets if x < 0)
    dm = moments(daily_pnl(sel, calendar, lo, hi)) if calendar else moments([])
    gross = [t["gross"] for t in sel if t["gross"] is not None]
    cost = [t["cost"] for t in sel if t["cost"] is not None]
    return {"trades": tm["n"], "net": round(sum(nets), 2), "pf": round(gp / gl, 3) if gl > 0 else (float("inf") if gp > 0 else 0.0),
            "mean_net": tm["mean"], "std_net": tm["std"], "sr_trade": tm["sr"],
            "mean_gross": sum(gross) / len(gross) if gross and len(gross) == len(sel) else float("nan"),
            "mean_cost": sum(cost) / len(cost) if cost and len(cost) == len(sel) else float("nan"),
            "n_days": dm["n"], "sr_daily": dm["sr"], "skew_daily": dm["skew"], "kurt_daily": dm["kurt"],
            "nets": nets, "gross": gross if len(gross) == len(sel) else None,
            "cost": cost if len(cost) == len(sel) else None}


# ───────────────────────────── the ledger ─────────────────────────────

def family_of(sid: str) -> str:
    """Module / catalogue prefix: r3_swing_x -> r3_swing; AUC-FADE-3PCT -> AUC-FADE; cs_doji_15m -> cs."""
    s = str(sid)
    if re.match(r"^r\d+_", s):
        return "_".join(s.split("_")[:2])
    if "-" in s and s.upper() == s:
        return "-".join(s.split("-")[:2])
    return s.split("_")[0] if "_" in s else s


def _fp(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:12]


def trial_key(row: dict) -> str:
    """Stable identity of one evaluation: the same source row backfilled twice is not a second trial row."""
    return _fp(row.get("strategy_id"), row.get("params_fp"), row.get("era"), row.get("protocol"), row.get("stage"),
               row.get("dataset"), row.get("period"), row.get("cost_model"), row.get("source"))


class TrialsLedger:
    """Append-only CSV. Rows are never rewritten or deleted; `append` skips rows whose trial_key is present."""

    def __init__(self, path: str | None = None):
        self.path = path or DEFAULT_LEDGER
        from .data import assert_outside_repo
        assert_outside_repo(self.path)

    def read(self) -> list:
        if not os.path.exists(self.path):
            return []
        with open(self.path, newline="") as fh:
            return list(csv.DictReader(fh))

    def append(self, rows) -> int:
        have = {r["trial_key"] for r in self.read()}
        new = []
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        for r in rows:
            r = dict(r)
            r.setdefault("recorded_at", now)
            r["trial_key"] = r.get("trial_key") or trial_key(r)
            if r["trial_key"] in have:
                continue
            have.add(r["trial_key"])
            new.append({k: _fmt(r.get(k)) for k in FIELDS})
        if not new:
            return 0
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        fresh = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if fresh:
                w.writeheader()
            w.writerows(new)
        return len(new)


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.6g}" if abs(v) < 1e-3 or abs(v) >= 1e7 else f"{v:.6f}".rstrip("0").rstrip(".")
    return v


def record_evaluation(results, trades_by_id, *, dataset: str, cost_model: str, ledger_path: str | None,
                      protocol: str = "2023-26", era: str = "2023-26", source: str = "", calendar=None) -> int:
    """Record one `protocol.evaluate` run (every strategy, passers or not) in the trials ledger. Called by
    stratlab.run when --trials / $STRATLAB_TRIALS is set. Returns the number of rows appended."""
    if not ledger_path:
        return 0
    rows = []
    for r in results:
        sp, split = r["spec"], r.get("split")
        base = {"strategy_id": r["id"], "family": family_of(r["id"]), "group": getattr(sp, "family", ""),
                "params_fp": getattr(sp, "fingerprint", ""), "era": era, "protocol": protocol, "dataset": dataset,
                "cost_model": cost_model, "source": source}
        tr = [{"entry": int(t[2]), "exit": _d8(t[6]) or int(t[2]), "net": float(t[12]), "gross": float(t[10]),
               "cost": float(t[11])} for t in trades_by_id.get(r["id"], [])]
        if split is None:
            continue
        cal = sorted(calendar) if calendar else []
        for stage, rng, verdict in (("screen", split.discovery, r["verdict"] if r["verdict"] in
                                     ("FAILED SCREEN", "INVALID", "NO VERDICT") else "PASSED SCREEN"),
                                    ("final", split.final, r["verdict"])):
            if stage == "final" and r.get("final") is None:
                continue
            st = series_stats(tr, cal, rng[0], rng[1])
            rows.append({**base, **_row_stats(st), "stage": stage, "period": f"{rng[0]}-{rng[1]}",
                         "verdict": verdict, "note": r.get("why", "") if stage == "screen" else r.get("final_note", "")})
    return TrialsLedger(ledger_path).append(rows)


def _row_stats(st: dict) -> dict:
    return {k: st[k] for k in ("trades", "net", "pf", "mean_net", "std_net", "sr_trade", "mean_gross", "mean_cost",
                               "n_days", "sr_daily", "skew_daily", "kurt_daily")}


# ───────────────────────────── calendars ─────────────────────────────

def calendar_3y(scratch: str) -> list:
    """The lab's sessions of the 2023-26 data (726, 20231110..20261109): the union of the session days of a few
    liquid stocks in the parsed-bars cache (sessions < 200 bars dropped, as the protocol does); fallback the
    NIFTY 50 daily file bars3y/_NIFTY50_day.json (which also holds special sessions)."""
    try:
        from . import data as D
        days: set = set()
        for cache in sorted(glob.glob(os.path.join(scratch, "stratlab_cache", "bars3y_*"))):
            for sym in ("RELIANCE", "HDFCBANK", "INFY", "SBIN", "TCS"):
                sd = D.load_symbol(cache, sym)
                if sd is not None:
                    days.update(int(x) for x in sd.days)
            if days:
                return sorted(days)
    except Exception:
        pass
    p = os.path.join(scratch, "bars3y", "_NIFTY50_day.json")
    try:
        with open(p) as fh:
            return sorted({_d8(b[0]) for b in json.load(fh)["bars"]})
    except Exception:
        return []


def calendar_old(scratch: str) -> list:
    """NSE sessions 2012-2023 from nse_old/index_daily.csv.gz (NIFTY 50 rows)."""
    p = os.path.join(scratch, "nse_old", "index_daily.csv.gz")
    out = set()
    try:
        with gzip.open(p, "rt", newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("index") in ("NIFTY", "NIFTY50", "NIFTY 50"):
                    out.add(int(r["date"]))
    except Exception:
        return []
    return sorted(out)


# ───────────────────────────── backfill ─────────────────────────────

SPLIT_3Y = {"disc": (20231110, 20251216), "final": (20251217, 20261109)}
OLD_ERA = (20120101, 20231109)
_SKIP_DIRS = ("venv311", "venv_check", "stratlab_cache", "node_modules", "__pycache__", "/trades", "_parts",
              "kronos_src", "ta_src", "final_intraday/algotrader_v4", "e2e_", "oldera_oos/run1")


def _skip(path: str) -> bool:
    return any(s in path for s in _SKIP_DIRS)


def _summary_meta(d: str) -> dict:
    """dataset, DISCOVERY and FINAL ranges from a run's summary.txt (stratlab.run / module layouts)."""
    meta = {"dataset": "", "disc": None, "final": None, "cost_model": ""}
    p = os.path.join(d, "summary.txt")
    if not os.path.exists(p):
        return meta
    with open(p, errors="replace") as fh:
        head = fh.read(6000)
    m = re.search(r"data=(\S+)", head)
    if m:
        meta["dataset"] = m.group(1)
    m = re.search(r"cost model[ =]+([^\s,|]+)", head)
    if m:
        meta["cost_model"] = m.group(1)
    m = re.search(r"DISCOVERY\s+(\d{8})\s*\.\.\s*(\d{8})", head)
    if m:
        meta["disc"] = (int(m.group(1)), int(m.group(2)))
    m = re.search(r"FINAL\s+(\d{8})\s*\.\.\s*(\d{8})", head)
    if m:
        meta["final"] = (int(m.group(1)), int(m.group(2)))
    return meta


def _ds_tag(dataset: str) -> str:
    b = os.path.basename(dataset.rstrip("/")) if dataset else ""
    return b or "bars3y"


def _trades_path(d: str, sid: str) -> str:
    for p in (os.path.join(d, "trades", sid + ".csv"), os.path.join(d, f"trades_{sid}.csv")):
        if os.path.exists(p):
            return p
    return ""


def _stats_for(path: str, cal: list, rng, cache: dict) -> dict:
    if not path or not rng or not cal:
        return {}
    if path not in cache:
        try:
            cache[path] = load_trades(path)
        except Exception:
            cache[path] = []
    return _row_stats(series_stats(cache[path], cal, rng[0], rng[1])) if cache[path] else {}


def _results_rows(scratch: str, cal3: list, tcache: dict) -> list:
    """Every results.csv written by a protocol run (stratlab.run, r3/r4 drivers, options modules)."""
    rows = []
    for p in sorted(glob.glob(os.path.join(scratch, "**", "results.csv"), recursive=True)):
        if _skip(p):
            continue
        d = os.path.dirname(p)
        rel = os.path.relpath(d, scratch)
        with open(p, newline="") as fh:
            rd = csv.DictReader(fh)
            if not rd.fieldnames or "id" not in rd.fieldnames or "disc_trades" not in rd.fieldnames:
                continue
            recs = list(rd)
        meta = _summary_meta(d)
        tag = _ds_tag(meta["dataset"])
        disc = meta["disc"] or (SPLIT_3Y["disc"] if tag == "bars3y" else None)
        smoke = "smoke" in rel.split(os.sep)
        for r in recs:
            sid = r["id"]
            tp = _trades_path(d, sid)
            st = _stats_for(tp, cal3, disc, tcache) if tag == "bars3y" else {}
            row = {"strategy_id": sid, "family": family_of(sid), "group": r.get("family", ""),
                   "params_fp": ("p:" + _fp(r["params"])) if r.get("params") else r.get("fingerprint", ""),
                   "era": "2023-26", "protocol": "2023-26 discovery/final", "stage": "screen-smoke" if smoke else "screen",
                   "dataset": tag, "period": f"{disc[0]}-{disc[1]}" if disc else "",
                   "cost_model": r.get("cost_model") or meta["cost_model"], "trades": _f(r.get("disc_trades")),
                   "net": _f(r.get("disc_net")), "pf": _f(r.get("disc_pf")),
                   "verdict": r.get("verdict", ""), "source": os.path.join(rel, "results.csv"),
                   "trades_file": tp, "note": r.get("why", "")}
            if st:                                           # prefer the trade list (exact) when it agrees
                row.update({k: v for k, v in st.items() if k not in ("trades", "net", "pf")})
            rows.append(row)
    return rows


def _report_rows(scratch: str, cal3: list, tcache: dict) -> list:
    """report.json files of the own-fill-model modules (auction, auction_r2, attention)."""
    rows = []
    for p in sorted(glob.glob(os.path.join(scratch, "**", "report.json"), recursive=True)):
        if _skip(p):
            continue
        try:
            with open(p) as fh:
                rep = json.load(fh)
        except Exception:
            continue
        if not isinstance(rep, dict) or not isinstance(rep.get("strategies"), dict):
            continue
        d, rel = os.path.dirname(p), os.path.relpath(os.path.dirname(p), _scratch_of(p, scratch))
        sp = rep.get("split") or {}
        disc = (sp["h1"][0], sp["h2"][1]) if sp.get("h1") and sp.get("h2") else SPLIT_3Y["disc"]
        for sid, v in rep["strategies"].items():
            dd = v.get("DISC") or {}
            tp = _trades_path(d, sid)
            st = _stats_for(tp, cal3, disc, tcache)
            row = {"strategy_id": sid, "family": family_of(sid), "group": "module", "params_fp": v.get("fingerprint", ""),
                   "era": "2023-26", "protocol": "2023-26 discovery/final", "stage": "screen", "dataset": "bars3y",
                   "period": f"{disc[0]}-{disc[1]}", "cost_model": rep.get("cost_model", ""),
                   "trades": dd.get("trades"), "net": dd.get("net"), "pf": dd.get("pf"),
                   "verdict": "FAILED SCREEN" if v.get("verdict") == "FAILED SCREEN" else "PASSED SCREEN",
                   "source": os.path.join(rel, "report.json"), "trades_file": tp, "note": v.get("why", "")}
            row.update({k: x for k, x in st.items() if k not in ("trades", "net", "pf")})
            rows.append(row)
    return rows


def _scratch_of(p, scratch):
    return scratch


def _ltr_rows(scratch: str, cal3: list, tcache: dict) -> list:
    rows = []
    for d in sorted(glob.glob(os.path.join(scratch, "**", "disc_excess_trades.csv"), recursive=True)):
        if _skip(d):
            continue
        dd = os.path.dirname(d)
        txt = open(os.path.join(dd, "summary.txt"), errors="replace").read() if os.path.exists(
            os.path.join(dd, "summary.txt")) else ""
        m = re.search(r"^(\S+)\s+\(stratlab/ltr\.py, cost model (\S+), fingerprint (\w+)\)", txt, re.M)
        sid, cm, fpr = (m.group(1), m.group(2).rstrip(","), m.group(3)) if m else ("LTR-12-1-LOSERS-CNC", "", "")
        verdict = "FAILED SCREEN" if "FAILED SCREEN" in txt else ("PASSED SCREEN" if "PASSED SCREEN" in txt else "")
        st = _stats_for(d, cal3, SPLIT_3Y["disc"], tcache)
        rows.append({"strategy_id": sid, "family": family_of(sid), "group": "ltr", "params_fp": fpr, "era": "2023-26",
                     "protocol": "2023-26 discovery/final", "stage": "screen", "dataset": "bars3y",
                     "period": "%d-%d" % SPLIT_3Y["disc"], "cost_model": cm, **st, "verdict": verdict,
                     "source": os.path.relpath(d, scratch), "trades_file": d,
                     "note": "excess over the equal-weight universe per name-period"})
    return rows


def _reproduces(trades: list, cal: list, lo: int, hi: int, want_trades, want_net) -> bool:
    """True when the trades with entry in [lo, hi] reproduce a ledger's trade count exactly and its net within
    rounding (the ledgers store net to 2 decimals). A file that does not is another run's (another dataset, cost
    model or code version) and must not lend its Sharpe / moments to the row."""
    wt, wn = _f(want_trades), _f(want_net)
    if wt is None or wn is None:
        return False
    st = series_stats(trades, cal, lo, hi)
    return st["trades"] == int(round(wt)) and abs(st["net"] - wn) <= 0.05 + 1e-6 * abs(wn)


def _final_rows(scratch: str, cal3: list, tcache: dict, screen_rows: list) -> list:
    """The one-look FINAL ledgers (authoritative FINAL verdicts).

    A FINAL entry borrows its daily Sharpe / moments from a full-run trade file of a screen row only when that row
    is of the SAME dataset, of the same cost model (where both record one: results.csv column, else the run's
    summary.txt; a run that records none is judged by the next test alone), and the file reproduces the entry's
    FINAL trade count and net. Otherwise the Sharpe columns stay blank: the trade count / net of the ledger
    must never be mixed with another run's statistics."""
    tp_by_id = {}
    for r in screen_rows:                     # a FULL-run trade file holds the FINAL trades of a screen passer
        if r.get("trades_file") and r.get("stage") == "screen":
            tp_by_id.setdefault(r["strategy_id"], []).append((r.get("dataset", ""), r.get("cost_model", ""),
                                                              r["trades_file"]))
    rows = []
    paths = sorted(set(glob.glob(os.path.join(scratch, "stratlab_cache", "final_ledger.json"))
                       + glob.glob(os.path.join(scratch, "stratlab_cache", "*", "final_ledger.json"))))
    for p in paths:
        with open(p) as fh:
            led = json.load(fh)
        for key, v in led.items():
            parts = key.split("|")
            sid, rng = parts[0], parts[1]
            lo, hi = (int(x) for x in rng.split("-"))
            ds = _ds_tag(v.get("dataset", ""))
            cm = v.get("cost_model") or ""
            f = v.get("final", {})
            st, tp = {}, ""
            if ds == "bars3y" and cal3:
                for cds, ccm, cand in tp_by_id.get(sid, []):
                    if cds != ds or (cm and ccm and ccm != cm) or not os.path.exists(cand):
                        continue
                    if cand not in tcache:
                        try:
                            tcache[cand] = load_trades(cand)
                        except Exception:
                            tcache[cand] = []
                    if _reproduces(tcache[cand], cal3, lo, hi, f.get("trades"), f.get("net")):
                        tp = cand
                        break
                st = _stats_for(tp, cal3, (lo, hi), tcache) if tp else {}
            verdict = ("PEEK" if v.get("peeked") else "RETEST" if v.get("tainted")
                       else "CONFIRMED" if f.get("confirmed") else "REJECTED ON FINAL")
            row = {"strategy_id": sid, "family": family_of(sid), "group": "", "params_fp": v.get("fingerprint", ""),
                   "era": "2023-26", "protocol": "2023-26 discovery/final", "stage": "final", "dataset": ds,
                   "period": rng, "cost_model": cm or "legacy", "trades": f.get("trades"),
                   "net": f.get("net"), "pf": f.get("pf"), "verdict": verdict,
                   "source": os.path.relpath(p, scratch) + "#" + key, "trades_file": tp, "note": f"seen {v.get('when', '')}"}
            row.update({k: x for k, x in st.items() if k not in ("trades", "net", "pf")})
            rows.append(row)
    return rows


def _oldera_rows(scratch: str, calo: list, tcache: dict) -> list:
    p = os.path.join(scratch, "stratlab_cache", "oldera_oos_ledger.json")
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        led = json.load(fh)
    tfiles = {os.path.splitext(os.path.basename(x))[0]: x
              for x in glob.glob(os.path.join(scratch, "oldera_oos", "run1", "b*", "trades", "*.csv"))}
    rows = []
    for key, v in led.items():
        sid = key.split("|")[0]
        res = v.get("result", {})
        era = res.get("era") or {}
        tp = tfiles.get(sid, "")
        st = _stats_for(tp, calo, OLD_ERA, tcache)
        row = {"strategy_id": sid, "family": family_of(sid), "group": "", "params_fp": v.get("fingerprint", ""),
               "era": "2012-23", "protocol": "2012-23 OOS", "stage": "oos", "dataset": "nse_old",
               "period": "%d-%d" % OLD_ERA, "cost_model": v.get("cost_model", ""), "trades": era.get("trades"),
               "net": era.get("net"), "pf": era.get("pf"), "verdict": res.get("verdict", ""),
               "source": "stratlab_cache/oldera_oos_ledger.json#" + key, "trades_file": tp, "note": res.get("why", "")}
        row.update({k: x for k, x in st.items() if k not in ("trades", "net", "pf")})
        rows.append(row)
    return rows


def _explore_rows(scratch: str) -> list:
    """Exploratory effect measurements (no protocol verdict, no trade list). Each named effect is one trial."""
    rows = []

    def add(src, name, n=None, mean=None, note="", group="explore"):
        rows.append({"strategy_id": "X:" + str(name)[:120], "family": "explore:" + src.split(os.sep)[-2]
                     if os.sep in src else "explore", "group": group, "params_fp": _fp(src, name),
                     "era": "2023-26", "protocol": "2023-26 exploratory", "stage": "explore", "dataset": "bars3y",
                     "period": "", "cost_model": "", "trades": n, "mean_net": mean, "verdict": "MEASURED",
                     "source": src, "note": note})

    def j(rel):
        p = os.path.join(scratch, rel)
        try:
            with open(p) as fh:
                return json.load(fh)
        except Exception:
            return None

    d = j("dd/round1_results.json")
    for x in (d or {}).get("facts", []):
        add("dd/round1_results.json", x.get("name"), x.get("n_obs"), None, f"t {x.get('t_stat')} (bps effect, gross)")
    for rel in sorted(glob.glob(os.path.join(scratch, "dd", "r2_intraday", "effects*.json"))):
        for x in j(os.path.relpath(rel, scratch)) or []:
            a = x.get("ALL", {})
            add(os.path.relpath(rel, scratch), x.get("name"), a.get("n"), a.get("net2"), f"t_day {a.get('t_day')} (bps net@2)")
    for rel in ("dd/r2_swing/effects.json", "dd/r2_swing/attention.json", "dd/r2_swing/reversal.json"):
        for x in j(rel) or []:
            a = x.get("all", {})
            add(rel, x.get("name"), a.get("n"), a.get("mean_bps"), f"t_cal {a.get('t_cal')} (bps gross)")
    for rel in sorted(glob.glob(os.path.join(scratch, "dd", "r2_fno", "m*.json"))):
        for x in j(os.path.relpath(rel, scratch)) or []:
            add(os.path.relpath(rel, scratch), x.get("name"), x.get("n"), x.get("mean"), f"t {x.get('t')} (bps gross)")
    for x in j("dd/xs/xs_results.json") or []:
        a = x.get("ALL", {})
        add("dd/xs/xs_results.json", x.get("name"), a.get("n_form"), a.get("ls_net"), f"ls_t {a.get('ls_t')}")
    for x in j("dd/xs/leadlag.json") or []:
        add("dd/xs/leadlag.json", f"leadlag k{x.get('k')} h{x.get('h')} d{x.get('delay')}",
            (x.get("n_H1") or 0) + (x.get("n_H2") or 0), None, f"excess t H1 {x.get('excess_H1_t')} H2 {x.get('excess_H2_t')}")
    for k, x in (j("dd/vol/opts_out.json") or {}).items():
        a = x.get("ALL", {}) if isinstance(x, dict) else {}
        add("dd/vol/opts_out.json", k, a.get("n"), a.get("net_lot"), f"t {a.get('t')}")
    seen = set()
    for k in (j("dd/vol/vrp_out.json") or {}):
        base = re.sub(r"_(H1|H2|ALL)$", "", k)
        if base not in seen:
            seen.add(base)
            add("dd/vol/vrp_out.json", base, None, None, "volatility risk premium measurement")
    for rel in ("dd/ev/effects.csv", "dd/r2_scalping/summary.csv", "dd/r2_scalping/summary_events3.pkl.csv"):
        p = os.path.join(scratch, rel)
        if os.path.exists(p):
            with open(p, newline="") as fh:
                for r in csv.DictReader(fh):
                    name = r.get("name") or r.get("effect")
                    add(rel, name, _f(r.get("n")) or ((_f(r.get("n1")) or 0) + (_f(r.get("n2")) or 0)) or None,
                        _f(r.get("net2")) if r.get("net2") is not None else _f(r.get("all")), f"t {r.get('t')}")
    for rel in ("pm_robust_edges/out3y/results.json",):
        for x in (j(rel) or {}).get("rows", []):
            dd = x.get("DISC", {})
            rows.append({"strategy_id": x["id"], "family": family_of(x["id"]), "group": "pm_robust_edges",
                         "params_fp": _fp(rel, x["id"]), "era": "2023-26", "protocol": "2023-26 discovery (screen-like)",
                         "stage": "screen", "dataset": "bars3y", "period": "%d-%d" % SPLIT_3Y["disc"],
                         "cost_model": "2026-11/f2", "trades": dd.get("trades"), "net": dd.get("net"), "pf": dd.get("pf"),
                         "mean_net": dd.get("net_per_trade"),
                         "verdict": "PASSED SCREEN" if x.get("screen_passed_2bps") else "FAILED SCREEN",
                         "source": rel, "note": str(x.get("screen_why", ""))})
    so = j("dd/screen_old.json")
    for sid, x in ((so or {}).get("results") or {}).items():
        rows.append({"strategy_id": sid, "family": family_of(sid), "group": x.get("kind", ""),
                     "params_fp": _fp("screen_old", sid), "era": "2023-26",
                     "protocol": "2023-26 discovery/final", "stage": "screen", "dataset": "bars3y(old code)",
                     "period": "%d-%d" % tuple(so["meta"]["discovery"]), "cost_model": "pre-2026-11",
                     "trades": x.get("trades"), "net": x.get("net"), "pf": x.get("pf"),
                     "mean_net": (x["net"] / x["trades"]) if x.get("trades") else None,
                     "mean_gross": (x["gross"] / x["trades"]) if x.get("trades") and x.get("gross") is not None else None,
                     "mean_cost": (x["cost"] / x["trades"]) if x.get("trades") and x.get("cost") is not None else None,
                     "verdict": "PASSED SCREEN" if x.get("passed") else "FAILED SCREEN",
                     "source": "dd/screen_old.json", "note": ",".join(x.get("why") or [])})
    return rows


def backfill(scratch: str = _SCRATCH_DEFAULT, ledger_path: str | None = None, verbose: bool = True) -> dict:
    """Rebuild the trials ledger from everything on disk. Idempotent: rows already present are not appended
    again. Returns counts by source kind."""
    cal3, calo = calendar_3y(scratch), calendar_old(scratch)
    tcache: dict = {}
    screen = _results_rows(scratch, cal3, tcache) + _report_rows(scratch, cal3, tcache) + _ltr_rows(scratch, cal3, tcache)
    final = _final_rows(scratch, cal3, tcache, screen)
    old = _oldera_rows(scratch, calo, tcache)
    expl = _explore_rows(scratch)
    led = TrialsLedger(ledger_path)
    n = led.append(screen + final + old + expl)
    out = {"screen_rows": len(screen), "final_rows": len(final), "oldera_rows": len(old), "explore_rows": len(expl),
           "appended": n, "ledger": led.path}
    if verbose:
        print(json.dumps(out, indent=1))
    return out


# ───────────────────────────── counting trials ─────────────────────────────

def count_trials(rows: list, era: str = "2023-26", stages=None) -> int:
    """Honest N: every strategy variant evaluated on the era's data in any stage — passers or not. A variant is a
    strategy id, and additionally each distinct parameter set run under the same id (results.csv `params`,
    fingerprint "p:..."; e.g. reviewer neighbour runs that kept the id). A re-run of the same id with the same
    parameters (smoke run, another cost model or slippage, a reviewer rebuild, another dataset) does not count
    twice. `stages` restricts the stages counted (e.g. {"final"})."""
    var: dict = {}
    for r in rows:
        if r.get("era") == era and (stages is None or r.get("stage") in stages):
            s = var.setdefault(r["strategy_id"], set())
            if str(r.get("params_fp", "")).startswith("p:"):
                s.add(r["params_fp"])
    return sum(max(1, len(s)) for s in var.values())


def _sr_values(rows, era, stage, dataset) -> list:
    by_id = {}
    for r in rows:
        if r.get("era") == era and r.get("stage") == stage and r.get("dataset") == dataset:
            v = _f(r.get("sr_daily"))
            if v is not None and r["strategy_id"] not in by_id:
                by_id[r["strategy_id"]] = v
    return list(by_id.values())


def var_sr(rows: list, era: str = "2023-26", stage: str = "screen", dataset: str = "bars3y") -> tuple:
    """Sample variance across trials of the daily Sharpe (one value per strategy id: its first row with a daily
    SR on `dataset`). Returns (variance, n, mean)."""
    x = _sr_values(rows, era, stage, dataset)
    if len(x) < 2:
        return float("nan"), len(x), float("nan")
    m = sum(x) / len(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - 1), len(x), m


def var_sr_robust(rows: list, era: str = "2023-26", stage: str = "screen", dataset: str = "bars3y") -> tuple:
    """(1.4826 x MAD)^2 of the same Sharpe values: the cross-trial dispersion without the tail of high-turnover
    strategies whose deterministic cost drag gives daily SRs of -1 to -5 (not luck). Returns (variance, n, median)."""
    x = sorted(_sr_values(rows, era, stage, dataset))
    if len(x) < 2:
        return float("nan"), len(x), float("nan")
    med = _median(x)
    mad = _median(sorted(abs(v - med) for v in x))
    return (1.4826 * mad) ** 2, len(x), med


def _median(x):
    n = len(x)
    return x[n // 2] if n % 2 else 0.5 * (x[n // 2 - 1] + x[n // 2])


# ───────────────────────────── DSR report ─────────────────────────────

DEPLOYED = ("AUC-FADE-3PCT", "SWING-ATTN-ON-3X", "VRP-STRANGLE-016-WINGED")


def report(ledger_path: str | None = None, out_dir: str | None = None, scratch: str = _SCRATCH_DEFAULT,
           verbose: bool = True) -> dict:
    """DSR / PSR and the deflation gate for the deployed strategies and every FINAL-CONFIRMED ledger entry, on
    the 3-year data (726 sessions) from each strategy's full-run trade list (DISCOVERY + FINAL trades).

    V[SR] — three estimates of the cross-trial variance of the daily Sharpe:
      robust (HEADLINE): (1.4826 MAD)^2 over the DISCOVERY Sharpes of all trials with a trade list;
      naive: their sample variance (dominated by the cost-drag tail, so SR0 is unreachably high);
      floor: 1/(T-1), the sampling variance of a Sharpe estimate of a zero-edge strategy — the most lenient value
             (it assumes the trials differ by noise only).
    For the FINAL period, N = the number of FINAL looks and V = the sample variance of their FINAL Sharpes."""
    rows = TrialsLedger(ledger_path).read()
    cal = calendar_3y(scratch)
    n_all = count_trials(rows, "2023-26")
    n_proto = count_trials(rows, "2023-26", {"screen", "screen-smoke", "final"})
    n_final = count_trials(rows, "2023-26", {"final"})
    v_naive, n_v, m_v = var_sr(rows, "2023-26", "screen")
    v_rob, _, med_v = var_sr_robust(rows, "2023-26", "screen")
    v_fin, n_vf, _ = var_sr(rows, "2023-26", "final", "bars3y")
    conf_rows = [r for r in rows if r.get("stage") == "final" and r.get("verdict") == "CONFIRMED"]
    confirmed = sorted({r["strategy_id"] for r in conf_rows})
    conf_ds = {}
    for r in conf_rows:
        conf_ds.setdefault(r["strategy_id"], set()).add(r.get("dataset", ""))
    targets = list(DEPLOYED) + [s for s in confirmed if s not in DEPLOYED]
    out = {"n_trials_all": n_all, "n_trials_protocol": n_proto, "n_final_looks": n_final,
           "var_sr_robust": v_rob, "median_sr_disc": med_v, "var_sr_naive": v_naive, "mean_sr_disc": m_v,
           "var_sr_disc_from": n_v, "var_sr_final": v_fin, "var_sr_final_from": n_vf,
           "calendar_sessions": len(cal), "strategies": []}
    lo, hi = SPLIT_3Y["disc"][0], SPLIT_3Y["final"][1]
    tl_cache: dict = {}

    def _tl(f):
        if f not in tl_cache:
            try:
                tl_cache[f] = load_trades(f)
            except Exception:
                tl_cache[f] = []
        return tl_cache[f]

    def _period(r):
        try:
            a, b = (int(x) for x in str(r.get("period", "")).split("-"))
            return a, b
        except ValueError:
            return None

    for sid in targets:
        # A 3-year trade file that holds FINAL trades (full run). The bars3y FINAL row's file comes first; any other
        # row's file is used only if it reproduces that row's own trades / net in its period, has the FINAL row's
        # cost model (where the FINAL row records one) and reproduces the FINAL row's trades / net. A file from
        # another dataset, cost model or code version must not stand in for the strategy.
        frows = [r for r in rows if r["strategy_id"] == sid and r.get("stage") == "final" and r.get("dataset") == "bars3y"]
        cands = [r for r in rows if r["strategy_id"] == sid and r.get("dataset") == "bars3y" and r.get("trades_file")
                 and os.path.exists(r["trades_file"]) and r.get("stage") in ("final", "screen", "screen-smoke")]
        cands.sort(key=lambda r: r.get("stage") != "final")
        tp = ""
        for r in cands:
            f, per = r["trades_file"], _period(r)
            tr = _tl(f)
            if not tr or not per or not cal or not _reproduces(tr, cal, per[0], per[1], r.get("trades"), r.get("net")):
                continue
            ok = True
            for fr in frows:
                fcm, fper = fr.get("cost_model", ""), _period(fr)
                if fcm and fcm != "legacy" and r.get("cost_model") and r.get("cost_model") != fcm:
                    ok = False
                elif fper and not _reproduces(tr, cal, fper[0], fper[1], fr.get("trades"), fr.get("net")):
                    ok = False
            if ok and (frows or any(t["entry"] >= SPLIT_3Y["final"][0] for t in tr)):
                tp = f
                break
        ent = {"id": sid, "deployed": sid in DEPLOYED, "confirmed": sid in confirmed, "trades_file": tp,
               "confirmed_on": sorted(conf_ds.get(sid, []))}
        if not tp or not cal:
            ent["error"] = ("no 3-year trade list with FINAL trades (FINAL confirmed on "
                            + ",".join(ent["confirmed_on"]) + "; the 3-year run never opened its FINAL)")
            out["strategies"].append(ent)
            continue
        tr = load_trades(tp)
        for name, rng in (("disc", SPLIT_3Y["disc"]), ("final", SPLIT_3Y["final"]), ("full", (lo, hi))):
            st = series_stats(tr, cal, *rng)
            sr, t, sk, ku = st["sr_daily"], st["n_days"], st["skew_daily"], st["kurt_daily"]
            e = {"trades": st["trades"], "net": st["net"], "pf": st["pf"], "mean_net": st["mean_net"],
                 "sr_daily": sr, "sr_ann": sr * math.sqrt(250) if not _bad(sr) else float("nan"),
                 "t": t, "skew": sk, "kurt": ku, "psr0": psr(sr, 0.0, t, sk, ku)}
            if name in ("disc", "full"):
                floor = 1.0 / (t - 1) if t and t > 1 else float("nan")
                e["sr0"] = expected_max_sr(n_all, v_rob)
                e["dsr"] = dsr(sr, t, sk, ku, n_all, v_rob)["dsr"]                     # headline
                e["dsr_protocol"] = dsr(sr, t, sk, ku, n_proto, v_rob)["dsr"]
                e["dsr_naive"] = dsr(sr, t, sk, ku, n_all, v_naive)["dsr"]
                e["sr0_floor"] = expected_max_sr(n_all, floor)
                e["dsr_floor"] = dsr(sr, t, sk, ku, n_all, floor)["dsr"]
            else:
                vf = v_fin if not _bad(v_fin) else 1.0 / (t - 1)
                e["sr0"] = expected_max_sr(n_final, vf)
                e["dsr"] = dsr(sr, t, sk, ku, n_final, vf)["dsr"]
                e["dsr_protocol"] = e["dsr_naive"] = float("nan")
                e["dsr_floor"] = dsr(sr, t, sk, ku, n_final, 1.0 / (t - 1))["dsr"] if t and t > 1 else float("nan")
            g = deflation_gate(st["nets"], st["gross"], st["cost"], e["dsr"])
            e["gate"] = {k: g[k] for k in ("mean_gross", "mean_cost", "haircut_mean", "haircut_basis", "haircut_ok",
                                           "dsr_ok", "passed", "either")}
            ent[name] = e
        out["strategies"].append(ent)
    text = format_report(out)
    if out_dir:
        from .data import assert_outside_repo
        out_dir = assert_outside_repo(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "dsr_report.json"), "w") as fh:
            json.dump(out, fh, indent=1, default=str)
        with open(os.path.join(out_dir, "dsr_report.txt"), "w") as fh:
            fh.write(text)
    if verbose:
        print(text)
    return out


def format_report(o: dict) -> str:
    sq = lambda v: math.sqrt(v) if v == v and v >= 0 else float("nan")   # noqa: E731
    L = ["DEFLATED SHARPE RATIO REPORT (Bailey & Lopez de Prado 2014) - an additional report; protocol verdicts unchanged",
         "=" * 132,
         f"N = {o['n_trials_all']} variants evaluated on 2023-26 data (every screened variant, passers or not, incl. "
         f"exploratory effect tests); protocol screens/finals only: {o['n_trials_protocol']}; FINAL looks: {o['n_final_looks']}",
         f"V[SR] daily over {o['var_sr_disc_from']} DISCOVERY trials with trade lists: robust (headline) "
         f"{o['var_sr_robust']:.6f} (sd {sq(o['var_sr_robust']):.4f}, median SR {o['median_sr_disc']:.4f}); naive "
         f"{o['var_sr_naive']:.6f} (sd {sq(o['var_sr_naive']):.4f}, mean {o['mean_sr_disc']:.4f}); floor 1/(T-1). "
         f"FINAL: {o['var_sr_final']:.6f} over {o['var_sr_final_from']} FINAL looks",
         "SR = daily P&L Sharpe (non-annualised; SRann = x sqrt(250)), T = sessions, kurt non-excess. "
         "SR0 = expected best-of-N Sharpe of zero-edge trials. DSR = P(true SR > SR0).",
         "Gate (additional): hc/tr = 0.5 x mean gross per trade - mean cost per trade must be > 0; DSR (headline) > 0.95.",
         ""]
    hdr = (f"{'strategy':<44}{'per':<6}{'tr':>5}{'net':>11}{'PF':>6}{'SRd':>8}{'SRann':>7}{'T':>5}{'skew':>7}"
           f"{'kurt':>7}{'PSR0':>6}{'SR0':>7}{'DSR':>6}{'DSRp':>6}{'DSRnv':>6}{'DSRfl':>6}{'hc/tr':>8} gate")
    L += [hdr, "-" * len(hdr)]
    for s in o["strategies"]:
        tag = s["id"] + (" [deployed]" if s.get("deployed") else "") + (" [CONF]" if s.get("confirmed") else "")
        if s.get("error"):
            L.append(f"{tag:<44}{s['error']}")
            continue
        for per in ("disc", "final", "full"):
            e = s[per]
            g = e["gate"]
            gs = "PASS" if g["passed"] else ("hc-only" if g["haircut_ok"] else ("dsr-only" if g["dsr_ok"] else "FAIL"))
            L.append(f"{(tag if per == 'disc' else ''):<44}{per:<6}{e['trades']:>5}{e['net']:>11,.0f}{_n(e['pf'], 6, 2)}"
                     f"{_n(e['sr_daily'], 8, 4)}{_n(e['sr_ann'], 7, 2)}{e['t']:>5}{_n(e['skew'], 7, 2)}{_n(e['kurt'], 7, 1)}"
                     f"{_n(e['psr0'], 6, 2)}{_n(e['sr0'], 7, 4)}{_n(e['dsr'], 6, 2)}{_n(e['dsr_protocol'], 6, 2)}"
                     f"{_n(e['dsr_naive'], 6, 2)}{_n(e['dsr_floor'], 6, 2)}{_n(g['haircut_mean'], 8, 0)} {gs}")
    L += ["", "Columns: DSR = N all variants, robust V (headline; final rows: N = FINAL looks, V of FINAL Sharpes); "
          "DSRp = N protocol screens only; DSRnv = naive V; DSRfl = floor V 1/(T-1) (most lenient).",
          "Reading: 'disc' deflates the in-sample Sharpe on which the selection was made; 'full' = DISCOVERY+FINAL; "
          "'final' deflates the hold-out Sharpe by the number of FINAL looks only. N treats correlated variants as "
          "independent (it overstates the effective N), but SR0 grows only like sqrt(2 ln N)."]
    return "\n".join(L) + "\n"


def _n(v, w, p):
    if v is None or (isinstance(v, float) and (math.isnan(v))):
        return f"{'-':>{w}}"
    if isinstance(v, float) and math.isinf(v):
        return f"{'inf':>{w}}"
    return f"{v:>{w},.{p}f}"


def summary(ledger_path: str | None = None) -> str:
    rows = TrialsLedger(ledger_path).read()
    from collections import Counter
    c = Counter((r["era"], r["stage"]) for r in rows)
    L = [f"trials ledger {TrialsLedger(ledger_path).path}: {len(rows)} rows"]
    for (era, st), k in sorted(c.items()):
        L.append(f"  {era:<8} {st:<13} {k:>6} rows  {count_trials(rows, era, {st}):>5} unique ids")
    L.append(f"  N 2023-26 (variants, all stages) = {count_trials(rows, '2023-26')}; "
             f"N 2012-23 = {count_trials(rows, '2012-23')}")
    v = Counter(r["verdict"] for r in rows if r["era"] == "2023-26" and r["stage"] == "final")
    L.append("  2023-26 FINAL verdicts: " + ", ".join(f"{k} {n}" for k, n in sorted(v.items())))
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.trials", description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("backfill", "summary", "report"):
        s = sub.add_parser(name)
        s.add_argument("--ledger", default=None, help=f"trials CSV (default {DEFAULT_LEDGER})")
        s.add_argument("--scratch", default=_SCRATCH_DEFAULT, help="scratch root holding the run outputs")
        if name == "report":
            s.add_argument("--out", default=None, help="write dsr_report.txt/json here (outside the repository)")
    a = ap.parse_args(argv)
    if a.cmd == "backfill":
        backfill(a.scratch, a.ledger)
    elif a.cmd == "summary":
        print(summary(a.ledger))
    else:
        report(a.ledger, a.out, a.scratch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
