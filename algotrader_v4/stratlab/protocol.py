"""
stratlab/protocol.py — the pre-registered judgement. These numbers were fixed before any strategy was run.
DO NOT TUNE THEM TO RESULTS: a threshold moved after seeing results is no longer a test.

  sessions (chronological)   DISCOVERY = first 70 %  (H1 = its first half, H2 = its second half)
                             FINAL     = last 30 %
  SCREEN, on DISCOVERY only  net after costs > 0 in H1 AND in H2
                             profit factor >= 1.15 over DISCOVERY
                             trades >= 150 (intraday, scalping, futures-intraday) or >= 60 (swing, futures-positional)
                             no single symbol contributes more than 35 % of the net
  FINAL, only for strategies that passed the screen, reported once per strategy
                             CONFIRMED = net > 0 and profit factor >= 1.05

A trade belongs to the period of its ENTRY date. With fewer than MIN_SESSIONS sessions the lab gives no
verdict at all ("too few sessions") and FINAL is never simulated.

FINAL results of screen failures are never computed: `evaluate()` only simulates the FINAL sessions for
strategies that passed, unless the caller explicitly passes peek_final=True (the CLI flag --peek-final), and a
peek is recorded in the ledger so that it can never later be presented as a confirmation.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

DISCOVERY_FRAC = 0.70
SCREEN_MIN_PF = 1.15
SCREEN_MIN_TRADES = {"intraday": 150, "positional": 60}     # by horizon
SCREEN_MAX_SYMBOL_SHARE = 0.35
FINAL_MIN_PF = 1.05
MIN_SESSIONS = 100              # below this there is no verdict, whatever the numbers look like
GATE_LUCK = 0.05                # about 5 % of useless strategies pass each coin-flip-like gate

_NET = 12                       # index of "net" in sim.TRADE_FIELDS
_SYM = 1
_DATE = 2


@dataclass
class Split:
    h1: tuple            # (first_date, last_date) inclusive, yyyymmdd
    h2: tuple
    final: tuple
    n: int
    n_discovery: int
    n_final: int

    @property
    def discovery(self) -> tuple:
        return (self.h1[0], self.h2[1])


def split_sessions(calendar) -> Split | None:
    """Chronological 70 / 30 split of the session dates, discovery halved. None if it cannot be split."""
    days = sorted(set(int(x) for x in calendar))
    n = len(days)
    nd = int(n * DISCOVERY_FRAC)
    nh = nd // 2
    if nh < 1 or nd - nh < 1 or n - nd < 1:
        return None
    return Split((days[0], days[nh - 1]), (days[nh], days[nd - 1]), (days[nd], days[-1]), n, nd, n - nd)


def stats(trades) -> dict:
    """trades, win %, profit factor and net (all after costs) for a list of trade tuples."""
    nets = [t[_NET] for t in trades]
    gp = sum(x for x in nets if x > 0)
    gl = -sum(x for x in nets if x < 0)
    n = len(nets)
    return {"trades": n, "win_pct": round(100.0 * sum(1 for x in nets if x > 0) / n, 1) if n else 0.0,
            "pf": round(gp / gl, 3) if gl > 0 else (float("inf") if gp > 0 else 0.0),
            "net": round(sum(nets), 2)}


def _in(trades, rng) -> list:
    return [t for t in trades if rng[0] <= t[_DATE] <= rng[1]]


def screen(trades, split: Split, horizon: str) -> dict:
    """The DISCOVERY screen. Returns {"passed", "reasons", "h1", "h2", "disc", "top_symbol", "top_share"}."""
    disc = _in(trades, split.discovery)
    h1, h2, d = stats(_in(disc, split.h1)), stats(_in(disc, split.h2)), stats(disc)
    by_sym: dict = {}
    for t in disc:
        by_sym[t[_SYM]] = by_sym.get(t[_SYM], 0.0) + t[_NET]
    top_sym, top_net = max(by_sym.items(), key=lambda kv: kv[1]) if by_sym else ("", 0.0)
    share = top_net / d["net"] if d["net"] > 0 else float("nan")
    need = SCREEN_MIN_TRADES[horizon]
    reasons = []
    if h1["net"] <= 0:
        reasons.append("H1 net <= 0")
    if h2["net"] <= 0:
        reasons.append("H2 net <= 0")
    if d["pf"] < SCREEN_MIN_PF:
        reasons.append(f"PF {d['pf']:.3f} < {SCREEN_MIN_PF}")
    if d["trades"] < need:
        reasons.append(f"{d['trades']} trades < {need}")
    if d["net"] > 0 and share > SCREEN_MAX_SYMBOL_SHARE:
        reasons.append(f"{top_sym} is {share:.0%} of net (> {SCREEN_MAX_SYMBOL_SHARE:.0%})")
    return {"passed": not reasons, "reasons": reasons, "h1": h1, "h2": h2, "disc": d,
            "top_symbol": top_sym, "top_share": share}


def final_verdict(final_trades) -> dict:
    f = stats(final_trades)
    f["confirmed"] = bool(f["net"] > 0 and f["pf"] >= FINAL_MIN_PF)
    return f


def _load_ledger(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def evaluate(specs, calendar, simulate, n_symbols: int = 0, ledger_path: str | None = None,
             peek_final: bool = False) -> tuple[list, dict]:
    """Judge strategies. `simulate(specs, lo, hi)` must return, for entry dates in [lo, hi] only,
    {id: {"trades": [...], "skipped": n, "lookahead": str|None, "error": str|None}}.

    Returns (results, trades_by_id). trades_by_id holds DISCOVERY trades for everyone and FINAL trades only
    for strategies whose FINAL was legitimately (or, with peek_final, explicitly) evaluated."""
    split = split_sessions(calendar)
    n_sessions = len(set(calendar))
    enough = split is not None and n_sessions >= MIN_SESSIONS
    results, trades_by_id = [], {}
    if split is None:
        for sp in specs:
            results.append({"id": sp.id, "spec": sp, "verdict": "NO VERDICT", "screen": None, "final": None,
                            "why": f"too few sessions ({n_sessions}) to split", "skipped": 0})
            trades_by_id[sp.id] = []
        return results, trades_by_id

    disc = simulate(specs, split.discovery[0], split.discovery[1])
    passers, by_id = [], {}
    for sp in specs:
        r = disc.get(sp.id, {"trades": [], "skipped": 0, "lookahead": None, "error": None})
        trades_by_id[sp.id] = list(r["trades"])
        sc = screen(r["trades"], split, sp.horizon)
        res = {"id": sp.id, "spec": sp, "screen": sc, "final": None, "skipped": r.get("skipped", 0),
               "final_note": ""}
        if r.get("error"):
            res["verdict"], res["why"] = "INVALID", "strategy raised: " + r["error"]
            sc["passed"] = False
        elif r.get("lookahead"):
            res["verdict"], res["why"] = "INVALID", "LOOK-AHEAD: " + r["lookahead"]
            sc["passed"] = False
        elif not enough:
            res["verdict"] = "NO VERDICT"
            res["why"] = (f"too few sessions ({n_sessions} < {MIN_SESSIONS})"
                          + ("; screen would pass" if sc["passed"] else "; screen: " + ", ".join(sc["reasons"])))
            sc["passed"] = False
        elif sc["passed"]:
            res["verdict"], res["why"] = "PASSED SCREEN", "all screen gates met"
            passers.append(sp)
        else:
            res["verdict"], res["why"] = "FAILED SCREEN", ", ".join(sc["reasons"])
        by_id[sp.id] = res
        results.append(res)

    want = list(passers)
    if peek_final and enough:
        want = [sp for sp in specs if by_id[sp.id]["verdict"] != "INVALID"]
    if want:
        ledger = _load_ledger(ledger_path) if ledger_path else {}
        fin = simulate(want, split.final[0], split.final[1])
        now = time.strftime("%Y-%m-%d %H:%M")
        for sp in want:
            res = by_id[sp.id]
            ftr = list(fin.get(sp.id, {}).get("trades", []))
            fv = final_verdict(ftr)
            passed = sp in passers
            key = f"{sp.id}|{split.final[0]}-{split.final[1]}|{n_sessions}s|{n_symbols}sym"
            old = ledger.get(key)
            if old is None:
                ledger[key] = {"fingerprint": sp.fingerprint, "when": now, "peeked": not passed,
                               "final": {k: fv[k] for k in ("trades", "win_pct", "pf", "net", "confirmed")}}
                clean = passed
            elif old["fingerprint"] == sp.fingerprint and not old.get("peeked"):
                clean = passed                               # same strategy, same data: the same single result
                res["final_note"] = f"first reported {old['when']}"
            else:
                clean = False                                # FINAL was already seen: this is not a fresh test
                res["final_note"] = (f"FINAL already seen on {old['when']} "
                                     f"({'peeked' if old.get('peeked') else 'earlier version'}); not a confirmation")
            res["final"] = fv
            trades_by_id[sp.id].extend(ftr)
            if not passed:
                res["final_note"] = "PEEK (failed screen; not a confirmation)"
            elif clean:
                res["verdict"] = "CONFIRMED" if fv["confirmed"] else "REJECTED ON FINAL"
            else:
                res["verdict"] = "RETEST (FINAL already used)"
        if ledger_path:
            os.makedirs(os.path.dirname(ledger_path) or ".", exist_ok=True)
            tmp = ledger_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(ledger, fh, indent=1)
            os.replace(tmp, ledger_path)
    for res in results:
        res["split"] = split
    return results, trades_by_id


def luck_reminder(results) -> str:
    """How many passed out of how many were tested, and how many of those passes luck alone would produce."""
    n = len(results)
    judged = [r for r in results if r["verdict"] not in ("NO VERDICT", "INVALID")]
    passed = [r for r in judged if r["verdict"] != "FAILED SCREEN"]
    conf = [r for r in judged if r["verdict"] == "CONFIRMED"]
    lines = [f"Tested {n} strategies: {len(judged)} judged, {len(passed)} passed the screen, "
             f"{len(conf)} confirmed on FINAL."]
    if len(judged) < n:
        lines.append(f"{n - len(judged)} got no verdict (too few sessions, or invalid).")
    lines.append(
        f"REMINDER: with N candidates some confirmations are luck. About {GATE_LUCK:.0%} of useless strategies "
        f"pass each coin-flip-like gate, so of {n} useless candidates about {n * GATE_LUCK:.1f} would pass a "
        f"single such gate by chance. A confirmation is a reason to paper-trade a strategy, not proof of edge.")
    return "\n".join(lines)
