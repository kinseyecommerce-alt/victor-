"""
stratlab/rerun_corrected.py — the corrected-data re-run (Nov 2026), pre-registered in prereg/rerun_corrected_v1.json
(byte-identical copy: <scratchpad>/audit/PREREGISTERED_RERUN.json) BEFORE any strategy was simulated on corrected
data.  Research only: reads bars, writes to <scratchpad>/audit.

    python -m stratlab.rerun_corrected --preregister          # writes both prereg copies (refuses to overwrite)
    python -m stratlab.rerun_corrected --stage screen         # every catalogue id, corrected 2023-26 DISCOVERY
    python -m stratlab.rerun_corrected --stage tfgrid         # all tfgrid combos, corrected 2023-25 DISCOVERY
    python -m stratlab.rerun_corrected --stage final          # FINAL: one look for new passers, recompute opened ones
    python -m stratlab.rerun_corrected --stage oldera         # 2012-23 daily (PIT NIFTY 100) + 2016-23 minute
    python -m stratlab.rerun_corrected --stage report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

_SCRATCH = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"
SCRATCH = os.environ.get("STRATLAB_SCRATCH", _SCRATCH)
AUDIT = os.path.join(SCRATCH, "audit")
PREREG = os.path.join(AUDIT, "PREREGISTERED_RERUN.json")
REPO_PREREG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prereg", "rerun_corrected_v1.json")
LEDGER_DIR = os.path.join(AUDIT, "ledgers")
FINAL_LEDGER = os.path.join(LEDGER_DIR, "final_corrected.json")
OLDERA_DAILY_LEDGER = os.path.join(LEDGER_DIR, "oldera_daily_corrected.json")
OLDERA_MIN_LEDGER = os.path.join(LEDGER_DIR, "oldera_min_corrected.json")
STAGE = "corrected_rerun"
DATA_TAG = "corr1"
CODE_FILES = ("corrected.py", "rerun_corrected.py", "audit_run.py", "refsim.py", "sim.py", "protocol.py",
              "protocol_oos.py", "protocol_min_oos.py", "tfgrid.py", "capital.py", "data.py", "data_daily.py",
              "auction.py", "attention.py", "auction_r2.py", "events_run.py", "trials.py")
SPECIAL = {"proven": ["SWING-ATTN-ON-3X", "AUC-FADE-2PCT"],
           "live_paper": ["AUC-FADE-3PCT", "SWING-ATTN-ON-3X", "VRP-STRANGLE-016-WINGED"],
           "event": ["ev_pead_long_h60"]}


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def preregister() -> dict:
    from . import catalog  # noqa: F401
    from . import corrected as C
    from . import tfgrid as TG
    from .api import REGISTRY
    if os.path.exists(PREREG) or os.path.exists(REPO_PREREG):
        raise SystemExit("pre-registration already written: refusing to overwrite (one plan, one look)")
    here = os.path.dirname(os.path.abspath(__file__))
    ids = sorted(REGISTRY)
    fp = {i: REGISTRY[i].fingerprint for i in ids}
    doc = {
        "title": "Corrected-data re-run of every lab strategy (same hypotheses, not new trials)",
        "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "why": "User: 'Test all strategies with corrected data - this much of strategies won't fail, simulation trade "
               "may be mistaken.' Part 1 of the work audits the simulator independently; this file freezes how the "
               "re-run on CORRECTED data is done and judged, before any strategy is simulated on corrected data.",
        "status_at_writing": {
            "simulator_audit": "refsim (independent engine) vs sim.py on 22 strategies x 97 symbols, DISCOVERY "
                               "2023-11-10..2025-12-16: every trade identical (fills, exits, qty, gross, cost) "
                               "wherever both engines read a strategy's rule the same way; all differences traced to "
                               "ambiguous one-line descriptions or undocumented hygiene filters, none to the engine. "
                               "30/30 hand-checked trades match the raw JSON rows and a manual Zerodha computation. "
                               "Gross/mirror/look-ahead/benchmark checks are still running when this was written: "
                               "if any of them finds a simulator BUG, the clause 'bug_rule' below applies.",
            "corrected_results_seen": "none (no strategy has been simulated on the corrected view)"},
        "data": {
            "definition": "stratlab/corrected.py (see its docstring): minute bars of both eras mapped per session "
                          "onto official NSE prints x F(d) (splits / bonuses / consolidations / rights / demergers; "
                          "dividends NOT back-adjusted = as-traded prices); affine repair of rupee-subtracted "
                          "dividends; Kite's stale 09:15 bar on special pre-open days dropped; sessions whose shape "
                          "disagrees with the official prints (> 1.5 % off, o/h/l ratios > 0.5 % apart) EXCLUDED; "
                          "unadjusted-volume windows rescaled and flagged; the 10 daily-file (nse_old) issues "
                          "repaired by official_factor / applied_factor. Index bars (NIFTY 50, NIFTY BANK, VIX) "
                          "unchanged.",
            "version": C.VERSION,
            "validation": "all 119 minute-era rows of bar_adjustment_issues.csv pass corrected.validate() "
                          "(overnight ratio = official gap / official factor; additive offsets < 0.2 % of price; "
                          "seam 2023-11-09/10 for IOC / ONGC / PETRONET within 0.5 % of the official gap)",
            "maps_sha256": {f: _sha(os.path.join(C.OUT, f)) for f in ("maps_3y.pkl", "maps_old.pkl")},
            "universe_minute": "today's 97 NIFTY-100 names (minute data exists only for them): the minute re-run "
                               "remains SURVIVORSHIP-BIASED, as before",
            "universe_daily_2012_23": "point-in-time NIFTY 100 membership from flowdata index_members "
                                      "(index_membership/index_members.csv.gz, reliable from 2012-01-02): a stock "
                                      "strategy may open a position on decision day D only if the stock is a NIFTY "
                                      "100 constituent in force on D (replaces the turnover top-100 of the original "
                                      "2012-23 round); event strategies keep their point-in-time top-500 universe"},
        "hypotheses": {
            "rule": "the SAME strategies with the SAME code, parameters and gates; nothing is added, dropped or "
                    "tuned. They are NOT new trials: ledger rows get stage 'corrected_rerun' / 'corrected_final' / "
                    "'corrected_oos' and the same params fingerprint, and are NOT added to N for the DSR.",
            "catalogue_ids": len(ids),
            "catalogue_fingerprints_sha256": hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest(),
            "tfgrid": "all 668 grid combos (38 rules x 8 timeframes x universes) + 608 natives with the 1:3 bracket "
                      "and Rs 10 lakh accounts exactly as prereg/tfgrid_v1.json",
            "tfgrid_prereg_sha256": _sha(TG.REPO_PREREG),
            "modules": ["AUC-FADE-2PCT", "AUC-FADE-3PCT", "SCALP-AUC-FADE-1TO3-0925", "SWING-ATTN-ON-3X"],
            "not_rerun": {"option / VRP strategies": "they read only index bars and NSE option files, which the "
                                                      "correction does not change: their results are unchanged by "
                                                      "construction (VRP-STRANGLE-016-WINGED reported as before)"},
            "special": SPECIAL},
        "judgement": {
            "screen_2023_26": "protocol.screen unchanged: DISCOVERY 2023-11-10..2025-12-16 (first 70 % of the "
                              "726-session calendar; H1 to 2024-11-29, H2 from 2024-12-02); net > 0 in H1 and H2, "
                              "PF >= 1.15, >= 150 (intraday) / 60 (positional) trades, top symbol <= 35 % of net "
                              "(not for one-instrument strategies); 2 bps (index 1 bps) slippage, costs.py 2026; "
                              "tfgrid: its own account screen unchanged",
            "result_of_record": "the corrected-data result is the result of record for every strategy",
            "final_one_look": "a strategy that passes the corrected DISCOVERY screen opens its FINAL "
                              "(2025-12-17..2026-11-09) ONLY if its FINAL was never opened before on the 3-year data "
                              "under any cost model, in any ledger (stratlab_cache/*/final_ledger.json, "
                              "tfgrid/final_ledger.json, events_daily ledger); the look is written once to "
                              "audit/ledgers/final_corrected.json. CONFIRMED = net > 0 and PF >= 1.05.",
            "opened_finals": "a FINAL that was already opened is recomputed on corrected data and reported side by "
                             "side; the ORIGINAL look stands (it is not a fresh look) unless bug_rule applies",
            "bug_rule": "if Part 1 finds a simulator bug, the corrected ENGINE's result stands for every affected "
                        "strategy and the report says so explicitly",
            "old_era": "for every strategy that passed the 2023-26 screen before (original verdict PASSED SCREEN / "
                       "CONFIRMED / REJECTED ON FINAL on bars3y, any tfgrid stage-1 passer) or passes now, plus "
                       "SPECIAL: 2012-23 daily (protocol_oos.judge gates unchanged, corrected daily file, PIT NIFTY "
                       "100) where eligible, and 2016-23 minute (protocol_min_oos gates unchanged, corrected old "
                       "minute view; its dirty-link exclusion = the original dirty set plus the session after "
                       "every excluded session). One judgement each, written to audit/ledgers/oldera_*_corrected.json; "
                       "the original one-look ledgers are not touched and their verdicts are reported next to it.",
            "proven": "FINAL-CONFIRMED on 2023-26 AND OOS-CONFIRMED on 2016-23 (tfgrid: also DSR >= 0.95), as before",
            "dsr": "deflated Sharpe with the SAME N as before the re-run (variants 2023-26 in trials_v2.csv, which "
                   "the re-run does not increase)"},
        "code_sha256": {f: _sha(os.path.join(here, f)) for f in CODE_FILES if os.path.exists(os.path.join(here, f))},
        "costs_sha256": _sha(os.path.join(os.path.dirname(here), "costs.py")),
    }
    blob = json.dumps(doc, indent=1, sort_keys=True).encode()
    for p in (PREREG, REPO_PREREG):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(blob)
    with open(PREREG + ".sha256", "w") as fh:
        fh.write(hashlib.sha256(blob).hexdigest() + "\n")
    return doc


def load_prereg() -> dict:
    a, b = open(PREREG, "rb").read(), open(REPO_PREREG, "rb").read()
    if a != b:
        raise SystemExit("the two pre-registration copies differ")
    return json.loads(a)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlab.rerun_corrected")
    ap.add_argument("--preregister", action="store_true")
    ap.add_argument("--stage", default="")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    if a.preregister:
        doc = preregister()
        print(f"written {PREREG} and {REPO_PREREG} (sha256 {_sha(PREREG)})")
        return 0
    raise SystemExit(f"unknown stage {a.stage!r}")


if __name__ == "__main__":
    sys.exit(main())
