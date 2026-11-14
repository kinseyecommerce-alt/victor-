"""
stratlab/tests_trials.py — tests for stratlab/trials.py (PSR / DSR worked values, moments, daily P&L, the
deflation gate, the append-only trials ledger, trial counting). Run by test_pipeline.py (section "STRATLAB:
trials ledger and Deflated Sharpe Ratio"), or alone: python -m stratlab.tests_trials
"""
from __future__ import annotations

import math
import os
import tempfile
from types import SimpleNamespace

from . import trials as T


def _close(a, b, tol):
    assert abs(a - b) <= tol, (a, b, tol)


def t_dsr_paper_worked_example():
    """Bailey & Lopez de Prado (2014), numerical example: annualised SR 2.5 over 5 years of daily data (T = 1250),
    N = 100 trials whose annualised SRs have variance 0.5, skew -3, kurtosis 10 -> SR0 = 0.1132 (daily) and
    DSR = 0.9004. By hand: E[max] factor = 0.4228 x Z(0.99) + 0.5772 x Z(1 - 1/(100e)) = 0.4228 x 2.3263 +
    0.5772 x 2.6799 = 2.5305; x sqrt(0.5 / 250) = 0.11317. z = (0.15811 - 0.11317) x sqrt(1249) /
    sqrt(1 + 3 x 0.15811 + 9/4 x 0.15811^2) = 1.5884 / 1.2372 = 1.2838 -> Phi = 0.9004."""
    sr = 2.5 / math.sqrt(250)
    d = T.dsr(sr, 1250, -3.0, 10.0, 100, 0.5 / 250)
    _close(d["sr0"], 0.1132, 5e-4)
    _close(d["sr0"] * math.sqrt(250), 1.7894, 2e-3)          # the same threshold, annualised
    _close(d["dsr"], 0.9004, 1e-3)
    assert d["psr0"] > 0.9999


def t_psr_hand_values():
    # SR 0.1, T 101, normal returns: z = 0.1 x 10 / sqrt(1 + 0.5 x 0.01) = 0.99751 -> Phi = 0.84074
    _close(T.psr(0.1, 0.0, 101, 0.0, 3.0), 0.84074, 1e-4)
    _close(T.psr(0.0, 0.0, 500, 0.0, 3.0), 0.5, 1e-12)        # SR equal to the benchmark -> one half
    # negative skew and fat tails make the same SR less convincing
    assert T.psr(0.1, 0.0, 101, -2.0, 9.0) < T.psr(0.1, 0.0, 101, 0.0, 3.0)
    assert math.isnan(T.psr(0.1, 0.0, 1))                     # undefined below 2 observations


def t_expected_max_sr_properties():
    assert T.expected_max_sr(1, 0.01) == 0.0                  # no selection, no deflation
    a, b, c = (T.expected_max_sr(n, 0.01) for n in (10, 100, 1000))
    assert 0 < a < b < c
    # scales with the cross-trial standard deviation
    _close(T.expected_max_sr(100, 0.04), 2 * T.expected_max_sr(100, 0.01), 1e-12)
    # with N = 1 the DSR is the PSR against zero
    d = T.dsr(0.05, 300, 0.0, 3.0, 1, 0.01)
    _close(d["dsr"], d["psr0"], 1e-12)


def t_moments_hand_values():
    m = T.moments([1, 2, 3, 4, 10])
    # mean 4; sample var (9+4+1+0+36)/4 = 12.5; population m2 = 10, m3 = 36, m4 = 278.8
    _close(m["mean"], 4.0, 1e-12)
    _close(m["std"], math.sqrt(12.5), 1e-12)
    _close(m["sr"], 4 / math.sqrt(12.5), 1e-12)
    _close(m["skew"], 36 / 10 ** 1.5, 1e-12)
    _close(m["kurt"], 2.788, 1e-12)


def t_daily_pnl_books_on_exit_and_zero_days():
    cal = [20240101, 20240102, 20240103, 20240104, 20240105]
    tr = [{"entry": 20240101, "exit": 20240103, "net": 10.0, "gross": 12.0, "cost": 2.0},
          {"entry": 20240102, "exit": 20240102, "net": -4.0, "gross": -3.0, "cost": 1.0},
          {"entry": 20231229, "exit": 20240102, "net": 99.0, "gross": 99.0, "cost": 0.0},     # entry before lo: excluded
          {"entry": 20240104, "exit": 20240110, "net": 5.0, "gross": 6.0, "cost": 1.0}]      # exit past hi: last day
    assert T.daily_pnl(tr, cal, 20240101, 20240105) == [0.0, -4.0, 10.0, 0.0, 5.0]
    st = T.series_stats(tr, cal, 20240101, 20240105)
    assert st["trades"] == 3 and st["n_days"] == 5 and st["net"] == 11.0
    _close(st["mean_gross"], 5.0, 1e-12)
    _close(st["mean_cost"], 4 / 3, 1e-12)


def t_deflation_gate_haircut_on_gross():
    # mean net 40 > 0, but half the gross edge (50) does not pay the cost (60): fails the haircut
    g = T.deflation_gate([40.0, 40.0], [100.0, 100.0], [60.0, 60.0], dsr_value=0.97)
    assert not g["haircut_ok"] and g["dsr_ok"] and not g["passed"] and g["either"]
    _close(g["haircut_mean"], -10.0, 1e-12)
    g = T.deflation_gate([140.0], [200.0], [60.0], dsr_value=0.5)
    assert g["haircut_ok"] and not g["dsr_ok"] and not g["passed"]
    _close(g["haircut_mean"], 40.0, 1e-12)
    g = T.deflation_gate([140.0], [200.0], [60.0], dsr_value=0.951)
    assert g["passed"]
    g = T.deflation_gate([5.0, -1.0])                         # nets only: flagged fallback
    assert g["haircut_basis"].startswith("net") and g["haircut_ok"] and not g["dsr_ok"]


def t_load_trades_layouts():
    with tempfile.TemporaryDirectory() as td:
        p1 = os.path.join(td, "sim.csv")
        with open(p1, "w") as fh:
            fh.write("id,symbol,date,side,entry_t,entry,exit_date,exit_t,exit,qty,gross,cost,net,reason\n"
                     "X,A,20240102,1,09:15,100,20240105,15:29,101,10,10,3,7,time\n")
        p2 = os.path.join(td, "opt.csv")
        with open(p2, "w") as fh:
            fh.write("date,entry,expiry,exit,gross,haircut,fees,net\n20240102,10:30,2024-01-04,20240103 15:29,100,10,5,85\n")
        p3 = os.path.join(td, "vrp.csv")
        with open(p3, "w") as fh:
            fh.write("date,expiry,gross,haircut,fees,net\n20240102,2024-01-04,100,10,5,85\n")
        a, = T.load_trades(p1)
        assert (a["entry"], a["exit"], a["net"], a["cost"]) == (20240102, 20240105, 7.0, 3.0)
        b, = T.load_trades(p2)
        assert (b["exit"], b["cost"]) == (20240103, 15.0)        # date-time exit beats expiry; cost = gross - net
        c, = T.load_trades(p3)
        assert c["exit"] == 20240104


def t_ledger_append_only_idempotent_and_outside_repo():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "trials.csv")
        led = T.TrialsLedger(path)
        r = {"strategy_id": "s1", "era": "2023-26", "stage": "screen", "dataset": "bars3y", "source": "x",
             "trades": 10, "net": -5.0, "sr_daily": float("nan")}
        assert led.append([r, dict(r, strategy_id="s2")]) == 2
        size = os.path.getsize(path)
        assert led.append([r]) == 0 and os.path.getsize(path) == size        # same evaluation: not appended
        assert led.append([dict(r, stage="final")]) == 1                      # a FINAL is another trial row
        rows = led.read()
        assert [x["strategy_id"] for x in rows] == ["s1", "s2", "s1"] and rows[0]["sr_daily"] == ""
        assert T.count_trials(rows) == 2 and T.count_trials(rows, stages={"final"}) == 1
    try:
        T.TrialsLedger(os.path.join(os.path.dirname(os.path.abspath(T.__file__)), "trials.csv"))
        raise AssertionError("a ledger inside the repository must be refused")
    except ValueError:
        pass


def t_count_trials_param_variants_and_reruns():
    rows = [{"strategy_id": "a", "era": "2023-26", "stage": "screen", "params_fp": "p:1"},
            {"strategy_id": "a", "era": "2023-26", "stage": "screen-smoke", "params_fp": "p:1"},   # rerun
            {"strategy_id": "a", "era": "2023-26", "stage": "screen", "params_fp": "p:2"},         # neighbour params
            {"strategy_id": "a", "era": "2023-26", "stage": "final", "params_fp": "codefp"},       # same variant
            {"strategy_id": "b", "era": "2023-26", "stage": "explore", "params_fp": "x"},
            {"strategy_id": "c", "era": "2012-23", "stage": "oos", "params_fp": "p:9"}]
    assert T.count_trials(rows) == 3 and T.count_trials(rows, "2012-23") == 1
    rob, n, med = T.var_sr_robust([{"strategy_id": k, "era": "2023-26", "stage": "screen", "dataset": "bars3y",
                                    "sr_daily": str(v)} for k, v in zip("abcde", (-5.0, -0.1, 0.0, 0.1, 0.2))])
    assert n == 5 and med == 0.0
    _close(rob, (1.4826 * 0.1) ** 2, 1e-12)                  # the -5 outlier does not inflate it


def t_var_sr_one_value_per_id():
    rows = [{"strategy_id": "a", "era": "2023-26", "stage": "screen", "dataset": "bars3y", "sr_daily": "0.1"},
            {"strategy_id": "a", "era": "2023-26", "stage": "screen", "dataset": "bars3y", "sr_daily": "0.9"},
            {"strategy_id": "b", "era": "2023-26", "stage": "screen", "dataset": "bars3y", "sr_daily": "-0.1"},
            {"strategy_id": "c", "era": "2023-26", "stage": "screen", "dataset": "bars12m", "sr_daily": "5"}]
    v, n, m = T.var_sr(rows)
    assert n == 2
    _close(m, 0.0, 1e-12)
    _close(v, 0.02, 1e-12)


def t_record_evaluation_counts_failures():
    from .protocol import Split
    split = Split((20240101, 20240102), (20240103, 20240104), (20240105, 20240105), 5, 4, 1)
    tr = ("A", "SYM", 20240101, 1, "09:15", 100.0, 20240101, "15:15", 101.0, 1, 1.0, 0.5, 0.5, "time")
    res = [{"id": "A", "spec": SimpleNamespace(family="f", fingerprint="fp1"), "split": split,
            "verdict": "FAILED SCREEN", "why": "PF", "final": None},
           {"id": "B", "spec": SimpleNamespace(family="f", fingerprint="fp2"), "split": split,
            "verdict": "CONFIRMED", "why": "ok", "final": {"trades": 0}, "final_note": ""}]
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.csv")
        n = T.record_evaluation(res, {"A": [tr], "B": []}, dataset="d", cost_model="cm", ledger_path=path,
                                calendar=[20240101, 20240102, 20240103, 20240104, 20240105])
        rows = T.TrialsLedger(path).read()
        assert n == 3 and [(r["strategy_id"], r["stage"]) for r in rows] == [("A", "screen"), ("B", "screen"), ("B", "final")]
        assert rows[0]["verdict"] == "FAILED SCREEN" and rows[1]["verdict"] == "PASSED SCREEN"
        assert rows[0]["trades"] == "1" and rows[0]["n_days"] == "4"
        assert T.record_evaluation(res, {}, dataset="d", cost_model="cm", ledger_path=None) == 0


def t_family_of():
    assert T.family_of("r3_swing_quiet_loser_rev") == "r3_swing"
    assert T.family_of("AUC-FADE-3PCT") == "AUC-FADE"
    assert T.family_of("cs_doji_15m") == "cs"


def t_final_rows_only_borrow_a_matching_trade_file():
    """A FINAL ledger entry takes Sharpe / moments only from a same-dataset, same-cost-model trade file that
    reproduces its FINAL trades and net; another dataset's file of the same id (or a non-reproducing one) is
    ignored and the Sharpe columns stay blank."""
    import json
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "stratlab_cache"))
        cal = [20251217, 20251218, 20251219, 20251222]

        def w(name, rows):
            p = os.path.join(d, name)
            with open(p, "w") as fh:
                fh.write("date,exit_date,net,gross,cost\n")
                for e, x, n in rows:
                    fh.write(f"{e},{x},{n},{n + 10},10\n")
            return p
        other = w("other_ds.csv", [(20251217, 20251218, -50.0)])                 # 12-month run, 1 trade
        wrong_cm = w("wrong_cm.csv", [(20251217, 20251218, 30.0), (20251219, 20251222, 70.0)])
        good = w("good.csv", [(20251217, 20251218, 30.0), (20251219, 20251222, 70.0)])
        stale = w("stale.csv", [(20251217, 20251218, 30.0), (20251219, 20251222, 75.0)])   # same count, net off
        led = {"S1|20251217-20251222|x": {"dataset": "/data/bars3y", "cost_model": "cm2", "fingerprint": "f",
                                          "final": {"trades": 2, "net": 100.0, "pf": 0, "confirmed": True}},
               "S2|20251217-20251222|x": {"dataset": "/data/bars3y",
                                          "final": {"trades": 2, "net": 100.0, "pf": 0, "confirmed": False}}}
        with open(os.path.join(d, "stratlab_cache", "final_ledger.json"), "w") as fh:
            json.dump(led, fh)
        scr = [{"strategy_id": "S1", "stage": "screen", "dataset": "bars12m", "cost_model": "cm2", "trades_file": other},
               {"strategy_id": "S1", "stage": "screen", "dataset": "bars3y", "cost_model": "cm1", "trades_file": wrong_cm},
               {"strategy_id": "S1", "stage": "screen", "dataset": "bars3y", "cost_model": "cm2", "trades_file": good},
               {"strategy_id": "S2", "stage": "screen", "dataset": "bars12m", "cost_model": "", "trades_file": other},
               {"strategy_id": "S2", "stage": "screen", "dataset": "bars3y", "cost_model": "", "trades_file": stale}]
        rows = {r["strategy_id"]: r for r in T._final_rows(d, cal, {}, scr)}
        assert rows["S1"]["trades_file"] == good and rows["S1"]["sr_daily"] is not None
        assert rows["S1"]["trades"] == 2 and rows["S1"]["net"] == 100.0
        assert rows["S2"]["trades_file"] == "" and "sr_daily" not in rows["S2"]      # no file reproduces it
        assert rows["S2"]["cost_model"] == "legacy"


TESTS = [
    ("trials: DSR reproduces Bailey & Lopez de Prado's worked example (SR0 0.1132, DSR 0.9004)",
     t_dsr_paper_worked_example),
    ("trials: PSR hand values (0.8407; 0.5 at the benchmark; skew/kurtosis penalty)", t_psr_hand_values),
    ("trials: expected max SR is 0 for N=1, grows with N, scales with sd", t_expected_max_sr_properties),
    ("trials: moments (sample std, population skew, non-excess kurtosis)", t_moments_hand_values),
    ("trials: daily P&L by exit session, zero days, entry-date period rule", t_daily_pnl_books_on_exit_and_zero_days),
    ("trials: deflation gate = 0.5 x mean gross - mean cost > 0 and/or DSR > 0.95", t_deflation_gate_haircut_on_gross),
    ("trials: trade CSV layouts (sim, options replay, VRP)", t_load_trades_layouts),
    ("trials: ledger is append-only, idempotent, refused inside the repo", t_ledger_append_only_idempotent_and_outside_repo),
    ("trials: N counts parameter variants once each, not re-runs; robust V ignores the cost-drag tail",
     t_count_trials_param_variants_and_reruns),
    ("trials: V[SR] uses one value per strategy id on one dataset", t_var_sr_one_value_per_id),
    ("trials: record_evaluation records failures as well as passers", t_record_evaluation_counts_failures),
    ("trials: family prefix", t_family_of),
    ("trials: FINAL rows borrow Sharpe only from a same-dataset/cost-model file reproducing trades and net",
     t_final_rows_only_borrow_a_matching_trade_file),
]


if __name__ == "__main__":
    for _n, _f in TESTS:
        _f()
        print("ok  ", _n)
