"""
parity_builder_promoted.py — lab parity of agents/builder_promoted.py against the builder engine (run in its OWN process
by test_pipeline.t_bp_lab_parity: building the lab engine changes module-level paths of stratlab.datagen_features that
later tests in the same process must not see). Usage: python parity_builder_promoted.py; exit 0 = parity, 2 = no data.
"""
import os as _os
import sys

import numpy as np

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

# the builder registry's promoted strategies of 25 Nov 2026 (server batches 0020-0046; replayed from the seeds,
# ids matched); bld0003 (the seed's) is checked by test_builder
_BP_RULES_25NOV = {
    'bld0020_sca_hig_17_21611f': {"agent": "scalping", "mtype": "HIGH_VOL", "block": "gap_fade", "params": [["g", 0.02], ["mode", "fade_day"]], "entry": "o0915", "exit": ["bracket_min", 0.005, 2.0, 30], "instrument": "STOCKS", "product": "mis", "structure": "", "universe": "NIFTY200", "size": "c10"},
    'bld0021_sca_sid_03_5ac9ef': {"agent": "scalping", "mtype": "SIDEWAYS", "block": "gap_fade", "params": [["g", 0.01], ["mode", "fade_day"]], "entry": "o0915", "exit": ["bracket_min", 0.005, 3.0, 30], "instrument": "STOCKS", "product": "mis", "structure": "", "universe": "NIFTY100", "size": "c10"},
    'bld0022_int_hig_02_41ce5b': {"agent": "intraday", "mtype": "HIGH_VOL", "block": "idio_gap", "params": [["g", 0.01]], "entry": "o0915", "exit": ["bracket", 0.005, 3.0], "instrument": "STOCKS", "product": "mis", "structure": "", "universe": "NIFTY200", "size": "c10"},
    'bld0025_sca_sid_11_638cd4': {"agent": "scalping", "mtype": "SIDEWAYS", "block": "gap_fade", "params": [["g", 0.01], ["mode", "fade_day"]], "entry": "o0915", "exit": ["bracket_min", 0.003, 3.0, 30], "instrument": "STOCKS", "product": "mis", "structure": "", "universe": "NIFTY200", "size": "c10"},
    'bld0026_sca_hig_03_591d6e': {"agent": "scalping", "mtype": "HIGH_VOL", "block": "gap_fade", "params": [["g", 0.02], ["mode", "fade_day"]], "entry": "o0915", "exit": ["bracket_min", 0.005, 3.0, 30], "instrument": "STOCKS", "product": "mis", "structure": "", "universe": "NIFTY200", "size": "c10"},
    'bld0030_sca_hig_13_38f07b': {"agent": "scalping", "mtype": "HIGH_VOL", "block": "gap_fade", "params": [["g", 0.02], ["mode", "fade_day"]], "entry": "o0915", "exit": ["bracket_min", 0.005, 3.0, 30], "instrument": "STOCKS", "product": "mis", "structure": "", "universe": "NIFTY100", "size": "c10"},
    'bld0030_sca_hig_18_69a1eb': {"agent": "scalping", "mtype": "HIGH_VOL", "block": "gap_fade", "params": [["g", 0.01], ["mode", "fade_day"]], "entry": "o0915", "exit": ["time_min", 25], "instrument": "STOCKS", "product": "mis", "structure": "", "universe": "NIFTY200", "size": "c10"},
    'bld0046_sca_sid_17_9715a7': {"agent": "scalping", "mtype": "SIDEWAYS", "block": "gap_fade", "params": [["g", 0.01], ["mode", "fade_day"]], "entry": "o0915", "exit": ["bracket_min", 0.005, 3.0, 30], "instrument": "STOCKS", "product": "mis", "structure": "", "universe": "NIFTY200", "size": "c10"},
}


def run():
    """agents/builder_promoted.py (added 25 Nov 2026): the live decision functions against the builder engine on the
    lab's 3-year minute data, for every promoted strategy of the 25 Nov registry (replayed from the committed seeds,
    ids matched): (1) market_type_by_day + regime_for == Engine.regime_m on every minute session; (2) plan_rule fed the
    engine's own gaps / previous-session flags / universe / dirty rows == Engine.plan's entries (symbol, side, order)
    on every session; (3) the walk tuple and exit deadline == the engine's; (4) every candidate is supported."""
    from stratlab import builder_engine as _BE, builder_grammar as _G
    from stratlab.builder_source import make_source as _ms
    from stratlab import datagen_run as _DR
    import agents.builder_promoted as _BP
    import validated_parity as _VP
    if not _os.path.isdir(_VP.PARITY_BARS):
        print("      SKIP (no lab data)")
        return
    rules = _BP_RULES_25NOV
    for cid, cand in rules.items():
        c = _G.Candidate.from_json(cand)
        assert _BP.supported(c) == (True, ""), cid
        assert cid.endswith(__import__("hashlib").sha1(c.key.encode()).hexdigest()[:6]), cid
    E = _BE.Engine(_ms("lab"))
    M = E.ctx.minute
    P_ = E.ctx.daily
    labels = _BP.market_type_by_day(E.dcal, P_.nifty["c"], P_.vix["c"])
    for t, d in enumerate(E.mcal.tolist()):
        code = int(E.regime_m[t])
        assert _BP.regime_for(int(d), labels) == (_G.MARKET_TYPES[code] if code >= 0 else ""), (d, code)
    gap, pok = E.ctx.m("gap"), E.ctx.m("prev_ok")
    six = M.six
    nrow = six["_NIFTY50_min"]
    total = 0
    for cid, cand in sorted(rules.items()):
        c = _G.Candidate.from_json(cand)
        sig = E.signals(c)
        rows = E.plan(c, sig, int(E.mcal[0]), int(E.mcal[-1]))
        want = {}
        for (s_, t, side, *_r) in rows:
            want.setdefault(int(t), []).append((M.syms[s_], int(side)))
        um = E.row_mask(c)
        names = [M.syms[i] for i in M.stocks if um is None or um[i]]
        code = _BE.MT_CODE[c.mtype]
        got = {}
        for t in range(len(E.mcal)):
            if int(E.regime_m[t]) != code or int(E.mcal[t]) in M.special:
                continue
            g_t = {M.syms[i]: float(gap[i, t]) for i in M.stocks
                   if bool(pok[i, t]) and np.isfinite(gap[i, t]) and np.isfinite(M.f["o0915"][i, t])}
            dirty = {M.syms[i] for i in M.stocks if _DR._dirty(E.ctx, i, t, t)}
            ng = float(gap[nrow, t]) if np.isfinite(gap[nrow, t]) else None
            p = _BP.plan_rule(c, g_t, ng, names, excluded=dirty)
            if p:
                got[t] = [(sym, side) for sym, side, _rk in p]
        assert got == want, (cid, sorted(set(got) ^ set(want))[:5],
                             [(t, got.get(t), want.get(t)) for t in sorted(set(got) | set(want))
                              if got.get(t) != want.get(t)][:3])
        w = _BP.walk_of(c)
        assert all(r[9] == w for r in rows), cid
        total += sum(len(v) for v in want.values())
        print(f"      {cid}: {sum(len(v) for v in want.values())} engine entries matched")
    assert total > 10000



if __name__ == "__main__":
    import validated_parity as _VP
    if not _os.path.isdir(_VP.PARITY_BARS):
        print("SKIP (no lab data)")
        sys.exit(2)
    run()
    print("PARITY OK")
