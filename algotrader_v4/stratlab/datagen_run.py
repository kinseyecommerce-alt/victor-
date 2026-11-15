"""
stratlab/datagen_run.py — the DATA-DRIVEN DISCOVERY study (prereg/datagen_v1.json): strategies CREATED from the lab's
own data, selected on old data only and judged on data the selection never saw. Research only; every output goes to
<scratchpad>/datagen (outside the repository).

    cd algotrader_v4
    python -m stratlab.datagen_run --data-check          # (a) signal counts per period, no price change, no return
    python -m stratlab.datagen_run --preregister         # (b) freeze candidates, N_search, splits, rule, gates, costs
    python -m stratlab.datagen_run --smoke               #     plumbing on a small sample: errors only, no P&L
    python -m stratlab.datagen_run --discovery           # (c) ONCE: every candidate on DISCOVERY, selection, controls
    python -m stratlab.datagen_run --validate            # (d) ONCE: the <= K carried candidates on VALIDATION
    python -m stratlab.datagen_run --final --trials <ledger>   # (e) ONE look at FINAL for validation passers + proof

SPLIT (strictly by time, fixed before any return was computed)
  DISCOVERY   minute rules 2016-01-01 .. 2023-11-09 (halves 2016-2019 / 2020-2023-11-09, protocol_min_oos.H1 / H2);
              daily rules 2012-01-01 .. 2023-11-09 (halves 2012-2017 / 2018-2023-11-09, protocol_oos.H1 / H2).
              The ONLY place returns are looked at before the selection is final.
  VALIDATION  the lab's 2023-26 DISCOVERY = protocol.split_sessions(trials.calendar_3y).discovery
              (2023-11-10 .. 2025-12-16), judged ONCE per carried candidate by protocol.screen (unchanged).
  FINAL       protocol.split_sessions(...).final (2025-12-17 .. 2026-11-09): protocol.evaluate's one-look FINAL for
              validation passers only (ledger <datagen>/final_ledger.json).

ENGINE (fill model "dg1", part of the cost-model key): every rule is an event rule with a FIXED exit, no stop, no
target, no optimisation. Rs 1 lakh per stock trade (whole shares at the entry price), index futures 1 lot (2026 lots,
sim.LOT_SIZES), slippage 2 bps per side for stocks / 1 bps for index futures against us on both fills (5 / 2.5 bps
stress), costs = costs.round_trip_cost (2026 Zerodha schedule): equity_mis intraday, equity_cnc delivery longs,
future for stock / index futures (stock futures simulated on the cash price, like attention.py). At most 10 entries
per candidate per day (largest rank first, ties by symbol), one open position per symbol and candidate.
  m_intra(a, b)  minute: enter at field a's bar open, exit at field b's bar open of the same session
  m_on           minute: enter at the 15:21 bar open, exit at the next session's 09:15 auction print
  m_hold(h)      minute: enter at the 15:21 open, exit at the 15:21 open h sessions later
  d_hold(h)      daily: decided in the evening of t, enter at the open of t+1, exit at the close of t+h (h = 1: the
                 same session's close); a hold that would pass the period end exits at the period's last close
  i_hold(h)      NIFTY 50 official daily bars as the index future, as d_hold
  i_on           index minute bars: 15:21 open -> next 09:15 open
Exclusions: daily = events_daily's corporate-action / unexplained-gap list (events_data.ca_excluded rule) and the
point-in-time universe gate (member on the decision day; product "fut" also needs a listed stock future that day);
minute = protocol_min_oos's dirty-link rule (old era) and ltr.CA_GAPS (2023-26), special sessions (median bar count
< 300) on the entry / exit day, a session whose previous calendar session is missing for gap rules, an overnight
exit beyond the period.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from costs import round_trip_cost  # noqa: E402

from . import datagen_features as F  # noqa: E402
from . import protocol as P  # noqa: E402
from . import protocol_min_oos as PM  # noqa: E402
from . import protocol_oos as PO  # noqa: E402
from . import sim  # noqa: E402
from . import trials as TR  # noqa: E402
from .api import Spec  # noqa: E402

DG = F.DG_DIR
PREREG = os.path.join(DG, "PREREGISTERED_DATAGEN.json")
PREREG_SHA_FILE = os.path.join(DG, "PREREGISTERED_DATAGEN.sha256")
REPO_PREREG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prereg", "datagen_v1.json")
DATA_CHECK = os.path.join(DG, "data_check.json")
DISC_OUT = os.path.join(DG, "discovery.json")
VAL_OUT = os.path.join(DG, "validation.json")
FIN_OUT = os.path.join(DG, "final.json")
FINAL_LEDGER = os.path.join(DG, "final_ledger.json")
PROGRESS = os.path.join(DG, "progress.txt")
FILL_MODEL = "dg1"
COST_MODEL = f"{sim.COST_MODEL}/{FILL_MODEL}"
NOTIONAL = 100_000.0
SLIP_BPS, SENS_BPS = 2.0, 5.0
IDX_SLIP_BPS, IDX_SENS_BPS = 1.0, 2.5
MAX_PER_DAY = 10
K_CARRY = 10                        # at most K candidates go to VALIDATION
FAMILY_CAP = 3                      # ... and at most 3 of one family
LEDGER_N0 = 2542                    # trials_v2.csv 2023-26 variant count when this study was designed (2026-11-15)
DSR_MIN = 0.95
CI_LEVEL = 0.95
LONG_BIASED = 0.70
SEED = 20261115
N_SHUFFLE = 5
N_PLACEBO = 50
BOOT_N = 2000
MINUTE_DISC = (20160101, 20231109)
DAILY_DISC = (20120101, 20231109)
MIN_HALVES = (PM.H1, PM.H2)
DAILY_HALVES = (PO.H1, PO.H2)
MIN_TRADES = dict(P.SCREEN_MIN_TRADES)       # intraday 150 / positional 60 (per DISCOVERY half, and in VALIDATION)
ENTRY_T = {"o0915": "09:15", "o0920": "09:20", "o0930": "09:30", "o0945": "09:45", "o1430": "14:30",
           "o1515": "15:15", "o1521": "15:21"}
_NET, _SYM, _DATE = 12, 1, 2


def log(msg: str) -> None:
    os.makedirs(DG, exist_ok=True)
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")


# ══════════════════════════════════════════════════════════════════════════
# the hypothesis space (declared up front; N_search = the candidates kept after the availability counts)
# ══════════════════════════════════════════════════════════════════════════
@dataclasses.dataclass(frozen=True)
class Cand:
    id: str
    family: str
    grid: str               # minute | daily | idx_daily | idx_minute
    rule: str               # function name in datagen_features
    params: tuple           # ((name, value), ...)
    exec: tuple             # see the module docstring
    product: str            # mis | cnc | fut
    story: str

    @property
    def horizon(self) -> str:
        if self.exec[0] == "m_intra" or (self.exec[0] == "d_hold" and self.product == "mis"):
            return "intraday"
        return "positional"

    @property
    def single(self) -> bool:
        return self.grid.startswith("idx")

    @property
    def kw(self) -> dict:
        return dict(self.params)

    def spec(self) -> Spec:
        fn = getattr(F, self.rule)
        kind = {"mis": "intraday", "cnc": "swing", "fut": "futures"}[self.product]
        return Spec(self.id, kind, self.family, self.story, fn,
                    {"params": list(self.params), "exec": list(self.exec), "product": self.product, "grid": self.grid,
                     "fill_model": FILL_MODEL},
                    horizon=self.horizon, underlying=("INDEX" if self.single else None), module="datagen_run")

    def describe(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.params)
        return f"{self.rule}({p}) -> {'/'.join(str(x) for x in self.exec)} [{self.product}]"


def _c(id, fam, grid, rule, params, ex, prod, story):
    return Cand(id, fam, grid, rule, tuple(params.items()), tuple(ex), prod, story)


def candidates() -> list:
    """Every candidate rule, in a fixed order. Grids of round thresholds and fixed horizons only."""
    C = []
    S_A = ("Opening call-auction imbalances overshoot; whether the first five minutes confirm or reverse the gap tells "
           "information (continues) from imbalance (reverts).")
    for g in (0.01, 0.02, 0.03):
        t = int(g * 100)
        C.append(_c(f"dg_a_gapgo_{t}", "A", "minute", "r_gap_path", {"g": g, "mode": "go"},
                    ("m_intra", "o0920", "o1515"), "mis", S_A))
        C.append(_c(f"dg_a_gaprev_{t}", "A", "minute", "r_gap_path", {"g": g, "mode": "revfollow"},
                    ("m_intra", "o0920", "o1515"), "mis", S_A))
        C.append(_c(f"dg_a_gapfadeday_{t}", "A", "minute", "r_gap_path", {"g": g, "mode": "fade_day"},
                    ("m_intra", "o0915", "o1515"), "mis", S_A))
    for r in (0.01, 0.02):
        t = int(r * 100)
        for mode in ("cont", "fade"):
            C.append(_c(f"dg_a_open5_{t}_{mode}", "A", "minute", "r_open5_shock", {"r": r, "mode": mode},
                        ("m_intra", "o0920", "o1515"), "mis", S_A))
    for g in (0.01, 0.02):
        C.append(_c(f"dg_a_idiogap_{int(g * 100)}", "A", "minute", "r_idio_gap", {"g": g},
                    ("m_intra", "o0920", "o1515"), "mis",
                    "A gap against the index's own gap is a stock-specific auction imbalance, not market news."))
    S_B = ("Abnormal volume = attention / news; attention-driven buying pressure and liquidity demand move prices and "
           "partly revert.")
    for k in (3.0, 5.0):
        for mode in ("cont", "fade"):
            C.append(_c(f"dg_b_earlyvol_{int(k)}_{mode}", "B", "minute", "r_early_vol", {"k": k, "mode": mode},
                        ("m_intra", "o0945", "o1515"), "mis", S_B))
    for mode in ("rev", "cont"):
        C.append(_c(f"dg_b_downattn_on_{mode}", "B", "minute", "r_down_attention", {"mode": mode}, ("m_on",), "fut",
                    S_B))
    for k in (3.0, 5.0):
        for mode in ("cont", "fade"):
            C.append(_c(f"dg_b_lasthour_{int(k)}_{mode}", "B", "minute", "r_lasthour_vol", {"k": k, "mode": mode},
                        ("m_on",), "fut", S_B))
    for day in ("up", "down"):
        C.append(_c(f"dg_b_attnswing_{day}_h5", "B", "minute", "r_attention_swing", {"day": day}, ("m_hold", 5),
                    "fut", S_B))
    for mode in ("lowvol_fade", "hivol_go"):
        C.append(_c(f"dg_b_gapvol_{mode}", "B", "minute", "r_gap_volume", {"mode": mode},
                    ("m_intra", "o0930", "o1515"), "mis", S_B))
    S_C = "Option writers' delta hedging and the unwinding of hedges around index expiries move the index overnight."
    for when in ("on", "pre"):
        for side, nm in ((1, "long"), (-1, "short")):
            C.append(_c(f"dg_c_expiry_{when}_{nm}", "C", "idx_minute", "r_expiry", {"when": when, "side": side},
                        ("i_on",), "fut", S_C))
    S_D = "A VIX spike is forced de-risking (margin calls, risk limits); the fear premium is paid to liquidity providers."
    for x in (0.10, 0.15):
        for h, prod in ((1, "fut"), (5, "fut")):
            C.append(_c(f"dg_d_vixspike_{int(x * 100)}_h{h}", "D", "daily", "r_vix_spike_losers", {"x": x},
                        ("d_hold", h), prod, S_D))
    C.append(_c("dg_d_vixhigh_losers_h5", "D", "daily", "r_vix_high_losers", {}, ("d_hold", 5), "fut", S_D))
    for h in (5, 20):
        C.append(_c(f"dg_d_vixhigh_nifty_h{h}", "D", "idx_daily", "r_vix_high_index", {}, ("i_hold", h), "fut", S_D))
    S_E = ("Delivery % separates investors who take shares home from intraday speculators: a delivery-heavy move is "
           "informed and drifts; a delivery-light spike is churn and fades.")
    for pp in (15.0, 25.0):
        for h in (5, 20):
            C.append(_c(f"dg_e_delivup_{int(pp)}_h{h}", "E", "daily", "r_delivery_shock", {"pp": pp, "day": "up"},
                        ("d_hold", h), "cnc", S_E))
            C.append(_c(f"dg_e_delivdn_{int(pp)}_h{h}", "E", "daily", "r_delivery_shock", {"pp": pp, "day": "down"},
                        ("d_hold", h), "fut", S_E))
    for h in (1, 5):
        C.append(_c(f"dg_e_lowdeliv_fade_h{h}", "E", "daily", "r_low_delivery_spike", {}, ("d_hold", h), "fut", S_E))
    C.append(_c("dg_e_delivtrend_h20", "E", "daily", "r_delivery_trend", {}, ("d_hold", 20), "cnc", S_E))
    S_F = "FIIs are the large, informed index-futures players; retail (Client) positioning is the contrarian side."
    for z in (1.0, 1.5):
        for h in (1, 5):
            C.append(_c(f"dg_f_fiiflow_z{str(z).replace('.', '')}_h{h}", "F", "idx_daily", "r_fii_flow", {"z": z},
                        ("i_hold", h), "fut", S_F))
    for h in (5, 20):
        C.append(_c(f"dg_f_clientcontra_h{h}", "F", "idx_daily", "r_client_contrarian", {}, ("i_hold", h), "fut",
                    S_F))
    S_G = ("The F&O ban forbids new positions until OI falls below 80 % of the market-wide limit: entry forces "
           "unwinding of a crowded trade; exit re-opens speculation.")
    for h in (5, 20):
        C.append(_c(f"dg_g_banentry_long_h{h}", "G", "daily", "r_ban", {"event": "entry", "side": 1}, ("d_hold", h),
                    "cnc", S_G))
    C.append(_c("dg_g_banentry_short_id", "G", "daily", "r_ban", {"event": "entry", "side": -1}, ("d_hold", 1), "mis",
                S_G))
    for h in (5, 20):
        C.append(_c(f"dg_g_banexit_long_h{h}", "G", "daily", "r_ban", {"event": "exit", "side": 1}, ("d_hold", h),
                    "cnc", S_G))
    C.append(_c("dg_g_banexit_short_h5", "G", "daily", "r_ban", {"event": "exit", "side": -1}, ("d_hold", 5), "fut",
                S_G))
    S_H = ("Futures OI tells new positions from closed ones: build-ups are new informed money (continue), short "
           "covering / long unwinding are forced closes (revert).")
    for quad in ("long_buildup", "short_buildup", "short_covering", "long_unwinding"):
        for thr in (0.10, 0.20):
            for h in (1, 5):
                C.append(_c(f"dg_h_{quad}_{int(thr * 100)}_h{h}", "H", "daily", "r_oi_quadrant",
                            {"quad": quad, "thr": thr}, ("d_hold", h), "fut", S_H))
    S_I = ("Large negotiated / bulk trades by a client class reveal who is accumulating; price pressure from big "
           "blocks partly reverts.")
    C.append(_c("dg_i_instbuy_h1", "I", "daily", "r_deals", {"cls": "inst", "flow": "buy", "side": 1}, ("d_hold", 1),
                "fut", S_I))
    C.append(_c("dg_i_instbuy_h5", "I", "daily", "r_deals", {"cls": "inst", "flow": "buy", "side": 1}, ("d_hold", 5),
                "cnc", S_I))
    C.append(_c("dg_i_instsell_rev_h5", "I", "daily", "r_deals", {"cls": "inst", "flow": "sell", "side": 1},
                ("d_hold", 5), "cnc", S_I))
    for h in (5, 20):
        C.append(_c(f"dg_i_otherbuy_h{h}", "I", "daily", "r_deals", {"cls": "other", "flow": "buy", "side": 1},
                    ("d_hold", h), "cnc", S_I))
    C.append(_c("dg_i_othersell_short_h5", "I", "daily", "r_deals", {"cls": "other", "flow": "sell", "side": -1},
                ("d_hold", 5), "fut", S_I))
    S_J = ("Results released outside market hours are absorbed by the next auction; under-reaction drifts, "
           "over-reaction reverts.")
    for g in (0.02, 0.04):
        for mode in ("cont", "fade"):
            C.append(_c(f"dg_j_resgap_{int(g * 100)}_{mode}", "J", "minute", "r_results_gap", {"g": g, "mode": mode},
                        ("m_intra", "o0920", "o1515"), "mis", S_J))
    C.append(_c("dg_j_react_up_h5", "J", "daily", "r_results_reaction", {"a": 0.04, "mode": "up"}, ("d_hold", 5),
                "cnc", S_J))
    C.append(_c("dg_j_react_down_h5", "J", "daily", "r_results_reaction", {"a": 0.04, "mode": "down"}, ("d_hold", 5),
                "fut", S_J))
    C.append(_c("dg_j_react_fade8_h1", "J", "daily", "r_results_reaction", {"a": 0.08, "mode": "fade"},
                ("d_hold", 1), "fut", S_J))
    S_K = ("Splits / bonuses draw retail attention to 'cheaper' shares; dividend-capture demand lifts cum-dividend "
           "prices.")
    C.append(_c("dg_k_splitbonus_runup", "K", "daily", "r_split_bonus", {"mode": "runup"}, ("d_hold", None), "cnc",
                S_K))
    C.append(_c("dg_k_splitbonus_post_h20", "K", "daily", "r_split_bonus", {"mode": "post"}, ("d_hold", None), "cnc",
                S_K))
    for y in (0.01, 0.02):
        C.append(_c(f"dg_k_divrunup_{int(y * 100)}", "K", "daily", "r_dividend_runup", {"y": y}, ("d_hold", None),
                    "cnc", S_K))
    S_M = ("One pre-declared model (L2 logistic, C = 1, fixed features, walk-forward inside DISCOVERY, frozen after it) "
           "combining the price / volume / delivery / OI features; it counts as three candidates.")
    C.append(_c("dg_m_ml_long_h5", "M", "daily", "r_ml", {"mode": "long"}, ("d_hold", 5), "fut", S_M))
    C.append(_c("dg_m_ml_short_h5", "M", "daily", "r_ml", {"mode": "short"}, ("d_hold", 5), "fut", S_M))
    C.append(_c("dg_m_ml_ls_h5", "M", "daily", "r_ml", {"mode": "ls"}, ("d_hold", 5), "fut", S_M))
    ids = [c.id for c in C]
    assert len(ids) == len(set(ids))
    return C


EXCLUDED_BY_DESIGN = {
    "auction fade with entry and exit inside 09:15-09:30": "neighbours of the proven AUC-FADE-2PCT / 3PCT (not new)",
    "high-volume UP-day overnight long": "the proven SWING-ATTN-ON-3X and its threshold neighbours (not new)",
    "index-weight changes": "NIFTY 50 / 100 reviews give < 60 events in VALIDATION (flows study: 37): cannot pass",
    "expiry-day intraday index trades": "< 150 index expiry sessions in VALIDATION: the intraday minimum cannot be met",
    "MWPL proximity": "no per-stock market-wide-position-limit series in the lab's data (only the ban list)",
    "VIX term structure": "no India VIX futures data in the lab",
    "options": "no candidate trades options; nothing uses VIX-modelled option prices",
}


# ══════════════════════════════════════════════════════════════════════════
# context, signals
# ══════════════════════════════════════════════════════════════════════════
_CTX: dict = {}


def context(need_minute: bool = True, need_daily: bool = True) -> F.Ctx:
    if "ctx" not in _CTX:
        from . import corrected
        corrected.install(minute=True, daily=True)
        daily = F.build_daily_panel() if need_daily else None
        minute = F.build_minute_panel() if need_minute else None
        _CTX["ctx"] = F.Ctx(daily, minute)
    return _CTX["ctx"]


def signals(c: Cand, ctx: F.Ctx) -> dict:
    """Sparse trigger cells of a candidate: {"s", "t", "side", "rank", "hold"} sorted by (t, -rank, symbol)."""
    key = "sig:" + c.id
    if key in ctx._c:
        return ctx._c[key]
    trig, side, rank, hold = getattr(F, c.rule)(ctx, **c.kw)
    s, t = np.nonzero(trig)
    h = (np.asarray(hold)[s, t] if hold is not None else np.full(len(s), _fixed_hold(c))).astype(np.int64)
    order = np.lexsort((_name_rank(c, ctx)[s], -rank[s, t], t))
    out = {"s": s[order].astype(np.int64), "t": t[order].astype(np.int64), "side": side[s, t][order].astype(np.int64),
           "rank": rank[s, t][order], "hold": h[order]}
    ctx._c[key] = out
    return out


def _fixed_hold(c: Cand) -> int:
    return int(c.exec[1]) if len(c.exec) > 1 and isinstance(c.exec[1], (int, np.integer)) else 0


def _name_rank(c: Cand, ctx: F.Ctx) -> np.ndarray:
    """Alphabetical rank of each row's symbol name (the tie-break of the daily cap)."""
    names = _names(c, ctx)
    r = np.empty(len(names), np.int64)
    r[np.argsort(np.array(names, dtype=object), kind="stable")] = np.arange(len(names))
    return r


def _names(c: Cand, ctx: F.Ctx) -> list:
    if c.grid in ("minute", "idx_minute"):
        return [{"_NIFTY50_min": "NIFTY", "_NIFTYBANK_min": "BANKNIFTY"}.get(x, x) for x in ctx.minute.syms]
    if c.grid == "idx_daily":
        return ["NIFTY"]
    return ctx.daily.syms


def calendar_of(c: Cand, ctx: F.Ctx) -> np.ndarray:
    return ctx.minute.cal if c.grid in ("minute", "idx_minute") else ctx.daily.cal


# ══════════════════════════════════════════════════════════════════════════
# the engine
# ══════════════════════════════════════════════════════════════════════════
def _dirty(ctx: F.Ctx, s: int, t0: int, t1: int) -> bool:
    """protocol_min_oos's DIRTY RULE on the minute grid: a dirty date of s in [cal t0, cal t1] or equal to the
    session right after t1."""
    M = ctx.minute
    dl = M.dirty.get(s)
    if dl is None or not len(dl):
        return False
    a, b = int(M.cal[t0]), int(M.cal[t1])
    j = int(np.searchsorted(dl, a))
    if j < len(dl) and dl[j] <= b:
        return True
    if t1 + 1 < len(M.cal):
        nx = int(M.cal[t1 + 1])
        return j < len(dl) and int(dl[j]) == nx
    return False


def _trade(cid, sym, d0, side, et, px_in, d1, xt, px_out, qty, kind, slip):
    fill = px_in * (1.0 + side * slip)
    xp = px_out * (1.0 - side * slip)
    gross = side * (xp - fill) * qty
    bv, sv = (fill * qty, xp * qty) if side > 0 else (xp * qty, fill * qty)
    cost = round_trip_cost(bv, sv, kind)
    return (cid, sym, int(d0), int(side), et, round(fill, 4), int(d1), xt, round(xp, 4), int(qty), round(gross, 2),
            cost, round(gross - cost, 2), "time")


def plan(c: Cand, ctx: F.Ctx, lo: int, hi: int, sig: dict | None = None, count_only: bool = False) -> list:
    """The entries a candidate takes with entry date in [lo, hi]: [(s, t_decision, side, t_entry, t_exit, px_in,
    px_out, entry_field, exit_field, bench_in, bench_out)] after every eligibility rule, the daily cap and the
    one-position-per-symbol rule. Prices are raw (no slippage). count_only skips nothing but the benchmark."""
    sig = signals(c, ctx) if sig is None else sig
    ex = c.exec[0]
    out = []
    busy: dict = {}
    per_day: dict = {}
    if c.grid in ("daily", "idx_daily"):
        Pn = ctx.daily
        cal = Pn.cal
        T = len(cal)
        hi_i = int(np.searchsorted(cal, hi, side="right")) - 1
        for s, t, side, h in zip(sig["s"].tolist(), sig["t"].tolist(), sig["side"].tolist(), sig["hold"].tolist()):
            e = t + 1
            if e >= T or e > hi_i or cal[e] < lo:
                continue
            if per_day.get(e, 0) >= (1 if c.single else MAX_PER_DAY):
                continue
            if busy.get(s, -1) >= e:
                continue
            if h is None or h < 1:
                continue
            x = min(t + int(h), hi_i)
            if c.grid == "daily":
                if not Pn.member[s, t]:
                    continue
                if c.product == "fut" and not Pn.fo[s, t]:
                    continue
                if side < 0 and c.product == "cnc":
                    continue
                o_row, c_row = Pn.o[s], Pn.c[s]
            else:
                o_row, c_row = Pn.nifty["o"], Pn.nifty["c"]
            pin = o_row[e]
            if not (np.isfinite(pin) and pin > 0):
                continue
            xs = x
            while xs >= e and not np.isfinite(c_row[xs]):
                xs -= 1
            if xs < e:
                continue
            if c.grid == "daily" and Pn.excluded(s, int(cal[t]), int(cal[xs])):
                continue
            bi, bo = Pn.nifty["o"][e], Pn.nifty["c"][xs]
            out.append((s, t, side, e, xs, float(pin), float(c_row[xs]), "o", "c", bi, bo))
            busy[s] = xs
            per_day[e] = per_day.get(e, 0) + 1
        return out
    M = ctx.minute
    cal = M.cal
    T = len(cal)
    has = np.isfinite(M.f["nb"])
    nifty = M.six["_NIFTY50_min"]
    for s, t, side, h in zip(sig["s"].tolist(), sig["t"].tolist(), sig["side"].tolist(), sig["hold"].tolist()):
        d = int(cal[t])
        if d < lo or d > hi:
            continue
        if per_day.get(t, 0) >= (2 if c.single else MAX_PER_DAY):
            continue
        if busy.get(s, -1) >= t:
            continue
        if d in M.special:
            continue
        if ex == "m_intra":
            fi, fo_ = c.exec[1], c.exec[2]
            pin, pout, x = M.f[fi][s, t], M.f[fo_][s, t], t
            bi, bo = M.f[fi][nifty, t], M.f[fo_][nifty, t]
        elif ex in ("m_on", "i_on"):
            x = t + 1
            if x >= T or not has[s, x] or int(cal[x]) > hi or int(cal[x]) in M.special:
                continue
            fi, fo_ = "o1521", "o0915"
            pin, pout = M.f["o1521"][s, t], M.f["o0915"][s, x]
            bi, bo = M.f["o1521"][nifty, t], M.f["o0915"][nifty, x]
        elif ex == "m_hold":
            x = t + int(c.exec[1])
            if x >= T or not has[s, x] or int(cal[x]) > hi or int(cal[x]) in M.special:
                continue
            fi, fo_ = "o1521", "o1521"
            pin, pout = M.f["o1521"][s, t], M.f["o1521"][s, x]
            bi, bo = M.f["o1521"][nifty, t], M.f["o1521"][nifty, x]
        else:
            raise ValueError(ex)
        if not (np.isfinite(pin) and np.isfinite(pout) and pin > 0 and pout > 0):
            continue
        if _dirty(ctx, s, t, x):
            continue
        out.append((s, t, side, t, x, float(pin), float(pout), fi, fo_, bi, bo))
        busy[s] = x
        per_day[t] = per_day.get(t, 0) + 1
    return out


def execute(c: Cand, ctx: F.Ctx, lo: int, hi: int, slip_bps: float | None = None, sig: dict | None = None,
            with_bench: bool = False):
    """Trades (sim.TRADE_FIELDS) of a candidate with entry date in [lo, hi]; with_bench also returns, per trade,
    the NIFTY 50 return over the same holding window (for the excess-return line)."""
    rows = plan(c, ctx, lo, hi, sig)
    names = _names(c, ctx)
    cal = calendar_of(c, ctx)
    idx = c.single
    if slip_bps is None:
        slip_bps = IDX_SLIP_BPS if idx else SLIP_BPS
    slip = slip_bps / 1e4
    kind = {"mis": "equity_mis", "cnc": "equity_cnc", "fut": "future"}[c.product]
    tr, bench = [], []
    for s, t, side, e, x, pin, pout, fi, fo_, bi, bo in rows:
        sym = names[s]
        qty = sim.LOT_SIZES[sym] if idx else max(1, int(NOTIONAL // pin))
        et = ENTRY_T.get(fi, "09:15")
        xt = "15:29" if fo_ == "c" else ENTRY_T.get(fo_, "15:15")
        tr.append(_trade(c.id, sym, cal[e], side, et, pin, cal[x], xt, pout, qty, kind, slip))
        bench.append(float(bo / bi - 1.0) if (np.isfinite(bi) and np.isfinite(bo) and bi > 0) else float("nan"))
    order = sorted(range(len(tr)), key=lambda k: (tr[k][2], tr[k][1]))
    tr = [tr[k] for k in order]
    bench = [bench[k] for k in order]
    return (tr, bench) if with_bench else tr


# ══════════════════════════════════════════════════════════════════════════
# statistics, the DISCOVERY selection rule
# ══════════════════════════════════════════════════════════════════════════
def halves(c: Cand):
    return MIN_HALVES if c.grid in ("minute", "idx_minute") else DAILY_HALVES


def disc_period(c: Cand):
    return MINUTE_DISC if c.grid in ("minute", "idx_minute") else DAILY_DISC


def _tdicts(trades):
    return [{"entry": int(t[2]), "exit": int(t[6]), "net": float(t[12]), "gross": float(t[10]),
             "cost": float(t[11])} for t in trades]


def block_stats(trades) -> dict:
    s = P.stats(trades)
    s["gross"] = round(sum(t[10] for t in trades), 2)
    s["cost"] = round(sum(t[11] for t in trades), 2)
    s["long"] = sum(1 for t in trades if t[3] > 0)
    return s


def judge_discovery(c: Cand, tr2: list, tr5: list) -> dict:
    """The pre-registered DISCOVERY gates: net > 0 in both halves; PF >= 1.15 over DISCOVERY; >= the protocol's minimum
    trades (intraday 150 / positional 60) in EACH half; top symbol <= 35 % of net (not for index candidates); and at
    5 bps (index 2.5 bps) net > 0 in both halves."""
    h1, h2 = halves(c)
    b1 = [t for t in tr2 if h1[0] <= t[2] <= h1[1]]
    b2 = [t for t in tr2 if h2[0] <= t[2] <= h2[1]]
    s1, s2, sa = block_stats(b1), block_stats(b2), block_stats(b1 + b2)
    f1 = sum(t[12] for t in tr5 if h1[0] <= t[2] <= h1[1])
    f2 = sum(t[12] for t in tr5 if h2[0] <= t[2] <= h2[1])
    by: dict = {}
    for t in b1 + b2:
        by[t[1]] = by.get(t[1], 0.0) + t[12]
    top, top_net = max(by.items(), key=lambda kv: kv[1]) if by else ("", 0.0)
    share = top_net / sa["net"] if sa["net"] > 0 else float("nan")
    need = MIN_TRADES[c.horizon]
    why = []
    if s1["net"] <= 0:
        why.append("H1 net <= 0")
    if s2["net"] <= 0:
        why.append("H2 net <= 0")
    if sa["pf"] < P.SCREEN_MIN_PF:
        why.append(f"PF {sa['pf']:.3f} < {P.SCREEN_MIN_PF}")
    if s1["trades"] < need:
        why.append(f"H1 {s1['trades']} trades < {need}")
    if s2["trades"] < need:
        why.append(f"H2 {s2['trades']} trades < {need}")
    if sa["net"] > 0 and share > P.SCREEN_MAX_SYMBOL_SHARE and not c.single:
        why.append(f"{top} is {share:.0%} of net")
    if f1 <= 0 or f2 <= 0:
        why.append(f"5 bps: H1 {f1:,.0f} / H2 {f2:,.0f}")
    return {"passed": not why, "why": why, "h1": s1, "h2": s2, "disc": sa, "top_symbol": top,
            "top_share": None if share != share else round(share, 3), "net5_h1": round(f1, 2), "net5_h2": round(f2, 2)}


def series(c: Cand, ctx: F.Ctx, trades, lo: int, hi: int) -> dict:
    cal = [int(x) for x in calendar_of(c, ctx)]
    st = TR.series_stats(_tdicts(trades), cal, lo, hi)
    return {k: st[k] for k in ("trades", "net", "pf", "mean_net", "std_net", "sr_trade", "mean_gross", "mean_cost",
                               "n_days", "sr_daily", "skew_daily", "kurt_daily")}


def robust_var(values) -> float:
    x = sorted(v for v in values if v == v)
    if len(x) < 2:
        return float("nan")
    med = TR._median(x)
    mad = TR._median(sorted(abs(v - med) for v in x))
    return (1.4826 * mad) ** 2


def dsr_of(st: dict, n: int, v: float) -> dict:
    if not st.get("n_days") or st.get("sr_daily") is None or st["sr_daily"] != st["sr_daily"]:
        return {"dsr": float("nan"), "psr0": float("nan"), "sr0": float("nan")}
    r = TR.dsr(st["sr_daily"], st["n_days"], st["skew_daily"], st["kurt_daily"], n, v)
    return {"dsr": r["dsr"], "psr0": r["psr0"], "sr0": r["sr0"]}


def select(rows: list, n_search: int, v: float, k: int = K_CARRY, cap: int = FAMILY_CAP) -> list:
    """Passers ranked by DSR (N = n_search + LEDGER_N0, V = the pre-registered V); the first k, at most `cap` per
    family. rows: [{"id", "family", "passed", "dsr"}]."""
    ok = [r for r in rows if r["passed"] and r.get("dsr") == r.get("dsr")]
    ok.sort(key=lambda r: (-r["dsr"], r["id"]))
    out, fam = [], {}
    for r in ok:
        if len(out) >= k:
            break
        if fam.get(r["family"], 0) >= cap:
            continue
        out.append(r["id"])
        fam[r["family"]] = fam.get(r["family"], 0) + 1
    return out


# ══════════════════════════════════════════════════════════════════════════
# controls (identical pipeline)
# ══════════════════════════════════════════════════════════════════════════
def eligible_cells(c: Cand, ctx: F.Ctx, lo: int, hi: int) -> tuple:
    """(s, t) of every decision cell whose entry date is in [lo, hi] and whose entry price exists, under the
    candidate's grid / product rules (the pool the placebo and the shuffle draw from)."""
    if c.grid == "daily":
        Pn = ctx.daily
        cal = Pn.cal
        e_ok = np.zeros(len(cal), bool)
        e_ok[:-1] = (cal[1:] >= lo) & (cal[1:] <= hi)
        o1 = np.full(Pn.shape, np.nan)
        o1[:, :-1] = Pn.o[:, 1:]
        m = Pn.member & np.isfinite(o1) & e_ok[None, :]
        if c.product == "fut":
            m &= Pn.fo
        return np.nonzero(m)
    if c.grid == "idx_daily":
        cal = ctx.daily.cal
        t = np.nonzero((np.r_[cal[1:], 0] >= lo) & (np.r_[cal[1:], 0] <= hi))[0]
        return np.zeros(len(t), np.int64), t
    M = ctx.minute
    rows = M.stocks if c.grid == "minute" else [M.six[x] for x in F.INDEX_MIN]
    inwin = (M.cal >= lo) & (M.cal <= hi)
    m = np.zeros(M.shape, bool)
    m[rows] = np.isfinite(M.f["nb"][rows]) & inwin[None, :]
    return np.nonzero(m)


def shuffled_signals(c: Cand, ctx: F.Ctx, lo: int, hi: int, seed: int) -> dict:
    """The SHUFFLE-THE-FEATURE control: inside [lo, hi] each symbol's trigger cells (with their side, rank, hold) are
    moved to dates drawn without replacement from that symbol's eligible dates - the same as permuting the rule's
    feature vector across dates within the symbol. Index rules: across the index's dates."""
    sig = signals(c, ctx)
    cal = calendar_of(c, ctx)
    rng = np.random.default_rng(seed)
    es, et = eligible_cells(c, ctx, lo, hi)
    pool: dict = {}
    for s, t in zip(es.tolist(), et.tolist()):
        pool.setdefault(s, []).append(t)
    off = 1 if c.grid in ("daily", "idx_daily") else 0
    inwin = [(lo <= int(cal[min(t + off, len(cal) - 1)]) <= hi) for t in sig["t"].tolist()]
    S, Tn, SD, RK, HD = [], [], [], [], []
    by: dict = {}
    for k, ok in enumerate(inwin):
        if ok:
            by.setdefault(int(sig["s"][k]), []).append(k)
    for s, ks in by.items():
        p = pool.get(s, [])
        if not p:
            continue
        n = min(len(ks), len(p))
        new_t = rng.choice(np.asarray(p), size=n, replace=False)
        for k, t in zip(ks[:n], new_t.tolist()):
            S.append(s), Tn.append(t), SD.append(int(sig["side"][k])), RK.append(float(sig["rank"][k]))
            HD.append(int(sig["hold"][k]))
    return _sorted_sig(c, ctx, S, Tn, SD, RK, HD)


def _sorted_sig(c, ctx, S, Tn, SD, RK, HD) -> dict:
    S, Tn = np.asarray(S, np.int64), np.asarray(Tn, np.int64)
    RK = np.asarray(RK, float)
    order = np.lexsort((_name_rank(c, ctx)[S], -RK, Tn)) if len(S) else np.arange(0)
    return {"s": S[order], "t": Tn[order], "side": np.asarray(SD, np.int64)[order], "rank": RK[order],
            "hold": np.asarray(HD, np.int64)[order]}


def placebo_signals(c: Cand, ctx: F.Ctx, lo: int, hi: int, n: int, long_share: float, hold: int,
                    seed: int) -> dict:
    """PLACEBO: n cells drawn uniformly (without replacement) from the candidate's eligible pool in [lo, hi], side long
    with probability long_share, random rank, the candidate's (median) hold."""
    rng = np.random.default_rng(seed)
    es, et = eligible_cells(c, ctx, lo, hi)
    if not len(es) or n <= 0:
        return _sorted_sig(c, ctx, [], [], [], [], [])
    k = rng.choice(len(es), size=min(n, len(es)), replace=False)
    sd = np.where(rng.random(len(k)) < long_share, 1, -1)
    if c.product == "cnc":
        sd[:] = 1
    return _sorted_sig(c, ctx, es[k].tolist(), et[k].tolist(), sd.tolist(), rng.random(len(k)).tolist(),
                       [hold] * len(k))


PLANTED = (
    # id, grid, exec, product, fraction of eligible cells, p(side = sign of the realised move)
    ("ctl_planted_min_intra_p60", "minute", ("m_intra", "o0920", "o1515"), "mis", 0.02, 0.60),
    ("ctl_planted_min_on_p60", "minute", ("m_on",), "fut", 0.02, 0.60),
    ("ctl_planted_daily_h5_p60", "daily", ("d_hold", 5), "fut", 0.01, 0.60),
    ("ctl_planted_nifty_h5_p60", "idx_daily", ("i_hold", 5), "fut", 0.30, 0.60),
    ("ctl_planted_daily_h5_p100", "daily", ("d_hold", 5), "fut", 0.01, 1.00),
    ("ctl_random_daily_h5_p50", "daily", ("d_hold", 5), "fut", 0.01, 0.50),
    ("ctl_random_min_intra_p50", "minute", ("m_intra", "o0920", "o1515"), "mis", 0.02, 0.50),
)


def planted_signals(c: Cand, ctx: F.Ctx, lo: int, hi: int, frac: float, p: float, seed: int) -> dict:
    """PLANTED EDGE: a random fraction of eligible cells; the side is the sign of the realised raw price move of the
    candidate's own entry -> exit with probability p (else the opposite). p = 0.5 is a pure random control."""
    rng = np.random.default_rng(seed)
    es, et = eligible_cells(c, ctx, lo, hi)
    k = rng.random(len(es)) < frac
    es, et = es[k], et[k]
    tmp = _sorted_sig(c, ctx, es.tolist(), et.tolist(), [1] * len(es), rng.random(len(es)).tolist(),
                      [_fixed_hold(c)] * len(es))
    rows = plan(dataclasses.replace(c, product="fut"), ctx, lo, hi, tmp)
    S, Tn, SD, RK, HD = [], [], [], [], []
    for s, t, _side, e, x, pin, pout, *_ in rows:
        mv = 1 if pout > pin else -1
        S.append(s), Tn.append(t), SD.append(mv if rng.random() < p else -mv), RK.append(rng.random())
        HD.append(_fixed_hold(c))
    return _sorted_sig(c, ctx, S, Tn, SD, RK, HD)


# ══════════════════════════════════════════════════════════════════════════
# (a) availability counts
# ══════════════════════════════════════════════════════════════════════════
def validation_split():
    cal3 = TR.calendar_3y(F.SCRATCH)
    return P.split_sessions(cal3), cal3


def data_check() -> dict:
    """Signal counts per candidate (after every eligibility rule, the cap and the one-position rule) in each
    DISCOVERY half and in VALIDATION. No price change and no return is computed. The keep rule (pre-registered with
    the counts): >= the minimum trades in EACH DISCOVERY half AND >= the minimum over VALIDATION."""
    ctx = context()
    split, _ = validation_split()
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "validation": list(split.discovery),
           "rule": "keep iff count >= protocol minimum (intraday 150 / positional 60) in each DISCOVERY half and over "
                   "VALIDATION", "candidates": {}}
    for c in candidates():
        h1, h2 = halves(c)
        n1 = len(plan(c, ctx, *h1))
        n2 = len(plan(c, ctx, *h2))
        nv = len(plan(c, ctx, *split.discovery))
        need = MIN_TRADES[c.horizon]
        keep = n1 >= need and n2 >= need and nv >= need
        out["candidates"][c.id] = {"family": c.family, "horizon": c.horizon, "disc_h1": n1, "disc_h2": n2,
                                   "validation": nv, "need": need, "keep": keep}
        log(f"data-check {c.id:34s} H1 {n1:6d} H2 {n2:6d} VAL {nv:6d} need {need:3d} {'KEEP' if keep else 'drop'}")
    out["n_declared"] = len(out["candidates"])
    out["n_search"] = sum(1 for v in out["candidates"].values() if v["keep"])
    with open(DATA_CHECK, "w") as f:
        json.dump(out, f, indent=1)
    # data coverage (counts only)
    P_ = ctx.daily
    M_ = ctx.minute
    cov = {"daily_symbols": len(P_.syms), "daily_sessions": len(P_.cal), "minute_names": len(M_.syms),
           "minute_sessions": len(M_.cal), "minute_special": sorted(M_.special),
           "delivery_cells": int(np.isfinite(ctx.d("dp")).sum()), "oi_cells": int(np.isfinite(ctx.d("doi")).sum()),
           "ban_cells": int(ctx.d("ban").sum()),
           "participant_days": int(np.isfinite(ctx.d("part")["fii_fut_idx_long"]).sum()),
           "inst_deal_cells": int((ctx.d("deals")["inst"] != 0).sum()),
           "other_deal_cells": int((ctx.d("deals")["other"] != 0).sum()),
           "results_reaction_cells": int(ctx.d("react").sum()),
           "results_premarket_minute_cells": int(ctx.m("pre").sum()),
           "expiries": {k: len(v) for k, v in ctx.m("exp").items()}}
    out["coverage"] = cov
    with open(DATA_CHECK, "w") as f:
        json.dump(out, f, indent=1)
    log(f"data-check: {out['n_search']} of {out['n_declared']} candidates kept")
    return out


# ══════════════════════════════════════════════════════════════════════════
# (b) pre-registration
# ══════════════════════════════════════════════════════════════════════════
CODE_FILES = ("datagen_features.py", "datagen_run.py", "protocol.py", "protocol_oos.py", "protocol_min_oos.py",
              "trials.py", "sim.py", "corrected.py", "data.py", "data_daily.py", "events_daily.py",
              "flows_strategies.py", "events_parse.py", "../costs.py")


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_sha() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    return {os.path.basename(f): _sha(os.path.join(here, f)) for f in CODE_FILES}


def kept_ids() -> list:
    with open(DATA_CHECK) as f:
        dc = json.load(f)
    return [c.id for c in candidates() if dc["candidates"][c.id]["keep"]]


def search_space() -> list:
    ids = set(kept_ids())
    return [c for c in candidates() if c.id in ids]


def prereg_doc() -> dict:
    with open(DATA_CHECK) as f:
        dc = json.load(f)
    split, cal3 = validation_split()
    cs = candidates()
    kept = [c for c in cs if dc["candidates"][c.id]["keep"]]
    return {
        "study": "datagen_v1 - data-driven discovery: strategies created from the lab's own data, selected on old data",
        "written": time.strftime("%Y-%m-%d %H:%M"),
        "written_before_any_return": True,
        "n_declared": len(cs), "n_search": len(kept),
        "families": {f: sum(1 for c in kept if c.family == f) for f in sorted({c.family for c in cs})},
        "splits": {"discovery_minute": list(MINUTE_DISC), "discovery_minute_halves": [list(h) for h in MIN_HALVES],
                   "discovery_daily": list(DAILY_DISC), "discovery_daily_halves": [list(h) for h in DAILY_HALVES],
                   "validation": list(split.discovery), "validation_halves": [list(split.h1), list(split.h2)],
                   "final": list(split.final), "calendar_3y_sessions": len(cal3)},
        "engine": {"fill_model": FILL_MODEL, "cost_model": COST_MODEL, "notional_rs": NOTIONAL,
                   "slippage_bps": SLIP_BPS, "stress_bps": SENS_BPS, "index_slippage_bps": IDX_SLIP_BPS,
                   "index_stress_bps": IDX_SENS_BPS, "max_per_day": MAX_PER_DAY, "lots": dict(sim.LOT_SIZES),
                   "costs": "costs.round_trip_cost: mis=equity_mis, cnc=equity_cnc, fut=future (2026 Zerodha)",
                   "doc": __doc__.split("ENGINE")[1].strip()},
        "selection": {
            "gates": "DISCOVERY: net > 0 in H1 and H2; PF >= 1.15; trades >= 150 intraday / 60 positional in EACH half; "
                     "top symbol <= 35 % of net (not index); at 5 bps (index 2.5) net > 0 in H1 and H2",
            "rank": "passers by DSR of the DISCOVERY daily net P&L, N = n_search + 2542, V = max(robust (1.4826 MAD)^2 of "
                    "the n_search DISCOVERY daily Sharpes, median 1/(T-1))",
            "K": K_CARRY, "family_cap": FAMILY_CAP,
            "full_table": "every candidate's DISCOVERY row is reported and appended to trials_v2.csv"},
        "validation": "protocol.evaluate(screen_only=True) on calendar_3y: protocol.screen on 2023-11-10..2025-12-16, "
                      "once (validation.json is never recomputed)",
        "final": "protocol.evaluate one-look FINAL for validation passers (ledger <datagen>/final_ledger.json); "
                 "CONFIRMED = net > 0 and PF >= 1.05",
        "proof": {"definition": "PROVEN = VALIDATION screen passed AND FINAL CONFIRMED AND DSR >= 0.95 of the full-history "
                                "daily net P&L (DISCOVERY start .. FINAL end) at N_full = (2023-26 variants in "
                                "trials_v2.csv before this study, 2542) + n_search and the selection V AND, for "
                                "long-biased rules (>= 70 % longs), the 95 % date-clustered bootstrap CI of the mean "
                                "per-trade excess over NIFTY 50 on VALIDATION + FINAL above 0 AND the candidate's "
                                "VALIDATION + FINAL net above the 95th percentile of 50 placebos AND every planted-edge "
                                "control (p >= 0.6) detected by the DISCOVERY gates",
                  "report_only": "DSR with the ledger's robust V and the 1/(T-1) floor; DSR on VALIDATION + FINAL only; "
                                 "5 bps; Rs 10 lakh account (capital.py, Rs 1 lakh per position)"},
        "controls": {"shuffle": f"{N_SHUFFLE} seeds: every candidate's trigger cells permuted across dates within symbol "
                                "in DISCOVERY, the selection gates re-applied: expected vs observed passers",
                     "placebo": f"{N_PLACEBO} seeds per carried candidate and period: same count, random eligible cells, "
                                "same long share",
                     "planted": [list(x[:3]) + [x[3], x[4], x[5]] for x in PLANTED]},
        "seed": SEED,
        "excluded_by_design": EXCLUDED_BY_DESIGN,
        "candidates": [{"id": c.id, "family": c.family, "grid": c.grid, "rule": c.describe(),
                        "rule_doc": (getattr(F, c.rule).__doc__ or "").strip(), "params": dict(c.params),
                        "exec": list(c.exec), "product": c.product, "horizon": c.horizon, "story": c.story,
                        "fingerprint": c.spec().fingerprint, "counts": dc["candidates"][c.id]} for c in kept],
        "dropped_on_availability": {c.id: dc["candidates"][c.id] for c in cs if not dc["candidates"][c.id]["keep"]},
        "ml": {"features": list(F.ML_FEATURES), "C": F.ML_C, "horizon": F.ML_HORIZON,
               "walk_forward": "yearly from 2014, fitted on decision days whose label window closed before the year; "
                               "frozen model (fitted through 2023-11-09) for every later day"},
        "data_check_sha256": _sha(DATA_CHECK),
        "code_sha256": code_sha(),
    }


def preregister() -> str:
    if os.path.exists(PREREG):
        raise SystemExit(f"{PREREG} exists: the pre-registration is written once")
    doc = prereg_doc()
    txt = json.dumps(doc, indent=1, sort_keys=False) + "\n"
    with open(PREREG, "w") as f:
        f.write(txt)
    sha = _sha(PREREG)
    with open(PREREG_SHA_FILE, "w") as f:
        f.write(sha + "\n")
    os.makedirs(os.path.dirname(REPO_PREREG), exist_ok=True)
    shutil.copyfile(PREREG, REPO_PREREG)
    log(f"pre-registered {doc['n_search']} candidates, sha256 {sha}")
    return sha


def check_prereg() -> dict:
    """Refuse to run unless the pre-registration exists, is byte-identical to the repo copy, matches its sha256 file,
    and the code / candidate fingerprints are unchanged."""
    with open(PREREG_SHA_FILE) as f:
        want = f.read().strip()
    if _sha(PREREG) != want or _sha(REPO_PREREG) != want:
        raise SystemExit("pre-registration sha256 mismatch (scratch / repo copy)")
    with open(PREREG) as f:
        doc = json.load(f)
    now = code_sha()
    for k in ("datagen_features.py", "datagen_run.py"):
        if doc["code_sha256"][k] != now[k]:
            raise SystemExit(f"{k} changed after the pre-registration")
    fp = {c.id: c.spec().fingerprint for c in candidates()}
    for row in doc["candidates"]:
        if fp.get(row["id"]) != row["fingerprint"]:
            raise SystemExit(f"{row['id']} fingerprint changed after the pre-registration")
    return doc


# ══════════════════════════════════════════════════════════════════════════
# (c) DISCOVERY, once
# ══════════════════════════════════════════════════════════════════════════
def _sens(c: Cand) -> float:
    return IDX_SENS_BPS if c.single else SENS_BPS


def run_candidate_discovery(c: Cand, ctx: F.Ctx, sig: dict | None = None) -> dict:
    lo, hi = disc_period(c)
    tr2 = execute(c, ctx, lo, hi, None, sig)
    tr5 = execute(c, ctx, lo, hi, _sens(c), sig)
    j = judge_discovery(c, tr2, tr5)
    st = series(c, ctx, tr2, lo, hi)
    return {"judge": j, "series": st, "trades": tr2}


def discovery() -> dict:
    if os.path.exists(DISC_OUT):
        log("discovery already run once: returning the stored result")
        with open(DISC_OUT) as f:
            return json.load(f)
    doc = check_prereg()
    ctx = context()
    space = search_space()
    assert len(space) == doc["n_search"]
    n_search = len(space)
    rows, trades = [], {}
    for c in space:
        r = run_candidate_discovery(c, ctx)
        trades[c.id] = r["trades"]
        rows.append({"id": c.id, "family": c.family, "grid": c.grid, "horizon": c.horizon,
                     "passed": r["judge"]["passed"], "why": r["judge"]["why"], "judge": r["judge"],
                     "series": r["series"]})
        log(f"discovery {c.id:34s} n={r['judge']['disc']['trades']:6d} net={r['judge']['disc']['net']:>12,.0f} "
            f"PF={r['judge']['disc']['pf']:.3f} {'PASS' if r['judge']['passed'] else 'fail: ' + '; '.join(r['judge']['why'])}")
    srs = [r["series"]["sr_daily"] for r in rows if r["series"]["n_days"] and r["series"]["sr_daily"] == r["series"]["sr_daily"]]
    v_mad = robust_var(srs)
    floor = float(np.median([1.0 / (r["series"]["n_days"] - 1) for r in rows if r["series"]["n_days"] > 1]))
    v = max(v_mad, floor)
    n_sel = n_search + LEDGER_N0
    for r in rows:
        r.update(dsr_of(r["series"], n_sel, v))
    carried = select(rows, n_search, v)
    out = {"run": time.strftime("%Y-%m-%d %H:%M"), "n_search": n_search, "n_sel": n_sel, "v_mad": v_mad,
           "v_floor": floor, "v": v, "passed": [r["id"] for r in rows if r["passed"]], "carried": carried,
           "rows": rows}
    os.makedirs(os.path.join(DG, "trades"), exist_ok=True)
    for cid, tr in trades.items():
        _write_trades(os.path.join(DG, "trades", f"disc_{cid}.csv.gz"), tr)
    with open(DISC_OUT, "w") as f:
        json.dump(out, f, indent=1, default=_js)
    log(f"discovery: {len(out['passed'])} of {n_search} passed the gates; carried {carried}")
    return out


def controls_discovery() -> dict:
    """Shuffle (all candidates, N_SHUFFLE seeds) and planted / random controls on DISCOVERY."""
    p = os.path.join(DG, "controls_discovery.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    check_prereg()
    ctx = context()
    space = search_space()
    shuffle = {}
    for k in range(N_SHUFFLE):
        seed = SEED + 1000 + k
        passed = []
        for c in space:
            lo, hi = disc_period(c)
            sig = shuffled_signals(c, ctx, lo, hi, seed)
            r = run_candidate_discovery(c, ctx, sig)
            if r["judge"]["passed"]:
                passed.append(c.id)
        shuffle[str(seed)] = passed
        log(f"shuffle seed {seed}: {len(passed)} of {len(space)} pass the DISCOVERY gates {passed}")
    planted = {}
    for k, (cid, grid, ex, prod, frac, p_) in enumerate(PLANTED):
        c = Cand(cid, "CTL", grid, "r_ml", (), ex, prod, "control")
        lo, hi = disc_period(c)
        sig = planted_signals(c, ctx, lo, hi, frac, p_, SEED + 2000 + k)
        r = run_candidate_discovery(c, ctx, sig)
        planted[cid] = {"p": p_, "frac": frac, "passed": r["judge"]["passed"], "why": r["judge"]["why"],
                        "disc": r["judge"]["disc"], "h1": r["judge"]["h1"], "h2": r["judge"]["h2"]}
        log(f"control {cid}: passed={r['judge']['passed']} PF={r['judge']['disc']['pf']} n={r['judge']['disc']['trades']}")
    out = {"shuffle": shuffle, "shuffle_mean_passers": float(np.mean([len(v) for v in shuffle.values()])),
           "planted": planted}
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=_js)
    return out


# ══════════════════════════════════════════════════════════════════════════
# (d) VALIDATION and (e) FINAL through protocol.evaluate
# ══════════════════════════════════════════════════════════════════════════
def _by_id() -> dict:
    return {c.id: c for c in candidates()}


def make_simulator(ctx: F.Ctx, slip_scale: float = 1.0):
    by = _by_id()

    def simulate(specs, lo, hi):
        out = {}
        for sp in specs:
            c = by[sp.id]
            bps = (IDX_SLIP_BPS if c.single else SLIP_BPS) * slip_scale
            try:
                out[sp.id] = {"trades": execute(c, ctx, lo, hi, bps), "skipped": 0, "lookahead": None, "error": None}
            except Exception as exc:          # pragma: no cover
                out[sp.id] = {"trades": [], "skipped": 0, "lookahead": None, "error": f"{type(exc).__name__}: {exc}"}
        return out
    return simulate


def _n_symbols(ctx) -> int:
    return len(ctx.daily.syms) + len(ctx.minute.syms)


def validate() -> dict:
    if os.path.exists(VAL_OUT):
        log("validation already run once: returning the stored result")
        with open(VAL_OUT) as f:
            return json.load(f)
    check_prereg()
    with open(DISC_OUT) as f:
        disc = json.load(f)
    ctx = context()
    by = _by_id()
    specs = [by[i].spec() for i in disc["carried"]]
    split, cal3 = validation_split()
    res, trades = P.evaluate(specs, cal3, make_simulator(ctx), n_symbols=_n_symbols(ctx), ledger_path=None,
                             dataset="datagen_v1", screen_only=True, cost_model=COST_MODEL)
    res5, _ = P.evaluate(specs, cal3, make_simulator(ctx, SENS_BPS / SLIP_BPS), n_symbols=_n_symbols(ctx),
                         ledger_path=None, dataset="datagen_v1", screen_only=True, cost_model=COST_MODEL)
    s5 = {r["id"]: r["screen"] for r in res5}
    rows = []
    for r in res:
        c = by[r["id"]]
        st = series(c, ctx, trades[r["id"]], *split.discovery)
        rows.append({"id": r["id"], "verdict": r["verdict"], "why": r.get("why", ""), "screen": r["screen"],
                     "series": st, "screen_5bps": {k: s5[r["id"]][k] for k in ("passed", "h1", "h2", "disc")}})
        _write_trades(os.path.join(DG, "trades", f"val_{r['id']}.csv.gz"), trades[r["id"]])
        log(f"validation {r['id']:34s} {r['verdict']}: {r.get('why', '')}")
    out = {"run": time.strftime("%Y-%m-%d %H:%M"), "period": list(split.discovery), "rows": rows,
           "passed": [x["id"] for x in rows if x["verdict"] == "PASSED SCREEN"]}
    with open(VAL_OUT, "w") as f:
        json.dump(out, f, indent=1, default=_js)
    return out


def final(trials_path: str | None = None) -> dict:
    if os.path.exists(FIN_OUT):
        log("final already run once: returning the stored result")
        with open(FIN_OUT) as f:
            return json.load(f)
    check_prereg()
    with open(VAL_OUT) as f:
        val = json.load(f)
    ctx = context()
    by = _by_id()
    split, cal3 = validation_split()
    specs = [by[i].spec() for i in val["passed"]]
    rows = []
    if specs:
        res, trades = P.evaluate(specs, cal3, make_simulator(ctx), n_symbols=_n_symbols(ctx),
                                 ledger_path=FINAL_LEDGER, dataset="datagen_v1", screen_only=False,
                                 cost_model=COST_MODEL)
        for r in res:
            ftr = [t for t in trades[r["id"]] if split.final[0] <= t[2] <= split.final[1]]
            vtr = [t for t in trades[r["id"]] if split.discovery[0] <= t[2] <= split.discovery[1]]
            assert r["verdict"] != "FAILED SCREEN", "validation passer failed the same screen on re-simulation"
            _write_trades(os.path.join(DG, "trades", f"fin_{r['id']}.csv.gz"), ftr)
            rows.append({"id": r["id"], "verdict": r["verdict"], "final": r["final"], "note": r.get("final_note", ""),
                         "n_val": len(vtr), "n_fin": len(ftr)})
            log(f"FINAL {r['id']:34s} {r['verdict']} {r['final']}")
    out = {"run": time.strftime("%Y-%m-%d %H:%M"), "period": list(split.final), "rows": rows}
    with open(FIN_OUT, "w") as f:
        json.dump(out, f, indent=1, default=_js)
    return out


# ══════════════════════════════════════════════════════════════════════════
# proof, controls on unseen periods, account view, trials ledger
# ══════════════════════════════════════════════════════════════════════════
def excess_ci(c: Cand, trades: list, bench: list, seed: int = SEED) -> dict:
    """Mean per-trade excess over NIFTY 50 (net - side x notional x NIFTY return over the same window) with a 95 %
    bootstrap CI clustered by entry date."""
    ex, dates = [], []
    for t, b in zip(trades, bench):
        if b != b:
            continue
        ex.append(t[12] - t[3] * t[5] * t[9] * b)
        dates.append(t[2])
    if not ex:
        return {"n": 0}
    by: dict = {}
    for d, e in zip(dates, ex):
        a = by.setdefault(d, [0.0, 0])
        a[0] += e
        a[1] += 1
    keys = list(by)
    sums = np.array([by[k][0] for k in keys])
    cnts = np.array([by[k][1] for k in keys])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(BOOT_N):
        k = rng.integers(0, len(keys), len(keys))
        bs.append(sums[k].sum() / cnts[k].sum())
    lo_, hi_ = np.percentile(bs, [100 * (1 - CI_LEVEL) / 2, 100 * (1 + CI_LEVEL) / 2])
    return {"n": len(ex), "mean_rs": round(float(np.mean(ex)), 2), "ci_rs": [round(float(lo_), 2),
                                                                           round(float(hi_), 2)],
            "mean_pct": round(100 * float(np.mean(ex)) / NOTIONAL, 3)}


def placebo_dist(c: Cand, ctx: F.Ctx, lo: int, hi: int, trades: list, seeds: int | None = None) -> dict:
    seeds = N_PLACEBO if seeds is None else seeds
    n = len(trades)
    ls = (sum(1 for t in trades if t[3] > 0) / n) if n else 1.0
    hold = int(np.median([x for x in signals(c, ctx)["hold"].tolist() if x > 0] or [1]))
    nets = []
    for k in range(seeds):
        sig = placebo_signals(c, ctx, lo, hi, n, ls, hold, SEED + 3000 + k)
        nets.append(sum(t[12] for t in execute(c, ctx, lo, hi, None, sig)))
    return {"n": n, "long_share": round(ls, 3), "nets": [round(x, 0) for x in nets],
            "p95": round(float(np.percentile(nets, 95)), 2) if nets else None,
            "median": round(float(np.median(nets)), 2) if nets else None,
            "actual": round(sum(t[12] for t in trades), 2)}


def account_view(c: Cand, trades: list, ctx: F.Ctx, lo: int, hi: int) -> dict:
    """Report only: the Rs 10 lakh account (capital.py) taking the candidate's trades at Rs 1 lakh per position."""
    from . import capital as CAP
    agent = {"mis": "intraday", "cnc": "swing", "fut": "futures"}[c.product]
    ag = dataclasses.replace(CAP.agents()[agent], fixed_notional=None if c.single else NOTIONAL)
    cands = []
    for t in trades:
        if not lo <= t[2] <= hi:
            continue
        hm = lambda s: int(s[:2]) * 60 + int(s[3:5])
        ek = t[2] * 10000 + hm(t[4])
        xk = t[6] * 10000 + hm(t[7]) + (1 if t[7] == "15:29" else 0)
        if xk <= ek:
            xk = ek + 1
        lot = sim.LOT_SIZES.get(t[1], 1)
        cands.append(CAP.cand(t[1], t[3], ek, t[5], None, xk, t[8], "time", 0.0,
                              "fut" if c.product == "fut" else c.product, lot, 1, c.id))
    res = CAP.simulate(cands, ag, c.id)
    cal = [int(x) for x in calendar_of(c, ctx)]
    m = CAP.metrics(res, cal, lo, hi)
    return {k: m[k] for k in ("trades", "candidates", "net", "return_pct_per_year", "max_dd_rs", "max_dd_pct",
                              "worst_month_rs", "pf", "sharpe", "skipped")}


def _write_trades(path: str, trades: list) -> None:
    import gzip
    with gzip.open(path, "wt") as f:
        f.write(",".join(sim.TRADE_FIELDS) + "\n")
        for t in trades:
            f.write(",".join(str(x) for x in t) + "\n")


def _read_trades(path: str) -> list:
    import gzip
    out = []
    if not os.path.exists(path):
        return out
    with gzip.open(path, "rt") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split(",")
            out.append((p[0], p[1], int(p[2]), int(p[3]), p[4], float(p[5]), int(p[6]), p[7], float(p[8]), int(p[9]),
                        float(p[10]), float(p[11]), float(p[12]), p[13]))
    return out


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if o != o else float(o)
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def proof(trials_path: str | None = None) -> dict:
    """Everything after the one FINAL look: proof gates, placebo on VALIDATION / FINAL, account view, trials rows."""
    p = os.path.join(DG, "proof.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    check_prereg()
    with open(DISC_OUT) as f:
        disc = json.load(f)
    with open(VAL_OUT) as f:
        val = json.load(f)
    with open(FIN_OUT) as f:
        fin = json.load(f)
    ctl = controls_discovery()
    ctx = context()
    by = _by_id()
    split, cal3 = validation_split()
    led = TR.TrialsLedger(trials_path) if trials_path else None
    n0 = TR.count_trials(led.read(), "2023-26") if led else LEDGER_N0
    n_full = n0 + disc["n_search"]
    rows_all = led.read() if led else []
    v_ledger = TR.var_sr_robust(rows_all)[0] if rows_all else float("nan")
    planted_ok = all(v["passed"] for k, v in ctl["planted"].items() if v["p"] >= 0.6)
    fin_by = {r["id"]: r for r in fin["rows"]}
    val_by = {r["id"]: r for r in val["rows"]}
    out = {"n_full": n_full, "n_ledger_before": n0, "v_selection": disc["v"], "v_ledger_robust": v_ledger,
           "planted_all_detected": planted_ok, "candidates": {}}
    unseen = (split.discovery[0], split.final[1])
    for cid in disc["carried"]:
        c = by[cid]
        dlo, dhi = disc_period(c)
        r = {"validation": val_by[cid]["verdict"], "final": fin_by.get(cid, {}).get("verdict", "not opened")}
        tr_v, b_v = execute(c, ctx, *split.discovery, None, None, True)
        r["placebo_disc"] = placebo_dist(c, ctx, dlo, dhi, _read_trades(os.path.join(DG, "trades",
                                                                                     f"disc_{cid}.csv.gz")))
        r["placebo_val"] = placebo_dist(c, ctx, *split.discovery, tr_v)
        tr_d = _read_trades(os.path.join(DG, "trades", f"disc_{cid}.csv.gz"))
        r["account_disc"] = account_view(c, tr_d, ctx, dlo, dhi)
        r["account_val"] = account_view(c, tr_v, ctx, *split.discovery)
        _, b_d = execute(c, ctx, dlo, dhi, None, None, True)
        r["excess_disc"] = excess_ci(c, tr_d, b_d)
        r["excess_val"] = excess_ci(c, tr_v, b_v)
        r["long_share_disc"] = round(sum(1 for t in tr_d if t[3] > 0) / max(1, len(tr_d)), 3)
        if r["final"] in ("CONFIRMED", "REJECTED ON FINAL"):
            tr_u, b_u = execute(c, ctx, *unseen, None, None, True)
            tr_f = [t for t in tr_u if t[2] >= split.final[0]]
            full = tr_d + tr_u
            cal = [int(x) for x in calendar_of(c, ctx)]
            st_full = TR.series_stats(_tdicts(full), cal, dlo, split.final[1])
            st_uns = TR.series_stats(_tdicts(tr_u), cal, *unseen)
            r["dsr_full"] = {"selection_v": dsr_of(st_full, n_full, disc["v"]),
                             "ledger_v": dsr_of(st_full, n_full, v_ledger),
                             "floor_v": dsr_of(st_full, n_full, 1.0 / max(1, st_full["n_days"] - 1)),
                             "sr_daily": st_full["sr_daily"], "n_days": st_full["n_days"]}
            r["dsr_unseen"] = {"selection_v": dsr_of(st_uns, n_full, disc["v"]), "sr_daily": st_uns["sr_daily"]}
            r["excess_unseen"] = excess_ci(c, tr_u, b_u)
            r["placebo_unseen"] = placebo_dist(c, ctx, *unseen, tr_u)
            r["account_final"] = account_view(c, tr_f, ctx, *split.final)
            ls = sum(1 for t in tr_u if t[3] > 0) / max(1, len(tr_u))
            gates = {"validation": r["validation"] == "PASSED SCREEN",
                     "final": r["final"] == "CONFIRMED",
                     "dsr": (r["dsr_full"]["selection_v"]["dsr"] or 0) >= DSR_MIN,
                     "excess": (ls < LONG_BIASED) or (r["excess_unseen"].get("ci_rs", [0])[0] > 0),
                     "placebo": r["placebo_unseen"]["actual"] > (r["placebo_unseen"]["p95"] or 0),
                     "planted": planted_ok}
            r["gates"] = gates
            r["proven"] = all(gates.values())
        else:
            r["proven"] = False
        out["candidates"][cid] = r
        log(f"proof {cid}: proven={r['proven']}")
    out["proven"] = [k for k, v in out["candidates"].items() if v["proven"]]
    if led:
        out["trials_appended"] = append_trials(led, disc, val, fin)
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=_js)
    return out


def append_trials(led, disc: dict, val: dict, fin: dict) -> int:
    """Every candidate's DISCOVERY row (era 2016-23 minute / 2012-23 daily), the VALIDATION screens and the FINAL looks
    (era 2023-26). Backed up first."""
    if os.path.exists(led.path):
        bak = led.path + ".bak_datagen_" + time.strftime("%Y%m%d_%H%M%S")
        shutil.copyfile(led.path, bak)
    by = _by_id()
    split, _ = validation_split()
    rows = []
    for r in disc["rows"]:
        c = by[r["id"]]
        era = "2016-23" if c.grid in ("minute", "idx_minute") else "2012-23"
        lo, hi = disc_period(c)
        rows.append({"strategy_id": c.id, "family": "datagen", "group": c.family, "params_fp": c.spec().fingerprint,
                     "era": era, "protocol": "datagen v1 discovery", "stage": "datagen-discovery",
                     "dataset": "datagen_v1", "period": f"{lo}-{hi}", "cost_model": COST_MODEL, **r["series"],
                     "verdict": ("CARRIED" if c.id in disc["carried"] else "PASSED GATES" if r["passed"]
                                 else "FAILED GATES"),
                     "source": "datagen/discovery.json#" + c.id, "note": "; ".join(r["why"])})
    fin_by = {r["id"]: r for r in fin["rows"]}
    for r in val["rows"]:
        c = by[r["id"]]
        rows.append({"strategy_id": c.id, "family": "datagen", "group": c.family, "params_fp": c.spec().fingerprint,
                     "era": "2023-26", "protocol": "datagen v1 validation (protocol.py)", "stage": "screen",
                     "dataset": "datagen_v1", "period": "%d-%d" % tuple(split.discovery), "cost_model": COST_MODEL,
                     **r["series"], "verdict": r["verdict"], "source": "datagen/validation.json#" + c.id,
                     "note": r["why"]})
        if c.id in fin_by:
            ftr = _read_trades(os.path.join(DG, "trades", f"fin_{c.id}.csv.gz"))
            st = series(c, _CTX["ctx"], ftr, *split.final)
            rows.append({"strategy_id": c.id, "family": "datagen", "group": c.family,
                         "params_fp": c.spec().fingerprint, "era": "2023-26", "protocol": "datagen v1 final",
                         "stage": "final", "dataset": "datagen_v1", "period": "%d-%d" % tuple(split.final),
                         "cost_model": COST_MODEL, **st, "verdict": fin_by[c.id]["verdict"],
                         "source": "datagen/final.json#" + c.id, "note": fin_by[c.id].get("note", "")})
    return led.append(rows)


# ══════════════════════════════════════════════════════════════════════════
# smoke, CLI
# ══════════════════════════════════════════════════════════════════════════
def smoke() -> int:
    """Plumbing: every candidate's signals and engine on one quarter of DISCOVERY; prints errors and trade counts
    only (no P&L)."""
    ctx = context()
    bad = 0
    for c in candidates():
        lo, hi = (20180101, 20180331)
        try:
            tr = execute(c, ctx, lo, hi)
            assert all(t[2] >= lo and t[2] <= hi for t in tr)
            print(f"smoke {c.id:34s} ok, {len(tr)} trades")
        except Exception as exc:
            bad += 1
            import traceback
            traceback.print_exc()
            print(f"smoke {c.id:34s} ERROR {type(exc).__name__}: {exc}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="data-driven discovery study (prereg/datagen_v1.json)")
    ap.add_argument("--data-check", action="store_true")
    ap.add_argument("--preregister", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--discovery", action="store_true")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--proof", action="store_true")
    ap.add_argument("--trials", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        for c in candidates():
            print(f"{c.id:34s} {c.family} {c.horizon:10s} {c.describe()}")
        print(len(candidates()), "declared")
        return 0
    if a.data_check:
        data_check()
    if a.preregister:
        preregister()
    if a.smoke:
        return 1 if smoke() else 0
    if a.discovery:
        discovery()
    if a.controls:
        controls_discovery()
    if a.validate:
        validate()
    if a.final:
        final(a.trials or None)
    if a.proof:
        proof(a.trials or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
