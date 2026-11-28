"""
agents/builder_promoted.py — the automatic strategy builder's PROMOTED strategies as one PAPER agent (key `bld_promoted`).

What it trades. Every strategy the builder proved (DSR >= 0.95 at the builder's N, VALIDATION + one FINAL look, controls)
is a "promoted" record in the builder's hash-chained registry (`builder.Builder(...).state()["promoted"]`, with the frozen
candidate, its universe as resolved on the batch's frozen data `universe_names`, and the registry hash). This agent reads
them at run time and trades the ones whose rule it can execute live:

    block      gap_fade (mode fade_day)  the stock opens >= g away from its previous close: trade against the gap
               idio_gap                  the stock gaps >= g against NIFTY 50's own opening gap: trade against it
    entry      o0915                     the 09:15 call-auction price (Kite quote ohlc.open at 09:15:05)
    exit       bracket / bracket_min / time_min   (stop / target / time, on the session's minute bars)
    product    MIS stock trades (the builder's size regime: Rs 5 lakh a trade under builder/v3 "c10", Rs 1 lakh v1/v2)

Anything else (daily or option rules) is listed in the status as "forward lab only" and never traded here. On
25 Nov 2026 the nine promoted strategies were all of the supported kind: bld0003_int_hig_02 (v1, Rs 1 lakh), and the
v3 batches' bld0020 / 0021 / 0022 / 0025 / 0026 / 0030 x2 / 0046 (Rs 5 lakh), all fades of the opening gap.

The decision is the builder engine's (`stratlab.builder_engine` / `datagen_features`): the same trigger (MAX_GAP 20 %,
the previous session's bars present), the market type of the PREVIOUS session (`builder_engine.market_types` on NIFTY 50
and India VIX day closes: the minute grid's regime_m), the universe mask, the ranking (|gap| descending, then the name's
alphabetical order), at most `builder_engine.MAX_PER_DAY` (10) trades per strategy per day, no trade on a special session,
and the exit is `builder_engine.walk_session` itself on the day's Kite minute bars (stop / target / time, raw prices).

Live-only guards (not in the tested rule): a price-adjusting corporate action on the day (event_calendar) excludes the
symbol, as for the auction fade; the agent's total gross notional is capped (VALIDATED_BUILDER_PROMOTED_MAX_GROSS,
default Rs 50 lakh: nine overlapping rules of up to 10 trades each would otherwise put several crore on one morning;
strategies are served in registry order, each in its own rank order, and an entry beyond the cap is skipped and counted).

Same weakness as the auction fade: the entry price is the 09:15 auction print, which is not known while pre-open orders
can be placed (not executable as tested). PAPER only; candidates (agents.strategy_agents.WHY_NOT_PROVEN).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from typing import Optional

import numpy as np
from loguru import logger

import validated_data as VD
from agents.validated_base import ValidatedAgent, PlannedTrade, ValidatedPosition, fingerprint, paper_fill_at, serialized
from config import settings
from ist_clock import SPECIAL_SESSIONS, is_trading_day
from stratlab import builder_engine as BE
from stratlab import builder_grammar as G
from stratlab import datagen_features as F

ENTRY_T = dtime(9, 15, 5)
SUPPORTED_BLOCKS = ("gap_fade", "idio_gap")
SUPPORTED_EXITS = ("bracket", "bracket_min", "time_min", "time", "signal", "signal_min") + tuple(G.TRAIL_KINDS) + \
    tuple(G.ATR_KINDS)   # time = the open of the 14:30 / 15:15 bar; signal = the VWAP cross; v4 / v5 trails
NIFTY_KEY = "_NIFTY50_min"
TAG = "VAL-BLD"

LAB_FINGERPRINT = fingerprint(
    BE.walk_session, BE.atr_walk, BE.atr_fraction, BE.market_types, BE.cost_kind, BE.trade_notional, F.r_gap_path, F.r_idio_gap,
    consts={"MAX_PER_DAY": BE.MAX_PER_DAY, "MAX_GAP": F.MAX_GAP, "SQUARE_MIN": BE.SQUARE_MIN,
            "ENTRY_MIN": sorted(G.ENTRY_MIN.items()), "SIZE_TABLES": repr(BE.SIZE_TABLES),
            "MT": (BE.SMA_N, BE.SLOPE_N, BE.VIX_N, BE.VIX_MIN, BE.VIX_Q), "MARKET_TYPES": list(G.MARKET_TYPES)})


# ── pure decision functions (the parity target) ───────────────────────────────
def is_swing(c: G.Candidate) -> bool:
    """A daily-grid STOCK rule of the swing or futures agent (decided on the evening's close, entered at the next
    open): swing longs in delivery (CNC); swing shorts and every futures-agent stock trade at the stock's own price
    with futures costs, exactly as the builder engine books them (builder_engine.cost_kind)."""
    return c.agent in ("swing", "futures") and G.BLOCK[c.block].grid == "daily" and c.instrument == "STOCKS"


def supported(c: G.Candidate) -> tuple[bool, str]:
    """Can this promoted candidate be traded live by this agent? (ok, reason)."""
    if is_swing(c):
        # entries: the builder forward run's pending_daily (the engine's own filters, F&O names only for shorts);
        # exits: the engine's own trade (any exit kind). PAPER books a short at the stock's price with futures costs
        # (the engine's booking); a LIVE short needs the futures contract order path (not built: LIVE is refused).
        return True, ""
    if c.block not in SUPPORTED_BLOCKS:
        return False, f"block {c.block}"
    if c.block == "gap_fade" and dict(c.params).get("mode") != "fade_day":
        return False, "gap_fade mode"
    if c.entry != "o0915":
        return False, f"entry {c.entry}"
    if not c.exit or c.exit[0] not in SUPPORTED_EXITS:
        return False, f"exit {c.exit[0] if c.exit else None}"
    if c.product != "mis" or c.instrument != "STOCKS":
        return False, f"{c.instrument}/{c.product}"
    return True, ""


def market_type_by_day(days, nifty_close, vix_close) -> dict[int, str]:
    """{session: market type name} with builder_engine.market_types (session t uses closes <= t only)."""
    codes = BE.market_types(np.asarray(nifty_close, float), np.asarray(vix_close, float))
    return {int(d): (G.MARKET_TYPES[k] if k >= 0 else "") for d, k in zip(days, codes.tolist())}


def regime_for(day: int, labels: dict[int, str]) -> str:
    """The market type the builder uses for a minute-grid entry on `day`: the label of the last daily session BEFORE
    it (builder_engine.Engine.regime_m), "" when unknown."""
    prev = [d for d in labels if d < int(day)]
    return labels[max(prev)] if prev else ""


def label_day_for(day: int, labels: dict[int, str]) -> int:
    """The session whose label regime_for uses for `day` (0 when none): must be the previous trading day."""
    prev = [d for d in labels if d < int(day)]
    return max(prev) if prev else 0


def _sgn(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def plan_rule(c: G.Candidate, gaps: dict, nifty_gap: Optional[float], names, excluded=(),
              max_per_day: int = BE.MAX_PER_DAY) -> list[tuple[str, int, float]]:
    """One strategy's entries for the day: [(symbol, side, |gap|)] in the builder's order. gaps = {symbol: gap} of
    the symbols with an auction print and a previous session; names = the strategy's universe; excluded = symbols the
    live-only guards drop (they do not use a slot: the engine drops dirty rows before counting)."""
    p = dict(c.params)
    g = float(p["g"])
    trig = []
    for sym in names:
        if sym in excluded or sym not in gaps:
            continue
        x = gaps[sym]
        if x is None or not np.isfinite(x) or abs(x) < g - 1e-12 or abs(x) > F.MAX_GAP:
            continue
        if c.block == "idio_gap":
            if nifty_gap is None or not np.isfinite(nifty_gap) or _sgn(x) != -_sgn(nifty_gap):
                continue
        trig.append((sym, -_sgn(x), abs(x)))
    trig.sort(key=lambda r: (-r[2], r[0]))                 # engine: rank desc, then the name's sorted order
    return trig[:max_per_day]


def walk_of(c: G.Candidate) -> tuple:
    """The builder's walk tuple of a candidate's exit (builder_engine.Engine._plan_minute). A "time" exit (the engine
    books the open of the first bar at / after 14:30 or 15:15, the panel's o1430 / o1515) is the time_min walk that
    ends at that minute."""
    m0 = G.ENTRY_MIN[c.entry]
    if c.exit[0] == "time":
        return (m0, "time_min", G.ENTRY_MIN[c.exit[1]] - m0)
    return (m0, c.exit[0]) + tuple(c.exit[1:])


def exit_deadline_min(c: G.Candidate) -> int:
    """Minutes after midnight at which the rule's own exit is due at the latest (walk_session's deadline)."""
    m0, kind = G.ENTRY_MIN[c.entry], c.exit[0]
    if kind == "time_min":
        d = m0 + int(c.exit[1])
    elif kind == "time":
        d = G.ENTRY_MIN[c.exit[1]]
    elif kind == "signal_min":
        d = m0 + int(c.exit[2])
    elif kind == "bracket_min":
        d = m0 + int(c.exit[3])
    elif kind in ("trail_min", "trail_be_min", "trail_atr_min", "trail_be_atr_min"):
        d = m0 + int(c.exit[2])
    else:
        d = BE.SQUARE_MIN
    return min(d, BE.SQUARE_MIN)


def trade_qty(c: G.Candidate, side: int, price: float) -> int:
    """builder_engine.build_trades: max(1, notional of the size regime // entry price)."""
    return max(1, int(BE.trade_notional(c, BE.cost_kind(c, side)) // float(price)))


# ── data (module level: tests patch these) ────────────────────────────────────
def promoted_rules() -> list[dict]:
    """The builder registry's promoted strategies: [{id, cand, universe_names, hash, ok, why}]."""
    try:
        import builder
        st = builder.Builder(builder.default_root()).state()
    except Exception as exc:
        logger.warning("[bld_promoted] builder registry unavailable: {}", exc)
        return []
    out = []
    for p in st.get("promoted") or []:
        try:
            c = G.Candidate.from_json(p["cand"])
        except Exception as exc:
            out.append({"id": p.get("id"), "ok": False, "why": f"bad candidate: {exc}"})
            continue
        ok, why = supported(c)
        out.append({"id": p["id"], "cand": p["cand"], "universe_names": list(p.get("universe_names") or []),
                    "universe": c.stock_universe, "hash": p.get("hash"), "ok": ok, "why": why})
    return out


def _server_source():
    from stratlab.builder_server_panels import ServerSource
    return ServerSource()


def universe_names(rule: dict) -> list[str]:
    """The strategy's frozen universe names (promotion record), else the data layer's lists for its universe."""
    if rule.get("universe_names"):
        return list(rule["universe_names"])
    try:
        ul = _server_source().universe_lists()
    except Exception as exc:
        logger.warning("[bld_promoted] universe lists unavailable: {}", exc)
        return list(VD.UNIVERSE)
    u = rule.get("universe") or "NIFTY100"
    return list(ul["base"]) + (list(ul["extra"]) if u == "NIFTY200" else [])


def prev_closes(names: list[str], today: date) -> dict[str, float]:
    """Close of each symbol's last stored session with >= MIN_SESSION_BARS bars, when that session is the previous
    trading day (the engine's pc + prev_ok), from the builder data layer's minute store."""
    from stratlab import data as D
    src = _server_source()
    st = src.store()
    want = int(_prev_td(today).strftime("%Y%m%d"))
    lo = int((today - timedelta(days=20)).strftime("%Y%m%d"))
    out = {}
    for s in names:
        try:
            z = st.raw(s, upto=want, lo=lo)
        except Exception:
            continue
        if not len(z["date"]):
            continue
        d = np.asarray(z["date"], np.int64)
        k = d == want
        if int(k.sum()) < D.MIN_SESSION_BARS:
            continue
        out[s] = float(np.asarray(z["c"], float)[k][-1])
    return out


def nifty_prev_close(today: date) -> Optional[float]:
    return prev_closes([NIFTY_KEY], today).get(NIFTY_KEY)


def market_labels(today: date) -> dict[int, str]:
    """Market types of the daily sessions before today (NIFTY 50 / India VIX day closes of the data layer)."""
    import os
    from stratlab.builder_server_panels import _bars_json
    src = _server_source()
    upto = int(_prev_td(today).strftime("%Y%m%d"))
    nf = _bars_json(os.path.join(src.nse, "bars", "_NIFTY50_day.json"), upto)
    vx = _bars_json(os.path.join(src.nse, "bars", "_VIX_day.json"), upto)
    if nf is None or vx is None:
        return {}
    days = np.asarray(nf["date"], np.int64)
    vmap = dict(zip(np.asarray(vx["date"], np.int64).tolist(), np.asarray(vx["c"], float).tolist()))
    vix = np.array([vmap.get(int(d), np.nan) for d in days], float)
    return market_type_by_day(days, nf["c"], vix)


def day_minutes(symbol: str, day: date) -> Optional[dict]:
    """The day's 1-minute bars of a symbol from Kite as arrays {tt (minutes after midnight), o, h, l, c, v}."""
    from kite_client import kite_client
    tok = kite_client.get_instrument_tokens([symbol], "NSE").get(symbol)
    if not tok:
        return None
    rows = kite_client.historical_data(int(tok), datetime.combine(day, dtime(9, 15)),
                                       datetime.combine(day, dtime(15, 30)), "minute") or []
    tt, o, h, l, c, v = [], [], [], [], [], []
    for r in rows:
        ts = r["date"] if not isinstance(r["date"], str) else datetime.fromisoformat(r["date"])
        if ts.date() != day:
            continue
        tt.append(ts.hour * 60 + ts.minute)
        o.append(float(r["open"]))
        h.append(float(r["high"]))
        l.append(float(r["low"]))
        c.append(float(r["close"]))
        v.append(float(r.get("volume") or 0))
    if not tt:
        return None
    return {"tt": np.array(tt, np.int64), "o": np.array(o), "h": np.array(h), "l": np.array(l), "c": np.array(c),
            "v": np.array(v)}


def prev_sessions_hlc(symbol: str, day: date, n: int = BE.ATR_N + 1) -> list:
    """The (high, low, close) of the symbol's last n sessions BEFORE `day` from Kite 1-minute bars, oldest first
    (special sessions skipped) - the builder engine's ATR input (builder/v5)."""
    from kite_client import kite_client
    tok = kite_client.get_instrument_tokens([symbol], "NSE").get(symbol)
    if not tok:
        return []
    start = day - timedelta(days=45)
    rows = kite_client.historical_data(int(tok), datetime.combine(start, dtime(9, 15)),
                                       datetime.combine(day - timedelta(days=1), dtime(15, 30)), "minute") or []
    by: dict = {}
    for r in rows:
        ts = r["date"] if not isinstance(r["date"], str) else datetime.fromisoformat(r["date"])
        d = ts.date()
        if d >= day or d in SPECIAL_SESSIONS:
            continue
        by.setdefault(d, []).append((float(r["high"]), float(r["low"]), float(r["close"])))
    out = []
    for d in sorted(by)[-n:]:
        bars = by[d]
        out.append((max(b[0] for b in bars), min(b[1] for b in bars), bars[-1][2]))
    return out


def forward_plan(day: date) -> Optional[dict]:
    """The builder forward run of `day` (07:05): {last_daily, pending {id: [[symbol, side, rank]]}, trades {id: [...]}}
    or None when it has not run."""
    import gzip
    import json
    try:
        import builder
        f = builder.default_root() / "forward" / "work" / f"forward_{day.strftime('%Y%m%d')}.json.gz"
        if not f.exists():
            return None
        with gzip.open(f, "rt") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("[bld_promoted] forward plan unreadable: {}", exc)
        return None


def fair_order(rules: list, plan_of) -> list:
    """The day's entries of every strategy, taken in turns: each strategy's 1st entry, then each one's 2nd, ... (each
    strategy keeps its own order). A gross cap that binds is then shared across the strategies instead of going to
    the first ones in the registry. [(rule, cand, symbol, side, rank)]."""
    plans = [(r, c, list(plan_of(r, c))) for r, c in rules]
    out = []
    for i in range(max((len(p) for _r, _c, p in plans), default=0)):
        for r, c, p in plans:
            if i < len(p):
                out.append((r, c) + tuple(p[i]))
    return out


def _prev_td(d: date) -> date:
    x = d - timedelta(days=1)
    while not is_trading_day(x):
        x -= timedelta(days=1)
    return x


# ── the agent ────────────────────────────────────────────────────────────────
@dataclass
class _Inputs:
    gaps: dict
    nifty_gap: Optional[float]
    opens: dict


class BuilderPromotedAgent(ValidatedAgent):
    name = "bld_promoted"
    product = "MIS"
    SETTING = "builder_promoted"             # settings.validated_builder_promoted_enabled (default off)
    COST_KIND = None                         # ledger trade_cost NSE MIS = equity_mis, the builder's cost kind
    LAB_ID = "builder/promoted"
    VERSION = "builder-promoted/v1"
    LAB_FINGERPRINT = LAB_FINGERPRINT
    LAB_EXPECTATION = {"source": "the builder registry's promoted records (VALIDATION + FINAL references per strategy)",
                       "live_only_guards": {"corporate_action": "a price-adjusting corporate action on the day "
                                                                "excludes the symbol",
                                            "gross_cap": "VALIDATED_BUILDER_PROMOTED_MAX_GROSS over all strategies"}}

    def __init__(self) -> None:
        super().__init__()
        self.last_rules: list[dict] = []

    def daily_loss_limit(self) -> float:
        return float(getattr(settings, "validated_builder_promoted_daily_loss", 0.0) or 0.0)

    def dd_limit(self) -> float:
        return float(getattr(settings, "validated_builder_promoted_dd_kill", 0.0) or 0.0)

    def _ledger_extra(self, pos: ValidatedPosition, exit_px: float) -> dict:
        return {"rule": pos.meta.get("rule"), "gap": pos.meta.get("gap"), "exit_reason": pos.meta.get("exit_reason"),
                "deadline_min": pos.meta.get("deadline_min")}

    def overnight_holdings(self) -> dict[tuple[str, str], int]:
        """The 15:15 square-off must leave an open position to its own exit (a bracket rule exits at the 15:15 bar
        open, read at 15:16); listed like an overnight holding until the exit books it."""
        out: dict[tuple[str, str], int] = {}
        for p in self._positions.values():
            k = (p.symbol, p.product)
            out[k] = out.get(k, 0) + p.qty * (1 if p.side == "BUY" else -1)
        return out

    # ── 09:15:05 entry ──
    @serialized
    async def enter(self, now: Optional[datetime] = None) -> dict:
        now = now or self._now()
        today = now.date()
        out = {"day": today.isoformat(), "entered": [], "skipped": {}}
        if not is_trading_day(today) or today in SPECIAL_SESSIONS:
            return out
        if self.done_today("entry"):
            return out
        block = self.entry_block_reason()
        if block:
            self._skip("*", block)
            out["reason"] = block
            return out
        rules = await asyncio.to_thread(promoted_rules)
        self.last_rules = [{k: r.get(k) for k in ("id", "ok", "why", "universe")} for r in rules]
        live = [r for r in rules if r.get("ok")]
        swing = [r for r in live if is_swing(G.Candidate.from_json(r["cand"]))]
        live = [r for r in live if r not in swing]
        if swing:
            out["swing"] = await self._swing(today, swing)
        if not live:
            self.mark_done("entry")
            out["reason"] = "no supported promoted strategy"
            return out
        labels = await asyncio.to_thread(market_labels, today)
        want = int(_prev_td(today).strftime("%Y%m%d"))
        have = label_day_for(int(today.strftime("%Y%m%d")), labels)
        if have != want:
            # the market type must be the PREVIOUS session's (the NSE archive and the minute store refresh apart)
            self.mark_done("entry")
            out["reason"] = f"data unavailable: market label for {want} missing (newest {have or 'none'})"
            self._skip("*", "market_label_stale")
            self._log("entry", "*", "info", out["reason"])
            return out
        mtype = regime_for(int(today.strftime("%Y%m%d")), labels)
        live = [r for r in live if G.Candidate.from_json(r["cand"]).mtype == mtype]
        out["market_type"] = mtype
        if not live:
            self.mark_done("entry")
            out["reason"] = f"no promoted strategy for market type {mtype or 'unknown'}"
            self._log("entry", "*", "info", out["reason"])
            return out
        names_of = {r["id"]: universe_names(r) for r in live}
        allnames = sorted({s for v in names_of.values() for s in v})
        pcs = await asyncio.to_thread(prev_closes, allnames + [NIFTY_KEY], today)
        q = await asyncio.to_thread(VD.quotes, [f"NSE:{s}" for s in allnames] + ["NSE:NIFTY 50"])
        opens = {s: float((q.get(s) or {}).get("open") or 0) for s in allnames}
        gaps = {s: opens[s] / pcs[s] - 1.0 for s in allnames if opens.get(s, 0) > 0 and pcs.get(s, 0) > 0}
        n_open = float((q.get("NIFTY 50") or {}).get("open") or 0)
        n_pc = pcs.get(NIFTY_KEY) or 0
        nifty_gap = n_open / n_pc - 1.0 if n_open > 0 and n_pc > 0 else None
        excluded = {s for s in allnames if VD.price_adjusting_actions(s, today)}
        cap = float(getattr(settings, "validated_builder_promoted_max_gross", 5_000_000.0)) * self._scale()
        gross = sum(p.entry_px * p.qty for p in self._positions.values())
        for r, c, sym, side, rank in fair_order(
                [(r, G.Candidate.from_json(r["cand"])) for r in live],
                lambda r, c: plan_rule(c, gaps, nifty_gap, names_of[r["id"]], excluded)):
                px = opens[sym]
                qty = max(1, int(trade_qty(c, side, px) * self._scale()))
                if gross + qty * px > cap:
                    out["skipped"][f"{r['id']}:{sym}"] = "gross_cap"
                    self._skip(sym, "gross_cap", log=False)
                    continue
                side_s = "BUY" if side > 0 else "SELL"
                try:
                    from sebi_compliance import sebi_compliance
                    ok, _algo, why = sebi_compliance.pre_order_check(
                        strategy=self.name, symbol=sym, exchange="NSE", transaction_type=side_s, quantity=qty,
                        order_type="MARKET", price_at_signal=px, signal_source=f"validated:{r['id']}",
                        regime=mtype or "n/a")
                    if not ok:
                        out["skipped"][f"{r['id']}:{sym}"] = f"sebi: {why}"
                        self._skip(sym, f"sebi: {why}", log=False)
                        continue
                    oid, fill = paper_fill_at(sym, "NSE", side_s, qty, "MIS", px, TAG, "AUCTION")
                    sebi_compliance.record_order_id(self.name, sym, oid)
                except Exception as exc:
                    self._error(f"entry {sym}", exc)
                    continue
                pt = PlannedTrade(symbol=sym, side=side, qty=qty, entry_ref=px, entry_t="09:15",
                                  exit_t=c.exit[0], key=r["id"])
                self._open(pt, "MIS", "NSE", fill, oid,
                           meta={"rule": r["id"], "gap": round(gaps[sym], 6), "rank": round(rank, 6),
                                 "cand": r["cand"], "deadline_min": exit_deadline_min(c), "day": today.isoformat()})
                gross += qty * px
                out["entered"].append(f"{r['id']}:{sym}")
        self.mark_done("entry")
        self._save_book()
        logger.info("[bld_promoted] {} entries ({} market): {} strategies, {} skipped", len(out["entered"]),
                    mtype, len(live), len(out["skipped"]))
        return out

    # ── exits: each position at its rule's own exit (walk_session on the day's minute bars) ──
    @serialized
    async def exit_due(self, now: Optional[datetime] = None) -> list[str]:
        now = now or self._now()
        mins = now.hour * 60 + now.minute
        done = []
        for pos in list(self._positions.values()):
            if not self._still_open(pos) or pos.meta.get("kind") == "swing":
                continue
            day = date.fromisoformat(pos.meta.get("day") or now.date().isoformat())
            if day == now.date() and mins <= int(pos.meta.get("deadline_min", BE.SQUARE_MIN)):
                continue
            c = G.Candidate.from_json(pos.meta["cand"])
            bars = await asyncio.to_thread(day_minutes, pos.symbol, day)
            res = None
            walk = walk_of(c)
            if walk[1] in G.ATR_KINDS:                # builder/v5: the distance from ATR(14) of the earlier sessions
                prev = await asyncio.to_thread(prev_sessions_hlc, pos.symbol, day)
                walk = BE.atr_walk(walk, prev)
                if walk is None:
                    self._deviation("no_atr_history", pos.symbol, "fewer than 15 earlier sessions: exit at the LTP")
            if bars is not None and walk is not None:
                res = BE.walk_session(bars["tt"], bars["o"], bars["h"], bars["l"], bars["c"], bars["v"], pos.sign,
                                      walk, int(day.strftime("%Y%m%d")))
            if res is None:
                q = await asyncio.to_thread(VD.quotes, [f"NSE:{pos.symbol}"])
                px = float((q.get(pos.symbol) or {}).get("ltp") or 0) or pos.entry_px
                self._deviation("no_minute_bars", pos.symbol, "exit at the LTP")
                reason, kind = "no_bars", "RECONCILE"
            else:
                _pin, px, _hm, reason = res
                kind = "CLOSE"
            pos.meta["exit_reason"] = reason
            back = "SELL" if pos.side == "BUY" else "BUY"
            _oid, fill = paper_fill_at(pos.symbol, "NSE", back, pos.qty, "MIS", round(float(px), 2), TAG, kind)
            self._close(pos, fill, f"{pos.meta.get('rule')}: {reason}")
            done.append(pos.symbol)
        if done:
            self._save_book()
        return done

    def _close(self, pos, exit_px, note, kind="normal"):
        """The ledger cost of a position is its own booking: a swing / futures-agent short at FUTURES costs (the
        builder engine's cost_kind), everything else by its product (MIS / CNC)."""
        ck = (pos.meta or {}).get("cost_kind")
        if ck == "future":
            old = self.COST_KIND
            self.COST_KIND = "future"
            try:
                return super()._close(pos, exit_px, note, kind)
            finally:
                self.COST_KIND = old
        return super()._close(pos, exit_px, note, kind)

    # ── swing rules (daily grid, CNC longs): exits mirror the engine's own trade, entries its pending_daily ──
    async def _swing(self, today: date, rules: list) -> dict:
        res = {"entered": [], "exited": [], "skipped": {}}
        plan = await asyncio.to_thread(forward_plan, today)
        want = int(_prev_td(today).strftime("%Y%m%d"))
        if plan is None or plan.get("last_daily") != want:
            res["reason"] = (f"data unavailable: forward plan of {today} "
                             + ("missing" if plan is None else f"ends {plan.get('last_daily')}, not {want}"))
            self._skip("*", "swing_plan_unavailable")
            self._log("entry", "*", "info", res["reason"])
            return res
        trades = plan.get("trades") or {}
        # 1) exits: each open swing position follows the engine's trade of the same rule, symbol and entry day
        for pos in list(self._positions.values()):
            if not self._still_open(pos) or pos.meta.get("kind") != "swing":
                continue
            if pos.meta.get("day") == today.isoformat():
                continue
            ed = int(date.fromisoformat(pos.meta["day"]).strftime("%Y%m%d"))
            tr = [t for t in trades.get(pos.meta.get("rule"), []) if t[1] == pos.symbol and int(t[2]) == ed]
            if tr and tr[0][-1] == "window_end":
                continue                                     # the engine still holds it
            if tr:
                px, reason, kind = float(tr[0][8]), f"{tr[0][-1]} on {tr[0][6]}", "CLOSE"
            else:
                q = await asyncio.to_thread(VD.quotes, [f"NSE:{pos.symbol}"])
                px = float((q.get(pos.symbol) or {}).get("ltp") or 0) or pos.entry_px
                reason, kind = "not in the engine's trades", "RECONCILE"
                self._deviation("swing_not_in_engine", pos.symbol, "exit at the LTP")
            pos.meta["exit_reason"] = reason
            back = "SELL" if pos.side == "BUY" else "BUY"
            _oid, fill = paper_fill_at(pos.symbol, "NSE", back, pos.qty, "CNC", round(px, 2), TAG, kind)
            self._close(pos, fill, f"{pos.meta.get('rule')}: {reason}")
            res["exited"].append(pos.symbol)
        # 2) entries at today's open: the engine's pending entries (longs only: supported())
        todo = [(r, p) for r in rules for p in (plan.get("pending") or {}).get(r["id"]) or []
                if isinstance(p, list)]
        if todo:
            syms = sorted({p[0] for _r, p in todo})
            q = await asyncio.to_thread(VD.quotes, [f"NSE:{x}" for x in syms])
            cap = float(getattr(settings, "validated_builder_promoted_max_gross", 5_000_000.0)) * self._scale()
            gross = sum(p.entry_px * p.qty for p in self._positions.values() if self._still_open(p))
            from sebi_compliance import sebi_compliance
            for r, (sym, side, rank) in todo:
                key = f"{r['id']}:{sym}"
                px = float((q.get(sym) or {}).get("open") or 0)
                if px <= 0:
                    res["skipped"][key] = "no open price"
                    continue
                side = 1 if int(side) > 0 else -1
                if VD.price_adjusting_actions(sym, today):
                    res["skipped"][key] = "corporate action"
                    continue
                if any(p.symbol == sym and p.meta.get("rule") == r["id"] for p in self._positions.values()
                       if self._still_open(p)):
                    res["skipped"][key] = "already held"
                    continue
                c = G.Candidate.from_json(r["cand"])
                qty = max(1, int(trade_qty(c, side, px) * self._scale()))
                ck = BE.cost_kind(c, side)                  # equity_cnc (swing long) or future (shorts, futures agent)
                side_s = "BUY" if side > 0 else "SELL"
                if gross + qty * px > cap:
                    res["skipped"][key] = "gross_cap"
                    self._skip(sym, "gross_cap", log=False)
                    continue
                try:
                    ok, _algo, why = sebi_compliance.pre_order_check(
                        strategy=self.name, symbol=sym, exchange="NSE", transaction_type=side_s, quantity=qty,
                        order_type="MARKET", price_at_signal=px, signal_source=f"validated:{r['id']}", regime="n/a")
                    if not ok:
                        res["skipped"][key] = f"sebi: {why}"
                        continue
                    oid, fill = paper_fill_at(sym, "NSE", side_s, qty, "CNC", px, TAG, "AUCTION")
                    sebi_compliance.record_order_id(self.name, sym, oid)
                except Exception as exc:
                    self._error(f"swing entry {sym}", exc)
                    continue
                pt = PlannedTrade(symbol=sym, side=side, qty=qty, entry_ref=px, entry_t="09:15", exit_t=c.exit[0],
                                  key=r["id"])
                self._open(pt, "CNC", "NSE", fill, oid, overnight=True,
                           meta={"rule": r["id"], "kind": "swing", "rank": round(float(rank), 6), "cand": r["cand"],
                                 "day": today.isoformat(), "cost_kind": ck,
                                 **({"paper_only": "short at the stock price with futures costs (engine booking)"}
                                    if ck == "future" else {})})
                gross += qty * px
                res["entered"].append(key)
        self._save_book()
        logger.info("[bld_promoted] swing: {} entered, {} exited, {} skipped", len(res["entered"]),
                    len(res["exited"]), len(res["skipped"]))
        return res

    async def catch_up(self, now: Optional[datetime] = None) -> dict:
        now = now or self._now()
        self.auto_start(now.date())
        return {"exited": await self.exit_due(now)}

    # ── status ──
    def rule_stats(self) -> dict:
        by: dict = {}
        for r in self.ledger_rows():
            k = r.get("rule") or "?"
            s = by.setdefault(k, {"n": 0, "net": 0.0, "wins": 0})
            s["n"] += 1
            s["net"] = round(s["net"] + float(r.get("net", 0)), 2)
            s["wins"] += float(r.get("net", 0)) > 0
        return by

    def live_readiness(self) -> dict:
        st = self.live_stats()
        ok = st.get("n", 0) >= 100 and st.get("net", 0) > 0 and (st.get("pf") or 0) >= 1.15
        return {"criterion": ">= 100 trades, net > 0 and PF >= 1.15 (all promoted strategies together); each "
                             "strategy is also judged by the builder's own forward family",
                "ready": bool(ok), "never_switches_to_live": True}

    def get_status(self) -> dict:
        st = super().get_status()
        st["rules"] = self.last_rules
        st["rule_stats"] = self.rule_stats()
        st["max_gross"] = float(getattr(settings, "validated_builder_promoted_max_gross", 5_000_000.0)) * self._scale()
        return st
