"""
agents/validated_attention.py — SWING-ATTN-ON-3X (stratlab/attention.py) as a PAPER agent.

The tested rule, unchanged (rule function and constants imported from stratlab.attention):
  15:20  cumulative volume 09:15..15:20 > 3.0 × the mean FULL-DAY volume of the symbol's previous 20 valid sessions,
         and the 15:20 price > the previous valid session's last close
  15:21  BUY the 10 largest volume ratios (ties by symbol), qty = floor(₹1 L / entry), one night
  09:15  next session: SELL at the opening call-auction print (a pre-open order — genuinely executable)
  exclusions: a special / short session on t, t+1 or in between; a price-adjusting corporate action on t or t+1
  no stop, no target, no other filter.

Carrier and costs. The lab traded ₹1 L "stock futures" priced off the cash price (no lot size, no basis) and costed
them as futures. The app has no stock futures (real lots are ₹5–15 L), so paper holds the position as a CNC delivery
long (the 15:15 MIS square-off never touches it) and books the ledger at the lab's FUTURES costs
(COST_KIND = "future"); every row also carries the CNC delivery cost (`cost_cnc`), and the status shows both.
At CNC costs the lab result is negative on FINAL (−7.5 bps/trade) — see live_readiness().

Live inputs ↔ lab bars: cum_vol_1520 = Kite quote "volume" at 15:21 (≈ the sum of the minute-bar volumes to 15:20),
close_1520 = the quote LTP, entry = the 15:21 ask (lab: 15:21 bar open), exit = the next session's official open
(lab: 09:15 bar open). prev_close / prev20 come from the session store (validated_data), built with the lab's
cleaning and its 200-bar rule. The 15:40 parity_audit recomputes the day's decision from the day's minute bars.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from typing import Callable, Iterable, Optional

from loguru import logger

import validated_data as VD
from agents.validated_base import (ValidatedAgent, PlannedTrade, ValidatedPosition, fingerprint, paper_marketable,
                                   paper_fill_at)
from config import settings
from costs import round_trip_cost
from ist_clock import now_ist, next_trading_day, is_trading_day
from stratlab import attention as T

ENTRY_T = dtime(15, 21)
ENTRY_DEADLINE = dtime(15, 24)          # the lab requires the entry bar to start by 15:24
EXIT_T = dtime(9, 15, 5)
EXIT_DEADLINE = dtime(9, 29, 50)        # the lab falls back to the first bar before 09:30
QUOTE_RETRY_GAPS_S = (10, 10)           # entry: whole-batch quote retries at +10 s and +20 s
EXIT_RETRY_S = 10
CARRIER = "CNC (paper proxy for a ₹1 L stock future; app has no stock futures, real lots ≥ ₹5 L)"

LAB_FINGERPRINT = fingerprint(T.attention_rule, consts={
    "VOL_MULT": T.VOL_MULT, "LOOKBACK": T.LOOKBACK, "MAX_PER_DAY": T.MAX_PER_DAY, "NOTIONAL": T.NOTIONAL,
    "DECISION_T": T.DECISION_T, "DECISION_EARLIEST": T.DECISION_EARLIEST, "ENTRY_LATEST": T.ENTRY_LATEST,
    "OPEN_T": T.OPEN_T, "EXIT_LATEST": T.EXIT_LATEST, "FILL_MODEL": T.FILL_MODEL})


# ── pure decision functions (the parity target) ───────────────────────────────
@dataclass(frozen=True)
class AttentionInput:
    symbol: str
    cum_vol_1520: float
    close_1520: float
    prev_close: float
    prev20_full_vols: tuple
    entry_px: float


def attention_exclusions(day: date, next_session: date, special: set, ca: Callable[[str, date], list],
                         symbols: Iterable[str]) -> dict[str, str]:
    """The lab's exclusions known at 15:20: every symbol when day t or the next session is special, or a special
    session lies strictly between them; a symbol with a price-adjusting corporate action on t or on t+1."""
    symbols = list(symbols)
    if day in special or next_session in special or any(day < s < next_session for s in special):
        return {s: "special_session" for s in symbols}
    out = {}
    for s in symbols:
        if ca(s, day) or ca(s, next_session):
            out[s] = "corporate_action"
    return out


def plan_attention(inputs: Iterable[AttentionInput], excluded: dict[str, str], notional: float = T.NOTIONAL,
                   max_per_day: int = T.MAX_PER_DAY) -> tuple[list[PlannedTrade], dict[str, str]]:
    """T.attention_rule per symbol, drop exclusions, rank (-ratio, symbol), first `max_per_day`."""
    cands, skips = [], {}
    for x in inputs:
        ratio = T.attention_rule(x.cum_vol_1520, x.prev20_full_vols, x.close_1520, x.prev_close, T.VOL_MULT)
        if ratio is None:
            skips[x.symbol] = "no_signal"
            continue
        if x.symbol in excluded:
            skips[x.symbol] = excluded[x.symbol]
            continue
        if not x.entry_px > 0:
            skips[x.symbol] = "no_data:quote"
            continue
        cands.append((ratio, x))
    cands.sort(key=lambda c: (-c[0], c[1].symbol))
    planned = []
    for ratio, x in cands[:max_per_day]:
        planned.append(PlannedTrade(symbol=x.symbol, side=1, qty=max(1, int(notional // x.entry_px)),
                                    entry_ref=float(x.entry_px), entry_t="15:21", exit_t="09:15", key=float(ratio)))
    for ratio, x in cands[max_per_day:]:
        skips[x.symbol] = "rank"
    return planned, skips


def scale_to_cap(planned: list[PlannedTrade], cap: float) -> tuple[list[PlannedTrade], Optional[float]]:
    """A capital cap never drops names: every quantity is scaled down proportionally (at least 1)."""
    gross = sum(p.qty * p.entry_ref for p in planned)
    if cap <= 0 or gross <= cap:
        return planned, None
    f = cap / gross
    return [PlannedTrade(p.symbol, p.side, max(1, int(p.qty * f)), p.entry_ref, p.entry_t, p.exit_t, p.key)
            for p in planned], f


def lab_ca(symbol: str, d: date) -> list[str]:
    """The lab's own corporate-action list (stratlab.ltr.CA_GAPS) as a ca() callable (parity tests)."""
    from stratlab.ltr import CA_GAPS
    k = d.year * 10000 + d.month * 100 + d.day
    return [CA_GAPS[(symbol, k)]] if (symbol, k) in CA_GAPS else []


# ── the agent ────────────────────────────────────────────────────────────────
class AttentionOvernightAgent(ValidatedAgent):
    name = "attn_on"
    product = "CNC"
    SETTING = "attention"
    COST_KIND = "future"                 # the lab's cost assumption (round_trip_cost(..., "future"))
    LAB_ID = "SWING-ATTN-ON-3X"
    VERSION = f"SWING-ATTN-ON-3X/{T.FILL_MODEL}/v1"
    LAB_FINGERPRINT = LAB_FINGERPRINT
    # r2_swing_out/summary.txt (2 bps/side slippage; futures costs unless stated)
    LAB_EXPECTATION = {
        "DISC": {"n": 580, "net_bps": 27.26, "pf": 2.13, "win_pct": 55.3},
        "FINAL": {"n": 194, "net_bps": 5.81, "pf": 1.18, "win_pct": 50.5},
        "DISC_cnc_costs": {"net_bps": 13.88, "pf": 1.46, "screen": "FAIL"},
        "FINAL_cnc_costs": {"net_bps": -7.54, "pf": 0.80},
        "FINAL_stress_5bps": {"net_bps": -0.19, "pf": 0.99},
    }

    def __init__(self) -> None:
        super().__init__()
        self.last_decision: dict = {}    # {"day", "symbols", "ratios"} of the latest 15:21 decision

    # ── ledger extras: the CNC cost next to the futures cost ──
    def _ledger_extra(self, pos: ValidatedPosition, exit_px: float) -> dict:
        return {"cost_cnc": round_trip_cost(pos.entry_px * pos.qty, float(exit_px) * pos.qty, "equity_cnc"),
                "carrier": CARRIER, "ratio": round(float(pos.meta.get("ratio", 0.0)), 3)}

    # ── 15:21 entry ──
    async def decide_and_enter(self) -> dict:
        now = self._now()
        today = now.date()
        out = {"day": today.isoformat(), "planned": [], "opened": []}
        if not is_trading_day(today):
            self._skip("*", "not_trading_day")
            return out
        if now.time().replace(tzinfo=None) >= ENTRY_DEADLINE:
            if not self.done_today("entry"):
                self._skip("*", "entry_window_missed")
                self.mark_done("entry")
            return out
        why = self.entry_block_reason()
        if why:
            self._skip("*", why)
            return out
        if self.done_today("entry"):
            self._skip("*", "already_entered")
            return out
        if not VD.store_current(today):
            self._skip("*", "no_data:store_stale")
            return out
        q = {}
        for wait in (0,) + QUOTE_RETRY_GAPS_S:
            if wait:
                await asyncio.sleep(wait)
                if self._now().time().replace(tzinfo=None) >= ENTRY_DEADLINE:
                    break
            q = await asyncio.to_thread(VD.quotes, [f"NSE:{s}" for s in VD.UNIVERSE])
            if q:
                break
        if not q:
            self._skip("*", "quote_failed")
            self.mark_done("entry")
            return out
        if self._now().time().replace(tzinfo=None) >= ENTRY_DEADLINE:
            self._skip("*", "entry_window_missed")
            self.mark_done("entry")
            return out
        inputs = []
        for sym in VD.UNIVERSE:
            qs = q.get(sym) or {}
            if not (float(qs.get("ltp") or 0) > 0 and float(qs.get("volume") or 0) > 0):
                self._skip(sym, "no_data:quote")
                continue
            pc, p20 = VD.prev_close(sym, today), VD.prev20_volumes(sym, today)
            if pc is None or p20 is None:
                self._skip(sym, "no_data:history")
                continue
            inputs.append(AttentionInput(sym, float(qs["volume"]), float(qs["ltp"]), pc, p20,
                                         float(qs.get("ask") or 0) or float(qs["ltp"])))
        nxt = next_trading_day(today)
        if not VD.ca_calendar_loaded():
            self.flags["ca_calendar_unavailable"] = today.isoformat()
            logger.warning("[attn_on] corporate-action calendar not loaded — proceeding without CA exclusions")
            self._log("entry", "*", "flag", "ca_calendar_unavailable")
        excluded = attention_exclusions(today, nxt, VD.special_sessions(), VD.price_adjusting_actions,
                                        [x.symbol for x in inputs])
        notional = T.NOTIONAL * self._scale()
        planned, skips = plan_attention(inputs, excluded, notional=notional)
        for sym, why in skips.items():
            self._skip(sym, why, log=why != "no_signal")
        cap = float(getattr(settings, "validated_attention_max_gross", 0.0) or 0.0) * self._scale()
        planned, f = scale_to_cap(planned, cap)
        if f is not None:
            self._log("entry", "*", "info", f"cap_scaled ×{f:.3f}")
            self.state.skip_counts["cap_scaled"] = self.state.skip_counts.get("cap_scaled", 0) + 1
        self.last_decision = {"day": today.isoformat(), "symbols": [p.symbol for p in planned],
                              "ratios": {p.symbol: round(p.key, 3) for p in planned}}
        out["planned"] = [(p.symbol, p.qty) for p in planned]
        from sebi_compliance import sebi_compliance
        for pt in planned:
            ok, _algo, reason = sebi_compliance.pre_order_check(
                strategy=self.name, symbol=pt.symbol, exchange="NSE", transaction_type="BUY", quantity=pt.qty,
                order_type="LIMIT", price_at_signal=pt.entry_ref, signal_source=self.LAB_ID, regime="n/a")
            if not ok:
                self._skip(pt.symbol, f"sebi: {reason}")
                continue
            try:
                oid, fill = paper_marketable(pt.symbol, "NSE", "BUY", pt.qty, "CNC", q[pt.symbol], "VAL-ATTN-IN")
            except Exception as exc:
                self._error(f"entry {pt.symbol}", exc)
                self._skip(pt.symbol, "order_failed")
                continue
            sebi_compliance.record_order_id(self.name, pt.symbol, oid)
            self._open(pt, "CNC", "NSE", fill, oid, overnight=True,
                       meta={"ratio": pt.key, "entry_day": today.isoformat(), "exit_due": nxt.isoformat()})
            out["opened"].append(pt.symbol)
        self.mark_done("entry")
        logger.info("[attn_on] {} decision: {} planned, opened {}", today, len(planned), out["opened"])
        return out

    # ── 09:15 exit ──
    def due_positions(self, today: date) -> list[ValidatedPosition]:
        return [p for p in self.positions() if p.opened[:10] < today.isoformat()]

    async def exit_at_open(self, q: Optional[dict] = None) -> list[str]:
        """Sell every overnight long at the session's official open (the call-auction print a pre-open SELL order
        fills at). Missing opens are retried every 10 s until 09:29:50; after that the LTP (late_exit)."""
        now = self._now()
        today = now.date()
        due = self.due_positions(today)
        if not due:
            return []
        if settings.trading_mode != "PAPER":
            self._skip("*", "live_mode_refused", stage="exit")
            return []
        prev_td = VD.previous_trading_day(today)
        closed = []

        def _exit(pos, px, kind, fill_kind, note):
            oid, fill = paper_fill_at(pos.symbol, pos.exchange, "SELL", pos.qty, pos.product, px,
                                      "VAL-ATTN-OUT", fill_kind)
            if kind != "normal":
                self._deviation(kind, pos.symbol, note)
            self._close(pos, fill, note, kind=kind)
            closed.append(pos.symbol)

        pending = list(due)
        q = q if q is not None else await asyncio.to_thread(VD.quotes, [f"NSE:{p.symbol}" for p in pending])
        while True:
            t = self._now().time().replace(tzinfo=None)
            late = t >= EXIT_DEADLINE
            rest = []
            for pos in pending:
                qs = (q or {}).get(pos.symbol) or {}
                missed = pos.opened[:10] < prev_td.isoformat()     # held over more than one session (downtime)
                if not late and float(qs.get("open") or 0) > 0:
                    kind = "late_exit" if missed else "normal"
                    _exit(pos, float(qs["open"]), kind, "AUCTION", "exit next-session open (09:15 auction)"
                          + (f"; missed_exit since {pos.meta.get('exit_due')}" if missed else ""))
                elif late and float(qs.get("ltp") or 0) > 0:
                    _exit(pos, float(qs["ltp"]), "late_exit", "LTP", f"late exit at LTP {t.strftime('%H:%M:%S')}")
                else:
                    rest.append(pos)
            pending = rest
            if not pending or late:
                break
            await asyncio.sleep(EXIT_RETRY_S)
            q = await asyncio.to_thread(VD.quotes, [f"NSE:{p.symbol}" for p in pending])
        for pos in pending:                       # no price at all: retried by catch_up / the next exit run
            self._skip(pos.symbol, "no_data:exit_quote", stage="exit")
        return closed

    # ── restart catch-up ──
    async def catch_up(self, now: Optional[datetime] = None) -> dict:
        now = now or self._now()
        today, t = now.date(), now.time().replace(tzinfo=None)
        out = {}
        if is_trading_day(today) and t >= EXIT_T and self.due_positions(today):
            out["exited"] = await self.exit_at_open()
        if is_trading_day(today) and ENTRY_T <= t < ENTRY_DEADLINE and not self.done_today("entry"):
            out["entry"] = await self.decide_and_enter()
        return out

    # ── 15:40 parity audit ──
    def parity_audit(self, day_rows: Optional[dict] = None) -> dict:
        """Recompute today's decision from the day's minute bars (volume 09:15..15:20, the 15:20 close, the 15:21
        open) and the same session store; log parity_mismatch where the symbol set differs from the quote-based
        decision (measures the 'quote volume ≈ minute-bar sum' assumption)."""
        today = self._now().date()
        if self.last_decision.get("day") != today.isoformat():
            return {"audited": False, "reason": "no decision today"}
        day_rows = VD.last_day_rows if day_rows is None else day_rows
        inputs = []
        for sym, rows in day_rows.items():
            feat = bar_features(rows)
            pc, p20 = VD.prev_close(sym, today), VD.prev20_volumes(sym, today)
            if feat is None or pc is None or p20 is None:
                continue
            inputs.append(AttentionInput(sym, feat["cum_vol"], feat["close_1520"], pc, p20, feat["entry_open"]))
        nxt = next_trading_day(today)
        excluded = attention_exclusions(today, nxt, VD.special_sessions(), VD.price_adjusting_actions,
                                        [x.symbol for x in inputs])
        planned, _ = plan_attention(inputs, excluded, notional=T.NOTIONAL * self._scale())
        bars_set, live_set = {p.symbol for p in planned}, set(self.last_decision.get("symbols") or [])
        res = {"audited": True, "bars_only": sorted(bars_set - live_set), "quotes_only": sorted(live_set - bars_set)}
        if res["bars_only"] or res["quotes_only"]:
            self._deviation("parity_mismatch", "*", f"bars-only {res['bars_only']} quotes-only {res['quotes_only']}")
        self.last_decision["audit"] = res
        return res

    # ── status ──
    def live_readiness(self) -> dict:
        rows = self.ledger_rows()
        cnc = [{**r, "net_cnc": float(r.get("gross", 0)) - float(r.get("cost_cnc", r.get("cost", 0)))} for r in rows]
        st = self.live_stats(cnc, net_key="net_cnc")
        ready = st.get("n", 0) >= 100 and st.get("net", 0) > 0 and (st.get("pf") or 0) >= 1.15
        return {"criterion": "≥ 100 trades, net > 0 and PF ≥ 1.15 at CNC costs (or a separately validated "
                             "stock-futures implementation with real lots)",
                "at_cnc_costs": st, "ready": bool(ready), "never_switches_to_live": True}

    def get_status(self) -> dict:
        st = super().get_status()
        rows = self.ledger_rows()
        st["live_cnc_costs"] = self.live_readiness()["at_cnc_costs"]
        st["carrier"] = CARRIER
        st["last_decision"] = self.last_decision
        st["cost_basis"] = "ledger net at FUTURES costs (lab assumption); live_cnc_costs at CNC delivery costs"
        st["open_value"] = round(sum(p.entry_px * p.qty for p in self.positions()), 2)
        st["ledger_trades"] = len(rows)
        return st

    def export_book(self) -> dict:
        return {**super().export_book(), "last_decision": dict(self.last_decision)}

    def import_book(self, d: dict, closed: set, stale: bool) -> None:
        super().import_book(d, closed, stale)
        if (d or {}).get("last_decision"):
            self.last_decision = dict(d["last_decision"])


def bar_features(rows: list[dict]) -> Optional[dict]:
    """The lab's decision/entry bars of one session from Kite minute candles (stratlab.attention.session_features):
    decision bar = last bar ≤ 15:20 (not before 15:15); entry = the next bar's open (must start by 15:24)."""
    bars = []
    for r in rows:
        ts = r["date"] if not isinstance(r["date"], str) else datetime.fromisoformat(r["date"])
        t = ts.hour * 60 + ts.minute
        if VD._SD.SESSION_OPEN <= t <= VD._SD.SESSION_LAST:
            bars.append((t, r))
    bars.sort(key=lambda b: b[0])
    j = max((i for i, (t, _) in enumerate(bars) if t <= T.DECISION_T), default=-1)
    if j < 0 or bars[j][0] < T.DECISION_EARLIEST or j + 1 >= len(bars) or bars[j + 1][0] > T.ENTRY_LATEST:
        return None
    return {"cum_vol": float(sum(float(b[1].get("volume") or 0) for b in bars[:j + 1])),
            "close_1520": float(bars[j][1]["close"]), "entry_open": float(bars[j + 1][1]["open"]),
            "entry_t": bars[j + 1][0]}
