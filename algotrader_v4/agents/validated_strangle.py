"""
agents/validated_strangle.py — VRP-STRANGLE-016-WINGED (stratlab/options_vrp.py) as a PAPER agent.

Why this variant. The lab protocol (DISCOVERY H1/H2 screen + one FINAL look, dd/vrp/final_run/summary.txt) confirmed
both VRP-STRANGLE-016-NAKED and VRP-STRANGLE-016-WINGED; the task prefers the defined-risk one, so only WINGED is
traded. VRP-WINGED-HIGHVIX failed the screen (53 trades < 60) and is not implemented.

The tested rule, unchanged (the decision is made by the lab's own objects: options_replay.ShortStrangle built by
stratlab.options_vrp.BY_ID["VRP-STRANGLE-016-WINGED"].strategy(), options_replay.next_expiry / MarketView / bs):
  10:20  on the first trading day after each weekly NIFTY expiry (skip the week when fewer than 3 trading days remain
         after the entry day, up to and including the expiry), in the next weekly expiry (Thursday; Tuesday from
         2025-09-01; a holiday moves it to the previous trading day):
         SELL the CE and the PE whose Black-Scholes |delta| is closest to 0.16 (flat vol = India VIX previous close,
         spot = NIFTY at 10:20, r = 6.5 %, calendar time to 15:30 on expiry, 50-point grid) and
         BUY the CE at short CE + 400 and the PE at short PE − 400 (defined risk); 2 lots (75, 65 from 2025-12-31).
  hold   to expiry; no stop, no target, no adjustment. One structure at a time.
  settle at intrinsic against NIFTY's expiry close = the mean of the 15:00-15:29 minute closes (NSE's last-30-minute
         rule as the lab models it); booked to the ledger per leg at intrinsic value, with option round-trip costs.

Paper execution. Real NFO contracts from Kite's instrument master (exact expiry and strike; Kite's nearest listed
weekly expiry must equal the lab's expiry, else the week is skipped as expiry_mismatch). Wings are bought first at
the ask, then the shorts are sold at the bid (marketable LIMITs; never a partial structure). Product NRML, held
overnight: base_agent.overnight_hold_qty lists the legs, so the 15:15 MIS square-off never touches them.

What paper cannot tell us yet (the key unknown): the lab's credit is a flat-VIX Black-Scholes model (no skew; real
far-OTM wings are richer, so the real credit is likely lower). Every structure records model_credit next to the
real_credit it got, and live_readiness() requires real ≥ 80 % of model before anything else.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from types import SimpleNamespace
from typing import Callable, Optional

from loguru import logger

import options_replay as OR
import validated_data as VD
from agents.validated_base import (ValidatedAgent, PlannedTrade, ValidatedPosition, fingerprint, paper_marketable,
                                   serialized)
from config import settings
from ist_clock import now_ist, is_trading_day
from stratlab import options_vrp as V

LAB_ID = "VRP-STRANGLE-016-WINGED"
SPEC = V.BY_ID[LAB_ID]
UNDERLYING = SPEC.underlying                    # "NIFTY"
SPOT_QUOTE = "NIFTY 50"
VIX_QUOTE = "INDIA VIX"
ENTRY_T = dtime(10, 20)
ENTRY_DEADLINE = dtime(10, 21)                  # a catch-up entry is allowed until 10:21:00 only (late_entry)
SETTLE_T = dtime(15, 31)
SETTLE_FROM, SETTLE_TO = OR.SETTLE_FROM, dtime(15, 29)
QUOTE_RETRY_GAPS_S = (10,)                      # one whole-batch retry, still inside the minute
SETTLE_N_CLOSES = 30                            # 15:00..15:29: the full window the lab averaged
SETTLE_ALERT_TD = 2                             # settlement pending this many trading days → status alert
NIFTY_50_TOKEN = 256265                         # fallbacks when the NSE instrument master has no row
INDIA_VIX_TOKEN = 264969

LAB_FINGERPRINT = fingerprint(
    OR.ShortStrangle, OR.MarketView, OR.next_expiry, OR._expiry_weekday, OR._adjust_back, OR.previous_trading_day,
    OR.trading_days_left, OR.bs, OR.bs_delta, OR.years_between, OR.fill_price, OR.haircut, OR._max_risk,
    OR.lot_size_on, OR.Replay._manage, V.OptSpec.strategy, V.OptSpec.margin,
    consts={"id": SPEC.id, "params": sorted(SPEC.params.items()), "lots": SPEC.lots, "LOT_SCHEDULE": V.LOT_SCHEDULE,
            "ELM_PCT": V.ELM_PCT, "OPT_FILL_MODEL": V.OPT_FILL_MODEL, "RISK_FREE": OR.RISK_FREE,
            "SETTLE_FROM": OR.SETTLE_FROM, "EXPIRY_CLOSE": OR.EXPIRY_CLOSE, "WEEKLY_SWITCH": OR.WEEKLY_SWITCH,
            "STRIKE_STEP": OR.STRIKE_STEP["NIFTY"], "SLIP_MIN": OR.SLIP_MIN_PTS["NIFTY"], "SLIP_PCT": OR.SLIP_PCT})


# ── pure decision functions (the parity target) ───────────────────────────────
@dataclass
class StranglePlan:
    entry: datetime
    expiry: date
    spot: float
    sigma: float
    lot: int
    qty: int
    legs: list                     # [(opt, strike, side ±1, model_px)] in the lab's order: S CE, S PE, B CE, B PE
    model_credit: float            # per unit, Σ −side·model_px (before the lab's fill haircut)
    lab_credit: float              # per unit, Σ −side·OR.fill_price(model_px): the lab's booked credit
    max_risk: float                # ₹, OR._max_risk on the lab fills (what the lab reports)
    margin_est: float              # ₹, SPEC.margin: max risk + ELM 2 % of spot notional
    lab_fills: list = field(default_factory=list)


def cycle_status(now: datetime, is_td: Callable[[date], bool]) -> tuple[Optional[date], str]:
    """(expiry, "") on a cycle day, else (None, reason): "weekend" / "not_cycle_day" / "short_week".
    Uses the lab strategy's own in_cycle (no spot or vol needed)."""
    if now.weekday() >= 5:
        return None, "weekend"
    strat = SPEC.strategy()
    exp = OR.next_expiry(UNDERLYING, now.date(), is_td, skip_today=True)
    view = OR.MarketView(UNDERLYING, now, 1.0, [], [], 0.1, exp, OR.STRIKE_STEP[UNDERLYING],
                         OR.years_between(now, exp), OR.RISK_FREE, is_td)
    if strat.in_cycle(view):
        return exp, ""
    return None, ("short_week" if strat.skipped_short_weeks else "not_cycle_day")


def plan_strangle(now: datetime, spot: float, vix_prev_close: float, is_td: Callable[[date], bool], lots: int,
                  lot_size: Optional[int] = None) -> tuple[Optional[StranglePlan], str]:
    """The lab's 10:20 decision (Replay's decision step with time_basis "calendar"), from spot and the previous
    VIX close. Returns (plan, "") or (None, reason)."""
    strat = SPEC.strategy()                     # ShortStrangle(10:20, expiry, 0.16, wings 400, weekly, min 3 days)
    if now.weekday() >= 5:
        return None, "weekend"
    if not spot or spot <= 0:
        return None, "no_data:spot"
    if not vix_prev_close or vix_prev_close <= 0:
        return None, "no_data:vix"
    exp = OR.next_expiry(UNDERLYING, now.date(), is_td, skip_today=True)
    view = OR.MarketView(UNDERLYING, now, float(spot), [], [], float(vix_prev_close) / 100.0, exp,
                         OR.STRIKE_STEP[UNDERLYING], OR.years_between(now, exp), OR.RISK_FREE, is_td)
    specs = strat.decide(view)
    if not specs:
        return None, ("short_week" if strat.skipped_short_weeks else "not_cycle_day")
    lot = int(lot_size or OR.lot_size_on(V.LOT_SCHEDULE, now.date()))
    qty = int(lots) * lot
    legs = [(s.opt, float(s.strike), int(s.side), OR.bs(view.spot, s.strike, view.T, view.sigma, s.opt, view.r))
            for s in specs]
    fills = [OR.fill_price(m, side, UNDERLYING) for _, _, side, m in legs]
    lab_credit = sum(-side * px for (_, _, side, _), px in zip(legs, fills))
    lab_legs = [OR.Leg(opt, k, side, px) for (opt, k, side, _), px in zip(legs, fills)]
    max_risk = round(OR._max_risk(lab_legs, lab_credit, qty), 2)
    margin = SPEC.margin(SimpleNamespace(spot_in=view.spot, qty=qty, max_risk=max_risk))
    return StranglePlan(entry=now, expiry=exp, spot=float(spot), sigma=view.sigma, lot=lot, qty=qty, legs=legs,
                        model_credit=round(sum(-side * m for _, _, side, m in legs), 4),
                        lab_credit=round(lab_credit, 4), max_risk=max_risk, margin_est=round(margin, 2),
                        lab_fills=fills), ""


def settle_strangle(legs, closes_from_1500: list[float]) -> tuple[float, list[float]]:
    """The lab's expiry settlement (options_replay.Replay._manage): S = mean of the closes stamped ≥ 15:00 on the
    expiry day; each leg at its intrinsic value against S."""
    S = sum(closes_from_1500) / len(closes_from_1500)
    return S, [max(0.0, S - k) if opt == "CE" else max(0.0, k - S) for opt, k, *_ in legs]


def max_structure_loss(legs_entry: list[tuple[str, float, int, float]], qty: int) -> float:
    """₹ worst expiry loss of the structure at its REAL fills (OR._max_risk), for the live-readiness check."""
    credit = sum(-side * px for _, _, side, px in legs_entry)
    return round(OR._max_risk([OR.Leg(o, k, s, px) for o, k, s, px in legs_entry], credit, qty), 2)


# ── Kite I/O (module level: tests patch these) ────────────────────────────────
def _token(symbol: str, fallback: int) -> int:
    from kite_client import kite_client
    try:
        return int(kite_client.get_instrument_tokens([symbol], "NSE").get(symbol) or fallback)
    except Exception:
        return fallback


def expiry_closes(expiry: date) -> list[float]:
    """NIFTY 50 minute closes of the expiry date stamped 15:00..15:29 (Kite historical candles). [] if unavailable."""
    from kite_client import kite_client
    rows = kite_client.historical_data(_token(SPOT_QUOTE, NIFTY_50_TOKEN), datetime.combine(expiry, dtime(9, 15)),
                                       datetime.combine(expiry, dtime(15, 30)), "minute") or []
    out = []
    for r in rows:
        ts = r["date"] if not isinstance(r["date"], str) else datetime.fromisoformat(r["date"])
        if ts.date() == expiry and SETTLE_FROM <= ts.time().replace(tzinfo=None) <= SETTLE_TO \
                and float(r.get("close") or 0) > 0:
            out.append((ts, float(r["close"])))
    return [c for _, c in sorted(out)]


def vix_prev_close_hist(today: date) -> Optional[float]:
    """Fallback for the quote's previous close: the last India VIX day candle strictly before `today`."""
    from kite_client import kite_client
    rows = kite_client.historical_data(_token(VIX_QUOTE, INDIA_VIX_TOKEN),
                                       datetime.combine(today - timedelta(days=12), dtime(0, 0)),
                                       datetime.combine(today, dtime(0, 0)), "day") or []
    best = None
    for r in rows:
        ts = r["date"] if not isinstance(r["date"], str) else datetime.fromisoformat(r["date"])
        if ts.date() < today and float(r.get("close") or 0) > 0 and (best is None or ts > best[0]):
            best = (ts, float(r["close"]))
    return best[1] if best else None


# ── the agent ────────────────────────────────────────────────────────────────
class WingedStrangleAgent(ValidatedAgent):
    name = "vrp_winged"
    product = "NRML"
    SETTING = "strangle"                    # settings.validated_strangle_enabled
    COST_KIND = None                        # ledger trade_cost: NFO option round trip per leg (as the lab)
    LAB_ID = LAB_ID
    VERSION = f"{LAB_ID}/{V.OPT_FILL_MODEL}/v1"
    LAB_FINGERPRINT = LAB_FINGERPRINT
    # dd/vrp/final_run/summary.txt (2 lots; bps of margin = max risk + 2 % of spot notional)
    LAB_EXPECTATION = {
        "DISC": {"n": 107, "net_per_trade": 1268, "bps_of_margin": 100.0, "pf": 1.24, "win_pct": 79.4,
                 "credit_pts": 57.0, "worst": -51922, "max_dd": 129760},
        "FINAL": {"n": 45, "net_per_trade": 1357, "bps_of_margin": 122.7, "pf": 1.27, "win_pct": 82.2,
                  "credit_pts": 60.4, "worst": -45439, "max_dd": 104495},
        "DISC_net_at_5": {"net": 39165, "pf": 1.07, "screen": "FAIL (H1 net <= 0)"},
        "t_bps": {"DISC": 0.81, "FINAL": 0.56},
        "model_limits": "flat-VIX Black-Scholes credit (no skew): the real credit is unverified",
    }

    def __init__(self) -> None:
        super().__init__()
        self.structures: dict[str, dict] = {}     # group id → entry record (expiry, credits, legs, settlement)
        self.marks: dict = {}                     # read-only leg marks for the status

    # ── limits: the strangle's scale is lots / 2 (the tested size) ──
    def _lots(self) -> int:
        return max(1, int(getattr(settings, "validated_vrp_lots", SPEC.lots) or SPEC.lots))

    def _lot_scale(self) -> float:
        return self._lots() / float(SPEC.lots)

    def daily_loss_limit(self) -> float:
        return 0.0                                # P&L lands once a week, at settlement: no daily stop

    def dd_limit(self) -> float:
        return float(getattr(settings, "validated_vrp_dd_kill", 0.0) or 0.0) * self._lot_scale()

    # ── helpers ──
    def _legs_of(self, gid: str) -> list[ValidatedPosition]:
        return [p for p in self.positions() if p.group == gid]

    def open_groups(self) -> list[str]:
        return sorted({p.group for p in self.positions()})

    def _ledger_extra(self, pos: ValidatedPosition, exit_px: float) -> dict:
        st = self.structures.get(pos.group, {})
        return {"group": pos.group, "opt": pos.meta.get("opt"), "strike": pos.meta.get("strike"),
                "expiry": pos.meta.get("expiry"), "model_px": pos.meta.get("model_px"),
                "lab_fill": pos.meta.get("lab_fill"), "real_credit_pts": st.get("real_credit"),
                "model_credit_pts": st.get("model_credit"), "lab_credit_pts": st.get("lab_credit"),
                "max_risk_real": st.get("max_risk_real")}

    # ── 10:20 entry ──
    @serialized
    async def decide_and_enter(self, late: bool = False) -> dict:
        now = self._now()
        today, t = now.date(), now.time().replace(tzinfo=None)
        out = {"day": today.isoformat(), "opened": [], "reason": ""}

        def skip(reason, sym="*", mark=True, log=True):
            self._skip(sym, reason, log=log)
            out["reason"] = reason
            if mark:
                self.mark_done("entry")
            return out

        if not is_trading_day(today):
            return skip("not_trading_day", mark=False)
        # a structure whose expiry has passed is settled first (the lab settled it before this entry)
        await self.settle_due(now)
        await self.unwind_orphans()
        # the cycle needs no market data: on the 4 non-cycle days a week nothing else is looked at
        sched = datetime.combine(today, ENTRY_T)             # the scheduled instant, naive IST (lab: the bar time)
        exp, why = cycle_status(sched, is_trading_day)
        if exp is None:
            if why == "not_cycle_day":                       # info, not a skip
                self._log("entry", "*", "info", why)
                out["reason"] = why
                return out
            return skip(why)
        if t >= ENTRY_DEADLINE:
            return skip("entry_window_missed")
        if t < dtime(10, 19):
            return skip("too_early", mark=False)
        block = self.entry_block_reason()
        if block:
            return skip(block, mark=False)                   # a blocked batch may still run inside the window
        if self.done_today("entry"):
            return skip("already_entered", mark=False)
        live = [g for g in self.open_groups() if self._legs_of(g)[0].meta.get("expiry", "9999") >= today.isoformat()]
        if live:
            return skip("position_open")                     # the lab holds one structure at a time
        if late:
            self._deviation("late_entry", "*", f"entry at {t.strftime('%H:%M:%S')}")
        # spot and the previous VIX close
        q = {}
        for wait in (0,) + QUOTE_RETRY_GAPS_S:
            if wait:
                await asyncio.sleep(wait)
            q = await asyncio.to_thread(VD.quotes, [f"NSE:{SPOT_QUOTE}", f"NSE:{VIX_QUOTE}"])
            if q:
                break
        if not q:
            return skip("quote_failed")
        spot = float((q.get(SPOT_QUOTE) or {}).get("ltp") or 0)
        vix = float((q.get(VIX_QUOTE) or {}).get("prev_close") or 0)
        if vix <= 0:
            try:
                vix = float(await asyncio.to_thread(vix_prev_close_hist, today) or 0)
            except Exception as exc:
                self._error("vix history", exc)
                vix = 0.0
        if spot <= 0:
            return skip("no_data:spot")
        if vix <= 0:
            return skip("no_data:vix")
        lots = self._lots()
        plan, why = plan_strangle(sched, spot, vix, is_trading_day, lots)
        if plan is None:
            return skip(why)
        # real contracts: exact expiry and strike from the instrument master
        import options_resolver as ORS
        listed = ORS.listed_expiries(UNDERLYING, today, skip_today=True)
        if not listed:
            return skip("no_contract")
        if listed[0] != plan.expiry:
            logger.warning("[vrp_winged] lab expiry {} vs Kite's nearest listed {} — week skipped",
                           plan.expiry, listed[0])
            return skip("expiry_mismatch")
        contracts = [ORS.resolve_exact(UNDERLYING, plan.expiry, k, opt) for opt, k, _, _ in plan.legs]
        if any(c is None for c in contracts):
            return skip("no_contract")
        kite_lot = int(contracts[0].lot_size or 0)
        if kite_lot and kite_lot != plan.lot:
            self.flags["lot_mismatch"] = f"{today.isoformat()}: lab {plan.lot} vs Kite {kite_lot}"
            self._log("entry", "*", "flag", f"lot_mismatch lab {plan.lot} vs Kite {kite_lot} (Kite used)")
            logger.warning("[vrp_winged] lot size: lab schedule {} vs Kite {} — using Kite's", plan.lot, kite_lot)
            plan, why = plan_strangle(sched, spot, vix, is_trading_day, lots, lot_size=kite_lot)
            if plan is None:
                return skip(why)
        scale = self._lot_scale()
        if plan.margin_est > float(settings.validated_vrp_max_margin) * scale or \
                plan.max_risk > float(settings.validated_vrp_max_loss) * scale:
            return skip("risk_cap")
        # the four leg quotes: every leg needs a two-sided quote, else no order at all
        lq = await asyncio.to_thread(VD.quotes, [f"NFO:{c.tradingsymbol}" for c in contracts])
        for c in contracts:
            x = (lq or {}).get(c.tradingsymbol) or {}
            if not (float(x.get("bid") or 0) > 0 and float(x.get("ask") or 0) > 0):
                return skip("no_data:leg_quote")
        # SEBI for every leg before the first order (a rejection never leaves a partial structure)
        from sebi_compliance import sebi_compliance
        for (opt, k, side, m), c in zip(plan.legs, contracts):
            ok, _algo, reason = sebi_compliance.pre_order_check(
                strategy=self.name, symbol=c.tradingsymbol, exchange="NFO",
                transaction_type="BUY" if side > 0 else "SELL", quantity=plan.qty, order_type="LIMIT",
                price_at_signal=float(lq[c.tradingsymbol]["ask" if side > 0 else "bid"]), signal_source=self.LAB_ID,
                regime="n/a")
            if not ok:
                return skip(f"sebi: {reason}")
        gid = f"VRP-{today.strftime('%Y%m%d')}"
        self.structures[gid] = {"gid": gid, "entry": sched.isoformat(), "expiry": plan.expiry.isoformat(),
                                "spot": plan.spot, "sigma": plan.sigma, "vix": vix, "lot": plan.lot, "qty": plan.qty,
                                "lots": lots, "model_credit": plan.model_credit, "lab_credit": plan.lab_credit,
                                "max_risk_lab": plan.max_risk, "margin_est": plan.margin_est, "status": "opening",
                                "late": bool(late)}
        # wings first (BUY at the ask), then the shorts (SELL at the bid)
        order = sorted(range(4), key=lambda i: 0 if plan.legs[i][2] > 0 else 1)
        filled: list[ValidatedPosition] = []
        for i in order:
            opt, k, side, m = plan.legs[i]
            c = contracts[i]
            tside = "BUY" if side > 0 else "SELL"
            try:
                oid, fill = paper_marketable(c.tradingsymbol, "NFO", tside, plan.qty, "NRML", lq[c.tradingsymbol],
                                             "VAL-VRP-W" if side > 0 else "VAL-VRP-S")
            except Exception as exc:
                self._error(f"entry {c.tradingsymbol}", exc)
                self._unwind(gid, filled, lq, "partial_structure" if side < 0 else "order_failed")
                return skip("partial_structure" if filled else "order_failed")
            sebi_compliance.record_order_id(self.name, c.tradingsymbol, oid)
            pt = PlannedTrade(symbol=c.tradingsymbol, side=side, qty=plan.qty, entry_ref=round(m, 4),
                              entry_t="10:20", exit_t="expiry", key=k)
            pos = self._open(pt, "NRML", "NFO", fill, oid, overnight=True, group=gid,
                             meta={"opt": opt, "strike": k, "model_px": round(m, 4),
                                   "lab_fill": round(plan.lab_fills[i], 4), "expiry": plan.expiry.isoformat(),
                                   "spot": plan.spot, "sigma": plan.sigma, "leg": i})
            filled.append(pos)
        legs_real = sorted(((p.meta["opt"], p.meta["strike"], 1 if p.side == "BUY" else -1, p.entry_px, p.meta["leg"])
                            for p in filled), key=lambda x: x[4])
        real_credit = round(sum(-s * px for _, _, s, px, _ in legs_real), 4)
        self.structures[gid].update({"status": "open", "real_credit": real_credit,
                                     "max_risk_real": max_structure_loss([x[:4] for x in legs_real], plan.qty),
                                     "legs": [p.oid for p in filled]})
        self.mark_done("entry")
        out["opened"] = [p.symbol for p in filled]
        out["gid"] = gid
        logger.info("[vrp_winged] {} opened: expiry {} qty {} credit real {:.2f} vs model {:.2f} (lab fill {:.2f})",
                    gid, plan.expiry, plan.qty, real_credit, plan.model_credit, plan.lab_credit)
        return out

    def _unwind(self, gid: str, filled: list[ValidatedPosition], lq: dict, kind: str) -> None:
        """A leg failed after others filled: close the filled legs at once (never hold a partial structure)."""
        if filled:
            logger.critical("[vrp_winged] {}: {} — closing {} filled leg(s)", gid, kind, len(filled))
        for pos in filled:
            if not self._still_open(pos):
                continue
            back = "SELL" if pos.side == "BUY" else "BUY"
            try:
                _oid, fill = paper_marketable(pos.symbol, "NFO", back, pos.qty, "NRML", lq.get(pos.symbol) or {},
                                              "VAL-VRP-UNWIND")
            except Exception as exc:
                self._error(f"unwind {pos.symbol}", exc)
                continue
            self._deviation(kind, pos.symbol, "unwound")
            self._close(pos, fill, f"{kind}: unwound", kind=kind)
        if not self._legs_of(gid):
            self.structures.get(gid, {})["status"] = kind

    async def unwind_orphans(self) -> list[str]:
        """A structure left in status "opening" by a process that died between legs (each leg is saved as it fills):
        never held as a partial structure — its filled legs are closed at the touch (deviation partial_structure)."""
        done = []
        for gid, st in list(self.structures.items()):
            if st.get("status") != "opening":
                continue
            legs = self._legs_of(gid)
            if not legs:
                st["status"] = "partial_structure"
                continue
            lq = await asyncio.to_thread(VD.quotes, [f"NFO:{p.symbol}" for p in legs])
            if not all(float((lq or {}).get(p.symbol, {}).get("bid" if p.side == "BUY" else "ask") or 0) > 0
                       or float((lq or {}).get(p.symbol, {}).get("ltp") or 0) > 0 for p in legs):
                self._skip(gid, "no_data:unwind_quote", stage="unwind")
                continue
            logger.critical("[vrp_winged] {} restored mid-entry with {} of 4 legs — unwinding", gid, len(legs))
            self._unwind(gid, legs, lq or {}, "partial_structure")
            done.append(gid)
            if str(st.get("entry", ""))[:10] == self._today().isoformat():
                self.mark_done("entry")          # as an in-flight unwind: no second structure the same day
        return done
    # ── expiry settlement ──
    def _due(self, now: datetime) -> list[str]:
        today, t = now.date(), now.time().replace(tzinfo=None)
        out = []
        for g in self.open_groups():
            exp = self._legs_of(g)[0].meta.get("expiry", "")
            if exp < today.isoformat() or (exp == today.isoformat() and t >= SETTLE_T):
                out.append(g)
        return out

    @serialized
    async def settle_due(self, now: Optional[datetime] = None) -> list[str]:
        """Settle every structure whose expiry close is known (15:31 on the expiry day or later): each leg is closed
        in the paper broker at its intrinsic value (fill_kind SETTLEMENT) and booked to the ledger at intrinsic.
        Never from a guessed price: without the 15:00-15:29 candles it retries (status alert after 2 trading days)."""
        now = now or self._now()
        settled = []
        due = self._due(now)
        if due and settings.trading_mode != "PAPER":
            self._skip("*", "live_mode_refused", stage="settle")
            return settled
        for gid in due:
            legs = sorted(self._legs_of(gid), key=lambda p: p.meta.get("leg", 0))
            exp = date.fromisoformat(legs[0].meta["expiry"])
            try:
                closes = await asyncio.to_thread(expiry_closes, exp)
            except Exception as exc:
                self._error(f"settlement candles {exp}", exc)
                closes = []
            if not closes:
                self._skip(gid, "settlement_pending", stage="settle")
                n_td, d = 0, exp
                while d < now.date():
                    d += timedelta(days=1)
                    n_td += bool(is_trading_day(d))
                if n_td >= SETTLE_ALERT_TD:
                    self.flags["settlement_pending"] = f"{gid} expiry {exp} ({n_td} trading days)"
                    logger.error("[vrp_winged] {} expiry {} still unsettled after {} trading days", gid, exp, n_td)
                continue
            if len(closes) < SETTLE_N_CLOSES:
                if exp >= now.date():
                    # Kite's historical feed can lag right after the close: settle only on the full window
                    self._skip(gid, f"settlement_pending: {len(closes)}/{SETTLE_N_CLOSES} closes", stage="settle")
                    continue
                # a later day: the window is final as Kite has it (the lab averaged the bars its data had)
                self._deviation("partial_settle_window", gid, f"{len(closes)}/{SETTLE_N_CLOSES} closes")
            S, intrinsic = settle_strangle([(p.meta["opt"], p.meta["strike"]) for p in legs], closes)
            from kite_client import kite_client
            for pos, px in zip(legs, intrinsic):
                if not self._still_open(pos):
                    continue
                back = "SELL" if pos.side == "BUY" else "BUY"
                kite_client.paper_fill_at(pos.symbol, "NFO", back, pos.qty, "NRML", round(px, 2), "VAL-VRP-SETTLE",
                                          "SETTLEMENT")
                self._close(pos, round(px, 2), f"expiry settle S={S:.2f}")
            st = self.structures.setdefault(gid, {})
            st.update({"status": "settled", "settle_S": round(S, 2), "settled_at": now.isoformat(timespec="seconds"),
                       "n_settle_closes": len(closes)})
            if self.flags.get("settlement_pending", "").startswith(gid):
                self.flags.pop("settlement_pending", None)
            self._save_book(force=True)
            settled.append(gid)
            logger.info("[vrp_winged] {} settled at S={:.2f} ({} closes)", gid, S, len(closes))
        return settled

    # ── read-only marks for the status ──
    async def mark(self) -> dict:
        if not self.positions():
            return {}
        q = await asyncio.to_thread(VD.quotes, [f"NFO:{p.symbol}" for p in self.positions()])
        if not q:
            return {}
        out = {}
        for gid in self.open_groups():
            legs = self._legs_of(gid)
            px = {p.symbol: float((q.get(p.symbol) or {}).get("ltp") or 0) for p in legs}
            if not all(v > 0 for v in px.values()):
                continue
            unreal = sum((px[p.symbol] - p.entry_px) * p.qty * (1 if p.side == "BUY" else -1) for p in legs)
            lot = int(self.structures.get(gid, {}).get("lot") or legs[0].qty // self._lots())
            shadow = sum((p.entry_px - px[p.symbol]) * lot for p in legs if p.side == "SELL")
            out[gid] = {"t": self._now().isoformat(timespec="seconds"), "unrealised": round(unreal, 2),
                        "shadow_naked_1lot": round(shadow, 2), "ltp": px}
        self.marks = out
        return out

    # ── restart catch-up ──
    @serialized
    async def catch_up(self, now: Optional[datetime] = None) -> dict:
        now = now or self._now()
        self.auto_start(now.date())            # the 08:55 preflight for a process started later (not after a Stop)
        out = {"settled": await self.settle_due(now)}
        out["unwound"] = await self.unwind_orphans()
        t = now.time().replace(tzinfo=None)
        if is_trading_day(now.date()) and t >= ENTRY_T and not self.done_today("entry"):
            exp, _why = cycle_status(datetime.combine(now.date(), ENTRY_T), is_trading_day)
            if exp is not None:
                out["entry"] = await self.decide_and_enter(late=True)
        return out

    # ── status ──
    def ledger_cycles(self) -> list[dict]:
        cyc: dict[str, dict] = {}
        for r in self.ledger_rows():
            g = r.get("group") or r.get("oid")
            c = cyc.setdefault(g, {"group": g, "net": 0.0, "legs": 0, "real_credit": r.get("real_credit_pts"),
                                   "model_credit": r.get("model_credit_pts"), "max_risk": r.get("max_risk_real"),
                                   "qty": r.get("qty")})
            c["net"] += float(r.get("net", 0))
            c["legs"] += 1
        return [dict(c, net=round(c["net"], 2)) for c in cyc.values()]

    def live_stats(self, rows=None, net_key: str = "net") -> dict:
        cyc = self.ledger_cycles()
        if not cyc:
            return {"n": 0}
        nets = [c["net"] for c in cyc]
        gp, gl = sum(x for x in nets if x > 0), -sum(x for x in nets if x < 0)
        rc = [float(c["real_credit"]) for c in cyc if c.get("real_credit") is not None]
        mc = [float(c["model_credit"]) for c in cyc if c.get("model_credit") is not None]
        return {"n": len(cyc), "net": round(sum(nets), 2), "net_per_cycle": round(sum(nets) / len(nets), 2),
                "pf": round(gp / gl, 2) if gl else None, "win_pct": round(100 * sum(x > 0 for x in nets) / len(nets), 1),
                "mean_real_credit": round(sum(rc) / len(rc), 2) if rc else None,
                "mean_model_credit": round(sum(mc) / len(mc), 2) if mc else None,
                "worst": round(min(nets), 2),
                "beyond_max_risk": sum(1 for c in cyc if c.get("max_risk") is not None
                                       and c["net"] < -float(c["max_risk"]) - 1e-6)}

    def live_readiness(self) -> dict:
        st = self.live_stats()
        n = st.get("n", 0)
        ratio = (st["mean_real_credit"] / st["mean_model_credit"]
                 if n and st.get("mean_real_credit") is not None and st.get("mean_model_credit") else None)
        checks = {"cycles>=26": n >= 26, "real_credit>=80%_of_model": ratio is not None and ratio >= 0.8,
                  "net>0": st.get("net", 0) > 0, "pf>=1.15": (st.get("pf") or 0) >= 1.15,
                  "no_loss_beyond_max_risk": st.get("beyond_max_risk", 0) == 0}
        return {"criterion": "≥ 26 cycles; mean real credit ≥ 80 % of mean model credit; net > 0 and PF ≥ 1.15; "
                             "no loss beyond the structure's max risk",
                "credit_ratio": round(ratio, 3) if ratio is not None else None, "checks": checks,
                "ready": all(checks.values()), "never_switches_to_live": True}

    def get_status(self) -> dict:
        st = super().get_status()
        st["structures"] = {g: s for g, s in self.structures.items() if s.get("status") in ("open", "opening")}
        st["recent_structures"] = sorted(self.structures.values(), key=lambda s: s.get("entry", ""))[-6:]
        st["marks"] = self.marks
        st["lots"] = self._lots()
        st["limits"].update({"max_margin": float(settings.validated_vrp_max_margin) * self._lot_scale(),
                             "max_loss": float(settings.validated_vrp_max_loss) * self._lot_scale()})
        try:
            exp, why = cycle_status(datetime.combine(self._now().date(), ENTRY_T), is_trading_day)
            st["today"] = {"cycle_day": exp is not None, "expiry": exp.isoformat() if exp else None, "reason": why}
        except Exception as exc:
            st["today"] = {"error": str(exc)[:120]}
        st["next_jobs"] = {"entry": "10:20:00 on the first trading day after a weekly expiry",
                           "settle": "15:31 on the expiry day (retried every 10 min to 16:51, then at restart)"}
        return st

    def export_book(self) -> dict:
        keep = {g: s for g, s in self.structures.items()
                if s.get("status") in ("open", "opening") or s.get("entry", "")[:10] >= (
                    self._now().date() - timedelta(days=30)).isoformat()}
        return {**super().export_book(), "structures": keep}

    def import_book(self, d: dict, closed: set, stale: bool) -> None:
        super().import_book(d, closed, stale)
        for g, s in ((d or {}).get("structures") or {}).items():
            self.structures.setdefault(g, dict(s))
