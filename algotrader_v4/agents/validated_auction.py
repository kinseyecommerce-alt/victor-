"""
agents/validated_auction.py — AUC-FADE (opening-auction fade) as a live PAPER agent, exactly as the lab tested it.

Lab: stratlab/auction.py (rules in its docstring), CONFIRMED by the pre-registered protocol on 3 years of NIFTY-100
minute bars (dd/auction_final/summary.txt). Variant AUC-FADE-3PCT by default (settings.validated_auction_variant
switches to 2PCT; never both): 3PCT is the only one still positive at 5 bps/side stress on FINAL.

    prev_close  close of the last 1-minute bar of the symbol's previous VALID session (>= 200 bars) — from
                validated_data (Kite historical minute candles through the same summarize_sessions as the parity
                test), not Kite's ohlc.close
    auction_px  the official open (Kite quote ohlc.open at 09:15:05) = the 09:15 bar's open
    signal      stratlab.auction.auction_rule (imported, not copied), ranked as stratlab.auction.select ranks:
                largest |gap| first, ties by symbol, at most MAX_PER_DAY; qty = max(1, int(NOTIONAL // auction_px))
    entry       a pre-open MIS order filled at the auction price (kite_client.paper_fill_at, fill_kind AUCTION)
    exit        09:20:00, a marketable LIMIT at the quote touch (bid for a long's SELL, ask for a short's BUY)
    no stop, no target, no other filter

Not executable with real money as tested: the signal reads the FINAL auction price, known only after pre-open
order entry closes (~09:08). The paper record measures whether the effect persists; forward_test.py's
"AUC-FADE-3PCT/live" rule (09:06:45 indicative price) measures the executable variant. See live_readiness().

Documented divergences from the lab (measured by the parity test, not hidden):
  * the lab drops a candidate whose 09:20..09:29 exit bar is missing (and a day whose session ends up < 200 bars)
    BEFORE ranking — look-ahead the live agent cannot have
  * corporate-action ex-dates (split / bonus / demerger ...) are skipped: the lab's adjusted data has no such gap
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Iterable, Optional

from loguru import logger

import stratlab.auction as A
import validated_data
from agents.validated_base import PlannedTrade, ValidatedAgent, paper_marketable, paper_row
from config import settings
from ist_clock import is_trading_day, now_ist  # noqa: F401  (now_ist: module-level clock the tests patch)

ENTRY_T = time(9, 15)
ENTRY_DEADLINE = time(9, 16)            # last moment an entry decision may be made (the fill is the official open)
EXIT_T = time(9, 20)
EXIT_LAST_RETRY = time(9, 29, 50)       # the lab falls back to the first bar before 09:30
ENTRY_RETRY_S = (15, 15, 15)            # 09:15:05 → +15 s, +30 s, +45 s for missing opens
EXIT_RETRY_S = 10
LOOSEST_GAP = min(sp.params["min_gap"] for sp in A.SPECS)     # the lab's candidates_for threshold
MIN_GAP = {"3PCT": 0.03, "2PCT": 0.02}
ENTRY_TAG, EXIT_TAG = "VAL-AUC-IN", "VAL-AUC-OUT"

# lab expectation (dd/auction_final/summary.txt, 2.1 bps/side slippage, 2026-11 costs)
LAB_EXPECTATIONS = {
    "3PCT": {"DISC": {"trades": 510, "net_bps": 46.63, "pf": 1.88, "win_pct": 61.0},
             "FINAL": {"trades": 252, "net_bps": 19.84, "pf": 1.32, "win_pct": 56.7},
             "FINAL_stress_5bps": {"net_bps": 14.06, "pf": 1.22}},
    "2PCT": {"DISC": {"trades": 1090, "net_bps": 23.69, "pf": 1.43, "win_pct": 56.4},
             "FINAL": {"trades": 525, "net_bps": 4.27, "pf": 1.08, "win_pct": 53.7},
             "FINAL_stress_5bps": {"net_bps": -1.52, "pf": 0.97}},
}
READINESS_MIN_TRADES = 150
READINESS_MIN_PF = 1.15


async def _sleep(s: float) -> None:          # tests replace it
    await asyncio.sleep(s)


def _fingerprint() -> str:
    src = "".join(inspect.getsource(f) for f in (A.auction_rule, A.select, A.settle, A.session_features))
    consts = repr((A.AUCTION_T, A.EXIT_T, A.EXIT_LATEST, A.MAX_ABS_GAP, A.MAX_PER_DAY, A.NOTIONAL, A.FILL_MODEL,
                   sorted((sp.id, sorted(sp.params.items())) for sp in A.SPECS)))
    return hashlib.sha1((src + consts).encode()).hexdigest()


LAB_FINGERPRINT = _fingerprint()


@dataclass(frozen=True)
class AuctionInput:
    symbol: str
    prev_close: float
    auction_px: float


def plan_auction(inputs: Iterable[AuctionInput], min_gap: float, notional: float = A.NOTIONAL,
                 max_per_day: int = A.MAX_PER_DAY) -> tuple[list[PlannedTrade], dict[str, str]]:
    """The day's trades, exactly as the lab selects them: auction_rule at the lab's loosest threshold, then
    min_gap < |gap| <= MAX_ABS_GAP (stratlab.auction.select), largest |gap| first (ties by symbol), at most
    max_per_day; qty = max(1, int(notional // auction_px)); fade the gap. Returns (plan, {symbol: skip reason})."""
    skips: dict[str, str] = {}
    cands = []
    for x in inputs:
        sig = A.auction_rule(x.prev_close, x.auction_px, LOOSEST_GAP, A.MAX_ABS_GAP)
        if sig is None or not (min_gap < abs(sig[1]) <= A.MAX_ABS_GAP):
            skips[x.symbol] = "no_signal"
            continue
        cands.append((abs(sig[1]), x.symbol, sig[0], x.auction_px))
    cands.sort(key=lambda c: (-c[0], c[1]))
    plan = [PlannedTrade(sym, side, max(1, int(notional // px)), px, "09:15", "09:20", g)
            for g, sym, side, px in cands[:max_per_day]]
    for _g, sym, _s, _p in cands[max_per_day:]:
        skips[sym] = "not_top10"
    return plan, skips


def apply_gross_cap(plan: list[PlannedTrade], cap: float) -> tuple[list[PlannedTrade], Optional[float]]:
    """Scale every quantity by cap / gross when the plan's gross exceeds `cap` (never drops a name)."""
    gross = sum(p.qty * p.entry_ref for p in plan)
    if gross <= cap or gross <= 0:
        return plan, None
    f = cap / gross
    return [PlannedTrade(p.symbol, p.side, max(1, int(p.qty * f)), p.entry_ref, p.entry_t, p.exit_t, p.key)
            for p in plan], f


class AuctionFadeAgent(ValidatedAgent):
    name = "auc_fade"
    product = "MIS"
    COST_KIND = None               # ledger trade_cost NSE MIS → equity_mis, the lab's cost kind
    ENABLE_SETTING = "validated_auction_enabled"
    SETTING = "auction"
    FLATTEN_CLOSED_ON_RESTORE = True
    LAB_FINGERPRINT = LAB_FINGERPRINT

    @property
    def variant(self) -> str:
        v = str(getattr(settings, "validated_auction_variant", "3PCT"))
        return v if v in MIN_GAP else "3PCT"

    @property
    def LAB_ID(self) -> str:                       # noqa: N802 (interface name)
        return f"AUC-FADE-{self.variant}"

    @property
    def VERSION(self) -> str:                      # noqa: N802
        return f"{self.LAB_ID}/{A.FILL_MODEL}/v1"

    @property
    def LAB_EXPECTATION(self) -> dict:             # noqa: N802
        return LAB_EXPECTATIONS[self.variant]

    @property
    def min_gap(self) -> float:
        return MIN_GAP[self.variant]

    def daily_loss_limit(self) -> float:
        return float(settings.validated_auction_daily_loss_stop) * float(settings.validated_notional_scale)

    def dd_limit(self) -> float:
        return float(settings.validated_auction_dd_kill) * float(settings.validated_notional_scale)

    # ── entry (09:15:05) ─────────────────────────────────────────────────────
    async def enter(self, q: Optional[dict] = None) -> dict:
        ds = self.day_state()
        today = self._today()
        reason = self.entry_block_reason()
        if reason:
            self._note("entry", "*", "skip", reason)
            return {"entered": 0, "skipped": reason}
        if ds.get("entry_done"):
            self._note("entry", "*", "skip", "already_entered")
            return {"entered": 0, "skipped": "already_entered"}
        if self._now().time() >= ENTRY_DEADLINE:
            self._note("entry", "*", "skip", "entry_window_missed")
            ds["entry_done"] = True
            return {"entered": 0, "skipped": "entry_window_missed"}

        pcs = {s: validated_data.prev_close(s, today) for s in validated_data.UNIVERSE}
        attempt = 0
        while True:
            if not q:
                q = await asyncio.to_thread(validated_data.quotes, validated_data.UNIVERSE)
            missing = [s for s in validated_data.UNIVERSE
                       if pcs.get(s) and not float((q.get(s) or {}).get("open") or 0) > 0] if q else []
            if q and not missing:
                break
            if attempt >= len(ENTRY_RETRY_S) or self._now().time() >= ENTRY_DEADLINE:
                break
            logger.info("[{}] {} — retry in {} s", self.name,
                        "quote batch failed" if not q else f"{len(missing)} opens missing", ENTRY_RETRY_S[attempt])
            await _sleep(ENTRY_RETRY_S[attempt])
            attempt += 1
            q = None
        ds["entry_done"] = True
        if not q:
            self._note("entry", "*", "skip", "quote_failed")
            self._save_book()
            return {"entered": 0, "skipped": "quote_failed"}

        inputs = []
        for s in validated_data.UNIVERSE:
            pc = pcs.get(s)
            if not pc:
                self._note("entry", s, "skip", "no_data:prev_close")
                continue
            ap = float((q.get(s) or {}).get("open") or 0.0)
            if ap <= 0:
                self._note("entry", s, "skip", "no_data:open")
                continue
            ca = validated_data.price_adjusting_actions(s, today)
            if ca:
                self._note("entry", s, "skip", f"corporate_action: {'; '.join(ca)[:80]}")
                continue
            inputs.append(AuctionInput(s, float(pc), ap))
        if not validated_data.ca_calendar_loaded():
            ds["ca_calendar_unavailable"] = True
            self._note("entry", "*", "info", "ca_calendar_unavailable (proceeding, flagged)")

        notional = A.NOTIONAL * float(settings.validated_notional_scale)
        plan, skips = plan_auction(inputs, self.min_gap, notional)
        for s, why in skips.items():
            self._note("entry", s, "skip", why, log=(why != "no_signal"))
        plan, f = apply_gross_cap(plan, float(settings.validated_auction_max_gross))
        if f is not None:
            self._note("entry", "*", "info", f"cap_scaled ×{f:.3f}")
            ds["cap_scaled"] = round(f, 4)
        ds["candidates"] = sorted(
            [{"symbol": x.symbol, "prev_close": x.prev_close, "auction_px": x.auction_px,
              "gap_pct": round((x.auction_px / x.prev_close - 1) * 100, 3)} for x in inputs],
            key=lambda c: -abs(c["gap_pct"]))[:20]
        ds["planned"] = [[p.symbol, p.side, p.qty] for p in plan]

        from sebi_compliance import sebi_compliance
        from kite_client import kite_client
        n = 0
        for p in plan:
            side = "BUY" if p.side > 0 else "SELL"
            try:
                ok, _algo, why = sebi_compliance.pre_order_check(
                    strategy=self.name, symbol=p.symbol, exchange="NSE", transaction_type=side, quantity=p.qty,
                    order_type="MARKET", price_at_signal=p.entry_ref, signal_source=f"validated:{self.LAB_ID}",
                    regime="n/a")
                if not ok:
                    self._note("entry", p.symbol, "skip", f"sebi: {why}")
                    continue
                oid = kite_client.paper_fill_at(p.symbol, "NSE", side, p.qty, "MIS", p.entry_ref,
                                                tag=ENTRY_TAG, fill_kind="AUCTION")
                sebi_compliance.record_order_id(self.name, p.symbol, oid)
                gap = p.entry_ref / next(x.prev_close for x in inputs if x.symbol == p.symbol) - 1
                self._open(p, "MIS", "NSE", p.entry_ref, oid, meta={"gap": round(gap, 6), "rank": n + 1})
                self._note("entry", p.symbol, "entered", f"{side} x{p.qty} @ {p.entry_ref} gap {gap * 100:+.2f}%")
                n += 1
            except Exception as exc:
                self.state.errors.append(f"entry {p.symbol}: {exc}"[:200])
                self._note("entry", p.symbol, "skip", f"order_error: {str(exc)[:80]}")
        ds["entered"] = n
        self._save_book()
        logger.info("[{}] {} entries: {} of {} planned", self.name, self.LAB_ID, n, len(plan))
        return {"entered": n, "planned": len(plan)}

    # ── exit (09:20:00) ──────────────────────────────────────────────────────
    def reconcile(self) -> list[str]:
        """Positions whose MIS row is already flat at the broker (e.g. the 15:15 square-off flattened it):
        booked at the row's last price, counted as a deviation."""
        out = []
        for p in self.positions():
            row = paper_row(p.symbol, p.product)
            if int((row or {}).get("quantity") or 0) * p.sign >= p.qty:
                continue
            px = float((row or {}).get("last_price") or 0.0) or p.entry_px
            self._deviation("flattened_by_squareoff", p.symbol)
            self._close(p, px, "flattened_by_squareoff", kind="reconcile")
            out.append(p.symbol)
        return out

    async def exit(self, late: bool = False) -> dict:
        """Close every open position at the quote touch; symbols without a usable quote are retried every 10 s
        until 09:29:50, then exited at their last price (late_exit)."""
        from kite_client import kite_client
        closed = 0
        while True:
            self.reconcile()
            open_ = self.positions()
            if not open_:
                break
            q = await asyncio.to_thread(validated_data.quotes, sorted({p.symbol for p in open_}))
            for p in open_:
                qs = q.get(p.symbol) or {}
                if not (float(qs.get("ltp") or 0) > 0 or float(qs.get("bid") or 0) > 0 or float(qs.get("ask") or 0) > 0):
                    continue
                try:
                    oid, fill = paper_marketable(p.symbol, p.exchange, "SELL" if p.sign > 0 else "BUY", p.qty,
                                                 p.product, qs, EXIT_TAG)
                except Exception as exc:
                    self.state.errors.append(f"exit {p.symbol}: {exc}"[:200])
                    logger.error("[{}] exit {} failed: {}", self.name, p.symbol, exc)
                    continue
                if late:
                    self._deviation("late_exit", p.symbol)
                self._close(p, fill, "late exit" if late else "09:20 exit", kind="late_exit" if late else "normal")
                closed += 1
            if not self.positions():
                break
            if self._now().time() >= EXIT_LAST_RETRY:
                for p in self.positions():            # no usable quote until 09:29:50: exit at the last price
                    row = paper_row(p.symbol, p.product) or {}
                    px = float((q.get(p.symbol) or {}).get("ltp") or 0) or float(row.get("last_price") or 0) \
                        or p.entry_px
                    kite_client.paper_fill_at(p.symbol, p.exchange, "SELL" if p.sign > 0 else "BUY", p.qty,
                                              p.product, px, tag=EXIT_TAG, fill_kind="RECONCILE")
                    self._deviation("late_exit", p.symbol, "no quote until 09:29:50")
                    self._close(p, px, "late exit (no quote)", kind="late_exit")
                    closed += 1
                break
            await _sleep(EXIT_RETRY_S)
        self.day_state()["exit_done"] = True
        self._save_book()
        return {"closed": closed, "open": len(self.positions())}

    # ── restart catch-up ─────────────────────────────────────────────────────
    async def catch_up(self, now: Optional[datetime] = None) -> dict:
        now = now or self._now()
        today, t = now.date(), now.time().replace(tzinfo=None)
        out: dict = {}
        if is_trading_day(today):
            self.state.running = True          # equivalent of the 08:55 preflight for a process started later
        self.reconcile()
        if is_trading_day(today) and ENTRY_T <= t < ENTRY_DEADLINE and not self.day_state().get("entry_done"):
            out["enter"] = await self.enter()
        if self.positions() and (t >= EXIT_T or any(p.opened[:10] < today.isoformat() for p in self.positions())):
            out["exit"] = await self.exit(late=True)
        return out

    def _on_stale_restore(self, positions) -> None:
        """An earlier session's snapshot still holds auction positions (the process was down through 09:20 and
        the 15:15 square-off): close each at the snapshot's last price, once."""
        from kite_client import kite_client
        for p in positions:
            row = paper_row(p.symbol, p.product) or {}
            px = float(row.get("last_price") or 0.0) or p.entry_px
            if int(row.get("quantity") or 0) * p.sign >= p.qty:
                kite_client.paper_fill_at(p.symbol, p.exchange, "SELL" if p.sign > 0 else "BUY", p.qty, p.product,
                                          px, tag=EXIT_TAG, fill_kind="RECONCILE")
            self._deviation("missed_exit", p.symbol, p.opened[:10])
            self._close(p, px, f"missed_exit {p.opened[:10]}", kind="missed_exit")

    # ── 15:40 parity audit ───────────────────────────────────────────────────
    def parity_audit(self, day: Optional[date] = None) -> dict:
        """Re-plan today from the stored minute bars (09:15 bar open, prev_close) and compare with the trades
        decided from quotes: differences measure quote-open vs bar-open drift."""
        day = day or self._today()
        ds = self.day_state()
        inputs = []
        for s in validated_data.UNIVERSE:
            ss = validated_data.session_on(s, day)
            pc = validated_data.prev_close(s, day)
            if not ss or not pc or ss.first_t != A.AUCTION_T:
                continue
            if validated_data.price_adjusting_actions(s, day):
                continue
            inputs.append(AuctionInput(s, pc, ss.first_open))
        plan, _ = plan_auction(inputs, self.min_gap, A.NOTIONAL * float(settings.validated_notional_scale))
        bars = sorted((p.symbol, p.side) for p in plan)
        live = sorted((s, side) for s, side, _q in ds.get("planned") or [])
        res = {"day": day.isoformat(), "bars": bars, "live": live, "match": bars == live}
        if not res["match"]:
            self._deviation("parity_mismatch", "*", f"bars {bars} vs live {live}"[:150])
        ds["parity_audit"] = res
        self._save_book()
        return res

    # ── status ───────────────────────────────────────────────────────────────
    def live_readiness(self) -> dict:
        """Read-only. Fixed before any live result was seen (spec §0): >= 150 paper trades with net > 0 and
        PF >= 1.15, AND forward_test's indicative-price rule (the only executable one) meeting the same bar."""
        own = self.live_stats()
        fwd = {}
        try:
            import forward_test
            fwd = forward_test.status()["summary"].get(f"{self.LAB_ID}/live") or {}
        except Exception:
            pass

        def ok(n, net, pf):
            return bool(n and n >= READINESS_MIN_TRADES and (net or 0) > 0 and (pf or 0) >= READINESS_MIN_PF)
        a = ok(own["trades"], own["net"], own["pf"])
        b = ok(fwd.get("trades"), fwd.get("net"), fwd.get("profit_factor"))
        return {"criteria": f">= {READINESS_MIN_TRADES} trades, net > 0, PF >= {READINESS_MIN_PF} on paper fills AND "
                            f"forward_test {self.LAB_ID}/live (indicative price) meeting the same bar",
                "paper": {**own, "met": a},
                "forward_live_rule": {"trades": fwd.get("trades"), "net": fwd.get("net"),
                                      "pf": fwd.get("profit_factor"), "met": b},
                "ready": a and b,
                "note": "read-only: nothing here switches to LIVE; the tested final-price rule is not executable"}

    def get_status(self) -> dict:
        st = super().get_status()
        st["variant"] = self.variant
        st["min_gap"] = self.min_gap
        st["candidates_today"] = self._day.get("candidates", [])
        try:
            import forward_test
            t = (forward_test.status().get("today") or {}).get("trades") or {}
            v = t.get(self.LAB_ID) or {}
            st["forward_test_today"] = {r: [(x["symbol"], x["side"], x["net"]) for x in v.get(r, [])]
                                        for r in ("final", "live")}
        except Exception:
            st["forward_test_today"] = None
        st["next_jobs"] = {"entry": "09:15:05", "exit": "09:20:00", "parity_audit": "15:40"}
        return st
