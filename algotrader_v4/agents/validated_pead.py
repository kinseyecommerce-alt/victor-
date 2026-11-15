"""
agents/validated_pead.py — EV-PEAD-LONG-H60 (post-earnings-announcement drift, long, 60 sessions) as a PAPER-only
CANDIDATE agent. Not proven, never live.

Why a candidate. The pre-registered event study (stratlab/events_strategies.py, prereg/events_v1.json, README
"Event study: promoter buying and earnings moves") CONFIRMED it on 2023-26 (DISCOVERY 529 trades PF 1.35; one-look
FINAL 296 trades PF 1.32, +₹5.68 L at ₹1 L a trade) and OOS-CONFIRMED it on 2012-23 (2,736 trades PF 1.93), but
it is NOT proven: deflated Sharpe 0.51 at the fair trial count N = 9 (proof needs >= 0.95), and its excess over
NIFTY on the same holding windows since 2018 is not significant (FINAL mean abnormal return +3.2 %, 95 % CI
-1.2 .. +7.7 %). Status shows `candidate: true, proven: false`; live_readiness() never switches anything to LIVE.

The tested rule, unchanged (rule functions imported from stratlab, never copied; LAB_FINGERPRINT pins them):
  event      quarterly results, timed on the FIRST public broadcast of the numbers (events_parse.build_results:
             the earliest of the structured results filing and a "Financial Result Updates" / results-day
             "Outcome of Board Meeting" announcement)
  reaction   the first session whose 15:30 close is after that broadcast (events_data.Asof.results_reacted_today:
             broadcast in [previous session 15:30, this session 15:30)) — results in hours react the same day,
             after the close or before the open react on the next session
  signal     (events_strategies._reaction / pead_long) on the reaction session D, a member of the point-in-time
             top-500-by-turnover universe (events_daily.monthly_universe, gate on D): the stock traded on the
             previous session, no excluded corporate-action day in (D-1, D], volume(D) > mean volume of its 20
             previous bars, and close(D)/close(D-1) - NIFTY(D)/NIFTY(D-1) >= +4 %
  entry      BUY at the next session's official open (a pre-open order, paper-filled at the open);
             qty = max(1, floor(₹1 L / (open × (1 + 2 bps)))) — sim.NOTIONAL and the lab's 2 bps slippage
  stop       catastrophe stop = 80 % of open × (1 + 2 bps); checked on DAILY bars exactly as sim.sim_positional
             does: on any held session (entry day included) whose LOW <= stop, exit at min(stop, that session's
             open) — booked in that evening's job from the official daily bar
  exit       the CLOSE (NSE LAST traded price, the lab's close) of the 60th session the stock traded, counting the
             entry session as the first (sim.sim_positional: natural = e_day + max_hold - 1, exit at its close)
  one at a time per stock: a stock held (or planned for tomorrow) is not entered again; a position closed today
             frees the stock for today's decision (the lab is flat again from the exit day's close)
Costs: the ledger books CNC delivery costs (costs.py "equity_cnc", the lab's cost kind for swing longs).

Live inputs (pead_data.py; each one fails SAFE — no entry that day, skip key in the status, no Telegram):
  results    NSE's public filing feeds (stratlab.events_fetch client, generic UA, polite) after the close, parsed by
             stratlab.events_parse; `no_event_feed` when a feed fails or the day's window holds no announcement
  universe   the lab's monthly_universe() on NSE sec_bhavdata_full files (nsearchives), `no_universe` otherwise
  bars       the same bhavcopy files through the lab's read_new() + new_factors() (adjusted, close = LAST),
             NIFTY 50 from NSE's index file (Kite day candles as fallback); `no_bars` / `no_nifty`
  opens      the Kite quote's ohlc.open at 09:15:10 (= the official open), retried to 09:29:50

Schedule (platform_scheduler, only while VALIDATED_PEAD_ENABLED=true or the agent holds positions): the evening
job at 19:35 / 20:35 / 21:35 / 22:35 IST (after NSE publishes the day's files) manages open positions and plans
tomorrow's entries; 08:20 retries a decision the evening could not make (the lab's order cut-off is 09:00);
09:15:10 enters at the open. Restart: catch_up() runs whatever is due.

Caps and limits (config; scale with VALIDATED_NOTIONAL_SCALE), from the lab's ev_pead_long_h60 2023-26 trades at
₹1 L / trade: concurrent open trades mean 57, p95 80, max 88 -> VALIDATED_PEAD_MAX_POSITIONS = 100 and
VALIDATED_PEAD_MAX_GROSS = ₹1 crore (lab peak open notional ₹87 L); max drawdown of realised P&L (by exit day)
₹11.19 L -> VALIDATED_PEAD_DD_KILL = ₹16.8 L (1.5x); worst exit day -₹1.60 L -> VALIDATED_PEAD_DAILY_LOSS_STOP =
₹2.4 L (1.5x), checked at the evening decision (after the day's exits) and at the open. When a cap binds, the
largest abnormal returns are entered first (ties by symbol) — a live-only rule (the lab had no cap).

Documented divergences (live-only guards; listed in LAB_EXPECTATION["live_only_guards"]):
  * provisional events: an announcement whose structured filing is not yet out is dated at the announcement (the
    lab, with the filing in hand, does the same); if the filing never comes the lab would have no event
  * renames: NSE's symbolchange.csv is applied with the lab's rename step (events_daily.apply_renames_fast); a held
    stock renamed during the hold follows the new symbol (position and paper row, deviation symbol_renamed)
  * corporate actions during a hold: a bonus / split confirmed by the ex-day gap (events_daily.decide_factor) keeps
    the position with the shares the holder receives (free shares paper-filled at ₹0, stop × factor) as the lab's
    adjusted series does; a demerger / scheme / large rights / consolidation exits at that session's close (the lab
    DROPS such trades: drop_excluded); an unexplained gap (open / previous close outside 0.6 .. 1.67) is flagged
  * a stock that does not trade on the planned entry session is not entered (lab: its next bar's open)
  * missed jobs (downtime): exits and stops are booked from the official daily bars of the missed sessions (flagged
    late_booking); a decision whose entry open has passed is skipped (decision_window_missed)
"""
from __future__ import annotations

import asyncio
import contextlib
import math
import threading
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from typing import Iterable, Optional

import numpy as np
from loguru import logger

import pead_data as PD
import validated_data as VD
from agents.validated_base import (ValidatedAgent, PlannedTrade, ValidatedPosition, fingerprint, paper_fill_at,
                                   serialized)
from config import settings
from ist_clock import now_ist, is_trading_day  # noqa: F401  (now_ist: module-level clock the tests patch)
from stratlab import data_daily as DD
from stratlab import events_daily as ED
from stratlab import events_data as E
from stratlab import events_parse as EP
from stratlab import events_strategies as ES
from stratlab import sim

SPEC = ES.BY_ID["ev_pead_long_h60"]
P = SPEC.p                               # min_ar_pct 4.0, vol_n 20, stop_pct 20.0, hold 60
NOTIONAL = sim.NOTIONAL                  # ₹1 L per trade
SLIP = sim.DEFAULT_SLIPPAGE_BPS / 1e4    # the lab's per-side slippage (events_run.SLIP_BPS)
FILL_MODEL = "ev1"                       # events_run.FILL_MODEL (not imported: events_run pulls the whole protocol)
ENTRY_T = dtime(9, 15, 10)
ENTRY_DEADLINE = dtime(9, 29, 50)
ENTRY_RETRY_S = 10
EVENING_FROM = dtime(18, 0)              # a session's files are complete only after NSE's evening publication
DECISION_CUTOFF = dtime(E.ORDER_CUTOFF // 100, E.ORDER_CUTOFF % 100)   # 09:00: the lab's order cut-off
BARS_SESSIONS = 30                       # history read for the reaction rule (needs vol_n + 2 = 22 bars)
ENTRY_TAG, EXIT_TAG, CA_TAG = "VAL-PEAD-IN", "VAL-PEAD-OUT", "VAL-PEAD-CA"
TOP_N = ED.TOP_N

LAB_FINGERPRINT = fingerprint(
    ES.pead_long, ES._reaction, E.Asof, E._select, E.ca_excluded, E.asof_swing, sim.sim_positional, sim._aligned,
    DD._rows_to_arrays, DD._Gated, ED.read_new, ED.monthly_universe, ED.new_factors, ED.decide_factor,
    ED.parse_action, EP.build_results, EP.build_boardmeet,
    consts={"id": SPEC.id, "params": sorted(SPEC.params.items()), "kind": SPEC.kind, "NOTIONAL": NOTIONAL,
            "SLIP_BPS": sim.DEFAULT_SLIPPAGE_BPS, "ORDER_CUTOFF": E.ORDER_CUTOFF, "CLOSE": E.CLOSE, "TOP_N": TOP_N,
            "FILL_MODEL": FILL_MODEL})

# stratlab/events run1 report.json (ev_pead_long_h60; 2 bps/side slippage, 2026-11 costs, CNC) and its trade CSVs
LAB_EXPECTATION = {
    "DISC_2023_26": {"trades": 529, "pf": 1.355, "net": 1151555.14, "win_pct": 48.8,
                     "h1": {"trades": 275, "pf": 1.537}, "h2": {"trades": 254, "pf": 1.138}},
    "FINAL": {"trades": 296, "pf": 1.317, "net": 567878.21, "win_pct": 51.7, "confirmed": True},
    "OLD_ERA_2012_23": {"trades": 2736, "pf": 1.926, "net": 15059703.24, "win_pct": 55.4, "verdict": "OOS-CONFIRMED"},
    "FINAL_vs_NIFTY": {"mean_ar_pct": 3.188, "ci95_mean_ar_pct": [-1.171, 7.658], "share_beating_nifty": 0.466},
    "dsr_fair": {"dsr": 0.5124, "n_fair": 9, "required": 0.95},
    "per_trade": {"mean_gross": 2415.79, "mean_cost": 238.94},
    "concurrency_2023_26": {"mean": 56.5, "p95": 80, "max": 88, "peak_open_notional": 8718543},
    "risk_2023_26": {"max_dd_by_exit_day": 1118594, "worst_exit_day": -159630},
    "proven": False,
    "why_not_proven": "DSR 0.51 at the fair trial count N = 9 (needs 0.95); excess vs NIFTY since 2018 not significant",
    "live_only_guards": {
        "event_feed": "no entries on a day whose NSE filing feeds failed or held no announcement (skip "
                      "'no_event_feed'); provisional events dated at an announcement whose structured filing is not "
                      "out yet",
        "universe": "monthly_universe() on live bhavcopy files with today's EQUITY_L minus the ETF list (lab: also "
                    "the old-era INE list); no universe -> no entries (skip 'no_universe')",
        "corporate_action": "bonus / split confirmed by the ex-day gap: shares received paper-filled at Rs 0 and the "
                            "stop scaled (the lab's adjusted series); demerger / scheme / rights / consolidation "
                            "during a hold: exit at that session's close (the lab drops such trades); unexplained "
                            "gap: flagged",
        "caps": "max positions / max gross (config): largest abnormal return first when a cap binds (lab: no cap)",
        "entry_session": "a stock that does not trade on the planned entry session is not entered (lab: its next "
                         "bar)",
        "missed_jobs": "exits / stops of missed sessions booked late from the official daily bars (late_booking); "
                       "a decision whose entry open has passed is skipped (decision_window_missed)",
    },
}
READINESS = {"min_trades": 100, "min_pf": 1.15}


# ══════════════════════════════════════════════════════════════════════════
# pure functions (the parity target)
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PeadEntry:
    symbol: str
    decision_day: int
    entry_day: int
    ar: float                 # the reaction session's abnormal return vs NIFTY


def results_table(events: Iterable) -> dict:
    """events [(symbol, first_ts 'YYYY-MM-DD HH:MM:SS', period_end)] or dicts -> events_data's results table."""
    by: dict = {}
    for ev in events:
        if isinstance(ev, dict):
            sym, ts, pe = ev["symbol"], ev["first_ts"], ev.get("period_end") or "0"
        else:
            sym, ts, pe = ev[0], ev[1], (ev[2] if len(ev) > 2 else "0")
        if not ts:
            continue
        by.setdefault(str(sym), []).append((E.ts_stamp(str(ts)), int(str(pe).replace("-", "") or 0)))
    out = {}
    for sym, rows in by.items():
        rows.sort()
        out[sym] = {"ts": np.array([r[0] for r in rows], np.int64),
                    "period_end": np.array([r[1] for r in rows], np.int64)}
    return out


_LAB_LOCK = threading.RLock()


@contextlib.contextmanager
def lab_context(calendar: Iterable[int], events: Iterable, exclusions: Iterable = ()):
    """stratlab.events_data's module state (calendar, results table, exclusions) set to the live inputs for one
    decision, restored afterwards (the lab functions read module globals)."""
    with _LAB_LOCK:
        saved = (list(E.calendar()), dict(E._T), {k: list(v) for k, v in E._EXC.items()})
        try:
            E.set_calendar(calendar)
            E._T.clear()
            E._T["results"] = results_table(events)
            E.set_exclusions(list(exclusions))
            yield
        finally:
            E.set_calendar(saved[0])
            E._T.clear()
            E._T.update(saved[1])
            E._EXC.clear()
            E._EXC.update(saved[2])


def _view(symbol: str, b, nifty_aux: dict):
    d = sim.DailyView(np.asarray(b.dates, np.int64), np.asarray(b.o, float), np.asarray(b.h, float),
                      np.asarray(b.l, float), np.asarray(b.c, float), np.asarray(b.v, float))
    d.symbol = symbol
    d.nifty = sim._aligned(d.date, nifty_aux) if nifty_aux else None
    d.vix = None
    return d


def _why_not(d, day: int, prv: int, symbol: str) -> str:
    """A readable reason for a candidate the lab rule rejected (diagnostics only; the decision is the lab's)."""
    if len(d) < P.vol_n + 2:
        return "short_history"
    if d.nifty is None or not (d.nifty[-1] > 0 and d.nifty[-2] > 0):
        return "no_nifty"
    if int(d.date[-2]) != prv:
        return "no_bar_prev_session"
    if E.ca_excluded(symbol, int(d.date[-2]), int(d.date[-1])):
        return "corporate_action"
    if not d.v[-1] > float(np.mean(d.v[-P.vol_n - 1:-1])):
        return "volume_not_above_average"
    return "below_threshold"


def plan_pead(day: int, calendar: Iterable[int], events: Iterable, bars: dict, nifty: dict, members,
              holding: Iterable[str] = (), exclusions: Iterable = ()) -> tuple[list[PeadEntry], dict]:
    """Tomorrow's entries exactly as the lab decides them on session `day` (yyyymmdd):
    calendar  the session calendar (must hold the previous and the next session)
    events    every known results event [(symbol, first_ts, period_end)] — the lab filters the reaction window
    bars      {symbol: pead_data.Bars} adjusted daily bars up to and including `day`
    nifty     {yyyymmdd: NIFTY 50 close}
    members   the universe of `day`'s month; holding: stocks held or already planned (one position per stock)
    exclusions [(symbol, yyyymmdd)] corporate-action / unexplained-gap days (events_data.set_exclusions)
    Returns (entries sorted by symbol, {symbol: skip reason})."""
    entries, skips = [], {}
    members, holding = set(members or ()), set(holding or ())
    nd = sorted(int(k) for k, v in nifty.items() if v and float(v) > 0)
    aux = {"date": np.asarray(nd, np.int64), "c": np.asarray([float(nifty[k]) for k in nd], float)} if nd else None
    with lab_context(calendar, events, exclusions):
        nxt, prv = E.next_session(day), E.prev_session(day)
        if nxt is None or prv is None:
            return [], {"*": "no_calendar"}
        lo, hi = E.stamp(prv, E.CLOSE), E.stamp(day, E.CLOSE)
        cands = sorted(s for s in E._T["results"] if len(E.results(s, hi, lo)))
        for s in cands:
            if s not in members:                      # data_daily._Gated: member on the decision day
                skips[s] = "not_member"
                continue
            if s in holding:
                skips[s] = "holding"
                continue
            b = bars.get(s)
            if b is None or not b.dates or int(b.dates[-1]) != day:
                skips[s] = "no_bar_today"
                continue
            d = _view(s, b, aux)
            with E.cutoff(E.stamp(nxt, E.ORDER_CUTOFF)):      # the lab's look-ahead guard (events_run._Guard)
                sig = ES.pead_long(d, P)
                ar = ES._reaction(d, P)
            if sig is None:
                skips[s] = _why_not(d, day, prv, s) if ar is None else "below_threshold"
                continue
            entries.append(PeadEntry(s, int(day), int(nxt), float(ar)))
    return entries, skips


def entry_fill_lab(open_px: float) -> float:
    """The lab's fill: the official open plus 2 bps slippage (sizing and the stop are computed from it)."""
    return float(open_px) * (1.0 + SLIP)


def entry_qty(open_px: float, notional: float = NOTIONAL) -> int:
    return max(1, int(notional // entry_fill_lab(open_px)))


def stop_price(open_px: float) -> float:
    return entry_fill_lab(open_px) * (1.0 - P.stop_pct / 100.0)


def hold_step(stop: float, held_sessions: int, bar: tuple, hold: int = P.hold) -> Optional[tuple[str, float]]:
    """One held session of sim.sim_positional (no target): bar = (open, high, low, close) of the k-th session the
    stock traded (entry session = 1). Stop first (a gap through it fills at the open), then the clock at the close
    of session `hold`. Returns (reason, raw exit price) or None (still held)."""
    o, _h, l, c = (float(x) for x in bar)
    if stop is not None and l <= stop:
        return "stop", min(float(stop), o)
    if held_sessions >= hold:
        return "max_hold", c
    return None


def apply_caps(entries: list[PeadEntry], open_px: dict, n_open: int, gross_open: float, max_positions: int,
               max_gross: float, notional: float) -> tuple[list[tuple[PeadEntry, int, float]], dict]:
    """Live-only caps: largest abnormal return first (ties by symbol). Returns ([(entry, qty, open)], skips)."""
    out, skips = [], {}
    n, gross = int(n_open), float(gross_open)
    for e in sorted(entries, key=lambda x: (-x.ar, x.symbol)):
        px = float(open_px.get(e.symbol) or 0.0)
        if px <= 0:
            skips[e.symbol] = "no_data:open"
            continue
        q = entry_qty(px, notional)
        if max_positions > 0 and n >= max_positions:
            skips[e.symbol] = "cap_positions"
            continue
        if max_gross > 0 and gross + q * px > max_gross:
            skips[e.symbol] = "cap_gross"
            continue
        out.append((e, q, px))
        n += 1
        gross += q * px
    return out, skips


# ══════════════════════════════════════════════════════════════════════════
# the agent
# ══════════════════════════════════════════════════════════════════════════
class PeadCandidateAgent(ValidatedAgent):
    name = "pead_h60"
    product = "CNC"
    SETTING = "pead"                      # settings.validated_pead_enabled / _daily_loss_stop / _dd_kill
    COST_KIND = "equity_cnc"              # stratlab api.Spec.cost_kind of a swing long
    LAB_ID = "EV-PEAD-LONG-H60"
    VERSION = f"EV-PEAD-LONG-H60/{FILL_MODEL}/v1"
    LAB_FINGERPRINT = LAB_FINGERPRINT
    LAB_EXPECTATION = LAB_EXPECTATION
    CANDIDATE = True
    PROVEN = False

    def __init__(self) -> None:
        super().__init__()
        self.pending: dict = {}           # {"decision_day", "entry_day", "entries": [{symbol, ar}], "made"}
        self.processed_through: int = 0   # last session whose daily bar was applied to the open positions
        self.last_decision: dict = {}     # {"day", "entries", "skips", "inputs"}
        self.inputs: dict = {}            # last input refresh status (feeds, universe, files)

    # ── settings ──
    def enabled_by_setting(self) -> bool:
        return bool(getattr(settings, "validated_pead_enabled", False))

    def _setting(self, suffix: str, default=None):
        if suffix == "enabled":
            return bool(getattr(settings, "validated_pead_enabled", False))
        return super()._setting(suffix, default)

    @property
    def shadow_enrol(self) -> bool:
        """shadow_registry enrols this candidate only once it is switched on (the Bonferroni family then grows by
        one, for good)."""
        return self.enabled_by_setting()

    def max_positions(self) -> int:
        return int(getattr(settings, "validated_pead_max_positions", 100) or 0)

    def max_gross(self) -> float:
        return float(getattr(settings, "validated_pead_max_gross", 0.0) or 0.0) * self._scale()

    def wanted(self) -> bool:
        """Jobs are needed while the flag is on or positions / a plan remain."""
        return self.enabled_by_setting() or bool(self._positions) or bool(self.pending)

    # ── ledger extras ──
    def _ledger_extra(self, pos: ValidatedPosition, exit_px: float) -> dict:
        m = pos.meta
        ex = {"candidate": True, "decision_day": m.get("decision_day"), "entry_day": m.get("entry_day"),
              "exit_day": m.get("exit_day"), "ar": m.get("ar"), "stop": m.get("stop"),
              "held_sessions": m.get("held_sessions"), "exit_reason": m.get("exit_reason")}
        no, nc = m.get("nifty_open_entry"), m.get("nifty_close_exit")
        if no and nc and pos.entry_px > 0:
            r = float(exit_px) / pos.entry_px - 1.0
            ex["nifty_ret"] = round(nc / no - 1.0, 6)
            ex["excess_vs_nifty"] = round(r - (nc / no - 1.0), 6)
        return ex

    # ── calendar helpers ──
    def _last_complete_session(self, now: datetime) -> Optional[int]:
        today = now.date()
        cal = PD.sessions_between(today - timedelta(days=20), today)
        if not cal:
            return None
        if cal[-1] == PD.ymd(today) and now.time().replace(tzinfo=None) < EVENING_FROM:
            cal = cal[:-1]
        return cal[-1] if cal else None

    @staticmethod
    def _next_session(day: int) -> Optional[int]:
        d = PD.to_date(day)
        cal = [s for s in PD.sessions_between(d + timedelta(days=1), d + timedelta(days=20))]
        return cal[0] if cal else None

    def _decision_open(self, last: int, now: datetime) -> bool:
        """A decision on session `last` can still reach its entry open (before the 09:00 cut-off)."""
        nxt = self._next_session(last)
        if nxt is None:
            return False
        today = PD.ymd(now.date())
        return today < nxt or (today == nxt and now.time().replace(tzinfo=None) < DECISION_CUTOFF)

    # ── input gathering (blocking: run in a thread) ──
    def _gather(self, now: datetime, last: int, todo: list[int], deciding: bool, held: dict) -> dict:
        """Downloads and computes everything the evening needs. held = {symbol: entry_day}."""
        out: dict = {"status": {}, "rows": {}, "factors": {}, "traded": {}, "nifty": {}, "renames": {}}
        st = PD.load_state()
        today = now.date()
        d_last = PD.to_date(last)
        changes = PD.ensure_symbol_changes(st, today)
        # a held stock renamed since its entry is followed to the new symbol (the bhavcopy and the feeds use it)
        ren = {o: n for d, o, n in changes if o in held and held[o] < d <= last}
        out["renames"] = ren
        held = {ren.get(s, s): ed for s, ed in held.items()}
        cal = PD.sessions_between(d_last - timedelta(days=60), d_last)
        need = set(todo)
        for s in todo:                                   # the previous session of each managed day (ex-day gap)
            i = cal.index(s) if s in cal else -1
            if i > 0:
                need.add(cal[i - 1])
        if held:
            lo = min(held.values())
            need |= set(PD.sessions_between(PD.to_date(lo), d_last))
        if deciding:
            need |= set(cal[-BARS_SESSIONS:])
        out["status"]["bhav"] = bh = PD.ensure_bhav(need)
        missing = sorted(d for d, s in bh.items() if s not in ("ok", "have"))
        out["status"]["bhav_missing"] = missing
        feeds = None
        if deciding:
            feeds = PD.fetch_feeds(today)
        elif held:
            feeds = PD.fetch_feeds(today, ("ca",))
        if feeds is not None:
            # fetch_feeds loads and saves the state file itself: carry its backfill markers into this snapshot so the
            # final save below does not wipe them (otherwise every evening would repeat the full 120-day backfill)
            st["feeds_backfilled"] = {**(st.get("feeds_backfilled") or {}),
                                      **(PD.load_state().get("feeds_backfilled") or {})}
        out["status"]["feeds"] = feeds
        nd = set(cal[-3:]) | set(todo) | set(held.values())
        out["nifty"] = PD.ensure_nifty(st, nd)
        actions = PD.corporate_actions()
        # managed sessions: rows, ex-day factors, traded-day counts
        hs = set(held)
        if hs:
            for s in todo:
                if s in missing:
                    continue
                out["rows"][s] = PD.session_rows(s, hs)
                i = cal.index(s) if s in cal else -1
                prev = cal[i - 1] if i > 0 else None
                if prev is None or prev in missing:
                    continue
                eq = PD.read_bhav(prev - 1, s)
                if eq is None:
                    continue
                eq = eq[eq["symbol"].isin(hs)].copy()
                eq["_pri"] = (eq["series"] != "EQ").astype(int)
                eq = eq.sort_values(["symbol", "date", "_pri"]).drop_duplicates(["symbol", "date"], keep="first")
                eq = eq.drop(columns="_pri").reset_index(drop=True)
                _applied, log = ED.new_factors(eq, [a for a in actions if a[0] in hs], lo=prev)
                for r in log:
                    out["factors"][(r["symbol"], int(r["first_session"]))] = (r["factor"], r["status"])
                prow = PD.session_rows(prev, hs)
                for sym, r in out["rows"][s].items():
                    pc = (prow.get(sym) or {}).get("close_price")
                    r["prev_close_stored"] = pc
            lo = min(held.values())
            eqh = PD.read_bhav(lo - 1, last)
            files_ok = all(PD.bhav_path(s).exists() for s in PD.sessions_between(PD.to_date(lo), d_last))
            for sym, ed in held.items():
                if eqh is None:
                    break
                days = sorted(int(x) for x in eqh.loc[eqh["symbol"] == sym, "date"])
                out["traded"][sym] = {"days": [x for x in days if x >= ed], "complete": files_ok}
        # decision inputs
        if deciding:
            dec = out["decision"] = {}
            prv = cal[-2] if len(cal) >= 2 else None
            if feeds is None or not feeds.get("ok"):
                dec["skip"] = "no_event_feed"
            elif last in missing or prv is None or prv in missing:
                dec["skip"] = "no_bars"
            else:
                lo_ts = datetime.combine(PD.to_date(prv), dtime(15, 30))
                hi_ts = datetime.combine(d_last, dtime(15, 30))
                if PD.window_has_announcements(None, lo_ts, hi_ts) == 0:
                    dec["skip"] = "no_event_feed"
            if "skip" not in dec:
                equity = PD.equity_symbols(st, today)
                members, ustat = (None, "no_equity_list") if not equity else self._universe(st, last, equity)
                out["status"]["universe"] = ustat
                if not members:
                    dec["skip"] = "no_universe"
                else:
                    events = PD.results_events()
                    lo_s, hi_s = PD.ymd(PD.to_date(prv) - timedelta(days=7)), PD.ymd(d_last + timedelta(days=2))
                    events = [e for e in events if lo_s * 10000 <= E.ts_stamp(e["first_ts"]) < hi_s * 10000]
                    syms = {e["symbol"] for e in events}
                    eq = PD.read_bhav(cal[-BARS_SESSIONS] - 1, last)
                    if eq is None:
                        dec["skip"] = "no_bars"
                    else:
                        eq = eq[eq["symbol"].isin(syms)]
                        bars, excl, applied = PD.build_bars(eq, actions, cal[-BARS_SESSIONS:])
                        ni = {k: v[1] for k, v in out["nifty"].items()}
                        if not (ni.get(last) and ni.get(prv)):
                            dec["skip"] = "no_nifty"
                        dec.update({"members": members, "events": events, "bars": bars, "exclusions": excl,
                                    "nifty": ni, "calendar": cal + [s for s in [self._next_session(last)] if s],
                                    "ca_applied": applied})
        keep_from = PD.sessions_between(d_last - timedelta(days=200), d_last)[-PD.KEEP_BHAV_SESSIONS:][0] \
            if PD.sessions_between(d_last - timedelta(days=200), d_last) else last
        if held:
            keep_from = min(keep_from, min(held.values()))
        out["status"]["pruned"] = PD.prune_bhav(keep_from)
        PD.save_state(st)
        return out

    def _universe(self, st: dict, day: int, equity: set) -> tuple[Optional[set], str]:
        d = PD.to_date(day)
        month = day // 100
        if not (st.get("universe") or {}).get(str(month)):
            cal = PD.sessions_between(d - timedelta(days=130), d, include_special=True)
            plain = [s for s in cal if not PD._is_special(s)]
            first = PD.month_first_session(day, plain)
            if first is not None:
                PD.ensure_bhav([s for s in cal if s < first][-60:] + [first])
        return PD.universe_for(day, st, equity, TOP_N)

    # ── the evening job ──
    @serialized
    async def evening(self, now: Optional[datetime] = None) -> dict:
        now = now or self._now()
        out: dict = {"managed": [], "closed": [], "planned": []}
        if not self.wanted():
            return {"skipped": "disabled"}
        if settings.trading_mode != "PAPER":
            self._skip("*", "live_mode_refused", stage="evening")
            return {"skipped": "live_mode_refused"}
        last = self._last_complete_session(now)
        if last is None:
            return {"skipped": "no_session"}
        out["session"] = last
        if not self._positions:
            self.processed_through = max(self.processed_through, last)     # nothing to manage up to `last`
        elif not self.processed_through:
            first = min(int(p.meta.get("entry_day") or last) for p in self.positions())
            self.processed_through = PD.ymd(PD.to_date(first) - timedelta(days=1))
        todo = [s for s in PD.sessions_between(PD.to_date(self.processed_through) + timedelta(days=1),
                                               PD.to_date(last))] if self.processed_through < last else []
        deciding = (self.enabled_by_setting() and self.last_decision.get("day") != last
                    and self._decision_open(last, now))
        if self.enabled_by_setting() and self.last_decision.get("day") != last and not deciding:
            self._skip("*", "decision_window_missed", stage="decision")
            self.last_decision = {"day": last, "entries": [], "skips": {"*": "decision_window_missed"}}
        held = {p.symbol: int(p.meta.get("entry_day") or last) for p in self.positions()}
        if not todo and not deciding:
            return {**out, "skipped": "nothing_due"}
        try:
            g = await asyncio.to_thread(self._gather, now, last, todo, deciding, held)
        except Exception as exc:
            self._error("gather", exc)
            return {**out, "skipped": "gather_failed"}
        self.inputs = {"at": now.isoformat(timespec="seconds"), "session": last,
                       "bhav_missing": g["status"].get("bhav_missing"),
                       "feeds": g["status"].get("feeds"), "universe": g["status"].get("universe")}
        for pos in self.positions():
            new = (g.get("renames") or {}).get(pos.symbol)
            if new:
                self._rename(pos, new)
        # 1. positions: every unprocessed session in order (stop, clock exit, corporate actions)
        for s in todo:
            if self._positions and s not in g["rows"]:
                self.flags["no_bars"] = str(s)
                self._skip("*", f"no_bars:{s}", stage="manage")
                break
            self._manage_session(s, g, late=(s < last))
            self.processed_through = s
        self._save_book(force=True)
        # 2. tomorrow's entries
        if deciding and self.processed_through >= last:
            self._decide(last, g.get("decision") or {}, now)
        elif deciding:
            self._skip("*", "positions_not_current", stage="decision")
        out["planned"] = [e["symbol"] for e in (self.pending.get("entries") or [])] \
            if self.pending.get("decision_day") == last else []
        return out

    def _manage_session(self, s: int, g: dict, late: bool) -> None:
        rows = g["rows"].get(s, {})
        nifty = g.get("nifty") or {}
        for pos in list(self.positions()):
            ed = int(pos.meta.get("entry_day") or 0)
            if s < ed or not self._still_open(pos):
                continue
            r = rows.get(pos.symbol)
            if not r:
                continue                                  # no bar: the lab walks the stock's own bars
            if s == ed and nifty.get(ed):
                pos.meta["nifty_open_entry"] = nifty[ed][0]
            # corporate actions (live-only guard; the lab adjusted bonus / split and dropped the rest)
            f, status = (g.get("factors") or {}).get((pos.symbol, s), (None, None))
            pc = r.get("prev_close_stored")
            gap = (r["open"] / pc) if pc else None
            if status == "applied" and f is not None and f < 1.0 and s > ed:
                self._apply_split(pos, float(f), s)
            elif status is not None and s > ed and (status.startswith("applied") or status == "not_applied_scheme"):
                # book the actual (unadjusted) close: a back-adjustment factor is not an executable price; any
                # entitlement (demerged shares, rights) is NOT booked, so the paper ledger stays conservative
                px = r["close"]
                self._exit(pos, px, s, "corporate_action_exit", nifty, kind="corporate_action_exit",
                           note=f"corporate action {status} on {s} (factor {f}): exit at the actual close; "
                                f"entitlement not booked (lab drops such trades)")
                continue
            elif gap is not None and not (0.6 <= gap <= 1.67) and s > ed:
                self._deviation("unexplained_gap", pos.symbol, f"{s} open/prev close {gap:.3f}")
            held_days = (g.get("traded") or {}).get(pos.symbol) or {}
            days = [x for x in held_days.get("days", []) if x <= s]
            if not held_days.get("complete", False) or s not in days:
                days = sorted(set(days) | {s})
                if not held_days.get("complete", False):
                    self.flags["hold_count_approx"] = str(s)
            n = len(days)
            pos.meta.update({"held_sessions": n, "last_close": r["close"], "last_day": s})
            res = hold_step(float(pos.meta["stop"]), n, (r["open"], r["high"], r["low"], r["close"]))
            if res is None:
                continue
            reason, px = res
            if late:
                self._deviation("late_booking", pos.symbol, f"{reason} of {s} booked late")
            self._exit(pos, px, s, reason, nifty, kind="normal",
                       note=("catastrophe stop" if reason == "stop" else f"clock exit: close of session {n}")
                       + (f" ({s}, booked late)" if late else f" ({s})"))

    def _exit(self, pos: ValidatedPosition, px: float, s: int, reason: str, nifty: dict, kind: str, note: str) -> None:
        pos.meta.update({"exit_day": s, "exit_reason": reason,
                         "nifty_close_exit": (nifty.get(s) or (None, None))[1]})
        fk = "STOP" if reason == "stop" else ("RECONCILE" if kind != "normal" else "CLOSE")
        oid, fill = paper_fill_at(pos.symbol, pos.exchange, "SELL", pos.qty, pos.product, round(float(px), 2),
                                  EXIT_TAG, fk)
        if kind != "normal":
            self._deviation(kind, pos.symbol, note)
        self._close(pos, fill, note, kind=kind)

    def _rename(self, pos: ValidatedPosition, new: str) -> None:
        """NSE renamed a held stock: the position and its paper row follow the new symbol (live-only bookkeeping)."""
        from agents.validated_base import paper_row
        row = paper_row(pos.symbol, pos.product)
        old = pos.symbol
        if row is not None and paper_row(new, pos.product) is None:
            row["tradingsymbol"] = new
        pos.meta.setdefault("renamed_from", old)
        pos.symbol = new
        self._deviation("symbol_renamed", new, f"{old} -> {new}")
        self._save_book(force=True)

    def _apply_split(self, pos: ValidatedPosition, f: float, s: int) -> None:
        """Bonus / split confirmed on session s: the holder now has qty / f shares (free shares at ₹0), the stop
        and the average price scale by f — the lab's back-adjusted series."""
        new_qty = int(round(pos.qty / f))
        extra = new_qty - pos.qty
        if extra > 0:
            paper_fill_at(pos.symbol, pos.exchange, "BUY", extra, pos.product, 0.0, CA_TAG, "CORPORATE_ACTION")
        old = (pos.qty, pos.entry_px, pos.meta.get("stop"))
        pos.entry_px = round(pos.entry_px * pos.qty / new_qty, 4)
        pos.entry_ref = round(pos.entry_ref * f, 4)
        pos.qty = new_qty
        pos.meta["stop"] = float(pos.meta["stop"]) * f
        pos.meta.setdefault("ca_adjustments", []).append({"day": s, "factor": f, "before": list(old)})
        self._log("manage", pos.symbol, "corporate_action_adjusted", f"factor {f:.4f}: qty {old[0]} -> {new_qty}")
        self._save_book(force=True)

    def _decide(self, last: int, dec: dict, now: datetime) -> None:
        why = self.entry_block_reason()
        if why:
            self._skip("*", why, stage="decision")
            return
        if dec.get("skip"):
            self.flags[dec["skip"]] = str(last)
            self._skip("*", dec["skip"], stage="decision")
            self.last_decision = {"day_attempted": last, "skip": dec["skip"], "at": now.isoformat(timespec="seconds")}
            return
        holding = {p.symbol for p in self.positions()}
        entries, skips = plan_pead(last, dec["calendar"], dec["events"], dec["bars"], dec["nifty"], dec["members"],
                                   holding, dec["exclusions"])
        for s, w in skips.items():
            self._skip(s, w, stage="decision", log=w not in ("not_member", "below_threshold"))
        self.pending = {"decision_day": last, "entry_day": entries[0].entry_day if entries else self._next_session(last),
                        "entries": [{"symbol": e.symbol, "ar": round(e.ar, 6)} for e in entries],
                        "made": now.isoformat(timespec="seconds")}
        self.last_decision = {"day": last, "entries": [(e.symbol, round(e.ar * 100, 2)) for e in entries],
                              "skips": dict(list(skips.items())[:200]), "candidates": len(skips) + len(entries),
                              "universe": len(dec["members"]), "events": len(dec["events"])}
        for e in entries:
            self._log("decision", e.symbol, "planned", f"AR {e.ar * 100:+.2f}% -> buy at the {e.entry_day} open")
        self._save_book(force=True)
        logger.info("[{}] decision {}: {} planned for {}", self.name, last, len(entries), self.pending["entry_day"])

    # ── 09:15:10 entry ──
    @serialized
    async def enter_at_open(self, q: Optional[dict] = None) -> dict:
        now = self._now()
        today = PD.ymd(now.date())
        pend = self.pending or {}
        out = {"opened": [], "skipped": {}}
        if not pend or int(pend.get("entry_day") or 0) != today or not pend.get("entries"):
            return out
        if self.done_today("entry"):
            return out
        why = self.entry_block_reason()
        if why:
            self._skip("*", why)
            self.mark_done("entry")
            self.pending = {}
            return {**out, "skipped": {"*": why}}
        if now.time().replace(tzinfo=None) >= ENTRY_DEADLINE:
            self._deviation("entry_window_missed", "*", f"{len(pend['entries'])} planned entries not taken")
            self._skip("*", "entry_window_missed")
            self.mark_done("entry")
            self.pending = {}
            return {**out, "skipped": {"*": "entry_window_missed"}}
        held = {p.symbol for p in self.positions()}
        plan = [PeadEntry(e["symbol"], int(pend["decision_day"]), today, float(e["ar"])) for e in pend["entries"]
                if e["symbol"] not in held]
        opens: dict = {}
        while True:
            want = [e.symbol for e in plan if not float(opens.get(e.symbol) or 0) > 0]
            if want:
                qq = q if q is not None else await asyncio.to_thread(VD.quotes, [f"NSE:{s}" for s in want])
                q = None
                for s in want:
                    o = float(((qq or {}).get(s) or {}).get("open") or 0.0)
                    if o > 0:
                        opens[s] = o
            want = [e.symbol for e in plan if not float(opens.get(e.symbol) or 0) > 0]
            if not want or self._now().time().replace(tzinfo=None) >= ENTRY_DEADLINE:
                break
            await asyncio.sleep(ENTRY_RETRY_S)
        gross = sum(p.entry_px * p.qty for p in self.positions())
        take, skips = apply_caps(plan, opens, len(self.positions()), gross, self.max_positions(), self.max_gross(),
                                 NOTIONAL * self._scale())
        from sebi_compliance import sebi_compliance
        for e, qty, px in take:
            ok, _algo, reason = sebi_compliance.pre_order_check(
                strategy=self.name, symbol=e.symbol, exchange="NSE", transaction_type="BUY", quantity=qty,
                order_type="MARKET", price_at_signal=px, signal_source=f"validated:{self.LAB_ID}", regime="n/a")
            if not ok:
                skips[e.symbol] = f"sebi: {reason}"
                continue
            try:
                oid, fill = paper_fill_at(e.symbol, "NSE", "BUY", qty, "CNC", px, ENTRY_TAG, "AUCTION")
            except Exception as exc:
                self._error(f"entry {e.symbol}", exc)
                skips[e.symbol] = "order_failed"
                continue
            sebi_compliance.record_order_id(self.name, e.symbol, oid)
            pt = PlannedTrade(e.symbol, 1, qty, px, "09:15", "close of session 60", e.ar)
            self._open(pt, "CNC", "NSE", fill, oid, overnight=True,
                       meta={"decision_day": e.decision_day, "entry_day": today, "ar": round(e.ar, 6),
                             "fill_lab": round(entry_fill_lab(px), 4), "stop": round(stop_price(px), 4),
                             "held_sessions": 0})
            out["opened"].append(e.symbol)
        for s, w in skips.items():
            self._skip(s, w)
        out["skipped"] = skips
        self.mark_done("entry")
        self.pending = {}
        self._save_book(force=True)
        logger.info("[{}] entries at the {} open: {} of {} planned", self.name, today, len(out["opened"]), len(plan))
        return out

    # ── restart catch-up ──
    @serialized
    async def catch_up(self, now: Optional[datetime] = None) -> dict:
        now = now or self._now()
        today, t = now.date(), now.time().replace(tzinfo=None)
        out: dict = {}
        self.auto_start(today)
        if not self.wanted():
            return out
        if (self.pending or {}).get("entry_day") == PD.ymd(today) and ENTRY_T <= t < ENTRY_DEADLINE \
                and not self.done_today("entry"):
            out["entry"] = await self.enter_at_open()
        if t >= EVENING_FROM or t < DECISION_CUTOFF or t >= ENTRY_DEADLINE:
            out["evening"] = await self.evening(now)
        return out

    # ── status ──
    def live_readiness(self) -> dict:
        rows = self.ledger_rows()
        st = self.live_stats(rows)
        exc = [float(r["excess_vs_nifty"]) for r in rows if r.get("excess_vs_nifty") is not None]
        mean_exc = round(100 * sum(exc) / len(exc), 3) if exc else None
        ready = (st.get("n", 0) >= READINESS["min_trades"] and (st.get("net") or 0) > 0
                 and (st.get("pf") or 0) >= READINESS["min_pf"] and mean_exc is not None and mean_exc > 0)
        return {"criterion": f">= {READINESS['min_trades']} closed paper trades, net > 0, PF >= {READINESS['min_pf']} "
                             "AND mean excess return vs NIFTY over the same holding windows (NIFTY open of the entry "
                             "session -> close of the exit session) > 0",
                "paper": st, "excess_vs_nifty": {"n": len(exc), "mean_pct": mean_exc},
                "ready": bool(ready), "never_switches_to_live": True, "candidate": True, "proven": False,
                "note": "read-only: a CANDIDATE (DSR 0.51 at N = 9); meeting the bar is a reason to re-test, "
                        "not to trade real money"}

    def get_status(self) -> dict:
        st = super().get_status()
        st.update({"candidate": True, "proven": False, "pending": self.pending, "last_decision": self.last_decision,
                   "processed_through": self.processed_through, "inputs": self.inputs,
                   "caps": {"max_positions": self.max_positions(), "max_gross": self.max_gross(),
                            "notional": NOTIONAL * self._scale()},
                   "open_value": round(sum(p.entry_px * p.qty for p in self.positions()), 2),
                   "flag": "VALIDATED_PEAD_ENABLED", "flag_on": self.enabled_by_setting(),
                   "next_jobs": {"evening": "19:35/20:35/21:35/22:35", "morning_retry": "08:20",
                                 "entry": "09:15:10"} if self.wanted() else {}})
        return st

    def _mark(self, p: ValidatedPosition) -> Optional[float]:
        px = float(p.meta.get("last_close") or 0.0)
        return round(p.sign * (px - p.entry_px) * p.qty, 2) if px > 0 else None

    def export_book(self) -> dict:
        return {**super().export_book(), "pending": dict(self.pending), "processed_through": self.processed_through,
                "last_decision": {k: v for k, v in self.last_decision.items() if k != "skips"}}

    def import_book(self, d: dict, closed: set, stale: bool) -> None:
        super().import_book(d, closed, stale)
        d = d or {}
        if d.get("pending"):
            self.pending = dict(d["pending"])
        self.processed_through = max(self.processed_through, int(d.get("processed_through") or 0))
        if d.get("last_decision"):
            self.last_decision = dict(d["last_decision"])
