"""
regime_v2.py — today's market type, known in time to act on it, and what that means for each strategy.

A day is one of TREND_UP, TREND_DOWN, RANGE, VOLATILE (judged against NIFTY's normal range over the last
20 sessions, so a 1 % day means the same in a calm and a wild month). Two reads per day:

  pre-open  (09:08)  gap, NIFTY vs its 20-day EMA, yesterday's range and type, India VIX level and change
  first hour (10:15) the first hour's range vs normal, how directional it was, its direction

Nothing here is hand-tuned. The research study (scratchpad research/regime_study.py) learns, on the first half
of 12 months of sessions, which feature values go with which day type and keeps a rule only when it beats the
naive "always the most common type" guess on the second half; it writes data/regime_model.json. A second
study replays every strategy and records its result per *predicted* type (what this module will actually
say live), in each half, in data/regime_scorecard.json.

regime_factor(strategy, regime) turns that into sizing: a strategy that made money for this predicted type
in both halves trades up to settings.regime_boost; one that lost in both halves trades at
settings.regime_cut; anything in between, too few trades, no model or no scorecard → 1.0. So without the
research files the system behaves exactly as before.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from loguru import logger

from config import settings
from ist_clock import now_ist

TYPES = ("TREND_UP", "TREND_DOWN", "RANGE", "VOLATILE")
_DATA = Path(os.getenv("PERFORMANCE_FILE") or Path(__file__).resolve().parent / "data" / "performance.json").parent
MODEL_FILE = Path(os.getenv("REGIME_MODEL_FILE") or _DATA / "regime_model.json")
SCORECARD_FILE = Path(os.getenv("REGIME_SCORECARD_FILE") or _DATA / "regime_scorecard.json")

_state: dict = {}          # {"date", "preopen": {...}, "first_hour": {...}}


# ── pure feature functions (shared with the research study) ──────────────────
def day_stats(bars: list) -> tuple[float, float, float, float, float, float]:
    """(open, close, high, low, range %, efficiency) of a list of [ts, o, h, l, c, ...] bars."""
    o, c = bars[0][1], bars[-1][4]
    hi, lo = max(b[2] for b in bars), min(b[3] for b in bars)
    return o, c, hi, lo, (hi / lo - 1) * 100, (abs(c - o) / (hi - lo) if hi > lo else 0.0)


def label(bars: list, avg_rng: float) -> str:
    """What a session (or part of one) turned out to be."""
    o, c, hi, lo, rng, eff = day_stats(bars)
    move = (c / o - 1) * 100
    if rng > 1.4 * avg_rng:
        return "VOLATILE"
    if eff >= 0.5 and abs(move) >= 0.4 * avg_rng:
        return "TREND_UP" if move > 0 else "TREND_DOWN"
    return "RANGE"


def preopen_features(prev_days: list[list], today_open: float, vix: Optional[float], vix_prev: Optional[float]) -> dict:
    """prev_days: the last ≥21 completed sessions, each a list of bars (daily bars work: one bar per day)."""
    last20 = [day_stats(d) for d in prev_days[-20:]]
    avg_rng = sum(s[4] for s in last20) / len(last20)
    closes = [s[1] for s in last20]
    ema = closes[0]
    for x in closes[1:]:
        ema += (x - ema) * 2 / 21
    before = [day_stats(d) for d in prev_days[-21:-1]]
    return {"avg_rng": avg_rng,
            "gap": (today_open / closes[-1] - 1) * 100 / avg_rng,
            "vs_ema20": (closes[-1] / ema - 1) * 100,
            "prev_rng": last20[-1][4] / avg_rng,
            "prev_label": label(prev_days[-1], sum(s[4] for s in before) / len(before) if before else avg_rng),
            "vix": vix, "vix_chg": (vix / vix_prev - 1) * 100 if vix and vix_prev else None}


def first_hour_features(first_hour_bars: list, avg_rng: float) -> dict:
    o, c, hi, lo, rng, eff = day_stats(first_hour_bars)
    return {"fh_rng": rng / avg_rng, "fh_eff": eff, "fh_dir": (c / o - 1) * 100 / avg_rng}


# ── the learned model and the scorecard ──────────────────────────────────────
def _read(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as exc:
        logger.warning("[regime] cannot read {}: {}", path, exc)
    return {}


def classify(features: dict, stage: str, model: Optional[dict] = None) -> Optional[str]:
    """The learned rule for this stage ("preopen" | "first_hour") applied to the features, or None when the
    study found no rule that beats the baseline out of sample (or a feature is missing)."""
    m = (model if model is not None else _read(MODEL_FILE)).get(stage)
    if not m:
        return None
    f = m["feature"]
    if f == "prev_label":
        return m["rule"].get(features.get("prev_label"))
    v = features.get(f)
    if not isinstance(v, (int, float)):
        return None
    a, b = m["cuts"]
    return m["rule"].get(str(0 if v < a else 1 if v < b else 2))


def regime_factor(strategy: str, regime: Optional[str], card: Optional[dict] = None) -> tuple[float, str]:
    """Size multiplier for `strategy` on a day predicted as `regime`, from the per-half scorecard."""
    if not settings.use_regime_routing or not regime:
        return 1.0, "no regime read"
    c = (card if card is not None else _read(SCORECARD_FILE)).get(strategy, {}).get(regime)
    if not c or c.get("trades", 0) < settings.regime_min_trades:
        return 1.0, f"too little evidence for {strategy} on {regime} days"
    h1, h2 = c.get("h1", 0.0), c.get("h2", 0.0)
    if h1 > 0 and h2 > 0:
        return settings.regime_boost, f"{strategy} made money on {regime} days in both halves (₹{h1:,.0f} / ₹{h2:,.0f})"
    if h1 < 0 and h2 < 0:
        return settings.regime_cut, f"{strategy} lost on {regime} days in both halves (₹{h1:,.0f} / ₹{h2:,.0f})"
    return 1.0, f"{strategy} mixed on {regime} days (₹{h1:,.0f} / ₹{h2:,.0f})"


# ── live reads ───────────────────────────────────────────────────────────────
def _daily_as_bars(rows: list[dict]) -> list[list]:
    return [[[str(r["date"]), r["open"], r["high"], r["low"], r["close"]]] for r in rows]


def refresh(stage: str) -> dict:
    """Compute today's read for `stage` from live data and remember it for the rest of the day."""
    global _state
    today = now_ist().date().isoformat()
    if _state.get("date") != today:
        _state = {"date": today}
    out: dict = {"stage": stage, "at": now_ist().strftime("%H:%M"), "regime": None, "features": {}}
    try:
        from daily_data import _completed_days
        from tick_engine import tick_engine
        nifty = _completed_days("NIFTY 50", 0.0)
        if len(nifty) < 22:
            out["why"] = "not enough NIFTY history"
            _state[stage] = out
            return out
        vix_rows = _completed_days("INDIA VIX", 0.0)
        vix_prev = vix_rows[-1]["close"] if vix_rows else None
        vix_prev2 = vix_rows[-2]["close"] if len(vix_rows) > 1 else None
        tick, _ = tick_engine.latest("NIFTY 50")
        buf = tick_engine._bufs_1min.get("NIFTY 50")
        today_bars = [[c.ts.isoformat(), c.open, c.high, c.low, c.close] for c in (buf.candles() if buf else [])
                      if c.ts.date().isoformat() == today]
        today_open = today_bars[0][1] if today_bars else (tick.open if tick and tick.open else (tick.ltp if tick else 0))
        if not today_open:
            out["why"] = "no NIFTY price yet"
            _state[stage] = out
            return out
        # pre-open uses yesterday's VIX change (today's VIX close is not known yet)
        feats = preopen_features(_daily_as_bars(nifty), today_open, vix_prev, vix_prev2)
        if stage == "first_hour":
            fh = [b for b in today_bars if b[0][11:16] < "10:15"]
            if len(fh) < 30:
                out["why"] = f"only {len(fh)} first-hour bars"
                _state[stage] = out
                return out
            feats.update(first_hour_features(fh, feats["avg_rng"]))
        out["features"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in feats.items()}
        out["regime"] = classify(feats, stage)
        if out["regime"] is None:
            out["why"] = "no validated rule for this stage" if not _read(MODEL_FILE).get(stage) else "feature missing"
    except Exception as exc:
        logger.warning("[regime] {} read failed: {}", stage, exc)
        out["why"] = f"error: {str(exc)[:80]}"
    _state[stage] = out
    logger.info("[regime] {} → {} {}", stage, out["regime"], out.get("why", ""))
    return out


def current() -> dict:
    """Today's best read: the first-hour one once it exists, else the pre-open one."""
    if _state.get("date") != now_ist().date().isoformat():
        return {"regime": None, "stage": None}
    for stage in ("first_hour", "preopen"):
        r = _state.get(stage)
        if r and r.get("regime"):
            return r
    return _state.get("first_hour") or _state.get("preopen") or {"regime": None, "stage": None}


def factor(strategy: str) -> float:
    return regime_factor(strategy, current().get("regime"))[0]


def status() -> dict:
    card = _read(SCORECARD_FILE)
    reg = current().get("regime")
    return {"today": _state if _state.get("date") == now_ist().date().isoformat() else {},
            "current": current(), "routing_enabled": settings.use_regime_routing,
            "model": _read(MODEL_FILE), "scorecard_strategies": sorted(card),
            "factors": {s: regime_factor(s, reg, card) for s in ("intraday", "scalping", "swing", "fno")}}


def reset() -> None:
    global _state
    _state = {}
