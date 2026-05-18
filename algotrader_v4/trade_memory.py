"""
trade_memory.py
After every closed trade, Claude Haiku analyzes what happened.
Builds a rolling knowledge base of what setups work in which market conditions.
Knowledge is fed back to the trade gate as institutional memory.
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import anthropic
from loguru import logger

from config import settings
from agents.base_agent import send_telegram


# ── Storage ───────────────────────────────────────────────────────────────────
MEMORY_FILE = Path("logs/trade_memory.jsonl")
MAX_ENTRIES = 500


# ── Anthropic client (lazy singleton) ────────────────────────────────────────
_client: Optional[anthropic.AsyncAnthropic] = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_log_dir() -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)


def _time_of_day(dt: Optional[datetime] = None) -> str:
    """Classify current (or given) time into a named market session."""
    t = (dt or datetime.now()).time()
    if t <= datetime.strptime("11:30", "%H:%M").time():
        return "morning"
    if t <= datetime.strptime("13:30", "%H:%M").time():
        return "midday"
    return "afternoon"


def _build_setup_summary(trade_record: Any, market_context: Optional[dict]) -> str:
    """
    Derive a plain-English setup summary from whatever indicator fields are
    present in *trade_record* or *market_context*.  Gracefully omits missing data.
    """
    parts: list[str] = []

    def _get(*keys: str) -> Any:
        """Try each key in trade_record first, then market_context."""
        for k in keys:
            v = None
            if hasattr(trade_record, k):
                v = getattr(trade_record, k)
            elif isinstance(trade_record, dict):
                v = trade_record.get(k)
            if v is None and isinstance(market_context, dict):
                v = market_context.get(k)
            if v is not None:
                return v
        return None

    rsi = _get("rsi_14", "rsi")
    if rsi is not None:
        parts.append(f"RSI={round(float(rsi), 1)}")

    ema9  = _get("ema9",  "ema_9")
    ema21 = _get("ema21", "ema_21")
    if ema9 and ema21:
        cross = "above" if float(ema9) > float(ema21) else "below"
        parts.append(f"EMA9 {cross} EMA21")

    macd_hist = _get("macd_hist", "macd_histogram")
    if macd_hist is not None:
        parts.append(f"MACD hist={round(float(macd_hist), 4)}")

    vwap    = _get("vwap")
    ltp     = _get("entry_price", "ltp")
    if vwap and ltp:
        pos = "above" if float(ltp) > float(vwap) else "below"
        parts.append(f"price {pos} VWAP")

    vol_ratio = _get("volume_ratio", "vol_ratio")
    if vol_ratio is not None:
        parts.append(f"vol_ratio={round(float(vol_ratio), 2)}x")

    trend = _get("trend")
    if trend:
        parts.append(f"trend={trend}")

    return ", ".join(parts) if parts else "indicators not available"


def _extract_field(trade_record: Any, *keys: str, default: Any = None) -> Any:
    """Pull the first matching key from a dataclass or dict trade record."""
    for k in keys:
        v = None
        if hasattr(trade_record, k):
            v = getattr(trade_record, k)
        elif isinstance(trade_record, dict):
            v = trade_record.get(k)
        if v is not None:
            return v
    return default


def _trim_file_to_max_entries() -> None:
    """Keep only the last MAX_ENTRIES lines in MEMORY_FILE."""
    try:
        lines = MEMORY_FILE.read_text(encoding="utf-8").splitlines()
        if len(lines) > MAX_ENTRIES:
            MEMORY_FILE.write_text(
                "\n".join(lines[-MAX_ENTRIES:]) + "\n", encoding="utf-8"
            )
    except Exception as exc:
        logger.debug("[memory] trim error: {}", exc)


# ── Core record function ──────────────────────────────────────────────────────

async def record_trade(
    trade_record: Any,
    market_context: Optional[dict] = None,
) -> None:
    """
    Analyse a closed trade with Claude Haiku and persist the insight to JSONL.

    *trade_record* may be a dataclass or a plain dict — all field access uses
    _extract_field() which handles both.  Never raises.
    """
    try:
        _ensure_log_dir()

        # ── Extract core fields ───────────────────────────────────────
        symbol       = _extract_field(trade_record, "symbol",       default="UNKNOWN")
        strategy     = _extract_field(trade_record, "strategy",     default="unknown")
        action       = _extract_field(trade_record, "action", "side", "transaction_type",
                                      default="UNKNOWN")
        entry_price  = _extract_field(trade_record, "entry_price",  default=0.0)
        exit_price   = _extract_field(trade_record, "exit_price",   default=0.0)
        pnl_pct      = _extract_field(trade_record, "pnl_pct", "pnl_percent", default=0.0)
        hold_bars    = _extract_field(trade_record, "hold_bars", "bars_held",  default=0)
        exit_reason  = _extract_field(trade_record, "exit_reason",  default="unknown")
        regime       = _extract_field(trade_record, "regime",       default="UNKNOWN")
        trade_time   = _extract_field(trade_record, "entry_time", "timestamp", default=None)

        tod = _time_of_day(trade_time if isinstance(trade_time, datetime) else None)
        setup_summary = _build_setup_summary(trade_record, market_context)

        outcome = "WIN" if float(pnl_pct) >= 0 else "LOSS"

        # ── Ask Claude Haiku for insight ──────────────────────────────
        insight, lesson = await _haiku_analyse(
            symbol=symbol, strategy=strategy, action=action,
            entry_price=entry_price, exit_price=exit_price,
            pnl_pct=pnl_pct, hold_bars=hold_bars,
            exit_reason=exit_reason, regime=regime,
            time_of_day=tod, setup_summary=setup_summary,
        )

        # ── Build and persist the memory entry ───────────────────────
        entry = {
            "ts":           datetime.now().isoformat(),
            "symbol":       symbol,
            "strategy":     strategy,
            "action":       action,
            "entry_price":  entry_price,
            "exit_price":   exit_price,
            "pnl_pct":      round(float(pnl_pct), 4),
            "hold_bars":    hold_bars,
            "exit_reason":  exit_reason,
            "regime":       regime,
            "time_of_day":  tod,
            "outcome":      outcome,
            "setup_summary": setup_summary,
            "insight":      insight,
            "lesson":       lesson,
        }

        with MEMORY_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

        _trim_file_to_max_entries()

        logger.info(
            "[memory] {} {} {} {:.2f}% | {}", outcome, action, symbol, float(pnl_pct), insight
        )

    except Exception as exc:
        logger.warning("[memory] record_trade error: {}", exc)


async def _haiku_analyse(
    symbol: str, strategy: str, action: str,
    entry_price: Any, exit_price: Any, pnl_pct: Any, hold_bars: Any,
    exit_reason: str, regime: str, time_of_day: str, setup_summary: str,
) -> tuple[str, str]:
    """
    Call Claude Haiku-4.5 to produce a one-sentence insight and a one-sentence
    actionable lesson for this trade.  Returns safe defaults on any error.
    """
    if not settings.anthropic_api_key:
        return ("API key not configured", "No lesson available")

    payload = {
        "symbol":        symbol,
        "strategy":      strategy,
        "action":        action,
        "entry_price":   entry_price,
        "exit_price":    exit_price,
        "pnl_pct":       pnl_pct,
        "hold_bars":     hold_bars,
        "exit_reason":   exit_reason,
        "regime":        regime,
        "time_of_day":   time_of_day,
        "setup_summary": setup_summary,
    }
    prompt = (
        "Analyse this NSE intraday trade and respond ONLY with valid JSON "
        "(no markdown, no text outside JSON):\n"
        '{"insight": "<1 sentence why it worked or failed>", '
        '"lesson": "<1 sentence actionable lesson for future trades>"}\n\n'
        f"Trade data:\n{json.dumps(payload)}"
    )

    try:
        resp = await asyncio.wait_for(
            _get_client().messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=8.0,
        )
        raw = resp.content[0].text.strip().lstrip("```json").rstrip("```").strip()
        d   = json.loads(raw)
        return d.get("insight", "No insight"), d.get("lesson", "No lesson")
    except asyncio.TimeoutError:
        logger.debug("[memory] Haiku timeout for {}", symbol)
        return ("Analysis timed out", "Review setup manually")
    except Exception as exc:
        logger.debug("[memory] Haiku error for {}: {}", symbol, exc)
        return (f"Analysis error: {str(exc)[:60]}", "Review setup manually")


# ── Query helpers ─────────────────────────────────────────────────────────────

def get_recent_insights(n: int = 10) -> list[dict]:
    """
    Return the last *n* memory entries as a list of dicts.
    Returns [] if the file does not exist or is unreadable.
    """
    try:
        if not MEMORY_FILE.exists():
            return []
        lines = MEMORY_FILE.read_text(encoding="utf-8").splitlines()
        entries: list[dict] = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(entries) >= n:
                break
        return list(reversed(entries))
    except Exception as exc:
        logger.debug("[memory] get_recent_insights error: {}", exc)
        return []


def get_regime_insights(regime: str, n: int = 5) -> list[dict]:
    """
    Return the last *n* entries whose *regime* field matches the given value.
    Comparison is case-insensitive.
    """
    try:
        if not MEMORY_FILE.exists():
            return []
        lines = MEMORY_FILE.read_text(encoding="utf-8").splitlines()
        matched: list[dict] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (entry.get("regime") or "").upper() == regime.upper():
                matched.append(entry)
        return matched[-n:]
    except Exception as exc:
        logger.debug("[memory] get_regime_insights error: {}", exc)
        return []


# ── Weekly synthesis ──────────────────────────────────────────────────────────

async def weekly_synthesis() -> str:
    """
    Read the last 100 trade memory entries, ask Claude Sonnet to synthesise them
    into 5 key patterns/lessons, send the result via Telegram, and return the text.

    Returns "" on any error.
    """
    try:
        entries = get_recent_insights(n=100)
        if not entries:
            logger.info("[memory] weekly_synthesis: no entries found")
            return ""

        if not settings.anthropic_api_key:
            logger.info("[memory] weekly_synthesis: no API key")
            return ""

        # Build a compact representation to stay within token budget
        compact = [
            {
                "symbol":      e.get("symbol"),
                "strategy":    e.get("strategy"),
                "action":      e.get("action"),
                "pnl_pct":     e.get("pnl_pct"),
                "outcome":     e.get("outcome"),
                "regime":      e.get("regime"),
                "time_of_day": e.get("time_of_day"),
                "exit_reason": e.get("exit_reason"),
                "insight":     e.get("insight"),
                "lesson":      e.get("lesson"),
            }
            for e in entries
        ]

        prompt = (
            "You are an NSE algorithmic trading analyst reviewing the last 100 completed trades.\n\n"
            "Synthesise these trade outcomes into exactly 5 key patterns or lessons. "
            "Focus on: which setups win, which lose, regime effects, time-of-day patterns, "
            "and specific actionable improvements.\n\n"
            "Respond in plain text, numbered 1–5. Be concise and specific.\n\n"
            f"Trade records:\n{json.dumps(compact, indent=2)}"
        )

        resp = await asyncio.wait_for(
            _get_client().messages.create(
                model="claude-sonnet-4-6",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=30.0,
        )
        synthesis = resp.content[0].text.strip()

        msg = (
            "<b>Weekly Trade Memory Synthesis</b>\n\n"
            f"{synthesis}\n\n"
            f"<i>Based on {len(entries)} trades — {datetime.now().strftime('%Y-%m-%d')}</i>"
        )
        await send_telegram(msg)
        logger.info("[memory] weekly synthesis sent via Telegram")
        return synthesis

    except asyncio.TimeoutError:
        logger.warning("[memory] weekly_synthesis: Sonnet timeout")
        return ""
    except Exception as exc:
        logger.warning("[memory] weekly_synthesis error: {}", exc)
        return ""


__all__ = [
    "record_trade",
    "get_recent_insights",
    "get_regime_insights",
    "weekly_synthesis",
    "MEMORY_FILE",
    "MAX_ENTRIES",
]
