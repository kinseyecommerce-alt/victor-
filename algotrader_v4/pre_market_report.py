"""
pre_market_report.py
Generates a pre-market intelligence briefing at 09:00 AM IST.
Pulls global cues, India indices, FII/DII, sector performance → Claude Sonnet synthesis.
Sends formatted report to Telegram.

Called by platform_scheduler at 09:00 IST (Mon–Fri).
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Optional

import anthropic
import yfinance as yf
from loguru import logger

from config import settings
from agents.base_agent import send_telegram
from market_data import nse_client


# ── Tickers ────────────────────────────────────────────────────────────────────

_GLOBAL_FUTURES: dict[str, str] = {
    "ES=F":      "S&P 500 Fut",
    "NQ=F":      "Nasdaq Fut",
    "CL=F":      "Crude Oil",
    "GC=F":      "Gold",
    "DX-Y.NYB":  "DXY (USD)",
}

_INDIA_INDICES: dict[str, str] = {
    "^NSEI":    "Nifty 50",
    "^NSEBANK": "Bank Nifty",
}

_SECTOR_ETFS: dict[str, str] = {
    "NIFTYBANK.NS":   "Banking",
    "NIFTYIT.NS":     "IT",
    "NIFTYPHARMA.NS": "Pharma",
    "NIFTYAUTO.NS":   "Auto",
    "NIFTYMETAL.NS":  "Metal",
    "NIFTYFMCG.NS":   "FMCG",
}

_NSE_FIIDII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"


# ── Data fetchers ──────────────────────────────────────────────────────────────

def _fetch_yf_change(ticker: str) -> Optional[dict]:
    """Synchronous yfinance fetch — run inside asyncio.to_thread."""
    try:
        df = yf.download(ticker, period="2d", interval="60m", progress=False, auto_adjust=True)
        if df.empty or len(df) < 2:
            return None
        # Flatten multi-index columns if present
        if hasattr(df.columns, "get_level_values"):
            df.columns = df.columns.get_level_values(0)
        closes = df["Close"].dropna()
        if len(closes) < 2:
            return None
        last_close = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2])
        change_pct = round((last_close - prev_close) / prev_close * 100, 2) if prev_close else 0.0
        return {
            "ticker":     ticker,
            "last":       round(last_close, 4),
            "prev":       round(prev_close, 4),
            "change_pct": change_pct,
        }
    except Exception as exc:
        logger.debug("[pre_market] yfinance {} fetch failed: {}", ticker, exc)
        return None


async def _fetch_global_futures() -> dict[str, Optional[dict]]:
    tasks = {
        ticker: asyncio.to_thread(_fetch_yf_change, ticker)
        for ticker in _GLOBAL_FUTURES
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    out: dict[str, Optional[dict]] = {}
    for ticker, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            logger.debug("[pre_market] {} exception: {}", ticker, result)
            out[ticker] = None
        else:
            out[ticker] = result
    return out


async def _fetch_india_indices() -> dict[str, Optional[dict]]:
    tasks = {
        ticker: asyncio.to_thread(_fetch_yf_change, ticker)
        for ticker in _INDIA_INDICES
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    out: dict[str, Optional[dict]] = {}
    for ticker, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            logger.debug("[pre_market] {} exception: {}", ticker, result)
            out[ticker] = None
        else:
            out[ticker] = result
    return out


async def _fetch_fiidii() -> Optional[dict]:
    """Fetch FII/DII net flows from NSE API. Returns None on any failure."""
    try:
        data = await nse_client.get(_NSE_FIIDII_URL)
        if not data:
            return None
        # Response is a list; first element is the latest trading day
        if isinstance(data, list) and data:
            latest = data[0]
        elif isinstance(data, dict):
            rows = data.get("data", data.get("fiiData", []))
            if not rows:
                return None
            latest = rows[0]
        else:
            return None

        fii_net = (
            latest.get("fiiNet")
            or latest.get("FII Net Purchase/Sales")
            or latest.get("fii_net")
        )
        dii_net = (
            latest.get("diiNet")
            or latest.get("DII Net Purchase/Sales")
            or latest.get("dii_net")
        )
        if fii_net is None and dii_net is None:
            return None

        return {
            "fii_net": float(str(fii_net).replace(",", "")) if fii_net is not None else None,
            "dii_net": float(str(dii_net).replace(",", "")) if dii_net is not None else None,
        }
    except Exception as exc:
        logger.debug("[pre_market] FII/DII fetch failed: {}", exc)
        return None


def _fetch_sector_change(ticker: str) -> Optional[dict]:
    """Synchronous sector ETF yesterday-vs-day-before change."""
    try:
        df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 2:
            return None
        if hasattr(df.columns, "get_level_values"):
            df.columns = df.columns.get_level_values(0)
        closes = df["Close"].dropna()
        if len(closes) < 2:
            return None
        yesterday  = float(closes.iloc[-1])
        day_before = float(closes.iloc[-2])
        change_pct = round((yesterday - day_before) / day_before * 100, 2) if day_before else 0.0
        return {"ticker": ticker, "change_pct": change_pct}
    except Exception as exc:
        logger.debug("[pre_market] sector {} fetch failed: {}", ticker, exc)
        return None


async def _fetch_sector_performance() -> dict[str, Optional[dict]]:
    tasks = {
        ticker: asyncio.to_thread(_fetch_sector_change, ticker)
        for ticker in _SECTOR_ETFS
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    out: dict[str, Optional[dict]] = {}
    for ticker, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            logger.debug("[pre_market] {} exception: {}", ticker, result)
            out[ticker] = None
        else:
            out[ticker] = result
    return out


# ── Claude synthesis ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an elite pre-market analyst for Indian equity markets (NSE/BSE).
Your job is to synthesise global cues, India index technicals, FII/DII flows, and sector
rotation data into a concise, actionable pre-market briefing.

Focus on: how global overnight moves translate to India open, where institutional money is
flowing, which sectors to favour or avoid, and concrete stock-level setups worth monitoring.

Output ONLY valid JSON — no markdown fences, no explanatory text outside the JSON:
{
  "sentiment": "bullish|bearish|neutral",
  "sentiment_confidence": <0-100>,
  "key_risks": ["<risk1>", "<risk2>"],
  "sector_focus": [
    {"sector": "<name>", "bias": "long|short|neutral", "reason": "<concise reason>"}
  ],
  "strategy_weights": {"intraday": <0-100>, "scalping": <0-100>, "swing": <0-100>, "fno": <0-100>},
  "stock_setups": [
    {"symbol": "<NSE symbol>", "bias": "long|short", "reason": "<concise reason>"}
  ],
  "summary": "<two sentence market briefing>"
}

RULES:
- strategy_weights must sum to 100.
- Provide 2–4 key_risks, 3–6 sector_focus entries, 2–5 stock_setups.
- If data is missing for a field, still produce the best possible estimate.
- Be specific and actionable — avoid generic statements."""


def _build_claude_payload(
    global_data: dict,
    india_data: dict,
    fiidii: Optional[dict],
    sector_data: dict,
) -> str:
    """Construct the user message for Claude from collected data."""
    now_ist = datetime.now().strftime("%Y-%m-%d %H:%M IST")

    # Global futures
    global_cues = []
    for ticker, label in _GLOBAL_FUTURES.items():
        d = global_data.get(ticker)
        if d:
            arrow = "▲" if d["change_pct"] >= 0 else "▼"
            global_cues.append(
                f"  {label}: {d['last']} ({arrow}{abs(d['change_pct'])}%)"
            )
        else:
            global_cues.append(f"  {label}: data unavailable")

    # India indices
    india_cues = []
    for ticker, label in _INDIA_INDICES.items():
        d = india_data.get(ticker)
        if d:
            arrow = "▲" if d["change_pct"] >= 0 else "▼"
            india_cues.append(
                f"  {label}: {d['last']} ({arrow}{abs(d['change_pct'])}%)"
            )
        else:
            india_cues.append(f"  {label}: data unavailable")

    # FII/DII
    if fiidii:
        fii_str = (
            f"₹{fiidii['fii_net']:+,.0f} Cr"
            if fiidii.get("fii_net") is not None
            else "N/A"
        )
        dii_str = (
            f"₹{fiidii['dii_net']:+,.0f} Cr"
            if fiidii.get("dii_net") is not None
            else "N/A"
        )
        fiidii_str = f"  FII Net: {fii_str} | DII Net: {dii_str}"
    else:
        fiidii_str = "  FII/DII data unavailable"

    # Sector performance
    sector_lines = []
    for ticker, label in _SECTOR_ETFS.items():
        d = sector_data.get(ticker)
        if d:
            arrow = "▲" if d["change_pct"] >= 0 else "▼"
            sector_lines.append(f"  {label}: {arrow}{abs(d['change_pct'])}%")
        else:
            sector_lines.append(f"  {label}: data unavailable")

    payload = (
        f"Pre-Market Data — {now_ist}\n\n"
        f"GLOBAL FUTURES (overnight):\n" + "\n".join(global_cues) + "\n\n"
        f"INDIA INDICES (prior session / ADR-implied):\n" + "\n".join(india_cues) + "\n\n"
        f"FII/DII FLOWS (latest session):\n{fiidii_str}\n\n"
        f"SECTOR ETF PERFORMANCE (1-day change):\n" + "\n".join(sector_lines) + "\n\n"
        f"Please generate the pre-market briefing JSON."
    )
    return payload


async def _call_claude(payload: str) -> Optional[dict]:
    """Call Claude Sonnet and parse the JSON response."""
    if not settings.anthropic_api_key:
        logger.warning("[pre_market] ANTHROPIC_API_KEY not set — skipping Claude synthesis")
        return None
    try:
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        resp = await asyncio.wait_for(
            client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": payload}],
            ),
            timeout=20.0,
        )
        raw = resp.content[0].text.strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except asyncio.TimeoutError:
        logger.warning("[pre_market] Claude call timed out")
        return None
    except json.JSONDecodeError as exc:
        logger.warning("[pre_market] Claude JSON parse error: {}", exc)
        return None
    except Exception as exc:
        logger.warning("[pre_market] Claude call failed: {}", exc)
        return None


# ── Telegram formatting ────────────────────────────────────────────────────────

def _sentiment_emoji(sentiment: str) -> str:
    return {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(sentiment.lower(), "⚪")


def _change_arrow(pct: Optional[float]) -> str:
    if pct is None:
        return ""
    return f"▲{abs(pct):.2f}%" if pct >= 0 else f"▼{abs(pct):.2f}%"


def _format_telegram(
    analysis: dict,
    global_data: dict,
    india_data: dict,
    fiidii: Optional[dict],
    sector_data: dict,
) -> str:
    now_ist = datetime.now().strftime("%d %b %Y, %H:%M IST")
    sentiment    = analysis.get("sentiment", "neutral").capitalize()
    confidence   = analysis.get("sentiment_confidence", 0)
    summary      = analysis.get("summary", "")
    emoji        = _sentiment_emoji(analysis.get("sentiment", "neutral"))

    lines: list[str] = [
        f"<b>Pre-Market Briefing — {now_ist}</b>",
        f"{emoji} <b>Sentiment:</b> {sentiment} ({confidence}%)",
        "",
        f"<b>Summary:</b> {summary}",
        "",
    ]

    # Global cues
    lines.append("<b>Global Cues:</b>")
    for ticker, label in _GLOBAL_FUTURES.items():
        d = global_data.get(ticker)
        val = f"{d['last']}  {_change_arrow(d['change_pct'])}" if d else "N/A"
        lines.append(f"  {label}: {val}")

    lines.append("")
    lines.append("<b>India Indices:</b>")
    for ticker, label in _INDIA_INDICES.items():
        d = india_data.get(ticker)
        val = f"{d['last']}  {_change_arrow(d['change_pct'])}" if d else "N/A"
        lines.append(f"  {label}: {val}")

    # FII/DII
    lines.append("")
    lines.append("<b>FII/DII Flows:</b>")
    if fiidii:
        fii = fiidii.get("fii_net")
        dii = fiidii.get("dii_net")
        fii_str = f"₹{fii:+,.0f} Cr" if fii is not None else "N/A"
        dii_str = f"₹{dii:+,.0f} Cr" if dii is not None else "N/A"
        lines.append(f"  FII Net: {fii_str}  |  DII Net: {dii_str}")
    else:
        lines.append("  Data unavailable")

    # Sector performance
    lines.append("")
    lines.append("<b>Sector Performance:</b>")
    for ticker, label in _SECTOR_ETFS.items():
        d = sector_data.get(ticker)
        val = _change_arrow(d["change_pct"]) if d else "N/A"
        lines.append(f"  {label}: {val}")

    # Strategy weights
    weights = analysis.get("strategy_weights", {})
    if weights:
        lines.append("")
        lines.append("<b>Strategy Weights:</b>")
        lines.append(
            f"  Intraday {weights.get('intraday', 0)}%  |  "
            f"Scalping {weights.get('scalping', 0)}%  |  "
            f"Swing {weights.get('swing', 0)}%  |  "
            f"F&O {weights.get('fno', 0)}%"
        )

    # Key risks
    risks = analysis.get("key_risks", [])
    if risks:
        lines.append("")
        lines.append("<b>Key Risks:</b>")
        for r in risks:
            lines.append(f"  ⚠ {r}")

    # Sector focus
    sectors = analysis.get("sector_focus", [])
    if sectors:
        lines.append("")
        lines.append("<b>Sector Focus:</b>")
        for s in sectors:
            bias_emoji = {"long": "🟢", "short": "🔴", "neutral": "🟡"}.get(
                s.get("bias", "neutral").lower(), "⚪"
            )
            lines.append(
                f"  {bias_emoji} {s.get('sector', '')}: {s.get('reason', '')}"
            )

    # Stock setups
    setups = analysis.get("stock_setups", [])
    if setups:
        lines.append("")
        lines.append("<b>Watch List Setups:</b>")
        for s in setups:
            bias_emoji = "🟢" if s.get("bias", "").lower() == "long" else "🔴"
            lines.append(
                f"  {bias_emoji} <code>{s.get('symbol', '')}</code>  {s.get('reason', '')}"
            )

    lines.append("")
    lines.append("<i>AlgoTrader Pro — Pre-Market Intelligence</i>")
    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────────────

async def generate_pre_market_report() -> dict:
    """
    Fetch all data in parallel, call Claude for synthesis,
    format and send Telegram report.

    Returns the full analysis dict (or partial dict on failure).
    Always resolves — never raises.
    """
    logger.info("[pre_market] starting report generation")

    # Fetch all data sources concurrently; each handles its own failures
    global_data, india_data, fiidii, sector_data = await asyncio.gather(
        _fetch_global_futures(),
        _fetch_india_indices(),
        _fetch_fiidii(),
        _fetch_sector_performance(),
        return_exceptions=False,
    )

    logger.info(
        "[pre_market] data fetched — global={}/{} india={}/{} fiidii={} sectors={}/{}",
        sum(1 for v in global_data.values() if v),
        len(global_data),
        sum(1 for v in india_data.values() if v),
        len(india_data),
        "ok" if fiidii else "unavailable",
        sum(1 for v in sector_data.values() if v),
        len(sector_data),
    )

    # Build Claude context and call
    payload  = _build_claude_payload(global_data, india_data, fiidii, sector_data)
    analysis = await _call_claude(payload)

    if analysis is None:
        # Fallback: minimal analysis so Telegram still gets a data-only report
        analysis = {
            "sentiment": "neutral",
            "sentiment_confidence": 50,
            "key_risks": ["Claude synthesis unavailable — data-only report"],
            "sector_focus": [],
            "strategy_weights": {"intraday": 25, "scalping": 25, "swing": 25, "fno": 25},
            "stock_setups": [],
            "summary": "Pre-market data collected. Claude analysis unavailable.",
        }
        logger.warning("[pre_market] Claude unavailable — sending data-only report")

    # Format and send Telegram message
    message = _format_telegram(analysis, global_data, india_data, fiidii, sector_data)
    await send_telegram(message)
    logger.info(
        "[pre_market] report sent — sentiment={} confidence={}",
        analysis.get("sentiment"),
        analysis.get("sentiment_confidence"),
    )

    return {
        "analysis":    analysis,
        "global_data": global_data,
        "india_data":  india_data,
        "fiidii":      fiidii,
        "sector_data": sector_data,
        "generated_at": datetime.now().isoformat(),
    }
