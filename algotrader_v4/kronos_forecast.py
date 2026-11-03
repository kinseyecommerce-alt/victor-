"""
kronos_forecast.py — short-horizon candlestick forecasts as one more input for the AI.

Kronos (shiyu-coder/Kronos, AAAI 2026) is an open-source foundation model trained on K-line
data from 45 exchanges: give it the last few hundred candles and it generates the next ones.
Its own paper reports modest skill — a little better than chance on direction — so it is
used here as evidence, never as a signal that trades by itself:

  * market_context gets a "forecast" block: expected move over the next N bars, the share of
    sampled paths that end up, and a band (p10–p90), with the source named.
  * GET /forecast/{symbol} shows it; GET /backtest/forecast-skill measures the forecaster's
    directional hit-rate on past bars against a 50% coin so you can see whether it has any
    edge on YOUR symbols before trusting it.

The model is optional. With torch + the Kronos repo installed (KRONOS_DIR) and
KRONOS_MODEL set, the real model runs (CPU is fine for a few symbols; a small GPU for many).
Without it, a labelled statistical fallback runs instead — drift + volatility Monte Carlo —
so every code path, test and dashboard field works and nothing silently pretends to be Kronos.
"""
from __future__ import annotations

import math
import os
import random
import sys
import threading
import time
from datetime import timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from config import settings

_predictor = None
_predictor_err: Optional[str] = None
_lock = threading.Lock()
_cache: dict[tuple, tuple[float, dict]] = {}


# ── model loading (lazy, optional) ────────────────────────────────────────

def _load_predictor():
    """Load Kronos once. Returns the predictor or None (reason in _predictor_err)."""
    global _predictor, _predictor_err
    if _predictor is not None or _predictor_err is not None:
        return _predictor
    with _lock:
        if _predictor is not None or _predictor_err is not None:
            return _predictor
        if not settings.kronos_model:
            _predictor_err = "KRONOS_MODEL not set"
            return None
        try:
            kdir = settings.kronos_dir
            if kdir and kdir not in sys.path:
                sys.path.insert(0, kdir)
            from model import Kronos, KronosTokenizer, KronosPredictor   # the Kronos repo's package
            tok = KronosTokenizer.from_pretrained(settings.kronos_tokenizer)
            mdl = Kronos.from_pretrained(settings.kronos_model)
            _predictor = KronosPredictor(mdl, tok, device=settings.kronos_device, max_context=512)
            logger.info("[kronos] loaded {} on {}", settings.kronos_model, settings.kronos_device)
        except Exception as exc:
            _predictor_err = f"{type(exc).__name__}: {str(exc)[:160]}"
            logger.warning("[kronos] not available ({}); statistical fallback in use", _predictor_err)
        return _predictor


def available() -> dict:
    p = _load_predictor()
    return {"kronos": p is not None, "model": settings.kronos_model or None, "device": settings.kronos_device,
            "reason": _predictor_err, "fallback": "drift+vol Monte Carlo" if p is None else None}


# ── forecasting ───────────────────────────────────────────────────────────

def _bars_df(bars: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{"timestamps": b["ts"], "open": b["open"], "high": b["high"], "low": b["low"],
                        "close": b["close"], "volume": float(b.get("volume") or 0)} for b in bars])
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    return df


def _future_stamps(last_ts, n: int, step_minutes: int) -> pd.Series:
    out, t = [], last_ts
    while len(out) < n:
        t = t + timedelta(minutes=step_minutes)
        if step_minutes < 60 * 24:
            # stay inside the session (09:15–15:30 IST): roll to the next day's open at the close
            if t.time() > pd.Timestamp("15:30").time():
                t = (t + timedelta(days=1)).replace(hour=9, minute=15)
            while t.weekday() >= 5:
                t = t + timedelta(days=1)
        out.append(t)
    return pd.Series(pd.to_datetime(out))


def _kronos(bars: list[dict], horizon: int, step_minutes: int, samples: int) -> Optional[dict]:
    p = _load_predictor()
    if p is None:
        return None
    df = _bars_df(bars[-settings.kronos_lookback:])
    x_ts = pd.Series(pd.to_datetime(df["timestamps"]))
    y_ts = _future_stamps(x_ts.iloc[-1], horizon, step_minutes)
    last = float(df["close"].iloc[-1])
    ends, paths = [], []
    for _ in range(samples):
        pred = p.predict(df=df[["open", "high", "low", "close", "volume", "amount"]], x_timestamp=x_ts,
                         y_timestamp=y_ts, pred_len=horizon, T=settings.kronos_temperature, top_p=0.9,
                         sample_count=1, verbose=False)
        closes = [float(c) for c in pred["close"].tolist()]
        paths.append(closes)
        ends.append(closes[-1])
    return _summarise(last, ends, paths, "kronos:" + (settings.kronos_model.split("/")[-1]))


def _fallback(bars: list[dict], horizon: int, samples: int, seed: Optional[int] = None) -> dict:
    """Drift + volatility Monte Carlo on the last 120 bars' log returns. Labelled; not Kronos."""
    closes = [b["close"] for b in bars[-121:]]
    rets = [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]
    mu = sum(rets) / len(rets) if rets else 0.0
    sd = math.sqrt(sum((r - mu) ** 2 for r in rets) / max(len(rets) - 1, 1)) if len(rets) > 1 else 0.001
    mu *= 0.5                                   # shrink the drift: recent drift is mostly noise
    rng = random.Random(seed if seed is not None else int(closes[-1] * 100) ^ len(bars))
    last = closes[-1]
    ends, paths = [], []
    for _ in range(samples):
        px, path = last, []
        for _ in range(horizon):
            px *= math.exp(mu + rng.gauss(0, sd))
            path.append(px)
        paths.append(path); ends.append(px)
    return _summarise(last, ends, paths, "fallback:drift+vol")


def _summarise(last: float, ends: list[float], paths: list[list[float]], source: str) -> dict:
    ends_sorted = sorted(ends)
    n = len(ends_sorted)
    q = lambda p: ends_sorted[min(n - 1, max(0, int(round(p * (n - 1)))))]
    median_path = [sorted(col)[len(col) // 2] for col in zip(*paths)]
    return {"last": round(last, 2), "expected_move_pct": round((sum(ends) / n / last - 1) * 100, 3),
            "p_up": round(sum(1 for e in ends if e > last) / n, 2),
            "p10_pct": round((q(0.10) / last - 1) * 100, 3), "p50_pct": round((q(0.50) / last - 1) * 100, 3),
            "p90_pct": round((q(0.90) / last - 1) * 100, 3), "samples": n, "source": source,
            "path_p50": [round(x, 2) for x in median_path]}


def model_loaded() -> bool:
    return _predictor is not None


def forecast(symbol: str, bars: Optional[list[dict]] = None, horizon: Optional[int] = None,
             step_minutes: int = 1, samples: Optional[int] = None, use_cache: bool = True,
             cached_only: bool = False) -> dict:
    """Forecast the next `horizon` bars for a symbol from its 1-minute buffer (or given bars)."""
    horizon = horizon or settings.kronos_horizon_bars
    samples = samples or settings.kronos_samples
    if bars is None:
        from tick_engine import tick_engine
        buf = tick_engine._bufs_1min.get(symbol)
        cs = buf.candles() if buf else []
        bars = [{"ts": c.ts, "open": c.open, "high": c.high, "low": c.low, "close": c.close, "volume": c.volume} for c in cs]
    if len(bars) < 30:
        return {"available": False, "reason": f"need ≥ 30 bars, have {len(bars)}", "symbol": symbol}
    key = (symbol, horizon, step_minutes, bars[-1]["ts"], round(bars[-1]["close"], 4))
    if use_cache:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < settings.kronos_cache_s:
            return hit[1]
        if cached_only and model_loaded():        # the real model is slow: never run it on the alert path
            recent = [(t, v) for k, (t, v) in _cache.items() if k[0] == symbol and time.time() - t < settings.kronos_cache_s]
            if recent:
                return max(recent, key=lambda x: x[0])[1]
            return {"available": False, "reason": "forecast not computed yet", "symbol": symbol}
    try:
        out = _kronos(bars, horizon, step_minutes, samples) or _fallback(bars, horizon, samples)
    except Exception as exc:
        logger.warning("[kronos] forecast {} failed: {}", symbol, exc)
        out = _fallback(bars, horizon, samples)
    out.update(available=True, symbol=symbol, horizon_bars=horizon, bar_minutes=step_minutes,
               as_of=str(bars[-1]["ts"]), model_available=_predictor is not None)
    if use_cache:
        if len(_cache) > 256:
            _cache.clear()
        _cache[key] = (time.time(), out)
    return out


def compact(symbol: str, cached_only: bool = False) -> dict:
    """The short form for the AI's market context and alerts."""
    f = forecast(symbol, cached_only=cached_only)
    if not f.get("available"):
        return {"available": False}
    return {k: f[k] for k in ("expected_move_pct", "p_up", "p10_pct", "p90_pct", "horizon_bars", "source")}


async def refresher() -> None:
    """Background: keep a fresh forecast per streamed NSE symbol when the real model is loaded
    (it is too slow for the alert path). The fallback is cheap and computed inline instead."""
    import asyncio
    from tick_engine import tick_engine
    while True:
        try:
            if model_loaded() or (settings.kronos_model and _predictor_err is None):
                syms = [s for s in tick_engine.symbols() if " " not in s and tick_engine._exchange.get(s, "NSE") == "NSE"]
                for s in syms[:settings.kronos_max_symbols]:
                    await asyncio.to_thread(forecast, s)
        except Exception as exc:
            logger.debug("[kronos] refresher: {}", exc)
        await asyncio.sleep(max(60, settings.kronos_cache_s // 2))


# ── skill check: does the forecaster beat a coin on this symbol? ──────────

def skill_check(symbol: str, bars: list[dict], horizon: Optional[int] = None, step: int = 30,
                lookback: int = 300, samples: int = 8) -> dict:
    """Walk forward through past bars: at every `step` bars, forecast `horizon` ahead from the
    bars so far, then compare the predicted direction with what happened. Reports hit-rate,
    a 95% band for a coin flip, and the correlation between expected and realised moves."""
    horizon = horizon or settings.kronos_horizon_bars
    hits, n, pairs = 0, 0, []
    i = max(lookback, 60)
    while i + horizon < len(bars):
        hist = bars[i - lookback:i]
        f = forecast(symbol, bars=hist, horizon=horizon, samples=samples, use_cache=False)
        if f.get("available"):
            exp = f["expected_move_pct"]
            real = (bars[i + horizon - 1]["close"] / bars[i - 1]["close"] - 1) * 100
            if abs(exp) > 0.02:                       # ignore "no opinion" forecasts
                n += 1
                hits += (exp > 0) == (real > 0)
                pairs.append((exp, real))
        i += step
    if n == 0:
        return {"symbol": symbol, "forecasts": 0, "verdict": "not enough bars"}
    hr = hits / n
    se = math.sqrt(0.25 / n)
    corr = None
    if len(pairs) > 3:
        ex = [p[0] for p in pairs]; re_ = [p[1] for p in pairs]
        mx, mr = sum(ex) / len(ex), sum(re_) / len(re_)
        num = sum((a - mx) * (b - mr) for a, b in pairs)
        den = math.sqrt(sum((a - mx) ** 2 for a in ex) * sum((b - mr) ** 2 for b in re_))
        corr = round(num / den, 3) if den > 0 else None
    better = hr > 0.5 + 1.96 * se
    return {"symbol": symbol, "forecasts": n, "hit_rate": round(hr, 3), "coin_flip_95pct_band": [round(0.5 - 1.96 * se, 3), round(0.5 + 1.96 * se, 3)],
            "beats_coin_flip": better, "expected_vs_realised_corr": corr, "horizon_bars": horizon,
            "source": forecast(symbol, bars=bars[-lookback:], horizon=horizon, samples=2, use_cache=False).get("source"),
            "verdict": ("some directional skill on this symbol" if better else
                        "no evidence of skill beyond chance — treat forecasts as noise here")}
