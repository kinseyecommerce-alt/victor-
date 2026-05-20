"""
kite_ticker.py
KiteConnect WebSocket wrapper for real-time tick streaming (LIVE mode only).
Feeds ticks directly into tick_engine._ingest_kite_tick().
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Callable, Optional

from loguru import logger

from config import settings
from tick_engine import Tick


class KiteTicker:
    """
    Wraps KiteConnect KiteTicker WebSocket.
    On each tick, calls `on_tick_callback(symbol, Tick)` as a coroutine.
    """

    def __init__(self) -> None:
        self._token_map: dict[str, int] = {}    # symbol → instrument_token
        self._rev_map:   dict[int, str] = {}    # instrument_token → symbol
        self._kws = None
        self._callback: Optional[Callable] = None
        self._loop:     Optional[asyncio.AbstractEventLoop] = None
        self._connected: bool = False

    @property
    def connected(self) -> bool:
        return self._connected

    def load_instruments(self, symbols: list[str]) -> None:
        try:
            from kite_client import kite_client
            instruments = kite_client.kite.instruments("NSE")
            for inst in instruments:
                sym = inst.get("tradingsymbol", "")
                if sym in symbols:
                    token = inst["instrument_token"]
                    self._token_map[sym]   = token
                    self._rev_map[token]   = sym
            logger.info("KiteTicker: loaded {} instrument tokens ({} requested)",
                        len(self._token_map), len(symbols))
        except Exception as exc:
            logger.warning("KiteTicker: instrument load failed: {}", exc)

    def start(
        self,
        symbols: list[str],
        on_tick_callback: Callable,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._callback = on_tick_callback
        self._loop     = loop
        self.load_instruments(symbols)

        if not self._token_map:
            logger.warning("KiteTicker: no tokens loaded — WebSocket not started")
            return
        if not settings.kite_api_key or not settings.kite_access_token:
            logger.warning("KiteTicker: missing api_key or access_token — WebSocket not started")
            return

        try:
            from kiteconnect import KiteTicker as _KT
            kws = _KT(
                api_key=settings.kite_api_key,
                access_token=settings.kite_access_token,
            )
            kws.on_ticks   = self._on_ticks
            kws.on_connect = self._on_connect
            kws.on_error   = self._on_error
            kws.on_close   = self._on_close
            self._kws = kws
            kws.connect(threaded=True)
            logger.info("KiteTicker: WebSocket connecting ({} symbols)…", len(symbols))
        except Exception as exc:
            logger.warning("KiteTicker: failed to start WebSocket: {}", exc)

    def stop(self) -> None:
        if self._kws:
            try:
                self._kws.close()
            except Exception:
                pass
        self._connected = False
        logger.info("KiteTicker: stopped")

    def _on_connect(self, ws, response) -> None:
        tokens = list(self._token_map.values())
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)
        self._connected = True
        logger.info("KiteTicker: connected — subscribed {} tokens in FULL mode", len(tokens))

    def _on_ticks(self, ws, ticks) -> None:
        if not self._callback or not self._loop:
            return
        for t in ticks:
            token = t.get("instrument_token")
            sym   = self._rev_map.get(token)
            if not sym:
                continue
            try:
                ohlc  = t.get("ohlc", {})
                depth = t.get("depth", {})
                buys  = depth.get("buy",  [{}]) if depth else [{}]
                sells = depth.get("sell", [{}]) if depth else [{}]
                bid = buys[0].get("price",  0) if buys  else 0
                ask = sells[0].get("price", 0) if sells else 0
                ltp = t.get("last_price", 0.0)
                tick = Tick(
                    symbol=sym, ltp=ltp,
                    bid=bid or ltp, ask=ask or ltp,
                    volume=t.get("volume_traded", 0),
                    change=t.get("change", 0.0),
                    change_pct=t.get("change", 0.0),
                    high=ohlc.get("high", ltp),
                    low=ohlc.get("low",  ltp),
                    open=ohlc.get("open", ltp),
                    timestamp=datetime.now(),
                )
                asyncio.run_coroutine_threadsafe(
                    self._callback(sym, tick), self._loop
                )
            except Exception as exc:
                logger.debug("KiteTicker: tick parse error for {}: {}", sym, exc)

    def _on_error(self, ws, code, reason) -> None:
        self._connected = False
        logger.warning("KiteTicker: error code={} reason={}", code, reason)

    def _on_close(self, ws, code, reason) -> None:
        self._connected = False
        logger.info("KiteTicker: connection closed code={} reason={}", code, reason)
