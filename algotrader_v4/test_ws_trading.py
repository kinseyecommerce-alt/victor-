"""
test_ws_trading.py — Live-WebSocket tick-trading wiring tests
Run: cd algotrader_v4 && python test_ws_trading.py

Verifies the master-agent → live Kite WebSocket → agent trade path:
  • tick_engine.start_ws()   — idempotent WebSocket start, gated on
                               broker-connected + use_kite_websocket +
                               not use_paper_simulator + symbols + loop
  • tick_engine.subscribe()  — triggers start_ws() so the boot-time gap
                               (WS never started because the watchlist was
                               empty at start_loop) is closed
  • WS tick → agent          — a tick ingested via _ingest_kite_tick() reaches
                               a subscriber queue as a MarketSnapshot
  • tick_engine.feed_status()— reports ws_active / source for WS / REST / SIM / NONE
  • REST fallback preserved  — connected + use_kite_websocket=False leaves
                               _use_ws False and source == KITE_REST
"""
from __future__ import annotations

import asyncio

# ── Harness ────────────────────────────────────────────────────────────────────
_results: list[tuple[str, bool, str]] = []
def ok(n):   _results.append((n, True, "")); print(f"  OK  {n}")
def fail(n, e): _results.append((n, False, e)); print(f"  XX  {n}: {e}")
def run(n, fn):
    try: fn(); ok(n)
    except Exception as exc: fail(n, str(exc)[:200])
def section(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")

from config import settings
from kite_client import kite_client
import mcx_instruments
import kite_ticker as kite_ticker_mod
from tick_engine import TickEngine, TickBuffer, Tick
from datetime import datetime
from datetime import timedelta
from ist_clock import now_ist


class FakeKite:
    """Minimal stand-in for KiteConnect used to simulate a live broker session."""
    def __init__(self, instruments=None, quotes=None, bars=None):
        self._instruments = instruments or []
        self._quotes = quotes or {}
        self._bars = bars or []
    def instruments(self, exchange): return self._instruments
    def quote(self, instruments): return self._quotes
    def ltp(self, instruments): return {k: {"last_price": 100.0} for k in instruments}
    def historical_data(self, **kw): return self._bars


class FakeTicker:
    """Records .start()/.stop() calls; never touches the network.

    `tokens` mirrors KiteTicker.start()'s real contract: it returns True when
    instrument tokens resolved (socket initiated) and False when none did.
    `is_connected` flips True after a successful start and False after stop,
    matching the real ticker so feed_status()/_ws_healthy() can be exercised.
    """
    instances: list["FakeTicker"] = []
    tokens: bool = True
    def __init__(self):
        self.started_with = None
        self.stopped = False
        self.is_connected = False
        FakeTicker.instances.append(self)
    def start(self, symbols, callback, loop, exchange="NSE"):
        self.started_with = {"symbols": list(symbols), "exchange": exchange,
                             "callback": callback, "loop": loop}
        self.is_connected = self.__class__.tokens
        return self.__class__.tokens
    def stop(self):
        self.stopped = True
        self.is_connected = False


class FakeTickerNoTokens(FakeTicker):
    """A ticker that finds no instrument tokens (start returns False, never connects)."""
    tokens = False


def _mcx_instruments():
    today = now_ist().date()
    near = today + timedelta(days=20)
    return [
        {"instrument_type": "FUT", "name": "CRUDEOIL", "tradingsymbol": "CRUDEOIL_NEAR",
         "instrument_token": 111, "expiry": near, "lot_size": 100, "exchange": "MCX"},
    ]


def _connect(fake):
    kite_client._kite = fake
    kite_client._instruments_cache.clear()
    mcx_instruments.clear_cache()

def _disconnect():
    kite_client._kite = None
    kite_client._instruments_cache.clear()
    mcx_instruments.clear_cache()


class _SettingsGuard:
    """Save/restore the settings we mutate so each test is isolated."""
    _keys = ("use_paper_simulator", "use_kite_websocket",
             "use_truedata_websocket", "truedata_username")
    def __enter__(self):
        self._saved = {k: getattr(settings, k) for k in self._keys}
        return self
    def __exit__(self, *exc):
        for k, v in self._saved.items():
            setattr(settings, k, v)


class _TickerPatch:
    """Swap kite_ticker.KiteTicker with FakeTicker (the class start_ws imports)."""
    def __enter__(self):
        FakeTicker.instances.clear()
        self._orig = kite_ticker_mod.KiteTicker
        kite_ticker_mod.KiteTicker = FakeTicker
        return self
    def __exit__(self, *exc):
        kite_ticker_mod.KiteTicker = self._orig


def _new_engine(symbols=("CRUDEOIL",), exchange="MCX"):
    eng = TickEngine()
    for s in symbols:
        eng._symbols.append(s)
        eng._exchange[s] = exchange
        eng._trading_symbol[s] = s
        eng._bufs_1min[s] = TickBuffer(60, maxlen=400)
        eng._bufs_5min[s] = TickBuffer(300, maxlen=200)
    eng._loop = asyncio.new_event_loop()   # a non-None loop; never run
    return eng


# ═══════════════════════════════════════════════════════════════════════════════
section("1. start_ws() STARTS THE WEBSOCKET WHEN CONNECTED + ENABLED")

def t_start_ws_starts_websocket():
    with _SettingsGuard(), _TickerPatch():
        settings.use_paper_simulator = False
        settings.use_kite_websocket = True
        settings.use_truedata_websocket = False
        _connect(FakeKite(instruments=_mcx_instruments()))
        try:
            eng = _new_engine(("CRUDEOIL",), "MCX")
            eng.start_ws()
            assert eng._use_ws is True, "expected _use_ws True after start"
            assert len(FakeTicker.instances) == 1, FakeTicker.instances
            tk = FakeTicker.instances[0]
            assert tk.started_with is not None, "ticker.start() not called"
            assert tk.started_with["symbols"] == ["CRUDEOIL"], tk.started_with
            assert tk.started_with["exchange"] == "MCX", tk.started_with["exchange"]
        finally:
            _disconnect()

run("start_ws() starts WS with symbols + exchange=MCX", t_start_ws_starts_websocket)


# ═══════════════════════════════════════════════════════════════════════════════
section("2. start_ws() IS A NO-OP WHEN PRECONDITIONS FAIL")

def t_noop_when_paper_simulator():
    with _SettingsGuard(), _TickerPatch():
        settings.use_paper_simulator = True
        settings.use_kite_websocket = True
        _connect(FakeKite(instruments=_mcx_instruments()))
        try:
            eng = _new_engine()
            eng.start_ws()
            assert eng._use_ws is False, "must not start WS with simulator on"
            assert FakeTicker.instances == [], "ticker must not be constructed"
        finally:
            _disconnect()

def t_noop_when_disconnected():
    with _SettingsGuard(), _TickerPatch():
        settings.use_paper_simulator = False
        settings.use_kite_websocket = True
        _disconnect()   # broker NOT connected
        eng = _new_engine()
        eng.start_ws()
        assert eng._use_ws is False, "must not start WS when broker disconnected"
        assert FakeTicker.instances == []

def t_noop_when_ws_disabled():
    with _SettingsGuard(), _TickerPatch():
        settings.use_paper_simulator = False
        settings.use_kite_websocket = False
        _connect(FakeKite(instruments=_mcx_instruments()))
        try:
            eng = _new_engine()
            eng.start_ws()
            assert eng._use_ws is False, "must not start WS when use_kite_websocket False"
            assert FakeTicker.instances == []
        finally:
            _disconnect()

def t_noop_when_no_symbols():
    with _SettingsGuard(), _TickerPatch():
        settings.use_paper_simulator = False
        settings.use_kite_websocket = True
        _connect(FakeKite(instruments=_mcx_instruments()))
        try:
            eng = TickEngine()
            eng._loop = asyncio.new_event_loop()   # loop present, but no symbols
            eng.start_ws()
            assert eng._use_ws is False
            assert FakeTicker.instances == []
        finally:
            _disconnect()

run("no-op when use_paper_simulator=True", t_noop_when_paper_simulator)
run("no-op when broker disconnected",      t_noop_when_disconnected)
run("no-op when use_kite_websocket=False", t_noop_when_ws_disabled)
run("no-op when no symbols subscribed",    t_noop_when_no_symbols)


# ═══════════════════════════════════════════════════════════════════════════════
section("3. subscribe() TRIGGERS start_ws() — BOOT-TIME GAP CLOSED")

def t_subscribe_triggers_start_ws():
    with _SettingsGuard(), _TickerPatch():
        settings.use_paper_simulator = False
        settings.use_kite_websocket = True
        settings.use_truedata_websocket = False
        _connect(FakeKite(instruments=_mcx_instruments()))
        try:
            eng = TickEngine()
            eng._loop = asyncio.new_event_loop()   # simulate loop already running
            eng.subscribe([{"symbol": "CRUDEOIL", "exchange": "MCX"}])
            assert eng._use_ws is True, "subscribe() must start WS when connected"
            assert len(FakeTicker.instances) == 1, FakeTicker.instances
            assert FakeTicker.instances[0].started_with["symbols"] == ["CRUDEOIL"]
        finally:
            _disconnect()

def t_subscribe_restarts_running_ticker():
    with _SettingsGuard(), _TickerPatch():
        settings.use_paper_simulator = False
        settings.use_kite_websocket = True
        settings.use_truedata_websocket = False
        _connect(FakeKite(instruments=_mcx_instruments()))
        try:
            eng = TickEngine()
            eng._loop = asyncio.new_event_loop()
            eng.subscribe([{"symbol": "CRUDEOIL", "exchange": "MCX"}])
            first = FakeTicker.instances[0]
            # simulate CRUDEOIL having been receiving live WS ticks
            import time as _t
            eng._ws_received.add("CRUDEOIL")
            eng._ws_last["CRUDEOIL"] = _t.monotonic()
            # Second subscribe should stop the first ticker and start a new one
            eng.subscribe([{"symbol": "GOLDM", "exchange": "MCX"}])
            assert first.stopped is True, "existing ticker must be stopped on restart"
            assert len(FakeTicker.instances) == 2, FakeTicker.instances
            assert set(FakeTicker.instances[1].started_with["symbols"]) == {"CRUDEOIL", "GOLDM"}
            # WS freshness must be cleared so REST covers every symbol during reconnect
            assert eng._ws_received == set(), "restart must clear _ws_received"
            assert eng._ws_last == {}, "restart must clear _ws_last"
        finally:
            _disconnect()

run("subscribe() starts WS (gap closed)", t_subscribe_triggers_start_ws)
run("subscribe() restarts ticker for expanded set", t_subscribe_restarts_running_ticker)


# ═══════════════════════════════════════════════════════════════════════════════
section("4. WS TICK → AGENT (end-to-end feed)")

def t_ws_tick_reaches_agent():
    async def _drive():
        eng = _new_engine(("CRUDEOIL",), "MCX")
        q = eng.add_subscriber("agent_intraday")
        tick = Tick(symbol="CRUDEOIL", ltp=6510.0, bid=6509.0, ask=6511.0,
                    volume=1234, change=10.0, change_pct=0.15,
                    high=6520.0, low=6490.0, open=6500.0, timestamp=datetime.now())
        await eng._ingest_kite_tick("CRUDEOIL", tick)
        assert not q.empty(), "no snapshot delivered to agent subscriber"
        snap = q.get_nowait()
        assert snap.symbol == "CRUDEOIL" and snap.tick.ltp == 6510.0, (snap.symbol, snap.tick.ltp)
        assert "CRUDEOIL" in eng._ws_received, "WS tick must mark symbol as ws_received"
    asyncio.run(_drive())

run("WS tick → MarketSnapshot on agent queue", t_ws_tick_reaches_agent)


# ═══════════════════════════════════════════════════════════════════════════════
section("5. feed_status() REPORTS THE ACTIVE SOURCE")

def t_feed_status_ws_active():
    with _SettingsGuard():
        settings.use_paper_simulator = False
        _connect(FakeKite())
        try:
            eng = _new_engine()
            eng._use_ws = True
            class _Connected:  # ticker reporting a live socket
                is_connected = True
            eng._kite_ticker = _Connected()
            st = eng.feed_status()
            assert st["ws_active"] is True and st["source"] == "KITE_WS", st
            assert st["subscribed"] == 1, st
        finally:
            _disconnect()

def t_feed_status_ws_flag_but_not_connected():
    # _use_ws True but the socket isn't actually connected → honest KITE_REST, not a lie
    with _SettingsGuard():
        settings.use_paper_simulator = False
        _connect(FakeKite())
        try:
            eng = _new_engine()
            eng._use_ws = True
            class _NotConnected:
                is_connected = False
            eng._kite_ticker = _NotConnected()
            st = eng.feed_status()
            assert st["source"] == "KITE_REST" and st["ws_active"] is False, st
        finally:
            _disconnect()

def t_feed_status_rest():
    with _SettingsGuard():
        settings.use_paper_simulator = False
        _connect(FakeKite())
        try:
            eng = _new_engine()
            eng._use_ws = False
            st = eng.feed_status()
            assert st["source"] == "KITE_REST" and st["ws_active"] is False, st
        finally:
            _disconnect()

def t_feed_status_sim():
    with _SettingsGuard():
        settings.use_paper_simulator = True
        _disconnect()
        eng = _new_engine()
        eng._use_ws = False
        st = eng.feed_status()
        assert st["source"] == "SIM", st

def t_feed_status_none():
    with _SettingsGuard():
        settings.use_paper_simulator = False
        _disconnect()
        eng = _new_engine()
        eng._use_ws = False
        st = eng.feed_status()
        assert st["source"] == "NONE", st

run("feed_status() → KITE_WS when WS active", t_feed_status_ws_active)
run("feed_status() → KITE_REST when WS flagged but socket down", t_feed_status_ws_flag_but_not_connected)
run("feed_status() → KITE_REST when connected, WS off", t_feed_status_rest)
run("feed_status() → SIM when simulator on", t_feed_status_sim)
run("feed_status() → NONE when disconnected", t_feed_status_none)


# ═══════════════════════════════════════════════════════════════════════════════
section("6. REST FALLBACK PRESERVED (use_kite_websocket=False)")

def t_rest_fallback_preserved():
    with _SettingsGuard(), _TickerPatch():
        settings.use_paper_simulator = False
        settings.use_kite_websocket = False   # WS disabled → REST path
        _connect(FakeKite(instruments=_mcx_instruments()))
        try:
            eng = _new_engine()
            eng.start_ws()
            assert eng._use_ws is False, "WS must stay off"
            assert FakeTicker.instances == [], "ticker must not be constructed"
            assert eng.feed_status()["source"] == "KITE_REST", eng.feed_status()
        finally:
            _disconnect()

run("REST fallback intact when WS disabled", t_rest_fallback_preserved)


# ═══════════════════════════════════════════════════════════════════════════════
section("7. WS DISCONNECT / STALENESS → REST RESUMES (no silent starvation)")

def t_stale_symbol_not_ws_covered():
    import time as _t
    eng = _new_engine(("CRUDEOIL",), "MCX")
    eng._use_ws = True
    now = _t.monotonic()
    # fresh tick → WS-covered (REST skips it)
    eng._ws_last["CRUDEOIL"] = now
    assert eng._ws_covers("CRUDEOIL", now) is True, "fresh WS symbol should be WS-covered"
    # tick older than WS_STALE_SEC (socket dropped) → NOT covered → REST resumes
    eng._ws_last["CRUDEOIL"] = now - (eng.WS_STALE_SEC + 1)
    assert eng._ws_covers("CRUDEOIL", now) is False, "stale symbol must fall back to REST"

def t_fetch_batch_includes_stale_symbols():
    # _fetch_kite_batch must re-fetch symbols that stopped receiving WS ticks
    async def _drive():
        eng = _new_engine(("CRUDEOIL",), "MCX")
        eng._use_ws = True
        import time as _t
        eng._ws_last["CRUDEOIL"] = _t.monotonic() - (eng.WS_STALE_SEC + 2)  # gone silent
        fetched = {}
        _connect(FakeKite(quotes={"MCX:CRUDEOIL": {"last_price": 6500.0}}))
        # spy: capture which symbols REST tried to quote
        orig = kite_client.quote_kite
        def _spy(insts):
            fetched["insts"] = insts
            return {}
        kite_client.quote_kite = _spy
        try:
            await eng._fetch_kite_batch()
        finally:
            kite_client.quote_kite = orig
            _disconnect()
        assert fetched.get("insts"), "REST must re-fetch a stale (silent) WS symbol"
    asyncio.run(_drive())

def t_no_token_start_uses_rest():
    with _SettingsGuard(), _TickerPatch():
        settings.use_paper_simulator = False
        settings.use_kite_websocket = True
        settings.use_truedata_websocket = False
        kite_ticker_mod.KiteTicker = FakeTickerNoTokens   # start() returns False
        _connect(FakeKite(instruments=_mcx_instruments()))
        try:
            eng = _new_engine()
            eng.start_ws()
            assert eng._use_ws is False, "no tokens → must not claim WS active"
            assert eng.feed_status()["source"] == "KITE_REST", eng.feed_status()
        finally:
            _disconnect()

run("stale symbol is not WS-covered",       t_stale_symbol_not_ws_covered)
run("_fetch_kite_batch re-fetches stale symbol", t_fetch_batch_includes_stale_symbols)
run("no-token WS start falls back to REST",  t_no_token_start_uses_rest)


# ── Summary ──────────────────────────────────────────────────────────────────
_disconnect()
passed = sum(1 for _, o, _ in _results if o)
failed = sum(1 for _, o, _ in _results if not o)
print(f"\n{'='*60}\n  RESULTS: {len(_results)} tests -- {passed} passed  {failed} failed\n{'='*60}")
if failed:
    print("\nFailed:")
    for n, o, e in _results:
        if not o: print(f"  XX {n}: {e}")
import sys
sys.exit(1 if failed else 0)
