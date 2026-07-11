"""
test_broker_data.py — Broker-only market data tests
Run: cd algotrader_v4 && python test_broker_data.py

Verifies that market data is sourced from the broker (Zerodha Kite) in BOTH
paper and live modes, and no longer from yfinance/NSE:
  • kite_client       — quote / ltp / historical gate on broker connection, not PAPER
  • mcx_instruments   — base name → nearest non-expired MCX futures contract
  • kite_ticker       — MCX token map keyed back to base names
  • tick_engine       — REST batch maps resolved tradingsymbol → base; historical
                        builds a DataFrame from broker bars; simulator off by default
"""
from __future__ import annotations

import asyncio
from datetime import timedelta

# ── Harness ────────────────────────────────────────────────────────────────────
_results: list[tuple[str, bool, str]] = []
def ok(n):   _results.append((n, True, "")); print(f"  OK  {n}")
def fail(n, e): _results.append((n, False, e)); print(f"  XX  {n}: {e}")
def run(n, fn):
    try: fn(); ok(n)
    except Exception as exc: fail(n, str(exc)[:150])
def section(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")

from config import settings
from kite_client import kite_client
import mcx_instruments
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


def _connect(fake):
    kite_client._kite = fake
    kite_client._instruments_cache.clear()
    mcx_instruments.clear_cache()

def _disconnect():
    kite_client._kite = None
    kite_client._instruments_cache.clear()
    mcx_instruments.clear_cache()


def _mcx_instruments():
    today = now_ist().date()
    near  = today + timedelta(days=20)
    far   = today + timedelta(days=50)
    old   = today - timedelta(days=10)
    return [
        {"instrument_type": "FUT", "name": "CRUDEOIL", "tradingsymbol": "CRUDEOIL_FAR",
         "instrument_token": 222, "expiry": far, "lot_size": 100, "exchange": "MCX"},
        {"instrument_type": "FUT", "name": "CRUDEOIL", "tradingsymbol": "CRUDEOIL_NEAR",
         "instrument_token": 111, "expiry": near, "lot_size": 100, "exchange": "MCX"},
        {"instrument_type": "FUT", "name": "CRUDEOIL", "tradingsymbol": "CRUDEOIL_OLD",
         "instrument_token": 333, "expiry": old, "lot_size": 100, "exchange": "MCX"},
        {"instrument_type": "OPT", "name": "CRUDEOIL", "tradingsymbol": "CRUDEOIL_CE",
         "instrument_token": 444, "expiry": near, "lot_size": 100, "exchange": "MCX"},
        {"instrument_type": "FUT", "name": "GOLDM", "tradingsymbol": "GOLDM_NEAR",
         "instrument_token": 555, "expiry": near, "lot_size": 10, "exchange": "MCX"},
    ]


# ═══════════════════════════════════════════════════════════════════════════════
section("1. DATA GATES ON BROKER CONNECTION, NOT TRADING MODE")

def t_disconnected_returns_empty():
    _disconnect()
    settings.trading_mode = "PAPER"
    assert kite_client.quote_kite(["MCX:CRUDEOIL_NEAR"]) == {}
    assert kite_client.ltp_kite(["MCX:CRUDEOIL_NEAR"]) == {}
    import datetime as dt
    assert kite_client.historical_data(111, dt.datetime.now()-timedelta(days=2),
                                       dt.datetime.now(), "minute") == []

def t_quote_flows_in_paper_when_connected():
    # PAPER mode, but broker connected → data must flow (broker-only feed)
    settings.trading_mode = "PAPER"
    _connect(FakeKite(quotes={"MCX:CRUDEOIL_NEAR": {"last_price": 6510.0}}))
    try:
        q = kite_client.quote_kite(["MCX:CRUDEOIL_NEAR"])
        assert q.get("MCX:CRUDEOIL_NEAR", {}).get("last_price") == 6510.0, q
    finally:
        _disconnect()

def t_historical_flows_in_paper_when_connected():
    settings.trading_mode = "PAPER"
    import datetime as dt
    _connect(FakeKite(bars=[{"date": dt.datetime.now(), "open": 6500, "high": 6520,
                             "low": 6490, "close": 6510, "volume": 1000}]))
    try:
        bars = kite_client.historical_data(111, dt.datetime.now()-timedelta(days=1),
                                           dt.datetime.now(), "minute")
        assert len(bars) == 1 and bars[0]["close"] == 6510, bars
    finally:
        _disconnect()

def t_ltp_flows_when_connected():
    _connect(FakeKite())
    try:
        got = kite_client.ltp_kite(["MCX:CRUDEOIL_NEAR"])
        assert got.get("MCX:CRUDEOIL_NEAR") == 100.0, got
    finally:
        _disconnect()

run("disconnected → quote/ltp/historical empty", t_disconnected_returns_empty)
run("quote flows in PAPER when broker connected", t_quote_flows_in_paper_when_connected)
run("historical flows in PAPER when connected",   t_historical_flows_in_paper_when_connected)
run("ltp flows when connected",                   t_ltp_flows_when_connected)


# ═══════════════════════════════════════════════════════════════════════════════
section("2. MCX INSTRUMENT RESOLUTION (from broker dump)")

def t_resolve_nearest_expiry():
    _connect(FakeKite(instruments=_mcx_instruments()))
    try:
        got = mcx_instruments.resolve(["CRUDEOIL"])
        rc = got["CRUDEOIL"]
        assert rc.tradingsymbol == "CRUDEOIL_NEAR" and rc.instrument_token == 111, rc
        assert rc.lot_size == 100
    finally:
        _disconnect()

def t_resolve_skips_expired():
    _connect(FakeKite(instruments=_mcx_instruments()))
    try:
        rc = mcx_instruments.resolve(["CRUDEOIL"])["CRUDEOIL"]
        assert rc.tradingsymbol != "CRUDEOIL_OLD", "must not pick an expired contract"
    finally:
        _disconnect()

def t_resolve_ignores_options():
    _connect(FakeKite(instruments=_mcx_instruments()))
    try:
        rc = mcx_instruments.resolve(["CRUDEOIL"])["CRUDEOIL"]
        assert rc.instrument_token != 444, "FUT only, not OPT"
    finally:
        _disconnect()

def t_token_and_symbol_maps():
    _connect(FakeKite(instruments=_mcx_instruments()))
    try:
        assert mcx_instruments.token_map(["CRUDEOIL", "GOLDM"]) == {"CRUDEOIL": 111, "GOLDM": 555}
        assert mcx_instruments.tradingsymbol_map(["GOLDM"]) == {"GOLDM": "GOLDM_NEAR"}
    finally:
        _disconnect()

def t_resolve_empty_when_disconnected():
    _disconnect()
    assert mcx_instruments.resolve(["CRUDEOIL"]) == {}

run("resolve picks nearest non-expired FUT", t_resolve_nearest_expiry)
run("resolve skips expired contracts",       t_resolve_skips_expired)
run("resolve ignores option instruments",    t_resolve_ignores_options)
run("token_map / tradingsymbol_map",         t_token_and_symbol_maps)
run("resolve empty when broker disconnected", t_resolve_empty_when_disconnected)


# ═══════════════════════════════════════════════════════════════════════════════
section("3. KITE TICKER — MCX token map keyed to base names")

def t_ticker_mcx_token_map():
    from kite_ticker import KiteTicker
    _connect(FakeKite(instruments=_mcx_instruments()))
    try:
        tk = KiteTicker()
        tk.load_instruments(["CRUDEOIL", "GOLDM"], exchange="MCX")
        assert tk._token_map.get("CRUDEOIL") == 111, tk._token_map
        assert tk._reverse_map.get(111) == "CRUDEOIL", tk._reverse_map
        assert tk._reverse_map.get(555) == "GOLDM"
    finally:
        _disconnect()

run("ticker maps MCX base → token → base", t_ticker_mcx_token_map)


# ═══════════════════════════════════════════════════════════════════════════════
section("4. TICK ENGINE — broker REST feed + historical DataFrame")

def t_fetch_batch_maps_tradingsymbol():
    from tick_engine import TickEngine, TickBuffer
    settings.trading_mode = "PAPER"
    eng = TickEngine()
    eng._symbols = ["CRUDEOIL"]
    eng._exchange = {"CRUDEOIL": "MCX"}
    eng._trading_symbol = {"CRUDEOIL": "CRUDEOIL_NEAR"}
    eng._bufs_1min = {"CRUDEOIL": TickBuffer(60, maxlen=400)}
    eng._bufs_5min = {"CRUDEOIL": TickBuffer(300, maxlen=200)}
    q = eng.add_subscriber("t")
    _connect(FakeKite(quotes={"MCX:CRUDEOIL_NEAR": {
        "last_price": 6510.0, "ohlc": {"open": 6500, "high": 6520, "low": 6490, "close": 6499},
        "volume_traded": 1234, "depth": {"buy": [{"price": 6509}], "sell": [{"price": 6511}]},
    }}))
    try:
        asyncio.run(eng._fetch_kite_batch())
        assert not q.empty(), "no snapshot produced from broker quote"
        snap = q.get_nowait()
        assert snap.symbol == "CRUDEOIL" and snap.tick.ltp == 6510.0, (snap.symbol, snap.tick.ltp)
    finally:
        _disconnect()

def t_get_historical_builds_df():
    from tick_engine import TickEngine
    import datetime as dt
    eng = TickEngine()
    eng._token = {"CRUDEOIL": 111}
    _connect(FakeKite(bars=[
        {"date": dt.datetime.now(), "open": 6500, "high": 6520, "low": 6490, "close": 6510, "volume": 1000},
        {"date": dt.datetime.now(), "open": 6510, "high": 6525, "low": 6505, "close": 6520, "volume": 1100},
    ]))
    try:
        df = eng.get_historical("CRUDEOIL", "MCX", "1m", "5d")
        assert len(df) == 2 and "timestamp" in df.columns and df.iloc[-1]["close"] == 6520, df
    finally:
        _disconnect()

def t_historical_empty_when_disconnected():
    from tick_engine import TickEngine
    _disconnect()
    df = TickEngine().get_historical("CRUDEOIL", "MCX", "1m", "5d")
    assert len(df) == 0 and list(df.columns) == ["timestamp","open","high","low","close","volume"]

def t_simulator_off_by_default():
    assert settings.use_paper_simulator is False, "broker feed must be the default (sim opt-in)"

def t_market_status_from_clock():
    from tick_engine import tick_engine
    st = asyncio.run(tick_engine.get_market_status())
    assert st["exchange"] == "MCX" and isinstance(st["open"], bool) and "broker_connected" in st

run("REST batch maps tradingsymbol → base + tick", t_fetch_batch_maps_tradingsymbol)
run("get_historical builds DataFrame from broker",  t_get_historical_builds_df)
run("get_historical empty when disconnected",       t_historical_empty_when_disconnected)
run("paper simulator off by default",               t_simulator_off_by_default)
run("market status derived from IST clock",         t_market_status_from_clock)


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
