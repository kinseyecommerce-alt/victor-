"""
test_option_bar_archiver.py — tests of option_bar_archiver.py (flag-off by default, read-only, bounded disk).

Run: cd algotrader_v4 && python test_option_bar_archiver.py     (also run by test_pipeline.py, section
"OPTION BAR ARCHIVER"; collectable by pytest: test_* functions)
"""
from __future__ import annotations

import asyncio
import csv
import gzip
import io
import json
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import mock

import option_bar_archiver as OBA

DAY = date(2026, 11, 13)                       # a Friday
HOLIDAYS = {date(2026, 11, 17)}                # a made-up Tuesday holiday for the window test


def _trading(d: date) -> bool:
    return d.weekday() < 5 and d not in HOLIDAYS


def _inst(sym, name, typ, expiry, strike=0.0, token=None):
    return {"tradingsymbol": sym, "name": name, "instrument_type": typ, "expiry": expiry, "strike": strike,
            "instrument_token": token or abs(hash(sym)) % 10**8, "lot_size": 65, "segment": "NFO-OPT"}


def _master():
    e0, e1, e2, far = DAY, date(2026, 11, 16), date(2026, 11, 18), date(2026, 11, 24)
    rows = [_inst("NIFTY26NOVFUT", "NIFTY", "FUT", e2), _inst("NIFTY26DECFUT", "NIFTY", "FUT", date(2026, 12, 29)),
            _inst("BANKNIFTY26NOVFUT", "BANKNIFTY", "FUT", far), _inst("RELIANCE26NOVFUT", "RELIANCE", "FUT", e2)]
    for k in (22000, 23000, 24000, 25000, 26000, 27000, 28000):
        rows.append(_inst(f"NIFTYW0{k}CE", "NIFTY", "CE", e0, k))            # expires today
        rows.append(_inst(f"NIFTYW2{k}PE", "NIFTY", "PE", e2, k))            # 2 sessions ahead (Mon, Tue holiday, Wed)
        rows.append(_inst(f"NIFTYFAR{k}CE", "NIFTY", "CE", far, k))          # outside the window
    rows.append(_inst("BANKNIFTYW161000CE", "BANKNIFTY", "CE", e1, 61000))
    rows.append(_inst("BANKNIFTYW170000CE", "BANKNIFTY", "CE", e1, 70000))   # > 8 % away
    rows.append(_inst("FINNIFTYW025000CE", "FINNIFTY", "CE", e0, 25000))     # not archived
    return rows


class FakeKite:
    """Exposes ONLY the three read-only calls the archiver may use; anything else raises."""
    ALLOWED = {"get_instruments", "get_instrument_tokens", "historical_data"}

    def __init__(self, master=None, empty=(), boom=()):
        object.__setattr__(self, "_d", {"master": master if master is not None else _master(), "calls": [],
                                        "empty": set(empty), "boom": set(boom)})

    def __getattr__(self, name):
        raise AssertionError(f"archiver touched kite_client.{name}")

    def get_instruments(self, exchange="NSE"):
        self._d["calls"].append(("get_instruments", exchange))
        return list(self._d["master"]) if exchange == "NFO" else []

    def get_instrument_tokens(self, symbols, exchange="NSE"):
        self._d["calls"].append(("get_instrument_tokens", tuple(symbols), exchange))
        return {"NIFTY 50": 256265, "NIFTY BANK": 260105}

    def historical_data(self, token, start, end, interval="day", continuous=False, oi=False):
        self._d["calls"].append(("historical_data", token, interval, oi))
        if interval == "day":
            return [{"date": datetime(2026, 11, 13), "open": 1, "high": 1, "low": 1,
                     "close": 25000.0 if token == 256265 else 61000.0, "volume": 0}]
        sym = next(r["tradingsymbol"] for r in self._d["master"] if r["instrument_token"] == token)
        if sym in self._d["boom"]:
            raise RuntimeError("kite down")
        if sym in self._d["empty"]:
            return []
        assert oi is True and interval == "minute"
        t0 = datetime(2026, 11, 13, 9, 15)
        return [{"date": t0 + timedelta(minutes=i), "open": 100 + i, "high": 101 + i, "low": 99 + i,
                 "close": 100.5 + i, "volume": 65 * (i + 1), "oi": 1000 + i} for i in range(5)]


def _read(path):
    return list(csv.reader(io.StringIO(gzip.decompress(Path(path).read_bytes()).decode())))


def test_flag_default_off_and_job_is_inert():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPTION_BAR_ARCHIVER_ENABLED", None)
        assert OBA.enabled() is False                                   # default off
        with tempfile.TemporaryDirectory() as td:
            k = FakeKite()
            out = OBA.daily_job(day=DAY, kite=k, root=Path(td) / "a", is_trading_day=_trading)
            assert "skipped" in out and k._d["calls"] == [] and not (Path(td) / "a").exists(), out
    for v, want in (("true", True), ("1", True), ("on", True), ("false", False), ("0", False), ("", False)):
        with mock.patch.dict(os.environ, {"OPTION_BAR_ARCHIVER_ENABLED": v}):
            assert OBA.enabled() is want, v


def test_scheduler_registers_the_job_only_with_the_flag():
    import platform_scheduler as ps

    class _S:
        def __init__(self):
            self.ids = []

        def add_job(self, *a, **k):
            self.ids.append(k.get("id"))

        def start(self):
            pass
    for flag, expect in ((None, False), ("false", False), ("true", True)):
        env = {} if flag is None else {"OPTION_BAR_ARCHIVER_ENABLED": flag}
        with mock.patch.dict(os.environ, env):
            if flag is None:
                os.environ.pop("OPTION_BAR_ARCHIVER_ENABLED", None)
            s = ps.PlatformScheduler()
            s._sched = _S()
            s.start()
            assert ("option_bar_archive" in s._sched.ids) is expect, (flag, s._sched.ids)
    # with the flag off the job wrapper itself does nothing (no Kite call, no file)
    with mock.patch.dict(os.environ, {"OPTION_BAR_ARCHIVER_ENABLED": "false"}), \
         mock.patch.object(OBA, "run_daily") as rd:
        asyncio.run(ps.PlatformScheduler()._option_bar_archive())
        assert rd.call_count == 0


def test_selection_window_strikes_and_underlyings():
    last = OBA.expiry_window(DAY, _trading)
    assert last == date(2026, 11, 18), last                 # Fri -> Mon (1), Tue holiday, Wed (2)
    assert OBA.expiry_window(date(2026, 11, 9), lambda d: d.weekday() < 5) == date(2026, 11, 11)
    rows = OBA.select_contracts(_master(), DAY, last, {"NIFTY": 25000.0, "BANKNIFTY": 61000.0})
    syms = [r["tradingsymbol"] for r in rows]
    assert "NIFTY26NOVFUT" in syms and "NIFTY26DECFUT" not in syms and "BANKNIFTY26NOVFUT" not in syms
    assert "RELIANCE26NOVFUT" not in syms and not any(s.startswith("FINNIFTY") for s in syms)
    assert not any("FAR" in s for s in syms)                               # expiry beyond the window
    assert "BANKNIFTYW161000CE" in syms and "BANKNIFTYW170000CE" not in syms
    ks = sorted({float(r["strike"]) for r in rows if r["name"] == "NIFTY" and r["instrument_type"] != "FUT"})
    assert ks == [23000.0, 24000.0, 25000.0, 26000.0, 27000.0], ks         # 8 % of 25000 = 2000 (inclusive)
    # no spot for an underlying -> its options are skipped, its future is not
    rows2 = OBA.select_contracts(_master(), DAY, last, {"BANKNIFTY": 61000.0})
    assert [r["tradingsymbol"] for r in rows2 if r["name"] == "NIFTY"] == ["NIFTY26NOVFUT"]
    # string expiries (as in Kite's CSV dump) work too
    m = [dict(r, expiry=r["expiry"].isoformat()) for r in _master()]
    assert len(OBA.select_contracts(m, DAY, last, {"NIFTY": 25000.0, "BANKNIFTY": 61000.0})) == len(rows)


def test_run_writes_gz_files_reads_only_and_is_idempotent():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "arch"
        k = FakeKite(empty={"NIFTYW024000CE"}, boom={"NIFTYW026000CE"})
        out = OBA.run_daily(day=DAY, kite=k, root=root, is_trading_day=_trading)
        assert out["contracts"] == 10 and out["empty"] == 1 and out["errors"] == 1, out      # 12 selected
        assert {c[0] for c in k._d["calls"]} <= FakeKite.ALLOWED
        assert ("get_instruments", "NFO") in k._d["calls"]
        f = root / "20261113" / "NIFTYW025000CE.csv.gz"
        rows = _read(f)
        assert rows[0] == ["ts", "open", "high", "low", "close", "volume", "oi"]
        assert rows[1] == ["2026-11-13T09:15", "100", "101", "99", "100.5", "65", "1000"] and len(rows) == 6
        assert (root / "20261118" / "NIFTY26NOVFUT.csv.gz").exists()
        assert not (root / "20261113" / "NIFTYW024000CE.csv.gz").exists()              # empty answer: no file
        man = [json.loads(x) for x in (root / "manifest.jsonl").read_text().splitlines()]
        assert len(man) == 1 and man[0]["selected"] == 12 and man[0]["contracts"] == 10
        files = sorted(p.name for p in root.rglob("*.csv.gz"))
        b0 = f.read_bytes()
        # a second run rewrites the same bytes; an empty answer never blanks an existing file
        k2 = FakeKite(empty={"NIFTYW025000CE"})
        out2 = OBA.run_daily(day=DAY, kite=k2, root=root, is_trading_day=_trading)
        assert f.read_bytes() == b0 and out2["empty"] == 1
        assert sorted(p.name for p in root.rglob("*.csv.gz")) == sorted(files + ["NIFTYW024000CE.csv.gz",
                                                                                 "NIFTYW026000CE.csv.gz"])
        assert not list(root.rglob("*.tmp"))
        # a non-trading day does nothing
        k3 = FakeKite()
        assert "skipped" in OBA.run_daily(day=date(2026, 11, 14), kite=k3, root=root, is_trading_day=_trading)
        assert k3._d["calls"] == []
        # no instrument master (no Kite session) -> skipped, nothing written
        k4 = FakeKite(master=[])
        n = len(list(root.rglob("*.csv.gz")))
        assert "skipped" in OBA.run_daily(day=DAY, kite=k4, root=root, is_trading_day=_trading)
        assert len(list(root.rglob("*.csv.gz"))) == n


def test_disk_cap_stops_writing():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "arch"
        root.mkdir()
        (root / "old.bin").write_bytes(b"x" * 4096)
        with mock.patch.dict(os.environ, {"OPTION_BAR_ARCHIVE_MAX_MB": "0.001"}):         # ~1 KB cap
            k = FakeKite()
            out = OBA.run_daily(day=DAY, kite=k, root=root, is_trading_day=_trading)
            assert "disk cap" in out.get("skipped", "") and k._d["calls"] == [], out
            assert not list(root.rglob("*.csv.gz"))
        with mock.patch.dict(os.environ, {"OPTION_BAR_ARCHIVE_MAX_MB": "0.0045"}):        # room for ~3 small files
            out = OBA.run_daily(day=DAY, kite=FakeKite(), root=root, is_trading_day=_trading)
            assert out.get("stopped") and 0 < out["contracts"] < 12, out
        st = OBA.status(root)
        assert st["enabled"] is False and OBA.archive_bytes(root) > 4096 and st["estimate_mb_per_week"] > 0


def test_module_never_references_orders_agents_or_telegram():
    src = (Path(__file__).resolve().parent / "option_bar_archiver.py").read_text()
    code = src.split('"""', 2)[2]                                       # skip the module docstring
    for banned in ("place_order", "modify_order", "cancel_order", "squareoff", "gtt", "send_telegram", "ALL_AGENTS",
                   "risk_manager", "bot_state", "paper_fill", "master_agent", "import agents", "from agents"):
        assert banned not in code, banned
    assert OBA.default_root().name == "option_bars"
    with mock.patch.dict(os.environ, {"PERFORMANCE_FILE": "/data/performance.json"}):
        os.environ.pop("OPTION_BAR_ARCHIVE_DIR", None)
        assert OBA.default_root() == Path("/data/option_bars")


TESTS = [
    ("option archiver: OPTION_BAR_ARCHIVER_ENABLED defaults to off; daily_job is inert when off",
     test_flag_default_off_and_job_is_inert),
    ("option archiver: the scheduler registers the job only with the flag on (nothing scheduled when off)",
     test_scheduler_registers_the_job_only_with_the_flag),
    ("option archiver: expiry today..+2 sessions, strikes within 8 %, NIFTY/BANKNIFTY only, expiring futures",
     test_selection_window_strikes_and_underlyings),
    ("option archiver: gz files with OI, manifest, idempotent; only read-only Kite calls (fake client)",
     test_run_writes_gz_files_reads_only_and_is_idempotent),
    ("option archiver: disk cap stops writing (never deletes)", test_disk_cap_stops_writing),
    ("option archiver: no order / agent / Telegram reference; root is the volume next to the ledger",
     test_module_never_references_orders_agents_or_telegram),
]

if __name__ == "__main__":
    bad = 0
    for name, fn in TESTS:
        try:
            fn()
            print("PASS", name)
        except Exception as exc:  # noqa: BLE001
            bad += 1
            print("FAIL", name, repr(exc))
    raise SystemExit(1 if bad else 0)
