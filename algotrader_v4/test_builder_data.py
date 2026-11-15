"""
test_builder_data.py — tests of the strategy builder's data layer (builder_data.py, builder_minute.py).

    cd algotrader_v4 && python test_builder_data.py      (also run from test_pipeline.py)

Every test works in a temporary directory with a fake network (no request leaves the machine). The layout-
compatibility tests feed the lab's own raw NSE files from the research scratchpad through the server's parsers and
compare what stratlab.nsedata returns from both directories; they are skipped when that scratchpad is absent.
Everything else uses small synthetic fixtures generated in the test.
"""
from __future__ import annotations

import gzip
import hashlib
import inspect
import io
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("PERFORMANCE_FILE", os.path.join(tempfile.mkdtemp(), "performance.json"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import settings  # noqa: E402
import builder_data as BD  # noqa: E402
import builder_minute as BM  # noqa: E402
import forward_lab as FL  # noqa: E402
import forward_lab_store as FS  # noqa: E402
from ist_clock import is_trading_day  # noqa: E402

SCRATCH = os.environ.get("STRATLAB_SCRATCH",
                         "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad")
LAB_NSE = os.path.join(SCRATCH, "nse")
LAB_OLD = os.path.join(SCRATCH, "nse_old")
LAB_BARS = os.path.join(SCRATCH, "bars3y")
_TMPS: list = []


def _tmp() -> Path:
    d = tempfile.mkdtemp(prefix="bdata_test_")
    _TMPS.append(d)
    return Path(d)


import atexit  # noqa: E402
atexit.register(lambda: [shutil.rmtree(d, ignore_errors=True) for d in _TMPS])


class Skip(Exception):
    pass


def _ymd(d) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def _d(k) -> date:
    return date(k // 10000, k // 100 % 100, k % 100)


def sessions(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


# ── fake network ──────────────────────────────────────────────────────────────────────────────────────────────────
class FakeNet:
    """URL -> (status, body) from a function; counts requests; never sleeps."""

    def __init__(self, fn) -> None:
        self.fn, self.urls = fn, []

    def get(self, url):
        self.urls.append(url)
        return self.fn(url)

    def http(self, budget=100000):
        return BD.Http(budget, get=self.get, pause=0.0, sleep=lambda s: None)


class NoFeeds:
    """pead_data._Client stand-in: every filing-feed window answers 'ok' with no rows."""

    def __init__(self):
        self.n = 0

    def fetch(self, feed, d0, d1):
        self.n += 1
        return "ok", [b"[]"] if feed != "ifr" else []


def _date_in(url: str) -> date | None:
    import re
    for pat, fmt in ((r"_(\d{8})_F_0000", "%Y%m%d"), (r"sec_bhavdata_full_(\d{8})", "%d%m%Y"),
                     (r"fao_participant_(?:oi|vol)_(\d{8})", "%d%m%Y"), (r"fo_secban_(\d{8})", "%d%m%Y"),
                     (r"ind_close_all_(\d{8})", "%d%m%Y"), (r"MTO_(\d{8})", "%d%m%Y"),
                     (r"fii_stats_(\d{2}-[A-Za-z]{3}-\d{4})", "%d-%b-%Y"), (r"/cm(\d{2}[A-Z]{3}\d{4})bhav", "%d%b%Y"),
                     (r"/fo(\d{2}[A-Z]{3}\d{4})bhav", "%d%b%Y")):
        m = re.search(pat, url)
        if m:
            return datetime.strptime(m.group(1).title() if "b" in fmt else m.group(1), fmt).date()
    return None


# ── synthetic NSE files (in the exact formats NSE publishes) ──────────────────────────────────────────────────────
SYN = ("INFY", "RELIANCE", "TATASTEEL", "ZOMATO", "NOTLAB")


def syn_sec_full(d: date, syms=SYN, deliv=lambda s, d: 40.0) -> bytes:
    lines = ["SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, "
             "TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER"]
    for i, s in enumerate(syms):
        px = 100.0 + 10 * i + (d.toordinal() % 7)
        q = 100000 + 1000 * i
        dp = deliv(s, d)
        lines.append(f"{s}, EQ, {d:%d-%b-%Y}, {px - 1:.2f}, {px - 0.5:.2f}, {px + 1:.2f}, {px - 2:.2f}, {px:.2f}, "
                     f"{px:.2f}, {px:.2f}, {q}, {q * px / 1e5:.2f}, {500 + i}, {int(q * dp / 100)}, {dp:.2f}")
        lines.append(f"{s}, BE, {d:%d-%b-%Y}, 1, 1, 1, 1, 1, 1, 1, 10, 0.01, 1, 10, 100.00")
    return ("\n".join(lines) + "\n").encode()


UDIFF_COLS = ["TradDt", "BizDt", "Sgmt", "Src", "FinInstrmTp", "FinInstrmId", "ISIN", "TckrSymb", "SctySrs", "XpryDt",
              "FininstrmActlXpryDt", "StrkPric", "OptnTp", "FinInstrmNm", "OpnPric", "HghPric", "LwPric", "ClsPric",
              "LastPric", "PrvsClsgPric", "UndrlygPric", "SttlmPric", "OpnIntrst", "ChngInOpnIntrst", "TtlTradgVol",
              "TtlTrfVal", "TtlNbOfTxsExctd", "SsnId", "NewBrdLotQty", "Rmks", "Rsvd1", "Rsvd2", "Rsvd3", "Rsvd4"]


def udiff_zip(rows: list[dict], d: date) -> bytes:
    buf = io.StringIO()
    df = pd.DataFrame([{c: r.get(c, "") for c in UDIFF_COLS} for r in rows], columns=UDIFF_COLS)
    df.to_csv(buf, index=False)
    zb = io.BytesIO()
    with zipfile.ZipFile(zb, "w") as z:
        z.writestr(f"BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv", buf.getvalue())
    return zb.getvalue()


def _fmt(x):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else x


def parts_to_udiff(fut: pd.DataFrame, opt: pd.DataFrame) -> list[dict]:
    """The lab's normalised part rows back to UDiFF columns (the inverse of fetch_fo.norm_udiff)."""
    m = {"FUTIDX": "IDF", "FUTSTK": "STF"}
    rows = []
    for r in fut.itertuples(index=False):
        rows.append({"TradDt": r.date, "FinInstrmTp": m[r.instrument], "TckrSymb": r.symbol, "XpryDt": r.expiry,
                     "StrkPric": "", "OptnTp": "", "OpnPric": _fmt(r.open), "HghPric": _fmt(r.high),
                     "LwPric": _fmt(r.low), "ClsPric": _fmt(r.close), "SttlmPric": _fmt(r.settle),
                     "TtlTradgVol": _fmt(r.contracts), "OpnIntrst": _fmt(r.oi), "ChngInOpnIntrst": _fmt(r.chg_oi),
                     "UndrlygPric": _fmt(r.underlying), "NewBrdLotQty": _fmt(r.lot)})
    for r in opt.itertuples(index=False):
        rows.append({"TradDt": r.date, "FinInstrmTp": "IDO" if r.symbol in ("NIFTY", "BANKNIFTY") else "STO",
                     "TckrSymb": r.symbol, "XpryDt": r.expiry, "StrkPric": _fmt(r.strike), "OptnTp": r.type,
                     "OpnPric": _fmt(r.open), "HghPric": _fmt(r.high), "LwPric": _fmt(r.low), "ClsPric": _fmt(r.close),
                     "SttlmPric": _fmt(r.settle), "TtlTradgVol": _fmt(r.contracts), "OpnIntrst": _fmt(r.oi),
                     "ChngInOpnIntrst": _fmt(r.chg_oi), "UndrlygPric": _fmt(r.underlying),
                     "NewBrdLotQty": _fmt(r.lot)})
    return rows


def syn_participant(d: date, kind: str, fii_long: int = 100, fii_short: int = 900) -> bytes:
    hdr = ("Client Type,Future Index Long,Future Index Short,Future Stock Long,Future Stock Short,Option Index Call Long,"
           "Option Index Put Long,Option Index Call Short,Option Index Put Short,Option Stock Call Long,"
           "Option Stock Put Long,Option Stock Call Short,Option Stock Put Short,Total Long Contracts,"
           "Total Short Contracts")
    title = "Open Interest (no. of contracts)" if kind == "oi" else "Volume (no. of contracts)"
    rows = []
    vals = {"Client": (500, 500), "DII": (200, 200), "FII": (fii_long, fii_short), "Pro": (300, 300)}
    tl = ts = 0
    for ct, (a, b) in vals.items():
        cols = [a, b] + [1000] * 10
        rows.append(",".join([ct] + [str(x) for x in cols] + [str(sum(cols[0::2])), str(sum(cols[1::2]))]))
    tot = [sum(v) for v in zip(*[[a, b] + [1000] * 10 for a, b in vals.values()])]
    rows.append(",".join(["TOTAL"] + [str(x) for x in tot] + [str(sum(tot[0::2])), str(sum(tot[1::2]))]))
    return (f'"Participant wise {title} in Equity Derivatives as on {d:%b %d, %Y}",,,,,,,,,,,,,,\n{hdr}\n'
            + "\n".join(rows) + "\n").encode()


def syn_ban(d: date, syms=()) -> bytes:
    lines = [f"Securities in Ban For Trade Date {d.strftime('%d-%b-%Y').upper()}:"]
    lines += [f"{i + 1},{s}" for i, s in enumerate(syms)] or ["NIL"]
    return ("\n".join(lines) + "\n").encode()


def syn_index(d: date, nifty=25000.0, bank=55000.0, vix=15.0) -> bytes:
    hdr = ("Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value,"
           "Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,Div Yield")
    rows = [f"Nifty 50,{d:%d-%m-%Y},{nifty},{nifty + 50},{nifty - 50},{nifty},0,0,1,1,20,3,1",
            f"Nifty Bank,{d:%d-%m-%Y},{bank},{bank + 90},{bank - 90},{bank},0,0,1,1,20,3,1",
            f"India VIX,{d:%d-%m-%Y},{vix},{vix},{vix},{vix},0,0,-,-,-,-,-",
            f"Nifty Midcap 50,{d:%d-%m-%Y},1,1,1,1,0,0,1,1,1,1,1"]
    return (hdr + "\n" + "\n".join(rows) + "\n").encode()


def syn_fo(d: date, expiry: date, syms=("INFY", "RELIANCE", "TATASTEEL")) -> bytes:
    rows = []
    k = d.toordinal()
    for s, lvl in (("NIFTY", 25000.0), ("BANKNIFTY", 55000.0)):
        for ex in (expiry, expiry + timedelta(days=28)):
            rows.append({"TradDt": d.isoformat(), "FinInstrmTp": "IDF", "TckrSymb": s, "XpryDt": ex.isoformat(),
                         "OpnPric": lvl, "HghPric": lvl + 10, "LwPric": lvl - 10, "ClsPric": lvl + 5,
                         "SttlmPric": lvl + 5, "TtlTradgVol": 1000, "OpnIntrst": 100000 + 100 * (k % 50),
                         "ChngInOpnIntrst": 100, "UndrlygPric": lvl, "NewBrdLotQty": 75})
        for strike in (lvl - 100, lvl, lvl + 100):
            for t in ("CE", "PE"):
                rows.append({"TradDt": d.isoformat(), "FinInstrmTp": "IDO", "TckrSymb": s, "XpryDt": expiry.isoformat(),
                             "StrkPric": strike, "OptnTp": t, "OpnPric": 100, "HghPric": 110, "LwPric": 90,
                             "ClsPric": 100, "SttlmPric": 100, "TtlTradgVol": 500, "OpnIntrst": 20000 + strike % 7,
                             "ChngInOpnIntrst": 10, "UndrlygPric": lvl, "NewBrdLotQty": 75})
    for i, s in enumerate(syms):
        px = 1000.0 + 100 * i
        rows.append({"TradDt": d.isoformat(), "FinInstrmTp": "STF", "TckrSymb": s, "XpryDt": expiry.isoformat(),
                     "OpnPric": px, "HghPric": px + 5, "LwPric": px - 5, "ClsPric": px + 1, "SttlmPric": px + 1,
                     "TtlTradgVol": 3000, "OpnIntrst": 500000 + 1000 * (k % 30), "ChngInOpnIntrst": 1000,
                     "UndrlygPric": px, "NewBrdLotQty": 500})
        for strike in (px - 20, px, px + 20):
            for t in ("CE", "PE"):
                rows.append({"TradDt": d.isoformat(), "FinInstrmTp": "STO", "TckrSymb": s, "XpryDt": expiry.isoformat(),
                             "StrkPric": strike, "OptnTp": t, "OpnPric": 10, "HghPric": 11, "LwPric": 9, "ClsPric": 10,
                             "SttlmPric": 10, "TtlTradgVol": 50, "OpnIntrst": 3000 + int(strike) % 11,
                             "ChngInOpnIntrst": 5, "UndrlygPric": px, "NewBrdLotQty": 500})
    return udiff_zip(rows, d)


def syn_net(days: list[date], expiry_of=None, fii_long=100, fii_short=900, ban=lambda d: ()):
    """A fake NSE answering for the given sessions (404 otherwise; feeds / lists answer 404)."""
    ds = set(days)
    expiry_of = expiry_of or (lambda d: d + timedelta(days=(3 - d.weekday()) % 7 or 7))

    def fn(url):
        d = _date_in(url)
        if d is None or d not in ds:
            return "missing", b""
        if "sec_bhavdata_full" in url:
            return "ok", syn_sec_full(d)
        if "BhavCopy_NSE_FO" in url:
            return "ok", syn_fo(d, expiry_of(d))
        if "fao_participant_oi" in url:
            return "ok", syn_participant(d, "oi", fii_long, fii_short)
        if "fao_participant_vol" in url:
            return "ok", syn_participant(d, "vol", fii_long, fii_short)
        if "fo_secban" in url:
            return "ok", syn_ban(d, ban(d))
        if "ind_close_all" in url:
            return "ok", syn_index(d)
        return "missing", b""
    return FakeNet(fn)


def _now(d: date, hh=23, mm=5) -> datetime:
    return datetime.combine(d, datetime.min.time()).replace(hour=hh, minute=mm)


_N_ORIG: list = []


def _use_nsedata(path, bars=None):
    """Point the imported stratlab.nsedata at `path` (as oldera_fo.use does); _reset_nsedata() restores it."""
    from stratlab import nsedata as N
    if not _N_ORIG:
        _N_ORIG.append((N.NSE_DIR, N.BARS_DIR))
    N.NSE_DIR = str(path)
    N.BARS_DIR = str(bars) if bars else os.path.join(str(path), "bars")
    N._S.clear()
    return N


def _reset_nsedata():
    from stratlab import nsedata as N
    if _N_ORIG:
        N.NSE_DIR, N.BARS_DIR = _N_ORIG[0]
        N._S.clear()


def _run_syn(root: Path, days: list[date], budget=100000, backfill=False, net=None, now=None):
    net = net or syn_net(days)
    b = BD.Builder(root)
    with patch.object(BD, "NSE_START", days[0]):
        r = b.run(now or _now(days[-1]), http=net.http(budget), feed_client=NoFeeds(), budget=budget,
                  backfill=backfill, log=lambda *a: None)
    return b, r, net


# ══ layout compatibility (the lab's own raw files) ═══════════════════════════════════════════════════════════════
SAMPLE = [date(2025, 6, 2), date(2025, 6, 3), date(2025, 6, 4), date(2025, 6, 5), date(2025, 6, 6)]
SAMPLE_SYMS = ["RELIANCE", "INFY", "HDFCBANK", "TATASTEEL", "ETERNAL", "UNITDSPR", "LODHA", "SBIN", "ADANIPOWER",
               "TCS"]


def _lab_net(days):
    """Serves the scratchpad's raw NSE files; the F&O bhavcopy is rebuilt as UDiFF from the lab's parts and the index
    file from the lab's NIFTY / NIFTY BANK bars (both raw sources were not kept by the lab's fetch)."""
    ds = set(days)
    nd = {int(b[0][:10].replace("-", "")): b for b in json.load(open(os.path.join(LAB_BARS, "_NIFTY50_day.json")))["bars"]}
    bn = {}
    for b in json.load(open(os.path.join(LAB_BARS, "_NIFTYBANK_min.json")))["bars"]:
        bn[int(b[0][:10].replace("-", ""))] = b

    def fn(url):
        d = _date_in(url)
        if d is None or d not in ds:
            return "missing", b""
        k = _ymd(d)
        if "sec_bhavdata_full" in url:
            return "ok", gzip.open(os.path.join(LAB_NSE, "raw_cm", f"sec_bhavdata_full_{k}.csv.gz")).read()
        for kind, sub in (("fao_participant_oi", "oi"), ("fao_participant_vol", "vol"), ("fo_secban", "ban")):
            if kind in url:
                p = os.path.join(LAB_NSE, "raw_part", sub, f"{k}.csv")
                return ("ok", open(p, "rb").read()) if os.path.exists(p) else ("missing", b"")
        if "fii_stats" in url:
            p = os.path.join(LAB_NSE, "raw_fii", f"{k}.xls")
            return ("ok", open(p, "rb").read()) if os.path.exists(p) else ("missing", b"")
        if "BhavCopy_NSE_FO" in url:
            fut = pd.read_csv(os.path.join(LAB_NSE, "fo_parts", "fut", f"{k}.csv.gz"), dtype={"symbol": str})
            oi = pd.read_csv(os.path.join(LAB_NSE, "fo_parts", "optidx", f"{k}.csv.gz"), dtype={"symbol": str,
                                                                                               "type": str})
            return "ok", udiff_zip(parts_to_udiff(fut, oi), d)
        if "ind_close_all" in url:
            a, b = nd[k], bn[k]
            return "ok", syn_index(d, nifty=a[4], bank=b[4]).replace(
                f"Nifty 50,{d:%d-%m-%Y},{a[4]},{a[4] + 50},{a[4] - 50},{a[4]}".encode(),
                f"Nifty 50,{d:%d-%m-%Y},{a[1]},{a[2]},{a[3]},{a[4]}".encode())
        return "missing", b""
    return FakeNet(fn)


def _lab_copy(dst: Path, days) -> Path:
    """The scratchpad's normalised files restricted to `days`, in nsedata's layout."""
    ks = {_ymd(d) for d in days}
    dst.mkdir(parents=True, exist_ok=True)
    rows = []
    for ch in pd.read_csv(os.path.join(LAB_NSE, "delivery.csv.gz"), chunksize=400000, dtype={"symbol": str,
                                                                                            "bars_symbol": str}):
        rows.append(ch[ch["date"].isin(ks)])
    pd.concat(rows).to_csv(dst / "delivery.csv.gz", index=False, compression="gzip")
    st = ["date,status,fmt,n_raw,n_fut,n_optidx,n_optstk_agg"]
    for line in open(os.path.join(LAB_NSE, "fo_parts", "status.csv")).read().splitlines()[1:]:
        if int(line[:10].replace("-", "")) in ks:
            st.append(line)
    for sub in ("fut", "optidx", "optstk", "optstk_full"):
        (dst / "fo_parts" / sub).mkdir(parents=True, exist_ok=True)
        if sub in ("fut", "optidx"):
            for k in ks:
                os.symlink(os.path.join(LAB_NSE, "fo_parts", sub, f"{k}.csv.gz"), dst / "fo_parts" / sub / f"{k}.csv.gz")
    (dst / "fo_parts" / "status.csv").write_text("\n".join(st) + "\n")
    for f in ("participant_oi.csv.gz", "participant_vol.csv.gz", "ban.csv.gz", "fii_stats.csv.gz"):
        shutil.copy(os.path.join(LAB_NSE, f), dst / f)
    (dst / "raw_part" / "ban").mkdir(parents=True)
    for k in ks:
        p = os.path.join(LAB_NSE, "raw_part", "ban", f"{k}.csv")
        if os.path.exists(p):
            shutil.copy(p, dst / "raw_part" / "ban" / f"{k}.csv")
    return dst


def _close(a, b, rel=1e-5) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return False
    both_nan = np.isnan(a) & np.isnan(b)
    return bool(np.all(both_nan | (np.abs(a - b) <= rel * np.maximum(1.0, np.abs(b)))))


def test_layout_compat_nsedata_reads_the_server_dir_like_the_lab_copy():
    """The lab's raw files of 5 sessions through the server's fetch / parse code -> nsedata over the server dir
    returns the same delivery, futures, index-option, participant (OI + volume), ban and FII rows as nsedata over the
    lab's own normalised copy of those sessions (prices: the lab's file rounds to 6 significant digits)."""
    if not os.path.isdir(os.path.join(LAB_NSE, "raw_cm")) or not os.path.exists(os.path.join(LAB_BARS,
                                                                                              "_NIFTY50_day.json")):
        raise Skip("the lab's NSE scratch data is absent")
    root = _tmp()
    b = BD.Builder(root)
    net = _lab_net(SAMPLE)
    with patch.object(BD, "NSE_START", SAMPLE[0]):
        r = b.run(_now(SAMPLE[-1]), http=net.http(), feed_client=NoFeeds(), backfill=False, log=lambda *a: None)
    assert all(st["cash"] == "ok" and st["fo"] == "ok" and st["poi"] == "ok" and st["fii"] == "ok"
               for k, st in b.state()["days"].items() if _d(int(k)) in SAMPLE), b.state()["days"]
    lab = _lab_copy(root / "labcopy", SAMPLE)
    before = 20250609
    res = {}
    for name, path, bars in (("server", b.nse, b.nse / "bars"), ("lab", lab, LAB_BARS)):
        N = _use_nsedata(path, bars)
        res[name] = {"deliv": {s: N.delivery(s, before, n=5) for s in SAMPLE_SYMS},
                     "fut": {s: N.fut_oi(s, before, n=5) for s in ("NIFTY", "BANKNIFTY", "RELIANCE", "INFY")},
                     "chain": N.options_chain("NIFTY", 20250606, "near"),
                     "poi": N.participant(before, n=5), "pvol": N.participant(before, n=5, kind="vol"),
                     "fii": N.fii_stats(before, n=5), "ban": [N.ban_list(_ymd(d)) for d in SAMPLE],
                     "fo_days": list(N.fo_sessions(before)), "alias": N.aliases()}
    s, l = res["server"], res["lab"]
    for sym in SAMPLE_SYMS:
        a, c = s["deliv"][sym], l["deliv"][sym]
        assert len(a) == len(c) == 5, (sym, len(a), len(c))
        for col in BD.DELIVERY_COLS[2:-1]:
            assert _close(a[col], c[col]), (sym, col, a[col], c[col])
    for sym in s["fut"]:
        a, c = s["fut"][sym], l["fut"][sym]
        assert len(a) == len(c) == 5, sym
        for col in a:
            if col == "lot":
                continue
            assert _close(a[col], c[col]), (sym, col, a[col], c[col])
    ca, cl = s["chain"], l["chain"]
    assert ca.date == cl.date and ca.expiry == cl.expiry and np.array_equal(ca.strike, cl.strike)
    for col in ("ce_close", "pe_close", "ce_oi", "pe_oi", "ce_vol", "pe_vol"):
        assert _close(getattr(ca, col), getattr(cl, col)), col
    for k in ("poi", "pvol", "fii"):
        a, c = s[k], l[k]
        assert len(a) == len(c) == 5, k
        for col in c:
            assert _close(a[col], c[col]), (k, col)
    assert s["ban"] == l["ban"] and s["fo_days"] == l["fo_days"]
    assert s["alias"] == l["alias"]                    # (ZOMATO -> ETERNAL etc. only before the 2025 renames)
    assert BD.bars_symbol("ZOMATO") == "ETERNAL" and BD.bars_symbol("RELIANCE") == "RELIANCE" and \
        BD.bars_symbol("NOTLAB") == ""


def test_old_era_cm_bhavcopy_and_mto_route():
    """Before 2020 (no sec_bhavdata_full) the CM bhavcopy + MTO delivery file give the same delivery rows as the
    lab's old-era build (nse_old parts -> nse_old_view/delivery.csv.gz); the old index names (CNX Nifty, CNX Bank)
    map to the index bars."""
    cmp_ = os.path.join(LAB_OLD, "parts", "cm", "20150105.csv.gz")
    view = os.path.join(SCRATCH, "nse_old_view", "delivery.csv.gz")
    if not (os.path.exists(cmp_) and os.path.exists(view)):
        raise Skip("the lab's old-era scratch data is absent")
    d = date(2015, 1, 5)
    cm = pd.read_csv(cmp_, dtype=str, keep_default_na=False)
    mto = pd.read_csv(os.path.join(LAB_OLD, "parts", "mto", "20150105.csv.gz"), dtype=str, keep_default_na=False)
    idx = pd.read_csv(os.path.join(LAB_OLD, "parts", "idx", "20150105.csv.gz"), dtype=str, keep_default_na=False)
    cm_csv = io.StringIO()
    pd.DataFrame({"SYMBOL": cm.symbol, "SERIES": cm.series, "OPEN": cm.open, "HIGH": cm.high, "LOW": cm.low,
                  "CLOSE": cm.close, "LAST": cm["last"], "PREVCLOSE": cm.prev_close, "TOTTRDQTY": cm.volume,
                  "TOTTRDVAL": cm.turnover, "TIMESTAMP": "05-JAN-2015", "TOTALTRADES": cm.trades,
                  "ISIN": cm.isin}).to_csv(cm_csv, index=False)
    zb = io.BytesIO()
    with zipfile.ZipFile(zb, "w") as z:
        z.writestr("cm05JAN2015bhav.csv", cm_csv.getvalue())
    mto_txt = "Record Type,Sr No,Name of Security,Type,Quantity Traded,Deliverable Quantity,% of Deliverable\n" + \
        "\n".join(f"20,{i + 1},{r.symbol},{r.series},{r.qty_traded},{r.deliv_qty},{r.deliv_pct}"
                  for i, r in enumerate(mto.itertuples(index=False)))
    idx_txt = ("Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value\n" +
               "\n".join(f"{r.index},{r.date},{r.open},{r.high},{r.low},{r.close}" for r in idx.itertuples(index=False)))

    def fn(url):
        if "cm05JAN2015bhav" in url:
            return "ok", zb.getvalue()
        if "MTO_05012015" in url:
            return "ok", mto_txt.encode()
        if "ind_close_all_05012015" in url:
            return "ok", idx_txt.encode()
        return "missing", b""
    root = _tmp()
    b = BD.Builder(root)
    net = FakeNet(fn)
    http = net.http()
    assert b.fetch_cash(http, d, set(BD.lab_names())) == "ok"
    assert b.fetch_idx(http, d) == "ok"
    got = b.delivery.read_day(20150105).set_index("symbol")
    lab = pd.read_csv(view, dtype={"symbol": str})
    lab = lab[lab.date == 20150105].set_index("symbol")
    common = sorted(set(got.index) & set(lab.index))
    assert len(common) > 200
    for col in ("open", "high", "low", "close", "prev_close", "volume", "turnover_lacs", "deliv_qty", "deliv_pct",
                "avg_price", "trades"):
        assert _close(got.loc[common, col], lab.loc[common, col], 1e-4), col
    rows = {r[0]: r for r in BD.parse_index_file(idx_txt.encode(), 20150105)}
    assert rows["NIFTY"][4] == 8378.4 and "BANKNIFTY" in rows and "INDIAVIX" in rows
    assert not any("sec_bhavdata_full" in u for u in net.urls)        # pre-2020: never asked


# ══ availability, resumability, budgets, guards ══════════════════════════════════════════════════════════════════
def test_availability_rule_unchanged():
    """The server files carry the trade date; nsedata's own `before` rule decides: a row dated D is not returned for
    before=D (intraday on D), is returned for before=D+1 (next-open decision on D's evening); the ban file of trade
    date D is usable at D's open."""
    days = sessions(date(2026, 3, 2), 6)
    root = _tmp()
    net = syn_net(days, ban=lambda d: ("TATASTEEL",) if d == days[3] else ())
    b, r, _ = _run_syn(root, days, net=net)
    N = _use_nsedata(b.nse)
    D = _ymd(days[3])
    assert int(N.delivery("INFY", D, n=10).date[-1]) == _ymd(days[2])
    assert int(N.delivery("INFY", D + 1, n=10).date[-1]) == D
    assert int(N.fut_oi("NIFTY", D, n=10).date[-1]) == _ymd(days[2])
    assert int(N.participant(D, n=10).date[-1]) == _ymd(days[2])
    assert N.banned("TATASTEEL", D) and not N.banned("TATASTEEL", _ymd(days[2]))
    with N.cutoff(D):
        assert int(N.delivery("INFY", D + 5, n=10).date[-1]) == _ymd(days[2])
        try:
            N.option_eod("NIFTY", _ymd(days[3] + timedelta(days=7)), 25000.0, "CE", D)
            raise AssertionError("look-ahead allowed")
        except N.LookAheadError:
            pass
    # the lab modules read by the builder are unchanged files (imports, not copies)
    src = inspect.getsource(BD)
    assert "stratlab.oldera_fo" in src or "oldera_fo as OF" in src
    assert "events_parse" in src and "build_corpactions.py" in src


def test_backfill_is_resumable_and_idempotent():
    """A run cut by its request budget resumes where it stopped; the result equals one uninterrupted run; a finished
    archive asks nothing again (only the last FINAL_AFTER_DAYS days' 404s are re-asked); a crashed delivery append is
    cut back to the committed length."""
    days = sessions(date(2026, 2, 2), 12)
    now = _now(days[-1] + timedelta(days=6))
    a = _tmp()
    ba, ra, na = _run_syn(a, days, budget=100000, backfill=True, now=now)
    assert ra["stopped"] is None, ra
    b = _tmp()
    runs = 0
    while True:
        bb, rb, nb = _run_syn(b, days, budget=40, backfill=True, now=now)
        runs += 1
        if not rb["stopped"]:
            break
        assert rb["stopped"].startswith("budget") and rb["requests"] == 40
        assert runs < 50
    assert runs > 3

    def digest(root):
        h = {}
        for p in sorted(Path(root, "nse").rglob("*")):
            if p.is_file() and not p.is_symlink() and p.suffix != ".json":
                data = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()   # (gzip mtime)
                h[str(p.relative_to(root))] = hashlib.sha256(data).hexdigest()
        return h
    da, db = digest(a), digest(b)
    # the delivery file's member ORDER differs (fetch order): compare its rows
    da.pop("nse/delivery.csv.gz"), db.pop("nse/delivery.csv.gz")
    assert da == db, sorted(k for k in set(da) | set(db) if da.get(k) != db.get(k))[:10]
    ra_rows = pd.read_csv(a / "nse" / "delivery.csv.gz").sort_values(["date", "symbol"]).reset_index(drop=True)
    rb_rows = pd.read_csv(b / "nse" / "delivery.csv.gz").sort_values(["date", "symbol"]).reset_index(drop=True)
    assert ra_rows.equals(rb_rows) and len(ra_rows) == len(days) * len(SYN)
    # idempotent: nothing final is asked again
    _, r2, n2 = _run_syn(a, days, budget=100000, backfill=True, now=now)
    asked = {_date_in(u) for u in n2.urls if _date_in(u)}
    assert all((now.date() - d).days <= BD.FINAL_AFTER_DAYS for d in asked), sorted(asked)
    # a crashed append (tail written, index not) is truncated before the next append
    df = BD.DeliveryFile(a / "nse")
    size = df.path.stat().st_size
    with open(df.path, "ab") as fh:
        fh.write(b"garbage-tail")
    extra = BD.parse_sec_full(syn_sec_full(days[-1] + timedelta(days=1)), _ymd(days[-1] + timedelta(days=1)))
    df.append(_ymd(days[-1] + timedelta(days=1)), extra)
    assert len(pd.read_csv(df.path)) == len(ra_rows) + len(extra)
    assert df.append(_ymd(days[-1] + timedelta(days=1)), extra) == 0          # same day twice: no-op
    assert df.path.stat().st_size > size


def _fake_kite(scale=lambda sym: 1.0):
    """historical_data stand-in: two sessions (the window's first two weekdays) of full minute bars; a short request
    (the corporate-action overlap re-read) gets every weekday."""
    calls = []

    def fetch(token, frm, to, interval):
        calls.append((token, frm, to, interval))
        if interval == "day":
            out, d = [], frm
            while d <= to:
                if d.weekday() < 5:
                    out.append({"date": datetime.combine(d, datetime.min.time()), "open": 100, "high": 101,
                                "low": 99, "close": 100.5, "volume": 0})
                d += timedelta(days=1)
            return out
        out, d, n = [], frm, 0
        sym = {v: k for k, v in TOKENS.items()}[token]
        short = (to - frm).days <= BM.OVERLAP_DAYS        # the corporate-action overlap re-read: every weekday
        while d <= to and (n < 2 or short):
            if d.weekday() < 5:
                for i in range(375):
                    t = datetime.combine(d, datetime.min.time()).replace(hour=9, minute=15) + timedelta(minutes=i)
                    px = round((1000.0 + i * 0.05) * scale(sym), 2)
                    out.append({"date": t, "open": px, "high": px + 0.5, "low": px - 0.5, "close": px, "volume": 10})
                n += 1
            d += timedelta(days=1)
        return out
    return fetch, calls


TOKENS = {"INFY": 1, "RELIANCE": 2, "_NIFTY50_min": 3, "_NIFTY50_day": 4, "_VIX_day": 5}
MSYMS = ["INFY", "RELIANCE", "_NIFTY50_min"]


def test_kite_minute_backfill_budget_resume_and_corporate_action_refetch():
    root = _tmp()
    mb = BM.MinuteBackfill(root / "minute")
    now = datetime(2026, 11, 14, 22, 0)                 # a Saturday night
    nwin = len(BM.windows(now.date()))
    assert nwin == 67 and all((w1 - w0).days == BM.WINDOW_DAYS - 1 for w0, w1 in BM.windows(now.date()))
    fetch, calls = _fake_kite()
    runs, tot = 0, 0
    while True:
        r = mb.run(now, fetch=fetch, tokens=TOKENS, max_calls=50, symbols=MSYMS, log=lambda *a: None)
        runs += 1
        assert r["calls"] <= 50
        tot += r["calls"]
        if r["stopped"] is None:
            break
        assert r["stopped"] == "call budget"
    minute_calls = [c for c in calls if c[3] == "minute" and (c[2] - c[1]).days > BM.OVERLAP_DAYS]
    overlap_calls = [c for c in calls if c[3] == "minute" and (c[2] - c[1]).days <= BM.OVERLAP_DAYS]
    assert len(minute_calls) == nwin * len(MSYMS) and runs >= 4, (len(minute_calls), runs)
    assert 0 < len(overlap_calls) <= runs * len(MSYMS)    # a resumed instrument re-reads one overlapping session
    assert all((c[2] - c[1]).days <= 59 for c in minute_calls)          # never more than 60 days per request
    p = mb.progress(mb.state(), now.date(), MSYMS, now=now)
    assert p["windows_final"] == p["windows_total"] == nwin * len(MSYMS) and p["progress_pct"] == 100.0
    sd = mb.store.load_symbol("INFY")
    assert sd is not None and sd.n_sessions == 2 * nwin and int(sd.days[0]) == 20160101
    # same reader as the forward lab: the store is a forward_lab_store.Store
    assert isinstance(mb.store, FS.Store) and mb.store.root == root / "minute"
    # the next day: only the newest (open) window is asked, one call per instrument (+ the two daily series)
    calls.clear()
    nxt = datetime(2026, 11, 16, 18, 40)
    r = mb.run(nxt, fetch=fetch, tokens=TOKENS, max_calls=1500, symbols=MSYMS, log=lambda *a: None)
    assert len([c for c in calls if c[3] == "minute"]) == len(MSYMS) and r["ca_refetch"] == []
    # a split goes ex: Kite back-adjusts its history -> every window of that instrument is fetched again
    calls.clear()
    fetch2, calls2 = _fake_kite(scale=lambda s: 0.5 if s == "RELIANCE" else 1.0)
    r = mb.run(nxt + timedelta(days=1), fetch=fetch2, tokens=TOKENS, max_calls=1500, symbols=MSYMS,
               log=lambda *a: None)
    assert r["ca_refetch"] == ["RELIANCE"], r
    rel = [c for c in calls2 if c[3] == "minute" and c[0] == TOKENS["RELIANCE"]]
    assert len(rel) == nwin                           # the detecting (newest) window, then every older one again
    sd = mb.store.load_symbol("RELIANCE")
    assert abs(float(sd.c[0]) - 500.0) < 1.0           # the whole stored window is adjusted


def test_corporate_action_at_the_start_of_a_new_window_is_detected():
    """Regression (review of PR #25): a split goes ex on the first session of a NEW 60-day window. The newest fetch
    then has no session in common with the stored bars; the check must still compare an OVERLAPPING stored session
    (one extra short request) and refetch the whole instrument on Kite's new basis — never leave the older (final)
    windows on the old price basis. Without any overlap it must not write the new window (retried next run)."""
    root = _tmp()
    mb = BM.MinuteBackfill(root / "minute")
    tok = {"INFY": 1}
    ws = BM.windows(date(2026, 11, 14))
    w_new = ws[-1][0]                                   # the start of the newest grid window
    split_day = w_new
    while split_day.weekday() >= 5:
        split_day += timedelta(days=1)
    state = {"adjusted": False}

    def fetch(token, frm, to, interval):
        out, d = [], frm
        while d <= to:
            if d.weekday() < 5:
                f = 0.5 if state["adjusted"] or d >= split_day else 1.0       # Kite back-adjusts after the ex-date
                for i in range(20):
                    t = datetime.combine(d, datetime.min.time()).replace(hour=9, minute=15) + timedelta(minutes=i)
                    px = round(1000.0 * f + i * 0.05, 2)
                    out.append({"date": t, "open": px, "high": px + 0.5, "low": px - 0.5, "close": px, "volume": 10})
            d += timedelta(days=1)
        return out
    before = w_new - timedelta(days=3)                 # the evening before the newest window starts
    r1 = mb.run(datetime.combine(before, datetime.min.time()).replace(hour=22), fetch=fetch, tokens=tok,
                max_calls=10000, symbols=["INFY"], log=lambda *a: None)
    assert r1["stopped"] is None and r1["ca_refetch"] == []
    old = mb.store.raw("INFY", upto=_ymd(before))
    assert abs(float(old["c"][0]) - 1000.0) < 1.0
    state["adjusted"] = True                            # the split went ex on split_day: Kite's whole history halves
    after = datetime.combine(split_day, datetime.min.time()).replace(hour=18, minute=40)
    r2 = mb.run(after, fetch=fetch, tokens=tok, max_calls=10000, symbols=["INFY"], log=lambda *a: None)
    assert r2["ca_refetch"] == ["INFY"], r2
    z = mb.store.raw("INFY")
    assert float(np.max(z["c"])) < 600.0, "older windows left on the pre-split price basis"
    # no overlapping session obtainable (Kite returns nothing for it): the new window is NOT written, retried later
    root2 = _tmp()
    mb2 = BM.MinuteBackfill(root2 / "minute")
    state["adjusted"] = False
    mb2.run(datetime.combine(before, datetime.min.time()).replace(hour=22), fetch=fetch, tokens=tok,
            max_calls=10000, symbols=["INFY"], log=lambda *a: None)
    state["adjusted"] = True
    last = int(mb2.state()["symbols"]["INFY"]["last"])

    def fetch_no_overlap(token, frm, to, interval):
        return [] if frm <= _d(last) <= to and to < w_new else fetch(token, frm, to, interval)
    r3 = mb2.run(after, fetch=fetch_no_overlap, tokens=tok, max_calls=10000, symbols=["INFY"], log=lambda *a: None)
    assert r3["ca_refetch"] == [] and int(mb2.state()["symbols"]["INFY"]["last"]) == last, r3
    assert str(_ymd(w_new)) not in mb2.state()["symbols"]["INFY"]["windows"]
    r4 = mb2.run(after + timedelta(hours=12), fetch=fetch, tokens=tok, max_calls=10000, symbols=["INFY"],
                 log=lambda *a: None)
    assert r4["ca_refetch"] == ["INFY"], r4


class _Resp:
    def __init__(self, url, status=200, body=b"{}"):
        self.url, self.status_code, self.content = url, status, body


class _FakeSession:
    """requests.Session stand-in under pead_data._Client: every IFR page is full (1,000 rows), so the client keeps
    paging (up to 30 pages) unless the builder's guard stops it."""

    def __init__(self, status=200):
        self.calls, self.status = [], status
        self.headers = {}

    def get(self, url, **kw):
        self.calls.append(url)
        if "/api/" not in url:
            return _Resp(url)
        if self.status != 200:
            return _Resp(url, self.status, b"")
        import stratlab.events_fetch as EF
        return _Resp(url, 200, json.dumps({"data": [{"x": i} for i in range(EF.IFR_PAGE)]}).encode())


def _guarded_ifr(budget=6, stop=None, status=200):
    import pead_data
    cl = pead_data._Client(0.0)
    sess = _FakeSession(status)
    cl.cl.s = sess
    get = sess.get
    cl.cl.s.get = lambda url, **kw: get(url, **{**kw, "timeout": 30})
    http = BD.Http(budget, get=lambda u: ("missing", b""), pause=0.0, sleep=lambda s: None, stop=stop)
    b = BD.Builder(_tmp())
    st = b.state()
    return b, st, cl, sess, http


def test_filing_feed_pagination_obeys_the_request_budget_and_time_guard():
    """Regression (review of PR #25): pead_data._Client.fetch pages the IFR feed up to 30 times and events_fetch's
    client adds warm-up / retry requests. EVERY underlying request must go through the run's budget, time-window
    and refusal checks; a guard that trips stops mid-pagination and the window stays pending (resumable)."""
    today = date(2026, 11, 16)
    with patch.object(BD, "FEEDS", ("ifr",)):
        b, st, cl, sess, http = _guarded_ifr(budget=6)
        try:
            b.fetch_events(http, st, today, newest_only=True, client=cl)
            raise AssertionError(f"budget not enforced: {len(sess.calls)} requests made")
        except BD.BudgetReached:
            pass
        assert len(sess.calls) <= 6 and http.n == len(sess.calls), (len(sess.calls), http.n)
        assert not any(k.startswith("ev:ifr") and v.get("status") == "ok" for k, v in st["windows"].items())
        # the PEAD window opens mid-pagination: no request after it
        n = {"k": 0}

        def stop():
            n["k"] += 1
            return "PEAD window" if n["k"] > 3 else None
        b, st, cl, sess, http = _guarded_ifr(budget=1000, stop=stop)
        try:
            b.fetch_events(http, st, today, newest_only=True, client=cl)
            raise AssertionError(f"time guard not enforced: {len(sess.calls)} requests made")
        except BD.StopRun:
            pass
        assert len(sess.calls) == 3, len(sess.calls)
        # refusals inside the client count too: three in a row end the run
        b, st, cl, sess, http = _guarded_ifr(budget=1000, status=403)
        try:
            b.fetch_events(http, st, today, newest_only=False, client=cl)
            raise AssertionError("refusals not counted")
        except BD.NSERefused:
            pass
        assert sum(1 for u in sess.calls if "/api/" in u) == BD.REFUSAL_STRIKES
    # the live PEAD agent's client is untouched (the guard is injected per builder run only)
    import pead_data
    fresh = pead_data._Client(0.0)
    assert not getattr(fresh.cl.s, "_builder_guarded", False)


def test_market_hours_guard():
    t = lambda h, m, d=date(2026, 11, 16): datetime.combine(d, datetime.min.time()).replace(hour=h, minute=m)
    assert not BD.in_market_hours(t(8, 59)) and BD.in_market_hours(t(9, 0)) and BD.in_market_hours(t(15, 44))
    assert not BD.in_market_hours(t(15, 45)) and not BD.in_market_hours(t(11, 0, date(2026, 11, 15)))   # Sunday
    saved = settings.builder_data_enabled
    settings.builder_data_enabled = True
    try:
        with patch.object(BM, "kite_fetch", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Kite called"))):
            r = BD.minute_job(now=t(10, 0), root=_tmp())
            assert r == {"skipped": "market hours (09:00-15:45 IST on a trading day)"}
    finally:
        settings.builder_data_enabled = saved
    # a run that reaches 09:00 stops before the next call
    root = _tmp()
    mb = BM.MinuteBackfill(root / "minute")
    fetch, calls = _fake_kite()
    clock = iter([t(8, 58)] * 5 + [t(9, 0)] * 1000)
    r = mb.run(t(8, 58), fetch=fetch, tokens=TOKENS, max_calls=1500, symbols=MSYMS, market_hours=BD.in_market_hours,
               clock=lambda: next(clock), log=lambda *a: None)
    assert r["stopped"] == "market hours" and r["calls"] == 5


def test_disk_cap_stops_writing_and_never_deletes():
    days = sessions(date(2026, 4, 6), 4)
    root = _tmp()
    b, r, _ = _run_syn(root, days[:2])
    before = {str(p): p.stat().st_mtime for p in root.rglob("*") if p.is_file() and not p.is_symlink()}
    saved = settings.builder_data_max_mb
    settings.builder_data_max_mb = BD.disk_bytes(root) / 1048576 * 0.9
    try:
        b2, r2, _ = _run_syn(root, days)
        assert r2["stopped"].startswith("disk_cap"), r2
        after = {str(p): p.stat().st_mtime for p in root.rglob("*") if p.is_file() and not p.is_symlink()}
        assert set(before) <= set(after)                       # nothing deleted
        changed = [p for p in before if after[p] != before[p] and not p.endswith(".json")]
        assert not changed and set(after) - set(before) <= {str(root / "state.json")}, (changed,
                                                                                         set(after) - set(before))
        mb = BM.MinuteBackfill(root / "minute")
        fetch, calls = _fake_kite()

        def cap():
            if BD.disk_bytes(root) >= BD.max_bytes():
                raise BD.DiskCapReached("cap")
        r3 = mb.run(datetime(2026, 11, 14, 22, 0), fetch=fetch, tokens=TOKENS, symbols=MSYMS, cap_check=cap,
                    log=lambda *a: None)
        assert r3["stopped"].startswith("disk_cap") and r3["calls"] == 0 and not (root / "minute" / "bars").exists()
        st = BD.Builder(root).status(_now(days[-1]))
        assert st["disk"]["cap_reached"] is True
    finally:
        settings.builder_data_max_mb = saved


def test_nse_refusal_is_recorded_and_fails_safe():
    root = _tmp()
    net = FakeNet(lambda url: ("refused", b"HTTP 403"))
    b = BD.Builder(root)
    r = b.run(_now(date(2026, 11, 13)), http=net.http(), feed_client=NoFeeds(), log=lambda *a: None)
    assert r["stopped"].startswith("refused") and len(net.urls) == BD.REFUSAL_STRIKES
    st = b.state()
    assert st["refused_until"] and st["last_error"]["error"].startswith("refused")
    # no retry for 24 h: the next run asks nothing
    r2 = b.run(_now(date(2026, 11, 14), 5, 30), http=net.http(), feed_client=NoFeeds(), log=lambda *a: None)
    assert r2["stopped"].startswith("refused: waiting") and len(net.urls) == BD.REFUSAL_STRIKES
    s = b.status(_now(date(2026, 11, 14)))
    assert s["refused_until"] and s["datasets"]["delivery"]["rows"] == 0
    # an unreachable NSE (timeouts / 5xx / proxy denial): the run ends after MAX_FAILURES failed requests in a row
    root2 = _tmp()
    net2 = FakeNet(lambda url: ("error", b"HTTP 503"))
    r3 = BD.Builder(root2).run(_now(date(2026, 11, 13)), http=net2.http(), feed_client=NoFeeds(), log=lambda *a: None)
    assert r3["stopped"].startswith("stopped:") and len(net2.urls) == BD.MAX_FAILURES * 3, (r3, len(net2.urls))
    # the forward lab's engine then simply gets no NSE dir (no view): nothing breaks
    assert BD.engine_env(root) == {} or os.getenv("STRATLAB_NSE_DIR")
    # the Http client never retries a refusal past the strikes and never changes its identity
    src = inspect.getsource(BD.Http) + inspect.getsource(BD._requests_get)
    assert "cookie" not in src.lower() and "proxy" not in BD._requests_get.__doc__.lower()


def test_flag_off_nothing_scheduled_and_no_network():
    saved = settings.builder_data_enabled
    try:
        settings.builder_data_enabled = False
        with patch.object(BD, "_requests_get", lambda *a: (_ for _ in ()).throw(AssertionError("network"))), \
                patch.object(BM, "kite_fetch", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Kite"))):
            assert BD.nse_job()["skipped"].startswith("BUILDER_DATA_ENABLED is off")
            assert BD.minute_job()["skipped"].startswith("BUILDER_DATA_ENABLED is off")
        import platform_scheduler as ps

        class _S:
            def __init__(self):
                self.ids = []

            def add_job(self, *a, **k):
                self.ids.append(k.get("id"))

            def start(self):
                pass

            def get_jobs(self):
                return self.ids
        for flag, expect in ((False, False), (True, True)):
            settings.builder_data_enabled = flag
            s = ps.PlatformScheduler()
            s._sched = _S()
            s.start()
            ids = [i for i in s._sched.ids if i and i.startswith("builder_")]
            assert bool(ids) is expect, (flag, ids)
            if expect:
                assert sorted(ids) == ["builder_minute", "builder_minute_early", "builder_minute_wkend",
                                       "builder_nse", "builder_nse_early", "builder_nse_wkend"]
        settings.builder_data_enabled = True
        saved_mode = settings.trading_mode
        settings.trading_mode = "LIVE"
        try:
            assert not BD.enabled()
        finally:
            settings.trading_mode = saved_mode
        # the NSE job stays out of the PEAD agent's fetch windows
        assert BD.nse_job(now=datetime(2026, 11, 16, 19, 40))["skipped"].startswith("PEAD")
        assert BD.nse_job(now=datetime(2026, 11, 16, 8, 20))["skipped"].startswith("PEAD")
    finally:
        settings.builder_data_enabled = saved


def test_forward_lab_unchanged():
    """The rule file and its pinned hash, the forward lab's store root and behaviour are untouched: the builder never
    writes under the forward-lab directory, and with no builder data the engine's environment is unchanged."""
    assert FL.file_sha256(FL.RULE_FILE) == FL.RULE_SHA256
    assert FL.load_rule()["_sha256"] == FL.RULE_SHA256
    root = _tmp()
    assert BD.engine_env(root) == {} or os.getenv("STRATLAB_NSE_DIR")
    import test_forward_lab as TFL
    assert len(TFL.TESTS) >= 21                        # (22 with the corporate-action review fix)
    fl_root = _tmp()
    with patch.object(FS, "default_root", lambda: fl_root):
        days = sessions(date(2026, 3, 2), 3)
        _run_syn(root, days)
        assert not any(fl_root.rglob("*"))
    assert BD.Builder(root).minute != FS.default_root() / "bars"
    if __name__ == "__main__":                           # standalone: run the forward lab's own tests too
        failed = 0
        for n, f in TFL.TESTS:
            try:
                f()
            except TFL.Skip:
                pass
            except Exception as exc:
                failed += 1
                print("forward lab test failed:", n, exc)
        assert failed == 0


# ══ the forward lab's nsedata-based catalogue ids ════════════════════════════════════════════════════════════════
R4_MODULES = ("stratlab.catalog.r4_delivery", "stratlab.catalog.r4_mechanics", "stratlab.catalog.r4_participant",
              "stratlab.catalog.r4_futoi")


def r4_ids() -> list[str]:
    import forward_lab_engine as E
    return sorted(i for i, sp in E.native_specs().items() if sp.fn.__module__ in R4_MODULES)


def test_forward_lab_nse_ids_produce_trades_with_the_server_archive():
    """The 85 catalogue ids that read NSE archive files: with no archive every one whose rule reaches nsedata on a
    session errors (the server today; 51 of 85 on the fixture session, the others stop at their own gates first); with the builder's archive (small synthetic fixtures: 3 stocks + the two indices, 70 sessions, FII
    index-futures long ratio 10 %) the forward lab's engine process — through engine_env(), the rule file untouched —
    runs every one of them without an error and they produce trades."""
    import test_forward_lab as TFL
    ids = r4_ids()
    assert len(ids) == 85, len(ids)
    days = sessions(date(2026, 1, 5), 70)
    tmp = _tmp()
    store = TFL.syn_store(tmp / "fl", [_ymd(d) for d in days])
    root = tmp / "builder"
    # FII index-futures long ratio 10 % every session; the NIFTY / BANKNIFTY / stock monthly contracts expire on the
    # second-last session (the evening of an expiry: the index "new series" rule)
    net = syn_net(days, fii_long=100, fii_short=900,
                  expiry_of=lambda d: days[-2] if d <= days[-2] else days[-2] + timedelta(days=28))
    b, r, _ = _run_syn(root, days, net=net, backfill=True)
    assert r["view"]["sessions"] == 70 and (b.view / "delivery.csv.gz").exists()
    env = BD.engine_env(root)
    D, prev = _ymd(days[-1]), _ymd(days[-2])
    # without the archive: the NSE ids error
    empty = tmp / "empty_nse"
    empty.mkdir()
    with patch.dict(os.environ, {"STRATLAB_NSE_DIR": str(empty), "STRATLAB_BARS_DIR": str(empty)}):
        assert BD.engine_env(root) == {}
        res0 = FL.run_engine_process(store.root, tmp / "state0", D, prev, D, tmp / "w0", time_limit_s=900, ids=ids)
    err0 = [i for i in ids if (res0["strategies"].get(i) or {}).get("error")]
    assert len(err0) >= 50, len(err0)               # every id whose rule reaches nsedata on this session
    # with the archive (the engine process gets STRATLAB_NSE_DIR from engine_env)
    assert env == {"STRATLAB_NSE_DIR": str(b.view), "STRATLAB_BARS_DIR": str(b.view / "bars")}
    saved = os.environ.pop("STRATLAB_NSE_DIR", None), os.environ.pop("STRATLAB_BARS_DIR", None)
    try:
        with patch.object(BD, "default_root", lambda: root):
            res = FL.run_engine_process(store.root, tmp / "state1", D, prev, D, tmp / "w1", time_limit_s=900, ids=ids)
    finally:
        for k, v in zip(("STRATLAB_NSE_DIR", "STRATLAB_BARS_DIR"), saved):
            if v is not None:
                os.environ[k] = v
    errs = {i: res["strategies"][i]["error"] for i in ids if (res["strategies"].get(i) or {}).get("error")}
    assert not errs, list(errs.items())[:3]
    traded = {i: len(res["strategies"][i]["closed"]) + len(res["strategies"][i]["open"]) for i in ids
              if res["strategies"][i]["closed"] or res["strategies"][i]["open"]}
    assert traded, "no trades"
    assert any(i.startswith("r4_participant_fii_lr") for i in traded), traded
    assert any(i.startswith("r4_mechanics") for i in traded), traded
    print(f"      r4 ids with trades on {D}: {len(traded)} ({sum(traded.values())} trades): "
          f"{', '.join(sorted(traded)[:6])}{' ...' if len(traded) > 6 else ''}")


def test_forward_lab_nse_ids_match_the_lab_on_real_sessions():
    """Real data: the forward engine (day by day, state carried) on 5 sessions of the lab's own bars, reading an NSE
    directory in the server layout that holds only files dated <= the last session, gives the lab's own trades for
    all 85 NSE-file ids (forward_lab_build.parity: sim.run_symbol over the same data), with trades."""
    import forward_lab_build as B
    cache = B.lab_cache(SCRATCH)
    if not (os.path.isdir(cache) and os.path.isdir(os.path.join(LAB_NSE, "fo_parts"))):
        raise Skip("the lab's 3-year bars / NSE scratch data are absent")
    ids = r4_ids()
    days = [20261026, 20261027, 20261028, 20261029, 20261030]
    st = [l.split(",") for l in open(os.path.join(LAB_NSE, "fo_parts", "status.csv")).read().splitlines()[1:]]
    ok = [_d(int(r[0].replace("-", ""))) for r in st if r[1] == "ok" and int(r[0].replace("-", "")) <= days[-1]]
    tmp = _tmp()
    copy = _lab_copy(tmp / "nse", ok[-250:])
    for sub in ("optstk", "optstk_full"):
        for d in ok[-250:]:
            f = os.path.join(LAB_NSE, "fo_parts", sub, f"{_ymd(d)}.csv.gz")
            if os.path.exists(f):
                os.symlink(f, copy / "fo_parts" / sub / os.path.basename(f))
    N = _use_nsedata(copy, LAB_BARS)
    try:
        r = B.parity(SCRATCH, days, ids, anchored=True, tmp=str(tmp / "par"), log=lambda *a: None)
    finally:
        _reset_nsedata()
    bad = {k: v for k, v in r["strategies"].items() if not v["same"]}
    assert not bad, json.dumps(bad, default=str)[:1500]
    traded = {k: v["fwd_closed"] + v["fwd_open"] for k, v in r["strategies"].items() if v["fwd_closed"] + v["fwd_open"]}
    assert r["matched"] == 85 and traded
    import forward_lab_engine as E
    mods = {E.native_specs()[k].fn.__module__.rsplit(".", 1)[-1] for k in traded}
    assert mods == {"r4_delivery", "r4_mechanics", "r4_participant", "r4_futoi"}, mods
    print(f"      85/85 NSE-file ids identical to the lab on 5 real sessions; {len(traded)} ids traded "
          f"({sum(traded.values())} trades/positions; modules {sorted(mods)})")


# ══ API, read-only ═══════════════════════════════════════════════════════════════════════════════════════════════
def test_v2_endpoint_shape():
    import main
    from fastapi.testclient import TestClient
    root = _tmp()
    _run_syn(root, sessions(date(2026, 3, 2), 3))
    with patch.object(BD, "default_root", lambda: root):
        c = TestClient(main.app)
        saved = settings.api_key
        settings.api_key = "k"
        try:
            assert c.get("/api/v2/builder/data").status_code == 401
            r = c.get("/api/v2/builder/data", headers={"X-API-Key": "k"})
            assert r.status_code == 200
            j = r.json()
            assert {"enabled", "root", "datasets", "disk", "schedule", "budgets", "last_run"} <= set(j), set(j)
            assert j["enabled"] is False and j["disk"]["cap_mb"] == 1800.0
            want = {"delivery", "fo_parts", "participant_oi", "participant_vol", "ban", "fii_stats", "index_spot",
                    "deals", "corpactions", "index_membership", "events", "minute", "nse_view"}
            assert set(j["datasets"]) == want, set(j["datasets"])
            for k, v in j["datasets"].items():
                assert {"from", "to", "files", "mb", "last_update", "last_error", "progress_pct"} <= set(v), k
            assert j["datasets"]["delivery"]["rows"] == 3 * len(SYN) and j["datasets"]["delivery"]["files"] == 3
        finally:
            settings.api_key = saved


def test_read_only_towards_trading():
    for mod in (BD, BM):
        src = inspect.getsource(mod)
        for bad in ("place_order", "paper_fill_at", "paper_marketable", "modify_order", "cancel_order",
                    "set_strategy(", "set_global(", "record_trade(", "set_agent_enabled", "squareoff", "ALL_AGENTS"):
            assert bad not in src, (mod.__name__, bad)


def _resetting(f):
    def go():
        try:
            return f()
        finally:
            _reset_nsedata()
    go.__name__ = f.__name__
    return go


TESTS = [(n, _resetting(f)) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]


def _main() -> int:
    failed = 0
    for n, f in TESTS:
        t0 = time.time()
        try:
            f()
            print(f"  ok    {n} ({time.time() - t0:.1f}s)")
        except Skip as exc:
            print(f"  skip  {n}: {exc}")
        except Exception as exc:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"  FAIL  {n}: {type(exc).__name__}: {exc}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
