"""
builder_universe.py — the STOCK UNIVERSE of the builder data layer's minute store (BUILDER_MINUTE_UNIVERSE).

  nifty100  the forward lab's list exactly (forward_lab_store.MINUTE_SYMBOLS: the lab's 97 NIFTY-100 names + NIFTY 50 +
            NIFTY BANK) — what builder_minute fetched before
  nifty200  (default) the same 99 instruments, FIRST and unchanged, then the ADDED names, sorted:
              * the current NIFTY 200 list from NSE Indices: the newest daily list the data layer saved
                (<root>/flow/raw/index/daily/NIFTY200/YYYYMMDD.csv, nsearchives ind_nifty200list.csv, fetched by
                builder_data.fetch_index from its first run on), else NIFTY200_SNAPSHOT below (that file as fetched on
                2026-11-16)
              * every historical NIFTY 100 member (NIFTY 50 + NIFTY NEXT 50, the data layer's month-end index files,
                <root>/flow/index_membership/index_members.csv.gz) whose membership reaches 2016 or later (the minute
                history starts 2016-01-01) — so names that LEFT the index are fetched too, when Kite still lists them
            NIFTY 200 membership is not point-in-time: NSE Indices' month-end files the data layer reads cover NIFTY 50
            / NEXT 50 only, so before the server's first daily NIFTY 200 list only the current list exists. Delisted /
            merged names are not served by Kite (no instrument token): they are recorded as "missing" and cannot be
            in the minute universe — SURVIVORSHIP BIAS of the added names, stated with every batch (rule builder/v2).

The resolved lists are written to <root>/minute/universe.json (read by stratlab.builder_server_panels.ServerSource:
"base" = the 97, "extra" = the added names). The forward lab's own store and universe are never touched.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

MODES = ("nifty100", "nifty200")
DEFAULT_MODE = "nifty200"
HIST_FROM = 20160101                  # the minute history's first day: older index members are not fetched
# nsearchives ind_nifty200list.csv as fetched on 2026-11-16 (200 EQ names; the data layer's own daily list replaces it)
NIFTY200_SNAPSHOT_DATE = 20261116
NIFTY200_SNAPSHOT = tuple("""
360ONE ABB ABCAPITAL ADANIENSOL ADANIENT ADANIGREEN ADANIPORTS ADANIPOWER AMBUJACEM APARINDS APLAPOLLO
APOLLOHOSP ASHOKLEY ASIANPAINT ASTRAL ATGL AUBANK AUROPHARMA AXISBANK BAJAJ-AUTO BAJAJFINSV BAJAJHLDNG
BAJFINANCE BANKBARODA BANKINDIA BDL BEL BHARATFORG BHARTIARTL BHEL BIOCON BLUESTARCO BOSCHLTD BPCL BRITANNIA BSE
CANBK CGPOWER CHOLAFIN CIPLA COALINDIA COCHINSHIP COFORGE COLPAL CONCOR CUMMINSIND DABUR DIVISLAB DIXON DLF
DMART DRREDDY EICHERMOT ENRIN ETERNAL EXIDEIND FEDERALBNK FORTIS GAIL GLENMARK GMRAIRPORT GODFRYPHLP GODREJCP
GODREJPROP GRASIM GROWW GVT&D HAL HAVELLS HCLTECH HDFCAMC HDFCBANK HDFCLIFE HEROMOTOCO HINDALCO HINDCOPPER
HINDPETRO HINDUNILVR HINDZINC HYUNDAI ICICIAMC ICICIBANK ICICIGI IDEA IDFCFIRSTB INDHOTEL INDIANB INDIGO
INDUSINDBK INDUSTOWER INFY IOC IRCTC IREDA IRFC ITC JINDALSTEL JIOFIN JSWENERGY JSWSTEEL JUBLFOOD KALYANKJIL KEI
KOTAKBANK LAURUSLABS LENSKART LGEINDIA LICHSGFIN LICI LODHA LT LTF LTM LUPIN M&M M&MFIN MAHABANK MANKIND MARICO
MARUTI MAXHEALTH MAZDOCK MCX MEESHO MFSL MOTHERSON MOTILALOFS MPHASIS MRF MUTHOOTFIN NATIONALUM NAUKRI NESTLEIND
NHPC NLCINDIA NMDC NTPC NYKAA OBEROIRLTY OFSS OIL ONGC PAGEIND PATANJALI PAYTM PERSISTENT PFC PHOENIXLTD
PIDILITIND PIIND PNB POLICYBZR POLYCAB POWERGRID POWERINDIA PREMIERENE PRESTIGE RADICO RECLTD RELIANCE RVNL SAIL
SBICARD SBILIFE SBIN SHRIRAMFIN SIEMENS SOLARINDS SRF SUNPHARMA SUPREMEIND SUZLON SWIGGY TATACAP TATACOMM
TATACONSUM TATAPOWER TATASTEEL TCS TECHM TIINDIA TITAN TMCV TMPV TORNTPHARM TRENT TVSMOTOR ULTRACEMCO UNIONBANK
UNITDSPR UPL VAML VBL VEDL VMM VOLTAS WAAREEENER WIPRO YESBANK ZYDUSLIFE
""".split())
ALIASES = {"ZOMATO": "ETERNAL", "MCDOWELL-N": "UNITDSPR", "MACROTECH": "LODHA"}   # old symbols of base names
MEMBERSHIP_NOTE = ("added names: the current NIFTY 200 list (not point-in-time; NSE Indices' month-end files cover "
                   "NIFTY 50 / NEXT 50 only) + historical NIFTY 100 members since 2016; delisted names are not on Kite "
                   "(survivorship bias)")
# Volume budget the cap is sized against (MB): forward lab, option-bar archiver, builder, builder data (its cap)
VOLUME_MB = 5000.0
VOLUME_CAPS_MB = {"forward_lab": 500.0, "option_bar_archiver": 1024.0, "builder": 400.0}
VOLUME_WORST_MAX_MB = 4600.0


def mode() -> str:
    try:
        from config import settings
        m = str(getattr(settings, "builder_minute_universe", DEFAULT_MODE) or DEFAULT_MODE)
    except Exception:
        m = os.getenv("BUILDER_MINUTE_UNIVERSE", DEFAULT_MODE)
    m = m.strip().lower()
    return m if m in MODES else DEFAULT_MODE


def _root(root=None) -> Path:
    if root is not None:
        return Path(root)
    import builder_data
    return builder_data.default_root()


def base_names() -> list:
    import forward_lab_store as FS
    return list(FS.UNIVERSE_STOCKS)


def index_names() -> list:
    import forward_lab_store as FS
    return list(FS.INDEX_MIN)


def parse_list_csv(body: bytes) -> list:
    """Symbols of an NSE Indices constituent list (Company Name, Industry, Symbol, Series, ISIN Code)."""
    out = []
    for r in csv.DictReader(io.StringIO(body.decode("utf-8-sig", errors="replace"))):
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
        s = r.get("symbol", "").upper()
        if s and r.get("series", "EQ").upper() in ("EQ", "BE", "BZ", ""):
            out.append(s)
    return out


def current_nifty200(root=None) -> tuple[list, dict]:
    """(symbols, source) of the newest NIFTY 200 list: the data layer's saved daily list, else the snapshot."""
    d = _root(root) / "flow" / "raw" / "index" / "daily" / "NIFTY200"
    files = sorted(d.glob("*.csv")) if d.exists() else []
    for f in reversed(files):
        try:
            syms = parse_list_csv(f.read_bytes())
        except Exception:
            continue
        if len(syms) >= 150:
            return sorted(set(syms)), {"source": "nse daily list", "date": int(f.name[:8]), "n": len(set(syms))}
    return sorted(NIFTY200_SNAPSHOT), {"source": "committed snapshot", "date": NIFTY200_SNAPSHOT_DATE,
                                       "n": len(NIFTY200_SNAPSHOT)}


def historical_nifty100(root=None, since: int = HIST_FROM) -> tuple[list, dict]:
    """(symbols, source): every NIFTY 50 / NEXT 50 / 100 member of the data layer's index membership file whose
    membership ends on or after `since` (or is open). Symbols are the file's latest symbols."""
    p = _root(root) / "flow" / "index_membership" / "index_members.csv.gz"
    if not p.exists():
        return [], {"source": "none (index membership not built yet)", "n": 0}
    out = set()
    with gzip.open(p, "rt") as fh:
        for r in csv.DictReader(fh):
            if r.get("index") not in ("NIFTY100", "NIFTY50", "NIFTYNEXT50"):
                continue
            to = (r.get("to_date") or "").strip()
            if to and int(to) < int(since):
                continue
            s = (r.get("symbol") or "").strip().upper()
            if s:
                out.add(s)
    return sorted(out), {"source": "index_members.csv.gz", "n": len(out), "since": int(since)}


def extra_names(root=None) -> tuple[list, dict]:
    """(the added names, sources): current NIFTY 200 + historical NIFTY 100 members, minus the 97, their old
    symbols and the index names. Sorted."""
    cur, src_c = current_nifty200(root)
    hist, src_h = historical_nifty100(root)
    skip = set(base_names()) | set(ALIASES) | set(index_names())
    ext = sorted((set(cur) | set(hist)) - skip)
    return ext, {"current_nifty200": src_c, "historical_nifty100": src_h,
                 "from_current": len(set(cur) - skip), "from_history_only": len(set(hist) - set(cur) - skip)}


def minute_symbols(m: Optional[str] = None, root=None) -> list:
    """The minute backfill's instrument list: the forward lab's 99 first (unchanged), then the added names."""
    import forward_lab_store as FS
    m = m or mode()
    if m != "nifty200":
        return list(FS.MINUTE_SYMBOLS)
    return list(FS.MINUTE_SYMBOLS) + extra_names(root)[0]


DUMP_MIN_INSTRUMENTS = 1000          # Kite's NSE dump has ~9,000 rows: fewer = a failed / truncated answer


def resolve_tokens(names: list, instruments_fn=None) -> dict:
    """Kite tokens of `names` from ONE NSE instrument dump, read-only. A name is "missing" only when a SUCCESSFUL dump
    (no error, >= DUMP_MIN_INSTRUMENTS rows) does not list it; a failed, empty or implausibly small dump (kite_client
    swallows its errors and returns []) leaves every name "unresolved" — never recorded as missing, retried next run.
    Returns {"tokens", "missing", "unresolved", "dump_size", "error"}."""
    names = sorted(set(names))
    if not names:
        return {"tokens": {}, "missing": [], "unresolved": [], "dump_size": None, "error": None}
    try:
        if instruments_fn is None:
            from kite_client import kite_client
            instruments_fn = lambda: kite_client.get_instruments("NSE")  # noqa: E731
        dump = list(instruments_fn() or [])
    except Exception as exc:
        return {"tokens": {}, "missing": [], "unresolved": names, "dump_size": 0,
                "error": f"instrument dump failed: {type(exc).__name__}: {str(exc)[:160]}"}
    if len(dump) < DUMP_MIN_INSTRUMENTS:
        return {"tokens": {}, "missing": [], "unresolved": names, "dump_size": len(dump),
                "error": f"instrument dump has {len(dump)} rows (< {DUMP_MIN_INSTRUMENTS}): treated as failed"}
    want, tok = set(names), {}
    for inst in dump:
        s = inst.get("tradingsymbol")
        if s in want and inst.get("instrument_token"):
            tok[s] = int(inst["instrument_token"])
    return {"tokens": tok, "missing": sorted(want - set(tok)), "unresolved": [], "dump_size": len(dump),
            "error": None}


def kite_tokens(names: list) -> dict:
    """{name: token} for the stock names among `names` (one instruments dump, read-only). Prefer resolve_tokens(),
    which tells a failed lookup from a missing instrument."""
    return resolve_tokens(names)["tokens"]


def write_universe_file(minute_root, missing: Optional[list] = None, root=None, m: Optional[str] = None,
                        symbols: Optional[list] = None, unresolved: Optional[list] = None,
                        lookup: Optional[dict] = None) -> dict:
    """<minute>/universe.json: what ServerSource resolves NIFTY100 / NIFTY200 from ("extra" = the run's added names,
    `symbols` beyond the 99, else extra_names()). `missing` only from a successful instrument dump; `unresolved` =
    names whose token lookup failed this run (retried next run). Rewritten only when it changes."""
    m = m or mode()
    ext, src = extra_names(root) if m == "nifty200" else ([], {})
    if symbols is not None and m == "nifty200":
        skip = set(base_names()) | set(index_names())
        ext = [s for s in symbols if s not in skip]
    unres = sorted(set(unresolved or []))
    doc = {"version": 1, "mode": m, "base": base_names(), "aliases": sorted(ALIASES), "extra": ext,
           "missing_on_kite": sorted(set(missing or []) - set(unres)), "unresolved": unres,
           "membership": MEMBERSHIP_NOTE, "sources": src}
    if lookup is not None:
        doc["token_lookup"] = {k: lookup.get(k) for k in ("dump_size", "error")}
    p = Path(minute_root) / "universe.json"
    old = None
    try:
        old = json.loads(p.read_text())
    except Exception:
        pass
    if old is not None and {k: v for k, v in old.items() if k != "written"} == doc:
        return old
    doc["written"] = datetime.now().isoformat(timespec="seconds")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
    os.replace(tmp, p)
    return doc


# ── status / projection (GET /api/v2/builder/data) ───────────────────────────────────────────────────────────────
def _months(a: int, b: int) -> int:
    return (b // 10000 - a // 10000) * 12 + (b // 100 % 100 - a // 100 % 100) + 1


def store_sizes(minute_root) -> dict:
    """Measured from the existing store: bytes and month files per instrument (allocated blocks)."""
    out = {}
    d = Path(minute_root) / "bars"
    if not d.exists():
        return out
    for sd in d.iterdir():
        if not sd.is_dir():
            continue
        files = list(sd.glob("*.npz"))
        if not files:
            continue
        b = 0
        for f in files:
            st = os.lstat(f)
            b += max(st.st_size, getattr(st, "st_blocks", 0) * 512)
        months = sorted(int(f.stem) for f in files)
        out[sd.name] = {"bytes": b, "months": len(files), "first_month": months[0], "last_month": months[-1]}
    return out


def projection(minute_root, names_to_add: list, today: Optional[date] = None, cap_mb: Optional[float] = None,
               used_mb: Optional[float] = None) -> dict:
    """Disk projection of the added names from the store's own measured size per instrument-month (falls back to the
    lab's measurement, 49 KB before 2023 / 55 KB after, when the store has no complete base instrument yet)."""
    today = today or date.today()
    sizes = store_sizes(minute_root)
    base = set(base_names())
    meas = [v for k, v in sizes.items() if k in base and v["months"] >= 12]
    if meas:
        kb_month = sum(v["bytes"] for v in meas) / sum(v["months"] for v in meas) / 1024.0
        how = f"measured on {len(meas)} stored base instruments"
    else:
        kb_month = (84 * 49 + 47 * 55) / 131.0
        how = "lab measurement (no stored base instrument yet)"
    months_full = _months(HIST_FROM, today.year * 10000 + today.month * 100 + 1)
    todo, have_b = 0, 0
    for n in names_to_add:
        s = sizes.get(n)
        have_b += s["bytes"] if s else 0
        todo += max(0, months_full - (s["months"] if s else 0))
    add_mb = todo * kb_month / 1024.0
    out = {"kb_per_instrument_month": round(kb_month, 1), "how": how, "months_per_instrument": months_full,
           "names": len(names_to_add), "stored_mb": round(have_b / 1048576, 1),
           "to_fetch_mb_worst_case": round(add_mb, 1),
           "note": "worst case: every added name with full history from 2016 (recent listings have less)",
           "growth_mb_per_year": round((len(names_to_add) + len(base) + 2) * 12 * kb_month / 1024.0, 1)}
    if used_mb is not None:
        out["builder_data_used_mb"] = round(used_mb, 1)
        out["builder_data_after_backfill_mb_worst_case"] = round(used_mb + add_mb, 1)
    if cap_mb is not None:
        out["cap_mb"] = cap_mb
        caps = dict(VOLUME_CAPS_MB, builder_data=cap_mb)
        out["volume"] = {"size_mb": VOLUME_MB, "caps_mb": caps, "worst_case_mb": round(sum(caps.values()), 1),
                         "worst_case_limit_mb": VOLUME_WORST_MAX_MB,
                         "within_limit": sum(caps.values()) <= VOLUME_WORST_MAX_MB}
    return out


def minute_status(root=None, now: Optional[datetime] = None) -> dict:
    """Coverage of the minute universe: mode, the added names, those Kite does not list, per-name from / to, the
    backfill progress of the added names and whether it is complete ("ready": the builder may freeze NIFTY200)."""
    import builder_minute as BM
    root = _root(root)
    now = now or datetime.now()
    m = mode()
    mroot = root / "minute"
    mb = BM.MinuteBackfill(mroot)
    st = mb.state()
    doc = read_universe_file(mroot)
    if doc is not None and doc.get("mode") == m:          # what the last minute run resolved (the store's own list)
        ext, src = (list(doc.get("extra") or []), doc.get("sources") or {}) if m == "nifty200" else ([], {})
        base_doc = list(doc.get("base") or [])
    else:                                                  # no minute run under this mode yet
        ext, src = extra_names(root) if m == "nifty200" else ([], {})
        base_doc = None
    # the last run's token lookup: "missing" only from a successful instrument dump, "unresolved" = its lookup failed
    unresolved = set((doc or {}).get("unresolved") or []) | set(st.get("unresolved") or [])
    missing = set((doc or {}).get("missing_on_kite") or []) if doc is not None else set(st.get("missing") or [])
    missing -= unresolved
    fetchable = [n for n in ext if n not in missing]
    unresolved_added = sorted(unresolved & set(ext))
    syms = st.get("symbols") or {}
    names_ready = 0
    pending = 0
    for n in fetchable:
        p = mb.pending(st, n, now.date(), now)
        # the newest (open) window is re-read on every run: complete = nothing older pending
        newest = BM.windows(now.date())[-1:]
        older = [w for w in p if w not in newest]
        pending += len(older)
        if not older and syms.get(n, {}).get("windows"):
            names_ready += 1
    total = len(fetchable) * max(1, len(BM.windows(now.date())) - 1)
    pct = round(100.0 * (1 - pending / total), 2) if total else 100.0
    coverage = {n: {"from": (syms.get(n) or {}).get("first"), "to": (syms.get(n) or {}).get("last")}
                for n in ext}
    base = base_doc if base_doc else base_names()
    out = {"mode": m, "base": len(base), "added": len(ext), "added_names": ext,
           "missing_on_kite": sorted(x for x in missing if x in set(ext) | set(base)),
           "unresolved": unresolved_added, "token_lookup": (doc or {}).get("token_lookup"),
           "names_fetchable": len(fetchable), "names_ready": names_ready, "progress_pct": pct,
           # ready = a minute run resolved the universe under this mode, no added name's token lookup is unresolved,
           # AND every added name Kite lists is backfilled
           "ready": bool(m == "nifty200" and doc is not None and base_doc is not None and not unresolved_added
                         and names_ready == len(fetchable)),
           "universe_file": doc is not None, "universe_written": (doc or {}).get("written"),
           "coverage": coverage, "sources": src, "membership": MEMBERSHIP_NOTE,
           "base_coverage": {n: {"from": (syms.get(n) or {}).get("first"), "to": (syms.get(n) or {}).get("last")}
                             for n in base}}
    return out


def read_universe_file(minute_root) -> Optional[dict]:
    try:
        return json.loads((Path(minute_root) / "universe.json").read_text())
    except Exception:
        return None
