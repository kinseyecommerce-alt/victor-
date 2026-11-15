"""Fetch 1-minute bars of currently-listed NFO contracts through the production server's read-only
GET /backtest/history-bars (see prod.py).  Research data only; GET only; <= 1 request/second.

Universe (from Kite's public instruments dump, raw/kite_instruments/instruments_NFO.csv):
  * NIFTY / BANKNIFTY futures, all listed expiries
  * stock futures of the lab's symbols (n100.txt), months given by --stock-months (1 = current month)
  * NIFTY / BANKNIFTY options, every listed expiry, strikes within +-PCT % of spot (spot = last daily
    close of the near-month future)
Per contract: one interval=day request (<= 800 days, walked further back while the window is full) to find the
first traded day, then minute requests in <= 60-day windows from that day to today.
Output: raw/optmin/<tradingsymbol>.json.gz  {symbol, meta, day:[...], minute:[...], windows:[[start,end,n]...]}
A contract whose DAY history is empty is NOT stored: it is noted in raw/optmin/_empty.json and retried on a
later pass (--retry-empty); it only becomes "confirmed empty" after two empty answers >= 30 min apart, each
bracketed by a passing session probe.
Stops when: market hours (IST weekday 08:45-15:45), session probe fails, persistent server errors, or free disk
falls below 1.2 GB.
Usage: python3 -I fetch_optmin.py [--pct 8] [--stock-months 1] [--retry-empty] [--only fut|stk|opt] [--max N]
"""
import argparse, csv, datetime as dt, gzip, json, os, re, shutil, sys, time, warnings
warnings.simplefilter('ignore', DeprecationWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prod  # noqa: E402

ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "raw", "optmin")
INSTR = os.path.join(ROOT, "raw", "kite_instruments", "instruments_NFO.csv")
N100 = os.path.join(os.path.dirname(ROOT), "n100.txt")
EMPTY = os.path.join(RAW, "_empty.json")
TODAY = dt.date(2026, 11, 14)
MIN_FREE = 1.2e9


def log(*a):
    print(dt.datetime.utcnow().strftime("%H:%M:%S"), *a, flush=True)


def guards():
    now = dt.datetime.utcnow() + dt.timedelta(hours=5, minutes=30)
    if now.weekday() < 5 and dt.time(8, 45) <= now.time() <= dt.time(15, 45):
        log("market hours, stopping"); sys.exit(0)
    if shutil.disk_usage(ROOT).free < MIN_FREE:
        log("free disk below 1.2 GB, stopping"); sys.exit(3)


def session_ok():
    try:
        n = prod.probe()
    except prod.Persistent as e:
        log("probe error", e); return False
    return n >= 100


def load_universe(pct, stock_months, spot):
    rows = list(csv.DictReader(open(INSTR)))
    lab = [s for s in re.split(r"[,\s]+", open(N100).read()) if s and not s.isdigit()]
    fut = [r for r in rows if r["instrument_type"] == "FUT"]
    months = sorted({r["expiry"] for r in fut})
    out = []
    for r in fut:
        if r["name"] in ("NIFTY", "BANKNIFTY"):
            out.append(("fut", r))
    for r in fut:
        if r["name"] in lab and r["expiry"] in months[:stock_months]:
            out.append(("stk", r))
    opts = []
    for r in rows:
        if r["name"] in ("NIFTY", "BANKNIFTY") and r["instrument_type"] in ("CE", "PE"):
            k = float(r["strike"]); s = spot[r["name"]]
            if abs(k / s - 1) <= pct / 100.0:
                opts.append((r["expiry"], abs(k - s), r["name"], r["instrument_type"], r))
    opts.sort(key=lambda x: x[:4])
    out += [("opt", o[-1]) for o in opts]
    missing = sorted(set(lab) - {r["name"] for r in fut})
    return out, missing


class SessionLost(Exception):
    pass


def fetch_contract(r):
    sym = r["tradingsymbol"]
    # daily history: walk back in 800-day windows while the window starts with data close to its left edge
    day, end = [], TODAY
    for _ in range(6):
        start = end - dt.timedelta(days=799)
        st, bars = prod.history(sym, start.isoformat(), end.isoformat(), interval="day")
        if st == 404:
            return 404, None
        bars = bars or []
        day = bars + day
        if not bars or dt.date.fromisoformat(bars[0][0][:10]) > start + dt.timedelta(days=10):
            break
        end = start - dt.timedelta(days=1)
    if not day:
        return 0, None
    first = dt.date.fromisoformat(day[0][0][:10])
    minute, windows, s = [], [], first
    while s <= TODAY:
        e = min(s + dt.timedelta(days=59), TODAY)
        if e <= s:
            e = s + dt.timedelta(days=1)
        st, bars = prod.history(sym, s.isoformat(), e.isoformat(), interval="minute")
        bars = bars or []
        dvol = sum((b[5] or 0) for b in day if s.isoformat() <= b[0][:10] <= e.isoformat())
        if not bars and dvol > 0:
            # the daily candles show trades in this window but no minute bar came back: a restarting server
            # answers like this.  Check the session; retry once; never store a contract fetched without a session.
            if not session_ok():
                raise SessionLost(sym)
            time.sleep(5)
            st, bars = prod.history(sym, s.isoformat(), e.isoformat(), interval="minute")
            bars = bars or []
            if not bars and not session_ok():
                raise SessionLost(sym)
        windows.append([s.isoformat(), e.isoformat(), len(bars)])
        minute += bars
        s = e + dt.timedelta(days=1)
    seen, uniq = set(), []
    for b in minute:
        if b[0] not in seen:
            seen.add(b[0]); uniq.append(b)
    uniq.sort(key=lambda b: b[0])
    meta = {k: r[k] for k in ("tradingsymbol", "name", "expiry", "strike", "lot_size", "instrument_type", "tick_size")}
    return 200, {"symbol": sym, "meta": meta, "day": day, "minute": uniq, "windows": windows,
                 "fetched_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=float, default=8.0)
    ap.add_argument("--stock-months", type=int, default=1)
    ap.add_argument("--retry-empty", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    guards()
    if not session_ok():
        log("Kite session not valid (probe failed), stopping"); sys.exit(1)
    spot = {}
    for u, f in (("NIFTY", "NIFTY26NOVFUT"), ("BANKNIFTY", "BANKNIFTY26NOVFUT")):
        st, bars = prod.history(f, "2026-11-02", "2026-11-14", interval="day")
        spot[u] = float(bars[-1][4])
    log("spot (near future close)", spot)
    uni, missing = load_universe(a.pct, a.stock_months, spot)
    log("universe", len(uni), {k: sum(1 for x in uni if x[0] == k) for k in ("fut", "stk", "opt")},
        "lab symbols without a listed future:", missing)
    json.dump({"spot": spot, "pct": a.pct, "n": len(uni), "lab_without_future": missing,
               "symbols": [r["tradingsymbol"] for _, r in uni]}, open(os.path.join(RAW, "_universe.json"), "w"))
    empty = json.load(open(EMPTY)) if os.path.exists(EMPTY) else {}
    done = n404 = nreq_contracts = 0
    fails = 0
    for i, (kind, r) in enumerate(uni):
        if a.only and kind != a.only:
            continue
        sym = r["tradingsymbol"]
        fn = os.path.join(RAW, sym.replace("&", "_") + ".json.gz")
        if os.path.exists(fn):
            continue
        e = empty.get(sym)
        if e:
            if not a.retry_empty or e.get("confirmed") or time.time() - e["ts"][-1] < 1800:
                continue
        if a.max and nreq_contracts >= a.max:
            break
        if nreq_contracts % 40 == 0:
            guards()
            if nreq_contracts and not session_ok():
                log("Kite session probe failed mid-run, stopping"); sys.exit(1)
        nreq_contracts += 1
        try:
            st, d = fetch_contract(r)
            fails = 0
        except SessionLost:
            log("minute window empty and the session probe failed -> stopping (nothing stored for", sym + ")")
            sys.exit(1)
        except prod.Persistent as ex:
            fails += 1
            log("FAIL", sym, ex)
            if fails >= 3:
                log("persistent server errors, stopping"); sys.exit(2)
            time.sleep(120)
            continue
        if st == 404:
            n404 += 1; log("404", sym); continue
        if st == 0:
            ok = session_ok()
            if not ok:
                log("empty answer and probe failed -> session invalid, stopping"); sys.exit(1)
            e = empty.setdefault(sym, {"ts": []})
            e["ts"].append(time.time())
            if len(e["ts"]) >= 2:
                e["confirmed"] = True
            json.dump(empty, open(EMPTY + ".tmp", "w")); os.replace(EMPTY + ".tmp", EMPTY)
            log("EMPTY", sym, "(attempt %d)" % len(e["ts"]))
            continue
        with gzip.open(fn + ".tmp", "wt") as fh:
            json.dump(d, fh, separators=(",", ":"))
        os.replace(fn + ".tmp", fn)
        if sym in empty:
            del empty[sym]; json.dump(empty, open(EMPTY + ".tmp", "w")); os.replace(EMPTY + ".tmp", EMPTY)
        done += 1
        if done % 25 == 0 or kind != "opt":
            log(f"{i+1}/{len(uni)} {sym} days={len(d['day'])} first={d['day'][0][0][:10]} min={len(d['minute'])} "
                f"win={len(d['windows'])}")
    log("PASS DONE stored", done, "404", n404, "empty-pending", sum(1 for v in empty.values() if not v.get("confirmed")),
        "empty-confirmed", sum(1 for v in empty.values() if v.get("confirmed")))


if __name__ == "__main__":
    main()
