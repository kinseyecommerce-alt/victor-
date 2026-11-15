"""Fetch NSE bulk and block deals (historical report CSV) per calendar quarter, 2012 -> today.
Plain GETs, generic User-Agent, no cookies, one request every 3 s, backoff on errors.  Resume-safe.
Usage: python3 -I fetch_deals.py OUTDIR      (writes OUTDIR/{bulk,block}/YYYYQn.csv)"""
import datetime as dt, os, sys, time, urllib.request, urllib.error
OUT = sys.argv[1]
B = "https://www.nseindia.com/api/historicalOR/bulk-block-short-deals"
UA = "Mozilla/5.0"
TODAY = dt.date(2026, 11, 14)


def get(url):
    for attempt in range(5):
        time.sleep(3)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return urllib.request.urlopen(req, timeout=180).read()
        except urllib.error.HTTPError as e:
            print("HTTP", e.code, url[-60:], flush=True)
            if e.code in (401, 403):
                raise
            time.sleep(20 * 2 ** attempt)
        except Exception as e:
            print("ERR", repr(e)[:100], flush=True); time.sleep(20 * 2 ** attempt)
    return None


for kind, opt in (("bulk", "bulk_deals"), ("block", "block_deals")):
    os.makedirs(os.path.join(OUT, kind), exist_ok=True)
    for y in range(2012, 2027):
        for q in range(4):
            s = dt.date(y, 3 * q + 1, 1)
            e = (dt.date(y + (q == 3), (3 * q + 3) % 12 + 1, 1) - dt.timedelta(days=1))
            if s > TODAY:
                continue
            final = e < TODAY - dt.timedelta(days=3)
            e = min(e, TODAY)
            fn = os.path.join(OUT, kind, f"{y}Q{q+1}.csv")
            if os.path.exists(fn):
                continue
            data = get(f"{B}?optionType={opt}&from={s:%d-%m-%Y}&to={e:%d-%m-%Y}&csv=true")
            if data is None or b"Date" not in data[:40]:
                print(kind, y, q + 1, "FAILED", (data or b"")[:80], flush=True); continue
            open(fn + ".tmp", "wb").write(data); os.replace(fn + ".tmp", fn)
            print(kind, y, q + 1, f"{s}..{e}", len(data), data.count(b"\n"), "rows", "" if final else "(partial quarter)", flush=True)
print("DONE", flush=True)
