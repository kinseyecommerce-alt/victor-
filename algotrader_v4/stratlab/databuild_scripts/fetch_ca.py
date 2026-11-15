"""Fetch NSE corporate actions (equities) per calendar quarter by EX-DATE, 2012 -> today + 60 days ahead.
Plain GETs, generic User-Agent, no cookies, one request every 3 s.  Resume-safe (closed quarters only).
Usage: python3 -I fetch_ca.py OUTDIR      (writes OUTDIR/YYYYQn.json)"""
import datetime as dt, json, os, sys, time, urllib.request, urllib.error
OUT = sys.argv[1]
B = "https://www.nseindia.com/api/corporates-corporateActions"
UA = "Mozilla/5.0"
TODAY = dt.date(2026, 11, 14)
os.makedirs(OUT, exist_ok=True)


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


for y in range(2012, 2027):
    for q in range(4):
        s = dt.date(y, 3 * q + 1, 1)
        e = (dt.date(y + (q == 3), (3 * q + 3) % 12 + 1, 1) - dt.timedelta(days=1))
        if s > TODAY:
            continue
        fn = os.path.join(OUT, f"{y}Q{q+1}.json")
        if os.path.exists(fn):
            continue
        data = get(f"{B}?index=equities&from_date={s:%d-%m-%Y}&to_date={e:%d-%m-%Y}")
        try:
            rows = json.loads(data)
            assert isinstance(rows, list)
        except Exception:
            print(y, q + 1, "FAILED", (data or b"")[:80], flush=True); continue
        open(fn + ".tmp", "wb").write(data); os.replace(fn + ".tmp", fn)
        print(y, q + 1, f"{s}..{e}", len(rows), "rows", flush=True)
print("DONE", flush=True)
