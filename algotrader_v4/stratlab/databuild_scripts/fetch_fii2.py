"""Complete the NSE FII derivative statistics: fetch fii_stats_DD-Mon-YYYY.xls for every weekday in
[START, END] that is not already in the existing raw dir (read-only) or in OUT.
404 -> OUT/_404.txt (holidays / not published).  One request per ~1.2 s, backoff, generic User-Agent.  Resume-safe.
Usage: python3 -I fetch_fii2.py EXISTING_RAW_FII OUT START END      (dates yyyymmdd)"""
import datetime as dt, gzip, os, sys, time, urllib.request, urllib.error
import numpy as np
OLD, OUT, START, END = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
UA = "Mozilla/5.0"
os.makedirs(OUT, exist_ok=True)
nfp = os.path.join(OUT, "_404.txt")
nf = set(open(nfp).read().split()) if os.path.exists(nfp) else set()
have = {f[:8] for f in os.listdir(OLD) if f.endswith(".xls")} | {f[:8] for f in os.listdir(OUT) if ".xls" in f}
SES = {int(x) for x in np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_px", "sessions.npy"))}
d = dt.datetime.strptime(START, "%Y%m%d").date()
end = dt.datetime.strptime(END, "%Y%m%d").date()
fails = 0
while d <= end:
    key = d.strftime("%Y%m%d")
    if (d.weekday() < 5 or int(key) in SES) and key not in have and key not in nf:
        url = f"https://nsearchives.nseindia.com/content/fo/fii_stats_{d:%d}-{d:%b}-{d:%Y}.xls"
        fn = os.path.join(OUT, key + ".xls.gz")
        for attempt in range(5):
            time.sleep(1.2)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                data = urllib.request.urlopen(req, timeout=40).read()
                if not data.startswith(b"\xd0\xcf\x11\xe0") and not data.startswith(b"PK"):
                    raise ValueError("not xls %r" % data[:40])
                open(fn + ".tmp", "wb").write(gzip.compress(data)); os.replace(fn + ".tmp", fn); print(key, "ok", flush=True)
                fails = 0
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    nf.add(key); open(nfp, "a").write(key + "\n"); print(key, "404", flush=True); fails = 0; break
                print(key, "HTTP", e.code, attempt, flush=True); time.sleep(10 * 2 ** attempt)
            except Exception as e:
                print(key, "ERR", repr(e)[:100], attempt, flush=True); time.sleep(10 * 2 ** attempt)
        else:
            print(key, "FAILED", flush=True); fails += 1
            if fails >= 4:
                print("persistent failures, stopping", flush=True); sys.exit(2)
    d += dt.timedelta(days=1)
print("DONE", flush=True)
