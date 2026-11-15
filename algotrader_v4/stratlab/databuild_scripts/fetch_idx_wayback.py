"""Historical NIFTY 50 / NIFTY 100 / NIFTY Next 50 constituent lists from Internet Archive snapshots of NSE's
own constituent CSV files.  Step 1: CDX index (one query per host x file), step 2: download each distinct
snapshot body (id_ = original bytes).  Polite: 1 request / 2.5 s, backoff.  Resume-safe.
Usage: python3 -I fetch_idx_wayback.py OUTDIR"""
import json, os, sys, time, urllib.parse, urllib.request, urllib.error
OUT = sys.argv[1]
UA = "Mozilla/5.0"
HOSTS_OLD = ["www.nseindia.com", "nseindia.com", "www1.nseindia.com", "archives.nseindia.com", "nsearchives.nseindia.com"]
HOSTS_NI = ["www.niftyindices.com", "niftyindices.com"]
FILES = {"NIFTY50": ["ind_nifty50list.csv", "ind_niftylist.csv"],
         "NIFTY100": ["ind_nifty100list.csv", "ind_cnx100list.csv"],
         "NIFTYNEXT50": ["ind_niftynext50list.csv", "ind_jrniftylist.csv", "ind_niftyjuniorlist.csv", "ind_cnxniftyjuniorlist.csv"]}
os.makedirs(os.path.join(OUT, "cdx"), exist_ok=True)
os.makedirs(os.path.join(OUT, "snap"), exist_ok=True)


def get(url, tries=6):
    for attempt in range(tries):
        time.sleep(30)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return urllib.request.urlopen(req, timeout=120).read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            print("HTTP", e.code, url[:110], flush=True); time.sleep(15 * 2 ** attempt)
        except Exception as e:
            print("ERR", repr(e)[:80], url[:110], flush=True); time.sleep(15 * 2 ** attempt)
    return None


index = []
for idx, files in FILES.items():
    for f in files:
        for h in HOSTS_OLD + HOSTS_NI:
            path = ("IndexConstituent/" if "niftyindices" in h else "content/indices/") + f
            fn = os.path.join(OUT, "cdx", f"{h}_{f}.json")
            if not os.path.exists(fn):
                q = urllib.parse.urlencode({"url": f"{h}/{path}", "output": "json", "filter": "statuscode:200",
                                            "fl": "timestamp,original,digest,length", "collapse": "digest"})
                data = get("https://web.archive.org/cdx/search/cdx?" + q)
                if data is None:
                    print("cdx FAILED", h, f, flush=True); continue
                open(fn, "wb").write(data)
            try:
                rows = json.loads(open(fn).read() or "[]")
            except Exception:
                rows = []
            print("cdx", idx, h, f, max(0, len(rows) - 1), flush=True)
            for r in rows[1:]:
                index.append((idx, r[0], r[1], r[2]))
seen = set()
todo = []
for idx, ts, orig, dig in sorted(index, key=lambda x: x[1]):
    if (idx, dig) in seen:
        continue
    seen.add((idx, dig)); todo.append((idx, ts, orig, dig))
print("distinct snapshots", len(todo), flush=True)
json.dump(todo, open(os.path.join(OUT, "snapshots.json"), "w"))
for idx, ts, orig, dig in sorted(todo, key=lambda x: -int(x[1])):       # newest first; pre-2011 not needed
    if ts < "2011":
        continue
    fn = os.path.join(OUT, "snap", f"{idx}_{ts}.csv")
    if os.path.exists(fn):
        continue
    data = get(f"https://web.archive.org/web/{ts}id_/{orig}", tries=2)
    if data is None:
        print("snap FAILED", idx, ts, flush=True); continue
    open(fn + ".tmp", "wb").write(data); os.replace(fn + ".tmp", fn)
    print("snap", idx, ts, len(data), flush=True)
print("DONE", flush=True)
