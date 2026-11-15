"""NSE PRddmmyy.zip for sessions in [START, END]: keep only the parsed Bc file (corporate actions as published that
day -> gives the first date an action was listed) and the Pd rows with a corporate-action indicator.
The zip itself is not stored.  One request per 1.5 s, backoff, generic User-Agent.  Resume-safe.
Usage: python3 fetch_pr2.py OUTDIR START END"""
import csv, datetime as dt, gzip, io, os, sys, time, urllib.request, urllib.error, zipfile
import numpy as np
OUT, START, END = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
UA = "Mozilla/5.0"
HERE = os.path.dirname(os.path.abspath(__file__))
SES = sorted(int(x) for x in np.load(os.path.join(os.path.dirname(HERE), "_px", "sessions.npy")))
days = [d for d in SES if START <= d <= END]
d = dt.date(SES[-1] // 10000, SES[-1] // 100 % 100, SES[-1] % 100) + dt.timedelta(days=1)
while int(d.strftime("%Y%m%d")) <= END:          # weekdays after the lab calendar's last date
    if d.weekday() < 5:
        days.append(int(d.strftime("%Y%m%d")))
    d += dt.timedelta(days=1)
for k in ("bc", "pr_ci"):
    os.makedirs(os.path.join(OUT, k), exist_ok=True)


def write(kind, key, header, rows):
    p = os.path.join(OUT, kind, key + ".csv.gz")
    with gzip.open(p + ".tmp", "wt", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    os.replace(p + ".tmp", p)


fails = 0
for day in days:
    key = str(day)
    if os.path.exists(os.path.join(OUT, "bc", key + ".csv.gz")) or os.path.exists(os.path.join(OUT, "bc", key + ".404")):
        continue
    ddmmyy = key[6:8] + key[4:6] + key[2:4]
    url = f"https://nsearchives.nseindia.com/archives/equities/bhavcopy/pr/PR{ddmmyy}.zip"
    data = None
    for attempt in range(5):
        time.sleep(1.5)
        try:
            data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60).read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                data = b""; break
            print(key, "HTTP", e.code, flush=True); time.sleep(10 * 2 ** attempt)
        except Exception as e:
            print(key, "ERR", repr(e)[:80], flush=True); time.sleep(10 * 2 ** attempt)
    if data is None:
        fails += 1; print(key, "FAILED", flush=True)
        if fails >= 4:
            print("persistent failures, stopping", flush=True); sys.exit(2)
        continue
    fails = 0
    if data == b"":
        open(os.path.join(OUT, "bc", key + ".404"), "w").close(); print(key, "404", flush=True); continue
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        print(key, "badzip", flush=True); continue
    names = {n.lower(): n for n in z.namelist()}
    # inside the zip the files are pdDDMMYY.csv / bcDDMMYY.csv, and pdDDMMYYYY.csv / bcDDMMYYYY.csv from 2025-10-13
    pd_n = next((v for k, v in names.items() if k.startswith("pd") and k.endswith(".csv")), None)
    bc_n = next((v for k, v in names.items() if k.startswith("bc") and k.endswith(".csv")), None)
    ci = []
    if pd_n:
        rd = csv.reader(io.StringIO(z.read(pd_n).decode("latin-1"))); next(rd, None)
        for f in rd:
            f = [x.strip() for x in f]
            if len(f) >= 13 and f[1] in ("EQ", "BE") and f[12]:
                ci.append([f[2], f[1], f[12], f[4], f[8]])
    write("pr_ci", key, ["symbol", "series", "corp_ind", "prev_close", "close"], ci)
    bc = []
    if bc_n:
        rd = csv.reader(io.StringIO(z.read(bc_n).decode("latin-1"))); next(rd, None)
        bc = [[x.strip() for x in f][:10] for f in rd if len(f) >= 10]
    write("bc", key, ["series", "symbol", "security", "record_dt", "bc_start", "bc_end", "ex_dt", "nd_start", "nd_end",
                      "purpose"], bc)
    print(key, "ok", len(bc), len(ci), "" if bc_n else "NO-BC", flush=True)
print("DONE", flush=True)
