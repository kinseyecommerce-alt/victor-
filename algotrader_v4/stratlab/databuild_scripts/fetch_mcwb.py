"""NSE Indices monthly "Market Capitalisation, Weightage and Beta for NIFTY 50 and NIFTY Next 50" archives
(mcwb_<mon><yy>.zip): the month-end constituents of both indices with their index weights.
One request per 3 s, generic User-Agent, niftyindices.com first, nsearchives.nseindia.com as fallback.  Resume-safe.
Usage: python3 -I fetch_mcwb.py OUTDIR [FIRST_YYYYMM] [LAST_YYYYMM]"""
import os, sys, time, urllib.request, urllib.error
OUT = sys.argv[1]
FIRST = int(sys.argv[2]) if len(sys.argv) > 2 else 201201
LAST = int(sys.argv[3]) if len(sys.argv) > 3 else 202610
UA = "Mozilla/5.0"
MON = "jan feb mar apr may jun jul aug sep oct nov dec".split()
BASES = ["https://www.niftyindices.com/Market_Capitalisation_Weightage_Beta_for_NIFTY_50_And_NIFTY_Next_50/",
         "https://nsearchives.nseindia.com/content/indices/"]
os.makedirs(OUT, exist_ok=True)
ym = FIRST
while ym <= LAST:
    y, m = divmod(ym, 100)
    name = f"mcwb_{MON[m - 1]}{y % 100:02d}.zip"
    fn = os.path.join(OUT, f"{ym}.zip")
    if not os.path.exists(fn) and not os.path.exists(fn + ".404"):
        got = False
        for base in BASES:
            for attempt in range(3):
                time.sleep(3)
                try:
                    data = urllib.request.urlopen(urllib.request.Request(base + name, headers={"User-Agent": UA}), timeout=60).read()
                    if data[:2] == b"PK":
                        open(fn + ".tmp", "wb").write(data); os.replace(fn + ".tmp", fn)
                        print(ym, "ok", len(data), base.split("/")[2], flush=True)
                        got = True
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        break
                    print(ym, "HTTP", e.code, flush=True); time.sleep(15 * 2 ** attempt)
                except Exception as e:
                    print(ym, "ERR", repr(e)[:80], flush=True); time.sleep(15 * 2 ** attempt)
            if got:
                break
        if not got:
            open(fn + ".404", "w").close()
            print(ym, "not found", flush=True)
    ym = ym + 1 if m < 12 else (y + 1) * 100 + 1
print("DONE", flush=True)
