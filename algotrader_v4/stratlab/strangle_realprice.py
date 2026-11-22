"""
stratlab/strangle_realprice.py — VRP-STRANGLE-016-WINGED re-priced on real NSE end-of-day option closes.
Usage: python -m stratlab.strangle_realprice <lab scratch dir>   (results: stratlab/STRANGLE_REALPRICE.md)
"""
# Re-price the lab's 152 VRP-STRANGLE-016-WINGED trades on real NSE end-of-day option closes.
# Same moment comparison: the lab's own model (Black-Scholes, sigma = the lab's VIX, calendar clock) at the
# entry day's close, against the real bhavcopy close of each leg on that day. Ratio real/model credit -> applied to
# the lab's 10:20 model credit to estimate the real-priced net. Exit (settlement at intrinsic) is unchanged.
import csv, gzip, sys, os
from datetime import datetime, date
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import options_replay as OR
S = sys.argv[1] if len(sys.argv) > 1 else "."   # the lab scratch dir (dd/vrp/final_run, nse*/fo_parts)
trades = list(csv.DictReader(open(f"{S}/dd/vrp/final_run/trades/VRP-STRANGLE-016-WINGED.csv")))
def day_file(d):
    for root in ("nse", "nse_old_view", "nse_old"):
        p = f"{S}/{root}/fo_parts/optidx/{d}.csv.gz"
        if os.path.exists(p): return p
rows = []
for t in trades:
    d = t["date"]; exp = t["expiry"]; p = day_file(d)
    if not p: continue
    chain = {}; und = None
    for r in csv.DictReader(gzip.open(p, "rt")):
        if r["symbol"] != "NIFTY" or r["expiry"] != exp: continue
        chain[(float(r["strike"]), r["type"])] = float(r["close"] or 0)
        und = und or float(r["underlying"] or 0)
    if not chain or not und: continue
    legs = []
    for L in t["legs"].split(" | "):
        side, k, px = L.split()
        strike, typ = float(k[:-2]), k[-2:]
        legs.append((side, strike, typ, float(px.split("->")[0])))
    now = datetime.strptime(d + " 15:30", "%Y%m%d %H:%M")
    T = OR.years_between(now, date.fromisoformat(exp))
    sig = float(t["vix"]) / 100.0
    real = model = 0.0; ok = True
    for side, k, typ, _ in legs:
        rp = chain.get((k, typ))
        if rp is None or rp <= 0: ok = False; break
        mp = OR.bs(und, k, T, sig, typ)
        sgn = 1 if side == "S" else -1
        real += sgn * rp; model += sgn * mp
    if not ok or model <= 0: continue
    rows.append({"date": d, "credit_lab": float(t["credit_pts"]), "real_eod": real, "model_eod": model,
                 "ratio": real / model, "qty": int(t["qty"]), "net": float(t["net"])})
n = len(rows)
import statistics as st
rat = [r["ratio"] for r in rows]
print(f"trades priced {n} of {len(trades)}")
print(f"real/model credit at the same close: mean {st.mean(rat):.3f}  median {st.median(rat):.3f}  "
      f"pooled {sum(r['real_eod'] for r in rows)/sum(r['model_eod'] for r in rows):.3f}")
lab_net = sum(r["net"] for r in rows)
adj = [r["net"] - (1 - r["ratio"]) * r["credit_lab"] * r["qty"] for r in rows]
pooled = sum(r['real_eod'] for r in rows)/sum(r['model_eod'] for r in rows)
adj_p = [r["net"] - (1 - pooled) * r["credit_lab"] * r["qty"] for r in rows]
gp = lambda xs: sum(x for x in xs if x > 0) / max(1e-9, -sum(x for x in xs if x < 0))
print(f"lab net (model) Rs {lab_net:,.0f}  PF {gp([r['net'] for r in rows]):.2f}")
print(f"real-priced net (per-trade ratio) Rs {sum(adj):,.0f}  PF {gp(adj):.2f}")
print(f"real-priced net (pooled ratio)    Rs {sum(adj_p):,.0f}  PF {gp(adj_p):.2f}")
for y in ("2023", "2024", "2025", "2026"):
    sub = [a for a, r in zip(adj_p, rows) if r["date"].startswith(y)]
    if sub: print(y, len(sub), f"{sum(sub):,.0f}")

# robustness: short legs only (the liquid ones), and only legs traded that day (contracts > 0)
rs, ms, n2 = 0.0, 0.0, 0
for t in trades:
    p = day_file(t["date"])
    if not p: continue
    ch = {}; und = None
    for r in csv.DictReader(gzip.open(p, "rt")):
        if r["symbol"] == "NIFTY" and r["expiry"] == t["expiry"]:
            ch[(float(r["strike"]), r["type"])] = (float(r["close"] or 0), int(float(r["contracts"] or 0)))
            und = und or float(r["underlying"] or 0)
    if not ch or not und: continue
    T = OR.years_between(datetime.strptime(t["date"] + " 15:30", "%Y%m%d %H:%M"), date.fromisoformat(t["expiry"]))
    for L in t["legs"].split(" | "):
        side, k, px = L.split()
        if side != "S": continue
        key = (float(k[:-2]), k[-2:])
        if key in ch and ch[key][0] > 0 and ch[key][1] > 0:
            rs += ch[key][0]; ms += OR.bs(und, key[0], T, float(t["vix"]) / 100, key[1]); n2 += 1
print(f"short legs traded that day: {n2}, real/model = {rs/ms:.3f}")

# robustness: short legs only (the liquid ones), and only legs traded that day (contracts > 0)
rs, ms, n2 = 0.0, 0.0, 0
for t in trades:
    p = day_file(t["date"])
    if not p: continue
    ch = {}; und = None
    for r in csv.DictReader(gzip.open(p, "rt")):
        if r["symbol"] == "NIFTY" and r["expiry"] == t["expiry"]:
            ch[(float(r["strike"]), r["type"])] = (float(r["close"] or 0), int(float(r["contracts"] or 0)))
            und = und or float(r["underlying"] or 0)
    if not ch or not und: continue
    T = OR.years_between(datetime.strptime(t["date"] + " 15:30", "%Y%m%d %H:%M"), date.fromisoformat(t["expiry"]))
    for L in t["legs"].split(" | "):
        side, k, px = L.split()
        if side != "B": continue
        key = (float(k[:-2]), k[-2:])
        if key in ch and ch[key][0] > 0 and ch[key][1] > 0:
            rs += ch[key][0]; ms += OR.bs(und, key[0], T, float(t["vix"]) / 100, key[1]); n2 += 1
print(f"wing (bought) legs traded that day: {n2}, real/model = {rs/ms:.3f}")
