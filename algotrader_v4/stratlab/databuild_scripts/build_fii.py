"""Complete FII derivative statistics -> databuild/fii_stats/{fii_stats.csv.gz, data_check.json}
Sources: <scratch>/nse/raw_fii/*.xls (existing, 2023-11-01..2026-02-09, read-only) + databuild/raw/fii_stats/*.xls.gz
(fetched now: 2014-12..2023-10 and 2026-02-10..2026-11-13), all from nsearchives.nseindia.com/content/fo/
fii_stats_DD-Mon-YYYY.xls.
Schema (identical to nse/fii_stats.csv.gz): date (ISO), instrument, buy_contracts, buy_amt_cr, sell_contracts,
sell_amt_cr, oi_contracts, oi_amt_cr.  The file named for trade date D is published in the evening of D.
Usage: python3 build_fii.py OLD_RAW NEW_RAW OUT_DIR"""
import collections, csv, glob, gzip, io, json, os, re, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cal

OLD, NEW, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
files = {}
for f in glob.glob(os.path.join(OLD, "*.xls")):
    files[os.path.basename(f)[:8]] = f
for f in glob.glob(os.path.join(NEW, "*.xls.gz")):
    files.setdefault(os.path.basename(f)[:8], f)
recs, problems, content_date_mismatch = [], [], []
MON = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}
for key in sorted(files):
    f = files[key]
    try:
        raw = gzip.open(f, "rb").read() if f.endswith(".gz") else open(f, "rb").read()
        x = pd.read_excel(io.BytesIO(raw), header=None)
    except Exception as e:
        problems.append(f"{key}: {e!r}"[:160]); continue
    date = f"{key[:4]}-{key[4:6]}-{key[6:]}"
    head = " ".join(str(v) for v in x.iloc[:3].values.ravel() if isinstance(v, str))
    m = re.search(r"(\d{1,2})-([A-Za-z]{3})-(\d{4})", head)
    if m:
        cd = f"{m.group(3)}-{MON[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"
        if cd != date:
            # NSE's archive holds another day's sheet under this name: using it would date tomorrow's numbers today
            content_date_mismatch.append((key, cd))
            continue
    n0 = len(recs)
    for _, r in x.iterrows():
        lab = r[0]
        if not isinstance(lab, str) or not re.search(r"(FUTURES|OPTIONS)\s*$", lab.strip().upper()):
            continue
        try:
            vals = [float(r[i]) for i in range(1, 7)]
        except Exception:
            problems.append(f"{key}: bad row {list(r)[:7]}"[:160]); continue
        recs.append({"date": date, "instrument": re.sub(r"\s+", " ", lab.strip().upper()), "buy_contracts": vals[0],
                     "buy_amt_cr": vals[1], "sell_contracts": vals[2], "sell_amt_cr": vals[3],
                     "oi_contracts": vals[4], "oi_amt_cr": vals[5]})
    if len(recs) == n0:
        problems.append(f"{key}: no instrument rows")
df = pd.DataFrame(recs).sort_values(["date", "instrument"])
df.to_csv(os.path.join(OUT, "fii_stats.csv.gz"), index=False, compression="gzip")

# ── data check
d_int = sorted({int(d.replace("-", "")) for d in df.date})
ses = cal.sessions()
first, last = d_int[0], d_int[-1]
exp = [s for s in ses if first <= s <= last]
nf = set()
p404 = os.path.join(NEW, "_404.txt")
if os.path.exists(p404):
    nf = {int(x) for x in open(p404).read().split()}
missing = [s for s in exp if s not in set(d_int)]
import datetime as dt
wk = {s: dt.date(s // 10000, s // 100 % 100, s % 100).weekday() for s in missing}
chk = {"file": "fii_stats.csv.gz", "rows": len(df), "days": len(d_int), "first_date": first, "last_date": last,
       "source": "nsearchives.nseindia.com/content/fo/fii_stats_DD-Mon-YYYY.xls (earliest file NSE serves: 2014-12-01)",
       "availability": "published in the evening of the trade date -> rows dated < before",
       "days_per_year": dict(sorted(collections.Counter(d // 10000 for d in d_int).items())),
       "lab_calendar_sessions_in_range": len(exp),
       "sessions_without_a_file": {"count": len(missing), "http_404": [s for s in missing if s in nf],
                                   "weekend_special_sessions": [s for s in missing if wk[s] >= 5],
                                   "other": [s for s in missing if s not in nf and wk[s] < 5]},
       "file_dates_not_in_lab_calendar": [d for d in d_int if d not in set(ses) and d <= ses[-1]],
       "duplicates": int(df.duplicated(["date", "instrument"]).sum()),
       "files_dropped_because_the_sheet_inside_is_dated_another_day": content_date_mismatch[:20],
       "parse_problems": problems[:30], "parse_problem_count": len(problems)}
inst = {}
for ins, g in df.groupby("instrument"):
    ds = sorted(int(x.replace("-", "")) for x in g.date)
    inst[ins] = {"days": len(ds), "first": ds[0], "last": ds[-1]}
chk["instrument_groups"] = inst
# second source: NSE participant-wise OI (FII long + short contracts) must equal fii_stats oi_contracts
part = None
for p in (os.path.join(cal.S, "nse", "participant_oi.csv.gz"),):
    if os.path.exists(p):
        part = pd.read_csv(p)
if part is not None:
    fi = part[part.client_type == "FII"].set_index("date")
    cmp_rows, agree, tot = [], 0, 0
    pairs = {"INDEX FUTURES": ["fut_idx_long", "fut_idx_short"], "STOCK FUTURES": ["fut_stk_long", "fut_stk_short"],
             "INDEX OPTIONS": ["opt_idx_call_long", "opt_idx_put_long", "opt_idx_call_short", "opt_idx_put_short"],
             "STOCK OPTIONS": ["opt_stk_call_long", "opt_stk_put_long", "opt_stk_call_short", "opt_stk_put_short"]}
    sub = df[df.instrument.isin(pairs)]
    for r in sub.itertuples(index=False):
        if r.date not in fi.index:
            continue
        ref = float(sum(fi.loc[r.date, c] for c in pairs[r.instrument]))
        ok = abs(ref - r.oi_contracts) <= max(2.0, 1e-4 * ref)
        tot += 1; agree += ok
        if (not ok and len(cmp_rows) < 10) or (ok and len(cmp_rows) < 4):
            cmp_rows.append({"date": r.date, "instrument": r.instrument, "fii_stats_oi_contracts": r.oi_contracts,
                             "participant_oi_fii_long_plus_short": ref, "agree": bool(ok)})
    import random
    random.seed(5)
    samp = sub[sub.date.isin(fi.index)].sample(10, random_state=5)
    spot = [{"date": r.date, "instrument": r.instrument, "fii_stats_oi_contracts": r.oi_contracts,
             "participant_oi_fii_long_plus_short": float(sum(fi.loc[r.date, c] for c in pairs[r.instrument]))}
            for r in samp.itertuples(index=False)]
    chk["cross_check_vs_participant_oi"] = {"rows_compared": tot, "agree": int(agree), "examples": cmp_rows,
                                            "spot_checks": spot,
                                            "note": "participant OI file covers 2023-11-01.. only (existing lab file)"}
json.dump(chk, open(os.path.join(OUT, "data_check.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in chk.items() if k not in ("instrument_groups",)}, indent=1)[:4000])
print({k: (v["days"], v["first"], v["last"]) for k, v in inst.items()})
