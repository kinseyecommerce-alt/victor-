"""Normalise NSE corporate actions -> databuild/corpactions/corpactions.csv.gz

Sources
  api   www.nseindia.com/api/corporates-corporateActions (index=equities), one request per quarter of EX-DATE,
        2012-01-01 .. 2026-12-31 (rows with a future ex-date are those NSE listed on the fetch date 2026-11-14)
  bc    the "Bc" file inside NSE's daily PR zip (the list of forthcoming corporate actions as published that day):
        nse_old/parts/bc (2012-01-02 .. 2023-11-09, existing) + databuild/raw/corpactions/pr/bc (2023-11-10 ..).
        Used for first_listed (= the first session whose Bc file carries the (symbol, ex-date)) and as a second
        source for the cross-check.

Schema: one row per (symbol, ex_date, kind)
  ex_date       int yyyymmdd
  symbol        NSE symbol as reported by the feed = the company's CURRENT symbol (old events of a renamed company
                appear under the new name: Tata Motors' 2015 rights issue is listed under TMPV)
  symbol_then   the symbol in force on the ex-date (rename list walked backwards) - the one found in price files
  lab_symbol / in_lab   as in deals.csv.gz (symmap.py)
  series, isin, face_value
  kind          dividend | bonus | split | consolidation | rights | demerger | capital_reduction | buyback
                (demerger = demerger / scheme of arrangement / amalgamation, and bonus issues of debentures or
                preference shares, which are distributions without a share-count factor)
  ratio_a, ratio_b   bonus a:b = a new shares for b held;  rights a:b = a offered for b held
  fv_from, fv_to     face value before / after (split, consolidation, capital reduction)
  dividend      Rs per share, the sum of all dividend components with that ex-date (blank when no amount is stated)
  rights_premium     Rs (issue price = premium + face value), blank when not stated
  factor        theoretical price factor of the action: prices BEFORE the ex-date x factor are comparable with prices
                from the ex-date on.  bonus b/(a+b); split / consolidation fv_to/fv_from; rights
                (b*P + a*(premium+face))/((a+b)*P) with P = official close of the session before the ex-date; blank
                for dividends (use dividend), demergers / schemes (no official ratio is published in the feed),
                capital reductions and buybacks
  record_date   int yyyymmdd or blank
  first_listed  int yyyymmdd: first session whose Bc file lists this (symbol, ex-date); blank if never seen in Bc
  known_date    the date used by the accessors' `before` rule: first_listed when present and < ex_date, else the
                session before the ex-date.  A row is visible to a decision when known_date < before
                (i.e. from the open of the session after it was published; at the latest from the ex-date's open).
  subject       NSE's text
Usage: python3 build_corpactions.py API_DIR NEW_BC_DIR OUT_DIR"""
import bisect, csv, datetime as dt, glob, gzip, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import symmap, cal

API, NEWBC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
OLDBC = os.path.join(cal.OLD, "parts", "bc")
os.makedirs(OUT, exist_ok=True)
NUM = r"([0-9]+(?:\.[0-9]+)?)"
RS = r"(?:rs\.?|re\.?|inr|rupees?|₹)?\s*"


def _d(s, fmt):
    try:
        return int(dt.datetime.strptime(s.strip(), fmt).strftime("%Y%m%d"))
    except Exception:
        return None


def parse(subject):
    """-> list of dicts (kind + fields) for the price-relevant components of an NSE corporate-action subject."""
    s = re.sub(r"\s+", " ", subject.replace("/-", " ").lower()).strip()
    out = []
    non_equity = re.search(r"ncrps|debenture|preference|\bncd", s)      # bonus paid in another security
    m = None if non_equity else re.search(r"bonus[^0-9]{0,30}?(\d+)\s*:\s*(\d+)", s)
    if non_equity and "bonus" in s:
        out.append({"kind": "demerger"})
    elif m:
        out.append({"kind": "bonus", "ratio_a": int(m.group(1)), "ratio_b": int(m.group(2))})
    elif "bonus" in s:
        out.append({"kind": "bonus"})
    if re.search(r"spli?t|sub-?\s?division", s):
        m = re.search(r"(?:spli?t|division)[^0-9]{0,40}?" + NUM + r".{0,40}?to\s*(?:face value\s*)?" + RS + NUM, s)
        out.append({"kind": "split", **({"fv_from": float(m.group(1)), "fv_to": float(m.group(2))} if m else {})})
    if "consolidation" in s:
        m = re.search(r"consolidation.{0,40}?" + RS + NUM + r".{0,30}?to\s*" + RS + NUM, s)
        out.append({"kind": "consolidation", **({"fv_from": float(m.group(1)), "fv_to": float(m.group(2))} if m else {})})
    m = re.search(r"rights?[^0-9]{0,30}?(\d+)\s*:\s*(\d+)", s)
    if m:
        r = {"kind": "rights", "ratio_a": int(m.group(1)), "ratio_b": int(m.group(2))}
        p = re.search(r"prem(?:ium)?\.?\s*(?:of\s*)?" + RS + NUM, s)
        if p:
            r["rights_premium"] = float(p.group(1))
        elif re.search(r"at par|@ par", s):
            r["rights_premium"] = 0.0
        out.append(r)
    elif re.search(r"\brights?\b", s):
        out.append({"kind": "rights"})
    if re.search(r"demerger|de-merger|scheme of arr|arrangement|spin[- ]?off|amalgamation", s) and \
            not any(o["kind"] == "demerger" for o in out):
        out.append({"kind": "demerger"})
    if re.search(r"capital reduction|reduction of capital", s):
        m = re.search(r"reduction.{0,40}?" + RS + NUM + r".{0,30}?to\s*" + RS + NUM, s)
        out.append({"kind": "capital_reduction", **({"fv_from": float(m.group(1)), "fv_to": float(m.group(2))} if m else {})})
    if re.search(r"buy\s?-?back", s):
        out.append({"kind": "buyback"})
    DIV = r"(?:\bdiv(?!ision)[a-z]*|ividend|distribution)"
    if re.search(DIV, s):
        sd = re.split(r"\(?\s*includ", s)[0]          # "... Rs 8.50 (Including Special Dividend Of Rs 2)" -> 8.50
        amts = [float(x) for x in re.findall(DIV + r"[^0-9/]{0,45}?" + NUM, sd)]
        pct = re.search(DIV + r"[^0-9]{0,20}?" + NUM + r"\s*%", sd)
        r = {"kind": "dividend"}
        if amts and not pct:
            r["dividend"] = round(sum(amts), 4)
        out.append(r)
    return out


# ── Bc files: first listing date per (symbol, ex-date) and the purposes seen
first, bc_purpose, bc_days = {}, {}, []
for d in (OLDBC, NEWBC):
    for fn in sorted(glob.glob(os.path.join(d, "*.csv.gz"))):
        day = int(os.path.basename(fn)[:8])
        bc_days.append(day)
        with gzip.open(fn, "rt") as fh:
            rd = csv.reader(fh)
            next(rd, None)
            for r in rd:
                if len(r) < 10:
                    continue
                ex = _d(r[6], "%d/%m/%Y") or _d(r[6], "%Y-%m-%d")       # ISO dates from 2025-10-13
                if ex is None:
                    continue
                k = (r[1].strip().upper(), ex)
                if k not in first:
                    first[k] = day
                bc_purpose.setdefault(k, set()).add(r[9].strip())
bc_days.sort()
ses = cal.sessions()


def prev_session(day):
    i = bisect.bisect_left(ses, day)
    if i > 0:
        return ses[i - 1]
    return int((dt.date(day // 10000, day // 100 % 100, day % 100) - dt.timedelta(days=1)).strftime("%Y%m%d"))


raw = []
for fn in sorted(glob.glob(os.path.join(API, "*.json"))):
    raw += json.load(open(fn))
seen, rows, dropped = set(), [], {"non_equity_series": 0, "no_price_relevant_component": 0, "bad_ex_date": 0,
                                  "duplicate": 0}
agg = {}
for r in raw:
    if r.get("series") not in ("EQ", "BE", "BZ"):
        dropped["non_equity_series"] += 1; continue
    ex = _d(r.get("exDate") or "", "%d-%b-%Y")
    if ex is None:
        dropped["bad_ex_date"] += 1; continue
    sym = r["symbol"].strip().upper()
    subj = re.sub(r"\s+", " ", r.get("subject") or "").strip()
    key = (sym, ex, subj)
    if key in seen:
        dropped["duplicate"] += 1; continue
    seen.add(key)
    comps = parse(subj)
    if not comps:
        dropped["no_price_relevant_component"] += 1; continue
    try:
        fv = float(r.get("faceVal") or "")
    except ValueError:
        fv = None
    for c in comps:
        k = (sym, ex, c["kind"])
        if k in agg:                      # two subjects with the same kind and ex-date: sum dividends, keep the rest
            a = agg[k]
            if c["kind"] == "dividend" and c.get("dividend") is not None:
                a["dividend"] = round((a.get("dividend") or 0) + c["dividend"], 4)
            a["subject"] += " || " + subj
            continue
        agg[k] = {**c, "ex_date": ex, "symbol": sym, "series": r["series"], "isin": r.get("isin") or "",
                  "face_value": fv, "record_date": _d(r.get("recDate") or "", "%d-%b-%Y"), "subject": subj}

for (sym, ex, kind), a in agg.items():
    f = None
    then = symmap.symbol_on(sym, ex)
    if kind == "bonus" and a.get("ratio_a") and a.get("ratio_b"):
        f = a["ratio_b"] / (a["ratio_a"] + a["ratio_b"])
    elif kind in ("split", "consolidation") and a.get("fv_from") and a.get("fv_to"):
        f = a["fv_to"] / a["fv_from"]
    elif kind == "rights" and a.get("ratio_a") and a.get("ratio_b") and a.get("rights_premium") is not None:
        p = cal.lookup(then, prev_session(ex))
        if p and p["close"] > 0:
            issue = a["rights_premium"] + (a["face_value"] or 0)
            f = (a["ratio_b"] * p["close"] + a["ratio_a"] * issue) / ((a["ratio_a"] + a["ratio_b"]) * p["close"])
            if f > 1:
                f = 1.0                   # issue price above market: no adjustment
    fl = first.get((then, ex)) or first.get((sym, ex))
    ps = prev_session(ex)
    known = fl if (fl is not None and fl < ex) else ps
    lab = symmap.lab_symbol(sym, ex)
    rows.append([ex, sym, then, lab, int(symmap.in_lab(lab)), a["series"], a["isin"],
                 "" if a["face_value"] is None else a["face_value"], kind,
                 a.get("ratio_a", ""), a.get("ratio_b", ""), a.get("fv_from", ""), a.get("fv_to", ""),
                 a.get("dividend", "") if a.get("dividend") is not None else "",
                 a.get("rights_premium", "") if a.get("rights_premium") is not None else "",
                 "" if f is None else round(f, 6), a["record_date"] or "", fl or "", known, a["subject"]])
rows.sort(key=lambda x: (x[0], x[1], x[8]))
hdr = ["ex_date", "symbol", "symbol_then", "lab_symbol", "in_lab", "series", "isin", "face_value", "kind", "ratio_a", "ratio_b",
       "fv_from", "fv_to", "dividend", "rights_premium", "factor", "record_date", "first_listed", "known_date", "subject"]
with gzip.open(os.path.join(OUT, "corpactions.csv.gz"), "wt", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(hdr); w.writerows(rows)
json.dump({"raw_api_rows": len(raw), "dropped": dropped, "rows": len(rows), "bc_files": len(bc_days),
           "bc_first": bc_days[0] if bc_days else None, "bc_last": bc_days[-1] if bc_days else None,
           "bc_keys": len(first)}, open(os.path.join(OUT, "_build.json"), "w"), indent=1)
# Bc purposes per key for the checker (second source)
with gzip.open(os.path.join(OUT, "_bc_keys.csv.gz"), "wt", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["symbol", "ex_date", "first_listed", "purposes"])
    for (s, e), d in sorted(first.items()):
        w.writerow([s, e, d, " || ".join(sorted(bc_purpose[(s, e)]))])
print("rows", len(rows), "dropped", dropped, "bc files", len(bc_days), bc_days[0], bc_days[-1])
