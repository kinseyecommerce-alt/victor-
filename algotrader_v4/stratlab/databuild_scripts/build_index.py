"""Historical NIFTY 50 / NIFTY NEXT 50 / NIFTY 100 membership, 2012 -> today.

Inputs (all under RAW_IDX_DIR)
  mcwb/YYYYMM.zip                NSE Indices' monthly "Market Capitalisation, Weightage and Beta for NIFTY 50 and
                                 NIFTY Next 50": the MONTH-END constituents of both indices with their weights
                                 (primary source for WHO was in the index; NIFTY 100 = NIFTY 50 + NIFTY Next 50)
  press/txt/ind_prs*.txt.gz      pdftotext of the NSE Indices press releases that could be obtained (primary source for
                                 the ANNOUNCEMENT date and the exact EFFECTIVE date of a change)
  wikipedia/NIFTY_50.html        Wikipedia's "List of replacements" (second source: NIFTY 50 effective dates)
  current/ind_<index>list.csv    today's constituent files (niftyindices.com, 2026-11-14)
  wayback/snap/*.csv             NSE's constituent files archived by the Internet Archive (independent anchors)
Outputs (index_membership/)
  index_members.csv.gz   index, symbol (latest NSE symbol), lab_symbol, in_lab, from_date, to_date (inclusive; blank =
                         still a constituent on the build date), from_precision, to_precision
  index_changes.csv      index, announce_date, effective_date, action (IN|OUT), symbol, lab_symbol, in_lab, company,
                         source, announce_source, effective_precision, note
  index_weights.csv.gz   month (yyyymm), snapshot_date (last session of the month), index (NIFTY50|NIFTYNEXT50),
                         symbol (as published), lab_symbol, in_lab, weight_pct, ff_mcap_cr
  data_check.json
How a change gets its dates.  Two consecutive month-end files differ -> the change happened in that month.
  effective_date   1. the press release that names the change (precision "day")
                   2. Wikipedia's replacement list (NIFTY 50; "day")
                   3. the same stock changing in the other index that month, or the single review date of that month
                      known from 1/2 ("inferred": semi-annual reviews of both indices take effect together)
                   4. otherwise the month-end snapshot date itself ("month": the LATEST the change can have happened,
                      so membership never shows a change earlier than it was true)
  announce_date    the press release date when the release was obtained (announce_source "press"; releases go out
                   after market hours, so a change is known from the next session); otherwise the session before the
                   effective date - the latest it can have been known (NSE announces about 4 weeks ahead) - and
                   announce_source says "none".
Not represented: constituents carried under a dummy symbol while a demerged company awaits listing.
Usage: python3 build_index.py RAW_IDX_DIR OUT_DIR"""
import bisect, collections, csv, datetime as dt, glob, gzip, html as _html, io, json, os, re, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cal, symmap

RAW, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
BUILD_DATE = 20261113
ORIGIN = 20120102              # first session of the lab's daily data
ALIASES = {"NIFTY50": {"s&p cnx nifty", "cnx nifty", "nifty 50", "nifty50"},
           "NIFTY100": {"cnx 100", "nifty 100", "nifty100"},
           "NIFTYNEXT50": {"cnx nifty junior", "nifty junior", "nifty next 50", "niftynext50", "nifty next50"}}
MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                          "september", "october", "november", "december"])}
MON = "|".join(MONTHS)
D1 = re.compile(r"(%s)\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})" % MON, re.I)
D2 = re.compile(r"(\d{1,2})(?:st|nd|rd|th)?\s+(%s)\s*,?\s*(\d{4})" % MON, re.I)
EFF = re.compile(r"(effective|w\.?\s?e\.?\s?f\.?|with effect)", re.I)
ROW = re.compile(r"^\s*(\d{1,3})[.)]?\s+(\S.*?)\s{2,}([A-Z0-9][A-Z0-9&\-\.\*]{1,19})\s*$")
IDXROW = re.compile(r"^\s*(\d{1,3})[.)]?\s+((?:Nifty|NIFTY|CNX|S&P CNX)\b.*?)\s*$")
HEAD = re.compile(r"^\s*(?:\(?[0-9a-zA-Z]{1,3}[).]\s+)?((?:S&P\s+)?(?:CNX|Nifty|NIFTY)[A-Za-z0-9&:%\-\. ]{0,60}?)(?:\s+Index)?\s*:?\s*$")
PREFIX = re.compile(r"^\s*\(?[0-9a-zA-Z]{1,3}[).]\s+")
ses = cal.sessions()


def _date(d):
    return dt.date(d // 10000, d // 100 % 100, d % 100)


def _int(d):
    return d.year * 10000 + d.month * 100 + d.day


def prev_session(day):
    i = bisect.bisect_left(ses, day)
    return ses[i - 1] if i > 0 else _int(_date(day) - dt.timedelta(days=1))


def which(name):
    n = re.sub(r"\s+", " ", name.lower().replace("index", " ")).strip(" :.-")
    for k, al in ALIASES.items():
        if n in al:
            return k
    return None


def find_date(txt):
    m = D1.search(txt)
    if m:
        return int(m.group(3)) * 10000 + MONTHS[m.group(1).lower()] * 100 + int(m.group(2))
    m = D2.search(txt)
    if m:
        return int(m.group(3)) * 10000 + MONTHS[m.group(2).lower()] * 100 + int(m.group(1))
    return None


def parse_release(lines, announce):
    """-> list of (index, effective, action, symbol, company, fmt)"""
    n = len(lines)
    eff_at = [None] * n
    cur = None
    for i, ln in enumerate(lines):
        m = EFF.search(ln)
        if m:
            d = find_date((ln[m.end():] + " " + (lines[i + 1] if i + 1 < n else "") + " " + (lines[i + 2] if i + 2 < n else ""))[:160])
            if d and announce - 10000 <= d <= announce + 10000:
                cur = d
        eff_at[i] = cur
    first_eff = next((e for e in eff_at if e and e >= announce), None) or next((e for e in eff_at if e), None)
    out = []
    index, mode = None, None
    para = []
    list_mode = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        low = s.lower()
        if not s:
            continue
        if re.search(r"sr\.?\s*no\.?\s+index name", low):
            txt = re.sub(r"\s+", " ", " ".join(para[-14:]))
            acts = [(m.start(), "OUT") for m in re.finditer(r"exclu(?:de|ded|sion)", txt, re.I)] + \
                   [(m.start(), "IN") for m in re.finditer(r"inclu(?:de|ded|sion)", txt, re.I)]
            syms = [(m.start(), m.group(1)) for m in re.finditer(r"\((?:Symbol\s*:\s*)?([A-Z0-9][A-Z0-9&\-]{1,19})\)", txt)]
            if acts and syms and not list_mode:
                eff = eff_at[i] if (eff_at[i] and eff_at[i] >= announce) else first_eff
                list_mode = (max(acts)[1], syms[-1][1], eff)
            continue
        if list_mode:
            m = IDXROW.match(ln)
            if m:
                ix = which(m.group(2))
                if ix:
                    out.append((ix, list_mode[2], list_mode[0], list_mode[1], "", "index_list"))
                continue
            if not re.match(r"^\s*\d", ln):
                list_mode = None
        m = HEAD.match(ln)
        if m and not PREFIX.match(ln):
            # an unnumbered line is a section header only if a change table follows directly (a wrapped line of a
            # note such as "Nifty 50 index" is not)
            nxt = next((x.strip().lower() for x in lines[i + 1:i + 4] if x.strip()), "")
            if not re.search(r"following compan|being excluded|being included", nxt):
                m = None
        if m and "exclu" not in low and "inclu" not in low and len(s) < 90:
            index = which(m.group(1))
            mode = None
            para.append(s)
            continue
        if "being excluded" in low or "will be excluded" in low or "to be excluded" in low or low.startswith("exclusion"):
            mode = "OUT" if index else None
        elif "being included" in low or "will be included" in low or "to be included" in low or low.startswith("inclusion"):
            mode = "IN" if index else None
        m = ROW.match(ln)
        if m and index and mode and m.group(3) not in ("SYMBOL", "NO", "NAME"):
            eff = eff_at[i] if (eff_at[i] and eff_at[i] >= announce) else first_eff
            out.append((index, eff, mode, m.group(3).strip("*"), re.sub(r"\s+", " ", m.group(2)), "section"))
            continue
        para.append(s)
    return out


# ══ 1. press releases
raw_rows, parsed_files, no_eff, empty_text = [], 0, [], []
for fn in sorted(glob.glob(os.path.join(RAW, "press", "txt", "*.txt.gz"))):
    base = os.path.basename(fn)
    m = re.match(r"ind_prs(\d{2})(\d{2})(\d{4})", base)
    if not m:
        continue
    announce = int(m.group(3)) * 10000 + int(m.group(2)) * 100 + int(m.group(1))
    text = gzip.open(fn, "rt", errors="replace").read()
    if len(text.strip()) < 50:
        empty_text.append(base.replace(".txt.gz", ".pdf"))
    parsed_files += 1
    for ix, eff, act, sym, comp, fmt in parse_release(text.splitlines(), announce):
        if eff is None:
            no_eff.append((base, ix, act, sym)); continue
        raw_rows.append({"index": ix, "announce_date": announce, "effective_date": eff, "action": act, "symbol": sym,
                         "company": comp, "source": base.replace(".txt.gz", ".pdf"), "fmt": fmt})
dropped = collections.Counter()
keep = []
for c in raw_rows:
    if c["symbol"].startswith("DUMMY"):
        dropped["dummy symbol of a demerged entity"] += 1; continue
    if c["fmt"] == "index_list":
        dropped["index-list format (demerger housekeeping; the monthly files decide)"] += 1; continue
    keep.append(c)
# the same (index, action, symbol) announced again within 45 days is a repeat or a revision (Yes Bank, March 2020):
# known from the FIRST announcement, effective on the date of the LAST one
keep.sort(key=lambda c: (c["index"], c["action"], c["symbol"], c["announce_date"], c["source"]))
press = []
for c in keep:
    m = press[-1] if press else None
    if m and (m["index"], m["action"], m["symbol"]) == (c["index"], c["action"], c["symbol"]) and \
            abs((_date(c["effective_date"]) - _date(m["effective_date"])).days) <= 45:
        if c["effective_date"] != m["effective_date"]:
            m["note"] = f"effective date revised from {m['effective_date']} by {c['source']}"
            m["effective_date"] = c["effective_date"]
        dropped["repeat of an earlier announcement"] += 1
        continue
    c["note"] = ""
    c["latest"] = symmap.latest_symbol(c["symbol"], c["announce_date"])
    press.append(c)

# ══ 2. Wikipedia (NIFTY 50)
WIKI = {"Reliance Communications": "RCOM", "Asian Paints": "ASIANPAINT", "Reliance Power": "RPOWER", "Bank of Baroda": "BANKBARODA",
        "Steel Authority of India": "SAIL", "Lupin": "LUPIN", "Sterlite Industries": "STER", "UltraTech Cement": "ULTRACEMCO",
        "Siemens India": "SIEMENS", "IndusInd Bank": "INDUSINDBK", "Wipro": "WIPRO", "NMDC": "NMDC",
        "Reliance Infrastructure": "RELINFRA", "JP Associates": "JPASSOCIAT", "Tech Mahindra": "TECHM",
        "Ranbaxy Laboratories": "RANBAXY", "United Spirits": "MCDOWELL-N", "Zee Entertainment Enterprises": "ZEEL", "DLF": "DLF",
        "Idea Cellular": "IDEA", "Jindal Steel & Power": "JINDALSTEL", "Yes Bank": "YESBANK", "IDFC": "IDFC",
        "Bosch India": "BOSCHLTD", "Adani Ports & SEZ": "ADANIPORTS", "Cairn India": "CAIRN", "Aurobindo Pharma": "AUROPHARMA",
        "Punjab National Bank": "PNB", "Bharti Infratel": "INFRATEL", "Vedanta": "VEDL", "Eicher Motors": "EICHERMOT",
        "BHEL": "BHEL", "Indiabulls Housing Finance": "IBULHSGFIN", "Indian Oil Corporation": "IOC",
        "Grasim Industries": "GRASIM", "ACC": "ACC", "Bajaj Finance": "BAJFINANCE", "Hindustan Petroleum": "HINDPETRO",
        "Tata Power": "TATAPOWER", "UPL": "UPL", "Ambuja Cements": "AMBUJACEM", "Bajaj Finserv": "BAJAJFINSV",
        "Titan Company": "TITAN", "JSW Steel": "JSWSTEEL", "Britannia Industries": "BRITANNIA", "Nestlé India": "NESTLEIND",
        "Shree Cement": "SHREECEM", "HDFC Life": "HDFCLIFE", "SBI Life Insurance Company": "SBILIFE",
        "Divi's Laboratories": "DIVISLAB", "GAIL": "GAIL", "Tata Consumer Products": "TATACONSUM",
        "Apollo Hospitals": "APOLLOHOSP", "Adani Enterprises": "ADANIENT", "HDFC": "HDFC", "LTIMindtree": "LTIM",
        "Shriram Finance": "SHRIRAMFIN", "Bharat Electronics": "BEL", "Trent": "TRENT", "Bharat Petroleum": "BPCL",
        "Jio Financial Services": "JIOFIN", "Eternal": "ETERNAL", "Hero MotoCorp": "HEROMOTOCO", "IndiGo": "INDIGO",
        "Max Healthcare": "MAXHEALTH", "BSE": "BSE"}
wiki, wiki_unmapped = [], []
wp = os.path.join(RAW, "wikipedia", "NIFTY_50.html")
if os.path.exists(wp):
    h = open(wp, encoding="utf8").read()
    i0 = h.find("List of replacements since 2005"); j0 = h.find("</table>", i0)
    date = None
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", h[i0:j0], re.S):
        cells = [re.sub(r"\[\d+\]", "", _html.unescape(re.sub(r"<[^>]+>", "", c))).strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
        if len(cells) < 2 or cells[0].startswith("Constituent"):
            continue
        if len(cells) >= 3 and re.search(r"\d{4}", cells[2]):
            date = find_date(cells[2])
        if not date or date < 20120101:
            continue
        for name, act in ((cells[0], "OUT"), (cells[1], "IN")):
            if name in WIKI:
                wiki.append({"effective_date": date, "action": act, "latest": symmap.latest_symbol(WIKI[name], date), "name": name})
            else:
                wiki_unmapped.append((date, act, name))

# ══ 3. monthly constituent files
snaps = {}                      # ym -> {"date": last session of the month, "NIFTY50": {latest symbol: row}, "NIFTYNEXT50": {...}}
weights_rows, snap_problems = [], []
for fn in sorted(glob.glob(os.path.join(RAW, "mcwb", "*.zip"))):
    ym = int(os.path.basename(fn)[:6])
    y, mth = divmod(ym, 100)
    last_day = _int((dt.date(y + (mth == 12), mth % 12 + 1, 1) - dt.timedelta(days=1)))
    i = bisect.bisect_right(ses, last_day)
    sdate = ses[i - 1] if i > 0 and ses[i - 1] // 100 == ym else last_day
    rec = {"date": sdate}
    try:
        z = zipfile.ZipFile(fn)
    except zipfile.BadZipFile:
        snap_problems.append(f"{ym}: bad zip"); continue
    for n in z.namelist():
        if not n.lower().endswith(".csv"):
            continue
        t = z.read(n).decode("latin-1").splitlines()
        title = (t[0] if t else "").lower()
        ix = "NIFTYNEXT50" if re.search(r"jr|junior|next\s*50", title + " " + n.lower()) else "NIFTY50"
        hi = next((k for k, l in enumerate(t[:8]) if "symbol" in l.lower()), None)
        if hi is None:
            snap_problems.append(f"{ym}: no header in {n}"); continue
        rd = csv.reader(io.StringIO("\n".join(t[hi:])))
        hdr = [x.strip().lower() for x in next(rd)]
        ks = next(k for k, x in enumerate(hdr) if "symbol" in x)
        kw = next((k for k, x in enumerate(hdr) if "weight" in x), None)
        km = next((k for k, x in enumerate(hdr) if "free float" in x or "market cap" in x), None)
        kn = next((k for k, x in enumerate(hdr) if "name" in x), None)
        d = {}
        for r in rd:
            if len(r) <= ks or not r[0].strip().isdigit() or not r[ks].strip():
                continue
            sym = r[ks].strip().upper()

            def num(k):
                try:
                    return float(r[k].replace(",", "")) if k is not None and len(r) > k else ""
                except ValueError:
                    return ""
            d[symmap.latest_symbol(sym, sdate)] = {"symbol": sym, "name": r[kn].strip() if kn is not None and len(r) > kn else "",
                                                   "weight": num(kw), "mcap": num(km)}
            lab = symmap.lab_symbol(sym, sdate)
            weights_rows.append([ym, sdate, ix, sym, lab, int(symmap.in_lab(lab)), num(kw), num(km)])
        rec[ix] = d
    if "NIFTY50" in rec and "NIFTYNEXT50" in rec:
        snaps[ym] = rec
    else:
        snap_problems.append(f"{ym}: one index missing")


def read_list(path):
    out = []
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":                   # archived copies keep the original gzip content-encoding
        raw = gzip.decompress(raw)
    with io.StringIO(raw.decode("utf-8-sig", errors="replace")) as fh:
        rd = csv.reader(fh)
        hdr = [x.strip().lower() for x in next(rd, [])]
        if "symbol" not in hdr:
            return None
        k = hdr.index("symbol")
        for r in rd:
            if len(r) > k and r[k].strip():
                out.append(r[k].strip().upper())
    return out


cur_lists = {ix: read_list(os.path.join(RAW, "current", f)) or [] for ix, f in
             {"NIFTY50": "ind_nifty50list.csv", "NIFTY100": "ind_nifty100list.csv", "NIFTYNEXT50": "ind_niftynext50list.csv"}.items()}
yms = sorted(snaps)
expect, ym = [], yms[0] if yms else 201112
while ym <= 202610:
    expect.append(ym)
    ym = ym + 1 if ym % 100 < 12 else (ym // 100 + 1) * 100 + 1
missing_months = [m for m in expect if m not in snaps]
# timeline of states: (date, {index: set}) ; the last state is today's constituent list
states = [(snaps[m]["date"], {ix: set(snaps[m][ix]) for ix in ("NIFTY50", "NIFTYNEXT50")}, m) for m in yms]
states.append((BUILD_DATE, {ix: {symmap.latest_symbol(s, BUILD_DATE) for s in cur_lists[ix]} for ix in ("NIFTY50", "NIFTYNEXT50")}, 0))

# ══ 4. date every change
dated = []          # dicts: index, action, latest, effective_date, precision, announce_date, announce_source, source, company, note
other = {"NIFTY50": "NIFTYNEXT50", "NIFTYNEXT50": "NIFTY50"}
used_press = set()
for k in range(1, len(states)):
    d0, s0, _ = states[k - 1]
    d1, s1, m1 = states[k]
    gap_months = 1 if k == len(states) - 1 else (m1 // 100 - states[k - 1][2] // 100) * 12 + (m1 % 100 - states[k - 1][2] % 100)
    pend = []
    for ix in ("NIFTY50", "NIFTYNEXT50"):
        for act, syms in (("OUT", s0[ix] - s1[ix]), ("IN", s1[ix] - s0[ix])):
            for sym in sorted(syms):
                if sym.startswith("DUMMY"):
                    continue
                c = {"index": ix, "action": act, "latest": sym, "effective_date": None, "precision": None,
                     "announce_date": None, "announce_source": "none", "source": f"monthly constituent files {states[k - 1][2] or ''}->{m1 or 'current list'}",
                     "company": (snaps.get(m1, {}).get(ix, {}).get(sym) or snaps.get(states[k - 1][2], {}).get(ix, {}).get(sym) or {}).get("name", ""),
                     "note": ""}
                hit = [p for p in press if p["index"] == ix and p["action"] == act and p["latest"] == sym and d0 < p["effective_date"] <= d1]
                if not hit:      # a release whose effective date fell just outside the window (holiday shifts)
                    hit = [p for p in press if p["index"] == ix and p["action"] == act and p["latest"] == sym
                           and abs((_date(p["effective_date"]) - _date(d1)).days) <= 6 and id(p) not in used_press]
                if hit:
                    p = hit[0]
                    used_press.add(id(p))
                    eff = min(max(p["effective_date"], _int(_date(d0) + dt.timedelta(days=1))), d1)
                    c.update(effective_date=eff, precision="day", announce_date=p["announce_date"], announce_source="press",
                             source=p["source"], note=p["note"], company=p["company"] or c["company"], press_symbol=p["symbol"])
                elif ix == "NIFTY50":
                    w = [x for x in wiki if x["action"] == act and x["latest"] == sym and d0 < x["effective_date"] <= d1]
                    if w:
                        c.update(effective_date=w[0]["effective_date"], precision="day",
                                 source="wikipedia:NIFTY_50 + " + c["source"])
                pend.append(c)
    # inferred dates: the same stock in the other index, else the month's single known review date
    known = sorted({c["effective_date"] for c in pend if c["effective_date"]})
    for c in pend:
        if c["effective_date"]:
            continue
        twin = [x for x in pend if x["index"] == other[c["index"]] and x["latest"] == c["latest"] and x["action"] != c["action"] and x["effective_date"]]
        if twin:
            c.update(effective_date=twin[0]["effective_date"], precision="inferred",
                     note="date of the same stock's change in " + other[c["index"]])
            if twin[0]["announce_source"] == "press":
                c.update(announce_date=twin[0]["announce_date"], announce_source="press (release for " + other[c["index"]] + ")")
        elif len(known) == 1 and gap_months == 1:
            c.update(effective_date=known[0], precision="inferred", note="the one review date known for that month")
        else:
            c.update(effective_date=d1, precision="month",
                     note="exact day unknown: dated at the month-end file that first shows the change (latest possible)")
    for c in pend:
        if c["announce_date"] is None:
            c["announce_date"] = prev_session(c["effective_date"])
            c["announce_source"] = "none: set to the session before the effective date"
        dated.append(c)
press_unconfirmed = [[p["index"], p["announce_date"], p["effective_date"], p["action"], p["symbol"], p["source"]]
                     for p in press if id(p) not in used_press and p["index"] in ("NIFTY50", "NIFTYNEXT50")
                     and ORIGIN <= p["effective_date"] <= BUILD_DATE]

# ══ 5. membership intervals
first_state = states[0][1]
intervals = {ix: [] for ix in ("NIFTY50", "NIFTYNEXT50")}
for ix in intervals:
    open_ = {s: (ORIGIN, "start") for s in first_state[ix] if not s.startswith("DUMMY")}
    for c in sorted([c for c in dated if c["index"] == ix], key=lambda c: (c["effective_date"], c["action"] == "IN")):
        if c["action"] == "OUT":
            st = open_.pop(c["latest"], None)
            if st:
                intervals[ix].append((c["latest"], st[0], _int(_date(c["effective_date"]) - dt.timedelta(days=1)), st[1], c["precision"]))
        else:
            open_[c["latest"]] = (c["effective_date"], c["precision"])
    for s, st in open_.items():
        intervals[ix].append((s, st[0], None, st[1], ""))
# NIFTY 100 = union of the two
merged100 = collections.defaultdict(list)
for ix in ("NIFTY50", "NIFTYNEXT50"):
    for s, f, t, pf, pt in intervals[ix]:
        merged100[s].append([f, t if t is not None else 99999999, pf, pt])
intervals["NIFTY100"] = []
for s, lst in merged100.items():
    lst.sort()
    cur = lst[0][:]
    for f, t, pf, pt in lst[1:]:
        if f <= _int(_date(min(cur[1], 99991230)) + dt.timedelta(days=1)) if cur[1] < 99999999 else True:
            if t >= cur[1]:
                cur[1], cur[3] = t, pt
        else:
            intervals["NIFTY100"].append((s, cur[0], None if cur[1] == 99999999 else cur[1], cur[2], cur[3]))
            cur = [f, t, pf, pt]
    intervals["NIFTY100"].append((s, cur[0], None if cur[1] == 99999999 else cur[1], cur[2], cur[3]))


def members_on(ix, day):
    return {s for s, f, t, _, _ in intervals[ix] if f <= day and (t is None or day <= t)}


# NIFTY 100 change rows: boundaries of its intervals, dated and sourced from the underlying change (or the press
# release for NIFTY 100 itself when there is one)
by_key = {(c["index"], c["action"], c["latest"], c["effective_date"]): c for c in dated}
for s, f, t, pf, pt in intervals["NIFTY100"]:
    for act, eff in (("IN", f if f > ORIGIN else None), ("OUT", _int(_date(t) + dt.timedelta(days=1)) if t else None)):
        if eff is None:
            continue
        base = by_key.get(("NIFTY50", act, s, eff)) or by_key.get(("NIFTYNEXT50", act, s, eff))
        if base is None:
            continue
        c = dict(base, index="NIFTY100")
        p = [x for x in press if x["index"] == "NIFTY100" and x["action"] == act and x["latest"] == s
             and abs((_date(x["effective_date"]) - _date(eff)).days) <= 6]
        if p:
            c.update(announce_date=p[0]["announce_date"], announce_source="press", source=p[0]["source"])
            used_press.add(id(p[0]))
        dated.append(c)
press100_unconfirmed = [[p["index"], p["announce_date"], p["effective_date"], p["action"], p["symbol"], p["source"]]
                        for p in press if id(p) not in used_press and p["index"] == "NIFTY100" and ORIGIN <= p["effective_date"] <= BUILD_DATE]

# ══ 6. write
dated = [c for c in dated if ORIGIN < c["effective_date"]]
dated.sort(key=lambda c: (c["index"], c["effective_date"], c["action"], c["latest"]))
with open(os.path.join(OUT, "index_changes.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["index", "announce_date", "effective_date", "action", "symbol", "lab_symbol", "in_lab", "company", "source",
                "announce_source", "effective_precision", "note"])
    for c in dated:
        sym = c.get("press_symbol") or symmap.symbol_on(c["latest"], c["effective_date"])
        lab = symmap.lab_symbol(c["latest"], BUILD_DATE)
        w.writerow([c["index"], c["announce_date"], c["effective_date"], c["action"], sym, lab, int(symmap.in_lab(lab)),
                    c["company"], c["source"], c["announce_source"], c["precision"], c["note"]])
mrows = []
for ix in ("NIFTY50", "NIFTY100", "NIFTYNEXT50"):
    for s, f, t, pf, pt in intervals[ix]:
        lab = symmap.lab_symbol(s, BUILD_DATE)
        mrows.append([ix, s, lab, int(symmap.in_lab(lab)), f, t or "", pf, pt])
mrows.sort(key=lambda r: (r[0], r[1], r[4]))
with gzip.open(os.path.join(OUT, "index_members.csv.gz"), "wt", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["index", "symbol", "lab_symbol", "in_lab", "from_date", "to_date", "from_precision", "to_precision"])
    w.writerows(mrows)
with gzip.open(os.path.join(OUT, "index_weights.csv.gz"), "wt", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["month", "snapshot_date", "index", "symbol", "lab_symbol", "in_lab", "weight_pct", "ff_mcap_cr"])
    w.writerows(weights_rows)

# ══ 7. data check
chk = {"built_through": BUILD_DATE, "origin": ORIGIN,
       "sources": {"monthly_constituent_files": "niftyindices.com mcwb_<mon><yy>.zip (NSE Indices), month-end NIFTY 50 and NIFTY Next 50 with weights",
                   "press_releases": "niftyindices.com/Press_Release/ind_prsDDMMYYYY*.pdf", "second_source": "en.wikipedia.org/wiki/NIFTY_50 (list of replacements)"},
       "availability": "a change is visible when announce_date < before; membership uses only changes with effective_date < before",
       "monthly_files": {"first": yms[0] if yms else None, "last": yms[-1] if yms else None, "count": len(yms),
                         "missing_months": missing_months, "problems": snap_problems,
                         "sizes": dict(collections.Counter(f"{len(snaps[m]['NIFTY50'])}/{len(snaps[m]['NIFTYNEXT50'])}" for m in yms))},
       "press_releases_parsed": parsed_files, "press_releases_without_text": empty_text,
       "press_rows_dropped": dict(dropped), "press_rows_without_effective_date": [list(x) for x in no_eff][:40]}
for ix in ("NIFTY50", "NIFTYNEXT50", "NIFTY100"):
    cs = [c for c in dated if c["index"] == ix]
    chk[ix] = {"reliable_from": ORIGIN if not missing_months else None,
               "current_constituents": len(cur_lists[ix]), "intervals": len(intervals[ix]), "change_rows": len(cs),
               "changes_per_year": dict(sorted(collections.Counter(str(c["effective_date"])[:4] for c in cs).items())),
               "effective_precision": dict(collections.Counter(c["precision"] for c in cs)),
               "announce_source": dict(collections.Counter(c["announce_source"].split(":")[0].split(" (")[0] for c in cs)),
               "month_precision_rows": [[c["effective_date"], c["action"], c["latest"]] for c in cs if c["precision"] == "month"],
               "members_today_equal_current_list": members_on(ix, BUILD_DATE) == {symmap.latest_symbol(s, BUILD_DATE) for s in cur_lists[ix]},
               "size_by_year_end": {str(d): len(members_on(ix, d)) for d in [s for s in ses if str(s)[4:6] == "12"][::-1] if d // 100 % 100 == 12 and d == max(x for x in ses if x // 10000 == d // 10000)}}
chk["press_rows_not_matched_to_a_monthly_change"] = {"NIFTY50_NEXT50": press_unconfirmed, "NIFTY100": press100_unconfirmed,
                                                      "note": "a release row with no matching difference between month-end files: a change undone "
                                                              "within the month, a second share class (TATAMTRDVR), or a parsing error"}
# Wikipedia comparison (NIFTY 50)
n50 = {(c["latest"], c["action"]): c for c in dated if c["index"] == "NIFTY50"}
same = diff = miss = 0
wdiff, wmiss = [], []
for x in wiki:
    c = next((c for c in dated if c["index"] == "NIFTY50" and c["latest"] == x["latest"] and c["action"] == x["action"]
              and abs((_date(c["effective_date"]) - _date(x["effective_date"])).days) <= 31), None)
    if c is None:
        miss += 1; wmiss.append([x["effective_date"], x["action"], x["latest"]])
    elif c["effective_date"] == x["effective_date"]:
        same += 1
    else:
        diff += 1; wdiff.append([x["latest"], x["action"], c["effective_date"], x["effective_date"], c["source"]])
wk = {(x["latest"], x["action"]) for x in wiki}
chk["nifty50_vs_wikipedia"] = {"wikipedia_rows_2012_on": len(wiki), "unmapped_names": wiki_unmapped, "same_stock_and_date": same,
                               "same_stock_other_date": wdiff, "in_wikipedia_not_in_files": wmiss,
                               "in_files_not_in_wikipedia": [[c["effective_date"], c["action"], c["latest"], c["source"]] for c in dated
                                                             if c["index"] == "NIFTY50" and (c["latest"], c["action"]) not in wk]}
# press N100 rows vs the union-derived NIFTY 100 changes
p100 = [p for p in press if p["index"] == "NIFTY100" and ORIGIN <= p["effective_date"] <= BUILD_DATE]
d100 = {(c["latest"], c["action"]): c for c in dated if c["index"] == "NIFTY100"}
agree = sum(1 for p in p100 if any(c["index"] == "NIFTY100" and c["latest"] == p["latest"] and c["action"] == p["action"]
                                   and abs((_date(c["effective_date"]) - _date(p["effective_date"])).days) <= 6 for c in dated))
chk["nifty100_press_rows_vs_union"] = {"press_rows": len(p100), "matched_in_union_derived_changes": agree}
# independent anchors
anch = []
for fn in sorted(glob.glob(os.path.join(RAW, "wayback", "snap", "*.csv"))):
    ix, ts = os.path.basename(fn)[:-4].rsplit("_", 1)
    day = int(ts[:8])
    lst = read_list(fn)
    if not lst or day < ORIGIN:
        continue
    snap = {symmap.latest_symbol(s, day) for s in lst}
    rec = members_on(ix, day)
    anch.append({"index": ix, "snapshot_date": day, "snapshot_size": len(snap), "reconstructed_size": len(rec),
                 "in_snapshot_not_reconstructed": sorted(snap - rec), "reconstructed_not_in_snapshot": sorted(rec - snap),
                 "match": snap == rec})
chk["internet_archive_anchors"] = {"files": len(anch), "matching": sum(1 for a in anch if a["match"]), "detail": anch,
                                   "note": "the Internet Archive rate-limited downloads (HTTP 429): only a few snapshots obtained"}
# spot checks: 10 NIFTY 50 changes with every source side by side
import random
random.seed(4)
cands = [c for c in dated if c["index"] == "NIFTY50" and c["announce_source"] == "press"]
spots = []
for c in random.sample(cands, min(10, len(cands))):
    w = next((x for x in wiki if x["latest"] == c["latest"] and x["action"] == c["action"]
              and abs((_date(c["effective_date"]) - _date(x["effective_date"])).days) <= 31), None)
    mb = max((m for m in yms if snaps[m]["date"] < c["effective_date"]), default=None)
    ma = min((m for m in yms if snaps[m]["date"] >= c["effective_date"]), default=None)
    spots.append({"symbol": c["latest"], "action": c["action"], "press_release": c["source"], "announce_date": c["announce_date"],
                  "effective_date_press": c["effective_date"], "effective_date_wikipedia": w["effective_date"] if w else None,
                  "in_month_end_file_before": (c["latest"] in snaps[mb]["NIFTY50"]) if mb else None, "file_before": mb,
                  "in_month_end_file_after": (c["latest"] in snaps[ma]["NIFTY50"]) if ma else None, "file_after": ma})
chk["spot_checks"] = spots
chk["rows"] = {"index_changes": len(dated), "index_members": len(mrows), "index_weights": len(weights_rows)}
json.dump(chk, open(os.path.join(OUT, "data_check.json"), "w"), indent=1)
print("monthly files", chk["monthly_files"])
print("press dropped", dict(dropped), "parsed", parsed_files)
for ix in ("NIFTY50", "NIFTYNEXT50", "NIFTY100"):
    c = chk[ix]
    print(ix, {k: c[k] for k in ("change_rows", "intervals", "effective_precision", "announce_source", "members_today_equal_current_list")})
    print("   per year", c["changes_per_year"])
    print("   month precision", c["month_precision_rows"][:30])
    print("   sizes", c["size_by_year_end"])
w = chk["nifty50_vs_wikipedia"]
print("wikipedia: same", w["same_stock_and_date"], "other date", w["same_stock_other_date"], "not in files", w["in_wikipedia_not_in_files"], "files only", w["in_files_not_in_wikipedia"])
print("press unmatched", len(press_unconfirmed), press_unconfirmed[:30])
print("press n100", chk["nifty100_press_rows_vs_union"], press100_unconfirmed[:20])
print("anchors", chk["internet_archive_anchors"]["files"], chk["internet_archive_anchors"]["matching"])
