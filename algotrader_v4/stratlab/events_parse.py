"""
stratlab/events_parse.py — normalise the raw NSE filing feeds (events_fetch.py) into three small tables.

Standalone (stdlib only) so it can run isolated on untrusted downloads, which are parsed as DATA only:

    python3 -I algotrader_v4/stratlab/events_parse.py --raw <scratch>/events/raw --out <scratch>/events

Outputs (gzip CSV, all timestamps IST "YYYY-MM-DD HH:MM:SS", symbols = the CURRENT NSE symbol after renames):

  insider.csv.gz    one row per disclosed transaction (SEBI PIT Reg 7(2))
      key, symbol, company, person, person_cat, is_promoter (1 = "Promoters" / "Promoter Group"), mode, txn (Buy /
      Sell / Pledge ...), sec_type, qty, value, from_date, to_date, intim_date, broadcast, exchange, source
      `broadcast` is the exchange's public dissemination time: the ONLY field a strategy may time on. from/to are the
      TRANSACTION dates (earlier, private until the broadcast) and must never be used for timing.
  results.csv.gz    one row per (symbol, period end): quarterly financial results (structured filings from
      /api/corporates-financial-results to 2025-03 and /api/integrated-filing-results from 2025-04)
      symbol, isin, period_end, relating_to, first_ts, src_first, fr_ts, ann_ts, n_fr, n_ann
      fr_ts   = earliest broadcast of a structured results filing for that period (in 2013-2018 this is often days
                after the numbers were out; from 2019 usually hours after)
      ann_ts  = the earliest announcement of the company after the period end and not later than fr_ts that is
                either filed under "Financial Result Updates", or filed under "Outcome of Board Meeting" on the date
                of a board meeting NOTIFIED for results (board-meeting feed)
      first_ts = min(fr_ts, ann_ts): the first moment the numbers were public on NSE. A wrong (too early / too
                late) first_ts mis-places the reaction day; it can never let a strategy see the future, because
                every use is "broadcast strictly before the decision".
  boardmeet.csv.gz  one row per board-meeting notice
      symbol, isin, meeting_date, purpose, is_results, notice_ts

Rows without a parseable broadcast time are dropped and counted in parse_report.json (never given a guessed time).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import gzip
import io
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                      "dec"), 1)}
_TS = re.compile(r"^\s*(\d{1,2})-([A-Za-z]{3})-(\d{4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?\s*$")


def parse_ts(s, need_time: bool = True):
    """'10-Jan-2024 21:21[:05]' -> datetime, or None. With need_time a bare date is rejected (no guessed times)."""
    if not isinstance(s, str):
        return None
    m = _TS.match(s)
    if not m:
        return None
    mon = MONTHS.get(m.group(2).lower())
    if mon is None:
        return None
    if m.group(4) is None:
        if need_time:
            return None
        hh = mm = ss = 0
    else:
        hh, mm, ss = int(m.group(4)), int(m.group(5)), int(m.group(6) or 0)
    try:
        return dt.datetime(int(m.group(3)), mon, int(m.group(1)), hh, mm, ss)
    except ValueError:
        return None


def parse_date(s):
    t = parse_ts(s, need_time=False)
    return None if t is None else t.date()


def fmt(t) -> str:
    return "" if t is None else t.strftime("%Y-%m-%d %H:%M:%S")


def num(x):
    try:
        v = float(str(x).replace(",", "").strip())
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def rows_of(payload):
    if isinstance(payload, dict):
        for k in ("data", "records", "rows"):
            if isinstance(payload.get(k), list):
                return payload[k]
        return []
    return payload if isinstance(payload, list) else []


def load_feed(raw: str, feed: str):
    for fn in sorted(glob.glob(os.path.join(raw, feed, "*.json.gz"))):
        try:
            with gzip.open(fn, "rb") as f:
                payload = json.loads(f.read())
        except (OSError, ValueError):
            continue
        for r in rows_of(payload):
            if isinstance(r, dict):
                yield r


# ── symbol renames ──────────────────────────────────────────────────────────
def load_changes(path: str) -> list:
    """NSE symbolchange.csv -> [(effective yyyymmdd, old, new)] in date order."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="latin-1", newline="") as f:
        rd = csv.reader(f)
        next(rd, None)
        for r in rd:
            if len(r) < 4:
                continue
            d = parse_date(r[3].strip().title())
            old, new = r[1].strip().upper(), r[2].strip().upper()
            if d is None or not old or not new or old == new:
                continue
            out.append((int(d.strftime("%Y%m%d")), old, new))
    out.sort()
    return out


def canon(sym: str, day: int, changes: list) -> str:
    """The current symbol of what was called `sym` on `day`: every later rename OLD -> NEW is followed. A rename
    dated on or before `day` is not applied (a symbol seen after its own rename date is a re-used ticker)."""
    s = (sym or "").strip().upper()
    for d, old, new in changes:
        if d > day and old == s:
            s, day = new, d
    return s


# ── insider ─────────────────────────────────────────────────────────────────
INS_COLS = ("key", "symbol", "company", "person", "person_cat", "is_promoter", "mode", "txn", "sec_type", "qty",
            "value", "from_date", "to_date", "intim_date", "broadcast", "exchange", "source")


def is_promoter(cat: str) -> bool:
    """'Promoters', 'Promoter Group', 'Promoter', 'Member of Promoter Group', ... (any category naming promoters)."""
    return "promoter" in (cat or "").lower()


def is_equity(sec_type: str) -> bool:
    return (sec_type or "").strip().lower().startswith("equity")


def insider_legacy_row(r: dict, changes: list):
    ts = parse_ts(r.get("date"))
    if ts is None:
        return None
    day = int(ts.strftime("%Y%m%d"))
    fd, td, idt = parse_date(r.get("acqfromDt")), parse_date(r.get("acqtoDt")), parse_date(r.get("intimDt"))
    cat = (r.get("personCategory") or "").strip()
    return {"key": f"L{r.get('did')}", "symbol": canon(r.get("symbol"), day, changes),
            "company": (r.get("company") or "").strip(), "person": (r.get("acqName") or "").strip(),
            "person_cat": cat, "is_promoter": int(is_promoter(cat)), "mode": (r.get("acqMode") or "").strip(),
            "txn": (r.get("tdpTransactionType") or "").strip(), "sec_type": (r.get("secType") or "").strip(),
            "qty": num(r.get("secAcq")), "value": num(r.get("secVal")),
            "from_date": "" if fd is None else fd.isoformat(), "to_date": "" if td is None else td.isoformat(),
            "intim_date": "" if idt is None else idt.isoformat(), "broadcast": fmt(ts),
            "exchange": (r.get("exchange") or "").strip(), "source": "pit"}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


# element local-names of the SEBI PIT XBRL taxonomy (Reg 7(2) "continual disclosure") -> our fields
_DOC_MAP = {
    "NameOfThePerson": "person", "CategoryOfPerson": "person_cat", "TypeOfInstrument": "sec_type",
    "SecuritiesAcquiredOrDisposedNumberOfSecurity": "qty", "SecuritiesAcquiredOrDisposedValueOfSecurity": "value",
    "SecuritiesAcquiredOrDisposedTransactionType": "txn", "ModeOfAcquisitionOrDisposal": "mode",
    "DateOfAllotmentAdviceOrAcquisitionOfSharesOrSaleOfSharesSpecifyFromDate": "from_date",
    "DateOfAllotmentAdviceOrAcquisitionOfSharesOrSaleOfSharesSpecifyToDate": "to_date",
    "DateOfIntimationToCompany": "intim_date", "ExchangeOnWhichTheTradeWasExecuted": "exchange",
}


def insider_doc_rows(xml_bytes: bytes, index_row: dict, changes: list) -> list:
    """One per-filing XBRL document of the 2026-04-21+ insider feed -> rows. The broadcast time comes from the INDEX
    row (the exchange's dissemination time), never from the document. Facts are grouped by their contextRef (one
    context per disclosed transaction). Parsed with the stdlib parser: no DTD / entity expansion is performed."""
    ts = parse_ts(index_row.get("broadcastDateTime")) or parse_ts(index_row.get("exchdisstime"))
    if ts is None:
        return []
    if b"<!DOCTYPE" in xml_bytes[:4096] or b"<!ENTITY" in xml_bytes[:4096]:
        return []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    day = int(ts.strftime("%Y%m%d"))
    by_ctx: dict = {}
    for el in root.iter():
        name = _local(el.tag)
        f = _DOC_MAP.get(name)
        ctx = el.attrib.get("contextRef")
        if f is None or ctx is None:
            continue
        by_ctx.setdefault(ctx, {})[f] = (el.text or "").strip()
    out = []
    sym = canon(index_row.get("symbol"), day, changes)
    for k, (ctx, d) in enumerate(sorted(by_ctx.items())):
        if "qty" not in d and "value" not in d:
            continue
        cat = d.get("person_cat", "")
        out.append({"key": f"D{index_row.get('appId')}-{k}", "symbol": sym,
                    "company": (index_row.get("companyName") or "").strip(), "person": d.get("person", ""),
                    "person_cat": cat, "is_promoter": int(is_promoter(cat)), "mode": d.get("mode", ""),
                    "txn": d.get("txn", ""), "sec_type": d.get("sec_type", ""), "qty": num(d.get("qty")),
                    "value": num(d.get("value")), "from_date": d.get("from_date", "")[:10],
                    "to_date": d.get("to_date", "")[:10], "intim_date": d.get("intim_date", "")[:10],
                    "broadcast": fmt(ts), "exchange": d.get("exchange", ""), "source": "pitgg"})
    return out


def dedupe_insider(rows: list) -> tuple[list, int]:
    """Drop exact repeats: same key, or the same transaction (symbol, person, mode, txn, qty, value, from, to)
    disclosed again (revised filings, NSE + BSE copies). The EARLIEST broadcast is kept."""
    rows = sorted(rows, key=lambda r: (r["broadcast"], r["key"]))
    seen_k, seen_t, out = set(), set(), []
    for r in rows:
        t = (r["symbol"], r["person"].lower(), r["mode"], r["txn"], r["qty"], r["value"], r["from_date"], r["to_date"])
        if r["key"] in seen_k or t in seen_t:
            continue
        seen_k.add(r["key"])
        seen_t.add(t)
        out.append(r)
    return out, len(rows) - len(out)


# ── results ─────────────────────────────────────────────────────────────────
RES_COLS = ("symbol", "isin", "period_end", "relating_to", "first_ts", "src_first", "fr_ts", "ann_ts", "n_fr", "n_ann")


def build_results(fr_rows, ann_fr_rows, ann_obm_rows, changes: list, ifr_rows=(),
                  results_meetings: dict | None = None) -> tuple[list, dict]:
    """fr_rows: /api/corporates-financial-results (to 2025-03); ifr_rows: /api/integrated-filing-results
    ("Integrated Filing - Financials", from 2025-04): both are structured results filings with a broadcast time.
    results_meetings: {symbol: set of dates of board meetings notified for results} (the board-meeting feed)."""
    rep = {"fr_rows": 0, "ifr_rows": 0, "fr_no_ts": 0, "fr_no_period": 0, "ann_rows": 0, "ann_no_ts": 0}
    per: dict = {}
    both = [(r, "fr") for r in fr_rows] + [(r, "ifr") for r in ifr_rows]
    for r, kind in both:
        rep[kind + "_rows"] += 1
        if kind == "fr":
            ts = parse_ts(r.get("broadCastDate")) or parse_ts(r.get("filingDate"))
            pe = parse_date(r.get("toDate"))
        else:
            ts = parse_ts(r.get("broadcast_Date")) or parse_ts(r.get("creation_Date"))
            pe = parse_date(r.get("qe_Date"))
        if ts is None:
            rep["fr_no_ts"] += 1
            continue
        if pe is None or not (0 <= (ts.date() - pe).days <= 200):
            rep["fr_no_period"] += 1                  # a late re-filing / revision of an old period is not an event
            continue
        sym = canon(r.get("symbol"), int(ts.strftime("%Y%m%d")), changes)
        e = per.setdefault((sym, pe), {"fr": [], "isin": "", "rel": ""})
        e["fr"].append(ts)
        e["isin"] = e["isin"] or (r.get("isin") or "").strip()
        e["rel"] = e["rel"] or (r.get("relatingTo") or "").strip()
    ann: dict = {}
    for kind, rows in (("fr", ann_fr_rows), ("obm", ann_obm_rows)):
        for r in rows:
            rep["ann_rows"] += 1
            ts = parse_ts(r.get("an_dt"))
            if ts is None:
                rep["ann_no_ts"] += 1
                continue
            sym = canon(r.get("symbol"), int(ts.strftime("%Y%m%d")), changes)
            ann.setdefault(sym, []).append((ts, kind))
    for v in ann.values():
        v.sort()
    meet = results_meetings or {}
    out = []
    for (sym, pe), e in sorted(per.items()):
        fr_ts = min(e["fr"])
        lo = dt.datetime.combine(pe, dt.time(23, 59, 59))
        md = meet.get(sym, ())
        cands = [t for t, kind in ann.get(sym, ())
                 if lo < t <= fr_ts and (kind == "fr" or t.date() in md)]
        ann_ts = min(cands) if cands else None
        first = fr_ts if ann_ts is None else min(fr_ts, ann_ts)
        out.append({"symbol": sym, "isin": e["isin"], "period_end": pe.isoformat(), "relating_to": e["rel"],
                    "first_ts": fmt(first), "src_first": "ann" if (ann_ts is not None and ann_ts < fr_ts) else "fr",
                    "fr_ts": fmt(fr_ts), "ann_ts": fmt(ann_ts), "n_fr": len(e["fr"]), "n_ann": len(cands)})
    return out, rep


# ── board meetings ──────────────────────────────────────────────────────────
BM_COLS = ("symbol", "isin", "meeting_date", "purpose", "is_results", "notice_ts")
_RESULT = re.compile(r"result", re.I)


def build_boardmeet(rows, changes: list) -> tuple[list, dict]:
    rep = {"bm_rows": 0, "bm_no_ts": 0, "bm_no_date": 0}
    seen, out = set(), []
    for r in rows:
        rep["bm_rows"] += 1
        ts = parse_ts(r.get("bm_timestamp"))
        md = parse_date(r.get("bm_date"))
        if md is None:
            rep["bm_no_date"] += 1
            continue
        if ts is None:
            rep["bm_no_ts"] += 1
            continue
        sym = canon(r.get("bm_symbol"), int(ts.strftime("%Y%m%d")), changes)
        purpose = " ".join(str(r.get("bm_purpose") or "").split())
        k = (sym, md, fmt(ts), purpose)
        if k in seen:
            continue
        seen.add(k)
        out.append({"symbol": sym, "isin": (r.get("sm_isin") or "").strip(), "meeting_date": md.isoformat(),
                    "purpose": purpose, "is_results": int(bool(_RESULT.search(purpose))), "notice_ts": fmt(ts)})
    out.sort(key=lambda r: (r["notice_ts"], r["symbol"]))
    return out, rep


def write_csv(path: str, cols, rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(cols), extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    with gzip.open(path + ".tmp", "wt", compresslevel=9, newline="") as f:
        f.write(buf.getvalue())
    os.replace(path + ".tmp", path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    raw, out = os.path.abspath(a.raw), os.path.abspath(a.out)
    changes = load_changes(os.path.join(raw, "ref", "symbolchange.csv"))
    rep = {"symbol_changes": len(changes)}

    ins, bad = [], 0
    n_leg = 0
    for r in load_feed(raw, "pit"):
        n_leg += 1
        x = insider_legacy_row(r, changes)
        if x is None:
            bad += 1
        else:
            ins.append(x)
    idx = {os.path.basename(str(r.get("xmlFileName") or "")): r for r in load_feed(raw, "pitgg")}
    n_doc = n_doc_rows = 0
    for fn in sorted(glob.glob(os.path.join(raw, "pitgg_docs", "*.xml.gz"))):
        row = idx.get(os.path.basename(fn)[:-3])
        if row is None:
            continue
        try:
            with gzip.open(fn, "rb") as f:
                body = f.read()
        except OSError:
            continue
        n_doc += 1
        rr = insider_doc_rows(body, row, changes)
        n_doc_rows += len(rr)
        ins.extend(rr)
    ins, ndup = dedupe_insider(ins)
    write_csv(os.path.join(out, "insider.csv.gz"), INS_COLS, ins)
    rep["insider"] = {"legacy_rows": n_leg, "legacy_no_timestamp": bad, "index_rows_2026": len(idx),
                      "documents_parsed": n_doc, "document_rows": n_doc_rows, "duplicates_dropped": ndup,
                      "rows_out": len(ins)}

    bm, r3 = build_boardmeet(load_feed(raw, "bm"), changes)
    write_csv(os.path.join(out, "boardmeet.csv.gz"), BM_COLS, bm)
    r3["rows_out"] = len(bm)
    rep["boardmeet"] = r3
    meetings: dict = {}
    for r in bm:
        if r["is_results"]:
            meetings.setdefault(r["symbol"], set()).add(dt.date.fromisoformat(r["meeting_date"]))

    res, r2 = build_results(load_feed(raw, "fr"), load_feed(raw, "ann_fr"), load_feed(raw, "ann_obm"), changes,
                            load_feed(raw, "ifr"), meetings)
    write_csv(os.path.join(out, "results.csv.gz"), RES_COLS, res)
    r2["events_out"] = len(res)
    rep["results"] = r2
    with open(os.path.join(out, "parse_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
