"""
stratlab/events_fetch.py — polite downloader for NSE's public corporate-filing feeds (RESEARCH ONLY).

Standalone on purpose (stdlib + requests, no package imports) so it can be run isolated:

    python3 -I algotrader_v4/stratlab/events_fetch.py --out <scratch>/events/raw --feeds pit,bm,fr,ann_fr,ann_obm \
        --start 2011-10 --end 2026-11

What it fetches (one gzip file per feed and calendar month, never overwritten once complete):

  pit      /api/corporates-pit                SEBI PIT Reg 7(2) insider / promoter trades with the exchange broadcast
                                              datetime (`date`), inline detail; the feed ends 2026-05-02
  pitgg    /api/corporates-pit-gg             the index of the same disclosures from 2026-05-03 (detail is in one XML
                                              document per filing: see --pit-docs)
  bm       /api/corporate-board-meetings      board-meeting notices: meeting date, purpose, intimation datetime
  fr       /api/corporates-financial-results  quarterly results filings with their broadcast datetime
  ann_fr   /api/corporate-announcements       subject "Financial Result Updates"  (broadcast datetime `an_dt`)
  ann_obm  /api/corporate-announcements       subject "Outcome of Board Meeting"
  ca       /api/corporates-corporateActions   ex-dates and purposes (bonus / split / dividend ...), for price adjustment
  ifr      /api/integrated-filing-results     "Integrated Filing - Financials" (results from 2025-04; paged, one file
                                              per month and page)

Etiquette / safety:
  * sequential, one request every --pause seconds (default 2.0), exponential backoff, a generic browser User-Agent,
    no credentials, no personal or project information in any request;
  * the public site sets its session cookies on an ordinary HTML page; the script loads that page first, exactly as
    a browser would. It does NOT try to defeat any bot protection: a 401 / 403 after one cookie refresh is recorded
    in the manifest as "refused" and the month is skipped;
  * a host blocked by the network policy (proxy 403 / 407 on CONNECT) is recorded and never retried;
  * responses are stored as received (gzip), parsed later as DATA only by events_parse.py.
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import gzip
import json
import os
import sys
import time

import requests

SITE = "https://www.nseindia.com"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
PAGES = {
    "pit": "/companies-listing/corporate-filings-insider-trading",
    "pitgg": "/companies-listing/corporate-filings-insider-trading",
    "bm": "/companies-listing/corporate-filings-board-meetings",
    "fr": "/companies-listing/corporate-filings-financial-results",
    "ann_fr": "/companies-listing/corporate-filings-announcements",
    "ann_obm": "/companies-listing/corporate-filings-announcements",
    "ca": "/companies-listing/corporate-filings-actions",
    "ifr": "/companies-listing/corporate-integrated-filing",
}
IFR_START = dt.date(2025, 4, 1)
IFR_PAGE = 1000
PIT_GG_START = dt.date(2026, 4, 21)         # the index + document form of the insider feed is asked from here
PIT_LEGACY_END = dt.date(2026, 5, 31)       # the inline feed still carried rows to 2026-05-02: ask it to here


def api_path(feed: str, d0: dt.date, d1: dt.date) -> str:
    w = f"index=equities&from_date={d0:%d-%m-%Y}&to_date={d1:%d-%m-%Y}"
    if feed == "pit":
        return f"/api/corporates-pit?{w}"
    if feed == "pitgg":
        return f"/api/corporates-pit-gg?{w}"
    if feed == "bm":
        return f"/api/corporate-board-meetings?{w}"
    if feed == "fr":
        return f"/api/corporates-financial-results?index=equities&period=Quarterly&from_date={d0:%d-%m-%Y}&to_date={d1:%d-%m-%Y}"
    if feed == "ann_fr":
        return f"/api/corporate-announcements?{w}&subject=Financial%20Result%20Updates"
    if feed == "ann_obm":
        return f"/api/corporate-announcements?{w}&subject=Outcome%20of%20Board%20Meeting"
    if feed == "ca":
        return f"/api/corporates-corporateActions?{w}"
    if feed == "ifr":
        return f"/api/integrated-filing-results?{w}&type=Integrated%20Filing-%20Financials&size={IFR_PAGE}"
    raise ValueError(feed)


def month_windows(start: str, end: str):
    y, m = (int(x) for x in start.split("-"))
    ye, me = (int(x) for x in end.split("-"))
    while (y, m) <= (ye, me):
        yield dt.date(y, m, 1), dt.date(y, m, calendar.monthrange(y, m)[1])
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


class Client:
    def __init__(self, pause: float):
        self.pause = pause
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9",
                               "Accept-Encoding": "gzip, deflate"})
        self.base = SITE
        self.warmed = set()
        self.last = 0.0

    def _wait(self):
        d = self.pause - (time.time() - self.last)
        if d > 0:
            time.sleep(d)
        self.last = time.time()

    def warm(self, page: str, force: bool = False):
        if page in self.warmed and not force:
            return
        self._wait()
        r = self.s.get(self.base + page, headers={"Accept": "text/html,application/xhtml+xml,*/*"}, timeout=60)
        # the site may answer from a sister host (a permanent redirect); follow it for the API calls too
        host = r.url.split("/companies-listing")[0]
        if host.startswith("https://") and host != self.base:
            self.base = host
        self.warmed.add(page)

    def get(self, path: str, page: str, binary: bool = False):
        """-> (status, body bytes | None). status: 'ok', 'refused', 'blocked', 'error'."""
        delay = 5.0
        refreshed = False
        for _ in range(5):
            try:
                self.warm(page)
                self._wait()
                url = path if path.startswith("http") else self.base + path
                r = self.s.get(url, headers={"Accept": "application/json, text/plain, */*",
                                             "Referer": self.base + page}, timeout=180)
            except requests.exceptions.ProxyError as e:      # policy denial on CONNECT: do not retry
                return "blocked", str(e)[:200].encode()
            except requests.RequestException:
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code == 200:
                return "ok", r.content
            if r.status_code in (401, 403):
                if refreshed:
                    return "refused", r.content[:200]
                refreshed = True
                self.warm(page, force=True)
                continue
            if r.status_code == 404:
                return "error", b"404"
            time.sleep(delay)
            delay *= 2
        return "error", None


def rows_of(payload):
    if isinstance(payload, dict):
        for k in ("data", "records", "rows"):
            if isinstance(payload.get(k), list):
                return payload[k]
        return []
    return payload if isinstance(payload, list) else []


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--feeds", default="pit,pitgg,bm,fr,ann_fr,ann_obm")
    ap.add_argument("--start", default="2011-10")
    ap.add_argument("--end", required=True)
    ap.add_argument("--pause", type=float, default=2.0)
    ap.add_argument("--pit-docs", default="", help="file with one symbol per line: fetch the per-filing XML documents "
                                                   "of the pitgg index for these symbols only")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    man_path = os.path.join(out, "manifest.jsonl")
    cl = Client(a.pause)
    today = dt.date.today()

    def log(rec):
        rec["ts"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        with open(man_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    for feed in [f for f in a.feeds.split(",") if f]:
        os.makedirs(os.path.join(out, feed), exist_ok=True)
        for d0, d1 in month_windows(a.start, a.end):
            if feed == "pit" and d0 > PIT_LEGACY_END:
                continue
            if feed == "pitgg" and d1 < PIT_GG_START:
                continue
            if d0 > today:
                continue
            if feed == "ifr":
                if d1 < IFR_START:
                    continue
                page = 1
                while page <= 60:
                    dst = os.path.join(out, feed, f"{d0:%Y%m}_p{page:02d}.json.gz")
                    if os.path.exists(dst) and d1 < today - dt.timedelta(days=3):
                        with gzip.open(dst, "rb") as f:
                            n = len(rows_of(json.loads(f.read())))
                    else:
                        status, body = cl.get(api_path(feed, d0, d1) + f"&page={page}", PAGES[feed])
                        n = -1
                        if status == "ok":
                            try:
                                n = len(rows_of(json.loads(body)))
                            except ValueError:
                                status = "error"
                        if status == "ok" and n > 0:
                            with gzip.open(dst + ".tmp", "wb", compresslevel=9) as f:
                                f.write(body)
                            os.replace(dst + ".tmp", dst)
                        log({"feed": feed, "month": f"{d0:%Y-%m}", "page": page, "status": status, "rows": n,
                             "bytes": len(body) if body else 0, "host": cl.base})
                        print(feed, f"{d0:%Y-%m}", page, status, n, flush=True)
                        if status != "ok":
                            break
                    if n < IFR_PAGE:
                        break
                    page += 1
                continue
            dst = os.path.join(out, feed, f"{d0:%Y%m}.json.gz")
            if os.path.exists(dst) and d1 < today - dt.timedelta(days=3):
                continue
            status, body = cl.get(api_path(feed, d0, d1), PAGES[feed])
            n = -1
            if status == "ok":
                try:
                    n = len(rows_of(json.loads(body)))
                except ValueError:
                    status = "error"
            if status == "ok":
                with gzip.open(dst + ".tmp", "wb", compresslevel=9) as f:
                    f.write(body)
                os.replace(dst + ".tmp", dst)
            log({"feed": feed, "month": f"{d0:%Y-%m}", "status": status, "rows": n,
                 "bytes": len(body) if body else 0, "host": cl.base})
            print(feed, f"{d0:%Y-%m}", status, n, flush=True)
            if status == "blocked":
                print("host blocked by network policy; stopping this feed", flush=True)
                break

    if a.pit_docs:
        want = {x.strip().upper() for x in open(a.pit_docs) if x.strip()}
        ddir = os.path.join(out, "pitgg_docs")
        os.makedirs(ddir, exist_ok=True)
        idx_dir = os.path.join(out, "pitgg")
        for fn in sorted(os.listdir(idx_dir)) if os.path.isdir(idx_dir) else []:
            with gzip.open(os.path.join(idx_dir, fn), "rb") as f:
                rows = rows_of(json.loads(f.read()))
            for r in rows:
                if str(r.get("symbol", "")).upper() not in want:
                    continue
                url = r.get("xmlFileName") or ""
                if not url.startswith("https://") or "nseindia" not in url.split("/")[2]:
                    continue
                name = os.path.basename(url.split("?")[0])
                if not name.lower().endswith(".xml") or "/" in name or name.startswith("."):
                    continue
                dst = os.path.join(ddir, name + ".gz")
                if os.path.exists(dst):
                    continue
                status, body = cl.get(url, PAGES["pitgg"])
                if status == "ok" and body:
                    with gzip.open(dst + ".tmp", "wb", compresslevel=9) as f:
                        f.write(body)
                    os.replace(dst + ".tmp", dst)
                log({"feed": "pitgg_docs", "doc": name, "symbol": r.get("symbol"), "status": status,
                     "bytes": len(body) if body else 0})
                if status == "blocked":
                    print("document host blocked by network policy; stopping", flush=True)
                    return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
