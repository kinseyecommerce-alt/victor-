"""Normalise NSE bulk / block deal quarter files -> databuild/deals/deals.csv.gz

Schema (one row per reported deal leg):
  date        int yyyymmdd, the TRADE date
  kind        bulk | block
  symbol      NSE symbol as reported (the symbol in force on the trade date)
  lab_symbol  the lab's name for that company (rename chain resolved, see symmap.py)
  in_lab      1 when lab_symbol is one of the lab's bar files
  security    security name as reported
  client      client name as reported (upper-cased, whitespace collapsed)
  side        BUY | SELL
  qty         shares (int)
  price       trade price / weighted average price (Rs, unadjusted)
  value_cr    qty * price / 1e7
  remarks     NSE remarks column ("" when "-")
Availability: NSE publishes the day's bulk and block deals after the close of the trade date; a row dated D is usable
from the open of the next session after D (accessors return rows dated strictly before `before`).
Usage: python3 -I build_deals.py RAW_DEALS_DIR OUT_DIR"""
import csv, datetime as dt, glob, gzip, io, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import symmap

RAW, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
rows, bad = [], []
for kind in ("bulk", "block"):
    for fn in sorted(glob.glob(os.path.join(RAW, kind, "*.csv"))):
        txt = open(fn, encoding="utf-8-sig").read()
        rd = csv.reader(io.StringIO(txt))
        hdr = [h.strip() for h in next(rd)]
        assert hdr[:7] == ["Date", "Symbol", "Security Name", "Client Name", "Buy / Sell", "Quantity Traded",
                           "Trade Price / Wght. Avg. Price"], (fn, hdr)
        for r in rd:
            if not r or not r[0].strip():
                continue
            try:
                d = int(dt.datetime.strptime(r[0].strip(), "%d-%b-%Y").strftime("%Y%m%d"))
                sym = r[1].strip().upper()
                side = r[4].strip().upper()
                qty = int(r[5].replace(",", "").strip())
                px = float(r[6].replace(",", "").strip())
                assert side in ("BUY", "SELL") and qty > 0 and px > 0 and sym
            except Exception as e:
                bad.append((os.path.basename(fn), r[:8])); continue
            client = re.sub(r"\s+", " ", r[3].strip().upper())
            rem = (r[7].strip() if len(r) > 7 else "")
            lab = symmap.lab_symbol(sym, d)
            rows.append((d, kind, sym, lab, int(symmap.in_lab(lab)), re.sub(r"\s+", " ", r[2].strip()), client, side,
                         qty, px, round(qty * px / 1e7, 4), "" if rem in ("-", "") else rem))
rows.sort(key=lambda x: (x[0], x[1], x[2], x[6], x[7]))
with gzip.open(os.path.join(OUT, "deals.csv.gz"), "wt", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["date", "kind", "symbol", "lab_symbol", "in_lab", "security", "client", "side", "qty", "price",
                "value_cr", "remarks"])
    w.writerows(rows)
print("rows", len(rows), "bad", len(bad))
for b in bad[:20]:
    print("BAD", b)
