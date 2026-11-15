"""Symbol mapping shared by the databuild scripts.

NSE reports every record under the symbol in force when the record was made.  The lab's bar files (bars3y) carry
one name per company (mostly the latest NSE symbol, but TATAMOTORS and LTIM kept their pre-rename names).
lab_symbol(sym, date) -> the lab's name for the company that traded as `sym` on `date`:
  1. follow NSE's rename list (symbolchange.csv: OLD -> NEW effective D) forward in date order from `date`
     (a rename applies to records dated before its effective date), giving the latest NSE symbol;
  2. map that latest symbol back to the lab's file name when the lab kept an older name.
Symbols that are not in the lab's universe are returned as their latest NSE symbol (so non-lab stocks are still
consistently named), with in_lab(sym) telling the two apart.
"""
import csv, datetime as dt, os

S = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"
SYMCHG = os.path.join(S, "databuild", "raw", "ref", "symbolchange.csv")
BARS = os.path.join(S, "bars3y")

_ren = {}        # old -> sorted [(effective yyyymmdd, new)]
_lab = None
_back = {}


def _load():
    global _lab
    if _lab is not None:
        return
    with open(SYMCHG, encoding="latin-1") as fh:
        rd = csv.reader(fh)
        next(rd)
        for r in rd:
            if len(r) < 4:
                continue
            old, new = r[1].strip(), r[2].strip()
            try:
                d = int(dt.datetime.strptime(r[3].strip(), "%d-%b-%Y").strftime("%Y%m%d"))
            except ValueError:
                continue
            if old and new and old != new:
                _ren.setdefault(old, []).append((d, new))
    for v in _ren.values():
        v.sort()
    _lab = sorted(f[:-5] for f in os.listdir(BARS) if f.endswith(".json") and not f.startswith("_"))
    for name in _lab:
        latest = latest_symbol(name, 19000101)
        if latest != name:
            _back[latest] = name


def latest_symbol(sym, date):
    _load() if _lab is None and not _ren else None
    cur, d = sym, int(date)
    for _ in range(12):
        nxt = [(e, n) for e, n in _ren.get(cur, ()) if e > d]
        if not nxt:
            break
        d, cur = nxt[0]
    return cur


def symbol_on(sym, date):
    """The symbol in force on `date` for the company that is `sym` today (rename chain walked backwards).
    NSE's corporate-action feed reports old events under the CURRENT symbol."""
    _load()
    cur, d = sym.strip().upper(), int(date)
    for _ in range(12):
        prev = [(e, old) for old, lst in _ren.items() for e, new in lst if new == cur and e > d]
        if not prev:
            break
        prev.sort()
        # the rename INTO cur that happened soonest after `date` (later ones belong to cur's own later history)
        e, old = prev[0]
        later_out = [x for x, _ in _ren.get(cur, ()) if d < x <= e]
        if later_out:
            break
        cur = old
    return cur


def lab_symbol(sym, date):
    _load()
    s = latest_symbol(sym.strip().upper(), date)
    return _back.get(s, s)


def in_lab(lab_sym):
    _load()
    return lab_sym in _lab


def lab_names():
    _load()
    return list(_lab)


def back_map():
    _load()
    return dict(_back)
