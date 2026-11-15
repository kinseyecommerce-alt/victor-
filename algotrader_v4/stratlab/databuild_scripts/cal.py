"""Session calendar and unadjusted NSE daily prices for the databuild data checks (read-only on the lab's files).

sessions()            sorted int yyyymmdd: NIFTY rows of nse_old/index_daily (2012-01-02..2023-11-09) + trade dates of
                      nse/delivery.csv.gz (2023-11-01..), + any extra dates passed by the caller
px()                  dict of numpy arrays for ALL NSE symbols, EQ/BE series: date, sym (codes), open, high, low, close,
                      prev_close, volume  (official, UNADJUSTED) ; lookup(sym, date) -> dict | None
"""
import os, sys
import numpy as np

S = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"
OLD = os.path.join(S, "nse_old")
NEW = os.path.join(S, "nse")
CACHE = os.path.join(S, "databuild", "_px")
_P = {}


def _build():
    import pandas as pd
    os.makedirs(CACHE, exist_ok=True)
    cols = ["date", "symbol", "open", "high", "low", "close", "prev_close", "volume"]
    a = pd.read_csv(os.path.join(OLD, "eq_daily.csv.gz"), usecols=cols + ["series"], dtype={"symbol": str, "series": str})
    a["pri"] = (a.series != "EQ").astype(int)
    a = a.sort_values(["symbol", "date", "pri"]).drop_duplicates(["symbol", "date"])[cols]
    b = pd.read_csv(os.path.join(NEW, "delivery.csv.gz"), usecols=cols, dtype={"symbol": str})
    b = b[b.date > a.date.max()]
    df = pd.concat([a, b], ignore_index=True).sort_values(["symbol", "date"])
    syms, codes = np.unique(df.symbol.to_numpy(str), return_inverse=True)
    np.savez_compressed(os.path.join(CACHE, "eq_all.npz"), syms=syms, code=codes.astype(np.int32),
                        date=df.date.to_numpy(np.int32),
                        **{k: df[k].to_numpy(np.float32) for k in cols[2:7]}, volume=df.volume.to_numpy(np.float64))
    idx = pd.read_csv(os.path.join(OLD, "index_daily.csv.gz"), usecols=["date", "index"])
    ses = sorted(set(idx[idx["index"] == "NIFTY"].date.astype(int)) | set(b.date.astype(int)))
    np.save(os.path.join(CACHE, "sessions.npy"), np.array(ses, np.int32))


def _ensure():
    if not os.path.exists(os.path.join(CACHE, "sessions.npy")):
        _build()


def sessions(extra=()):
    _ensure()
    return sorted(set(int(x) for x in np.load(os.path.join(CACHE, "sessions.npy"))) | set(int(x) for x in extra))


def px():
    if not _P:
        _ensure()
        z = np.load(os.path.join(CACHE, "eq_all.npz"), allow_pickle=False)
        _P.update({k: z[k] for k in z.files})
        syms = _P["syms"]
        _P["_ix"] = {s: i for i, s in enumerate(syms)}
        bounds = np.searchsorted(_P["code"], np.arange(len(syms) + 1))
        _P["_b"] = bounds
    return _P


def series(sym):
    """All rows of one NSE symbol (as reported on each date; no rename merging): dict of arrays or None."""
    p = px()
    i = p["_ix"].get(sym)
    if i is None:
        return None
    lo, hi = p["_b"][i], p["_b"][i + 1]
    return {k: p[k][lo:hi] for k in ("date", "open", "high", "low", "close", "prev_close", "volume")}


def lookup(sym, date):
    s = series(sym)
    if s is None:
        return None
    j = int(np.searchsorted(s["date"], date))
    if j >= len(s["date"]) or s["date"][j] != date:
        return None
    return {k: float(s[k][j]) for k in s}


if __name__ == "__main__":
    _build()
    print(len(sessions()), "sessions", sessions()[0], sessions()[-1], "rows", len(px()["date"]))
