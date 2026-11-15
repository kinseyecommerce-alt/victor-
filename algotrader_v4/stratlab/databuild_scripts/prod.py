"""Read-only client for the production server's GET /backtest/history-bars (research data only).

Base URL and API key come from the environment (ALGOTRADER_BASE_URL, ALGOTRADER_API_KEY) or, when unset, are read at
call time from the session's existing fetch script (<scratchpad>/fetch3y.sh).  Neither is stored in this file; the key
is sent ONLY to that base URL and is never printed, logged or written anywhere.  Only GET requests are issued, and
only to PATH below.
"""
import json, os, re, time, urllib.parse, urllib.request, urllib.error

_S = "/tmp/claude-0/-home-user-victor-/10a63b67-92f5-5498-8eb5-0d5b292e5470/scratchpad"
PATH = "/backtest/history-bars"
MIN_GAP = 1.1            # seconds between requests (<= 1 request / second)
_last = [0.0]
_cfg = {}


def _conf():
    if not _cfg:
        base, key = os.environ.get("ALGOTRADER_BASE_URL"), os.environ.get("ALGOTRADER_API_KEY")
        if not (base and key):
            txt = open(os.path.join(_S, "fetch3y.sh")).read()
            mk = re.search(r'X-API-Key: ([^"\s]+)', txt)
            mu = re.search(r'\bU=(https://[^\s"]+)', txt)
            if not (mk and mu):
                raise RuntimeError("base URL / api key not configured")
            base, key = base or mu.group(1), key or mk.group(1)
        _cfg["base"], _cfg["key"] = base.rstrip("/"), key
    return _cfg


class Persistent(Exception):
    pass


def history(symbol, start, end, interval="minute", exchange="NFO", timeout=180, tries=5):
    """-> (status, bars|None).  status 200 -> list of [ts,o,h,l,c,v]; 404 -> None (no instrument token).
    Backs off on 429/5xx/network errors; raises Persistent after `tries` failures."""
    c = _conf()
    q = urllib.parse.urlencode({"symbol": symbol, "start": start, "end": end, "interval": interval,
                                "exchange": exchange})
    url = c["base"] + PATH + "?" + q
    err = None
    for attempt in range(tries):
        wait = MIN_GAP - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers={"X-API-Key": c["key"], "Accept": "application/json"}, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            if "bars" not in d:
                raise ValueError("no bars key")
            return 200, d["bars"]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 404, None
            if e.code in (400, 401, 403, 422):
                raise Persistent(f"HTTP {e.code} for {symbol} {start}..{end}")
            err = f"HTTP {e.code}"
        except Exception as e:          # network / json
            err = type(e).__name__
        time.sleep(min(300, 15 * 2 ** attempt))
    raise Persistent(f"{err} for {symbol} {start}..{end}")


def probe():
    """Kite session check: a known busy equity week must return bars.  -> number of bars (0 = session invalid)."""
    st, bars = history("RELIANCE", "2026-11-09", "2026-11-13", exchange="NSE", tries=2)
    return len(bars or [])
