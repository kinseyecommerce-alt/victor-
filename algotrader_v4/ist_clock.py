"""
ist_clock.py — Central IST clock for AlgoTrader Pro.

All market-timing decisions must use now_ist() instead of datetime.now()
so they work correctly regardless of the server's local timezone (UTC in prod).
"""
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

_IST = ZoneInfo("Asia/Kolkata")

_MARKET_OPEN  = dtime(9, 15)
_MARKET_CLOSE = dtime(15, 30)


_demo_anchor: tuple[datetime, datetime] | None = None   # (real start, simulated start)


def _demo_now() -> datetime | None:
    """DEMO_MODE (paper simulator only): a clock that starts at 10:00 IST on a weekday when
    the server starts and then runs at real speed, so market-hours logic works any time."""
    global _demo_anchor
    try:
        from config import settings
        if not settings.demo_active:
            return None
    except Exception:
        return None
    real = datetime.now(_IST)
    if _demo_anchor is None:
        day = real.date()
        while day.weekday() >= 5:                       # weekend → previous Friday
            day -= timedelta(days=1)
        _demo_anchor = (real, datetime.combine(day, dtime(10, 0), tzinfo=_IST))
    return _demo_anchor[1] + (real - _demo_anchor[0])


def now_ist() -> datetime:
    """Current datetime in IST (Asia/Kolkata). In DEMO_MODE, the simulated demo clock."""
    return _demo_now() or datetime.now(_IST)


def ist_time() -> dtime:
    """Current wall-clock time in IST."""
    return now_ist().time()


def is_market_open() -> bool:
    """True if NSE equity market is currently open (Mon–Fri 09:15–15:30 IST)."""
    n = now_ist()
    if n.weekday() >= 5:
        return False
    t = n.time().replace(tzinfo=None)
    return _MARKET_OPEN <= t <= _MARKET_CLOSE


def minutes_since_open() -> int:
    """Minutes elapsed since 09:15 IST today (0 if before open)."""
    t = ist_time()
    open_mins = 9 * 60 + 15
    now_mins  = t.hour * 60 + t.minute
    return max(0, now_mins - open_mins)


def minutes_to_squareoff(squareoff_time: str) -> int:
    """Minutes remaining until squareoff_time (HH:MM IST). 0 if already past."""
    h, m = [int(x) for x in squareoff_time.split(":")]
    t = ist_time()
    sq_mins  = h * 60 + m
    now_mins = t.hour * 60 + t.minute
    return max(0, sq_mins - now_mins)
