"""
ist_clock.py — Central IST clock for AlgoTrader Pro.

All market-timing decisions must use now_ist() instead of datetime.now()
so they work correctly regardless of the server's local timezone (UTC in prod).
"""
from datetime import date, datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

_IST = ZoneInfo("Asia/Kolkata")

_MARKET_OPEN  = dtime(9, 15)
_MARKET_CLOSE = dtime(15, 30)

# NSE equity (capital market) trading holidays that fall on weekdays. Source: Zerodha and Groww
# holiday calendars for 2026 (the NSE page itself refused automated reads); both agree. Extend or
# correct without a deploy through MARKET_HOLIDAYS="YYYY-MM-DD:Name,YYYY-MM-DD:Name" (env/.env),
# and when a new year's list is published add it here. Weekend holidays and Muhurat sessions are
# not listed: weekends are closed anyway and Muhurat is a special evening session we do not trade.
NSE_HOLIDAYS: dict[date, str] = {
    date(2026, 1, 15):  "Municipal Corporation Election - Maharashtra",
    date(2026, 1, 26):  "Republic Day",
    date(2026, 3, 3):   "Holi",
    date(2026, 3, 26):  "Shri Ram Navami",
    date(2026, 3, 31):  "Shri Mahavir Jayanti",
    date(2026, 4, 3):   "Good Friday",
    date(2026, 4, 14):  "Dr. Baba Saheb Ambedkar Jayanti",
    date(2026, 5, 1):   "Maharashtra Day",
    date(2026, 5, 28):  "Bakri Eid",
    date(2026, 6, 26):  "Muharram",
    date(2026, 9, 14):  "Ganesh Chaturthi",
    date(2026, 10, 2):  "Mahatma Gandhi Jayanti",
    date(2026, 10, 20): "Dussehra",
    date(2026, 11, 10): "Diwali-Balipratipada",
    date(2026, 11, 24): "Prakash Gurpurb Sri Guru Nanak Dev",
    date(2026, 12, 25): "Christmas",
}
_HOLIDAY_YEARS = {2026}


def _extra_holidays() -> dict[date, str]:
    """Operator-supplied holidays from settings.market_holidays ("YYYY-MM-DD[:Name],...")."""
    out: dict[date, str] = {}
    try:
        from config import settings
        raw = settings.market_holidays or ""
    except Exception:
        return out
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        d, _, name = part.partition(":")
        try:
            out[date.fromisoformat(d.strip())] = name.strip() or "Market holiday"
        except ValueError:
            continue
    return out


def holiday_name(d: date | None = None) -> str | None:
    """The exchange holiday on `d` (default: today IST), or None on a normal day."""
    d = d or now_ist().date()
    return _extra_holidays().get(d) or NSE_HOLIDAYS.get(d)


def is_trading_day(d: date | None = None) -> bool:
    """Weekday that is not an NSE holiday."""
    d = d or now_ist().date()
    return d.weekday() < 5 and holiday_name(d) is None


def holiday_calendar_known(d: date | None = None) -> bool:
    """False once we are in a year whose NSE list has not been added (MARKET_HOLIDAYS still works)."""
    d = d or now_ist().date()
    return d.year in _HOLIDAY_YEARS or any(h.year == d.year for h in _extra_holidays())


def next_trading_day(d: date | None = None) -> date:
    d = (d or now_ist().date()) + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return d


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
        while not is_trading_day(day):                  # weekend/holiday → previous session
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
    """True if the NSE equity market is currently open: a trading day (Mon–Fri, not an exchange
    holiday) between 09:15 and 15:30 IST."""
    n = now_ist()
    if not is_trading_day(n.date()):
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
