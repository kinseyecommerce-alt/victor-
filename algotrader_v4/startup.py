#!/usr/bin/env python3
"""
startup.py
Run this ONCE before starting the server:
  python startup.py
It will:
  1. Validate .env keys are filled in
  2. Test Kite connection (if access token is set)
  3. Test Anthropic API connection
  4. Show a ready summary
"""
import os, sys
from dotenv import load_dotenv

load_dotenv()

REQUIRED = [
    "KITE_API_KEY",
    "KITE_API_SECRET",
    "ANTHROPIC_API_KEY",
]

OPTIONAL = [
    "KITE_ACCESS_TOKEN",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
]

print("\n" + "="*52)
print("  AlgoTrader Pro v2 — Startup Check")
print("="*52)

# ── Check required keys ────────────────────────────────
ok = True
for key in REQUIRED:
    val = os.getenv(key, "")
    if not val or val.startswith("your_"):
        print(f"  ✗  {key} — NOT SET")
        ok = False
    else:
        print(f"  ✓  {key} — ok ({val[:4]}…)")

for key in OPTIONAL:
    val = os.getenv(key, "")
    status = "set" if val and not val.startswith("your_") else "not set"
    print(f"  ○  {key} — {status}")

if not ok:
    print("\n  ✗ Fill in .env before starting.\n")
    sys.exit(1)

# ── Test Anthropic ────────────────────────────────────
print("\n  Testing Anthropic API…")
try:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": "say ok"}]
    )
    print("  ✓  Anthropic API — connected")
except Exception as e:
    print(f"  ✗  Anthropic API error: {e}")
    ok = False

# ── Test Kite (optional — needs daily token) ──────────────────────
token = os.getenv("KITE_ACCESS_TOKEN", "")
if token and not token.startswith("your_"):
    print("\n  Testing Zerodha Kite API…")
    try:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=os.getenv("KITE_API_KEY"))
        kite.set_access_token(token)
        profile = kite.profile()
        print(f"  ✓  Kite API — connected as {profile.get('user_name', '?')}")
    except Exception as e:
        print(f"  ✗  Kite API error: {e}")
        print("     (Access token may be stale — generate a fresh one via /auth/login-url)")

mode = os.getenv("TRADING_MODE", "PAPER")
print(f"\n  Trading mode : {mode}")
print(f"  Max daily loss: ₹{os.getenv('MAX_DAILY_LOSS', '5000')}")
print(f"  Max position : ₹{os.getenv('MAX_POSITION_SIZE', '50000')}")
print(f"  Backtest gate: win≥{os.getenv('BT_MIN_WIN_RATE','55')}% "
      f"sharpe≥{os.getenv('BT_MIN_SHARPE','1.0')} "
      f"dd≤{os.getenv('BT_MAX_DRAWDOWN_PCT','15')}%")

if ok:
    print("\n  ✓ All checks passed. Start the server:")
    print("    uvicorn main:app --reload --host 0.0.0.0 --port 8000\n")
else:
    print("\n  ✗ Fix the errors above before starting.\n")
    sys.exit(1)