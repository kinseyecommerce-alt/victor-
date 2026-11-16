"""
config.py — runtime settings (pydantic-settings) read from the environment / .env at startup.

Runtime changes are applied by mutating the `settings` singleton (they last until the process restarts and are
never written back to .env). Unknown variables (fields of the removed legacy engine still set on an old deployment)
are ignored, never an error.
"""
import re
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False,
                                      extra="ignore")

    # ── Zerodha Kite ──
    kite_api_key: str = ""
    kite_api_secret: str = ""
    kite_access_token: str = ""
    kite_user_id: str = ""          # automatic daily login (user id + password + TOTP seed, vault)
    kite_password: str = ""
    kite_totp_secret: str = ""
    kite_redirect_url: str = ""     # e.g. https://yourdomain.com/auth/kite/callback
    use_kite_websocket: bool = True     # Kite WebSocket for the price feed (REST quote batch as fallback)

    # ── Telegram ──
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # ── Security ──
    api_key: str = ""                   # X-API-Key header for mutating routes and the v2 API
    kill_switch_reset_secret: str = ""  # separate secret required to reset the SEBI kill switch
    admin_username: str = "admin"
    admin_password: str = ""            # plain fallback for dev only
    admin_password_hash: str = ""       # bcrypt hash — takes precedence
    jwt_secret_key: str = ""
    jwt_expire_hours: int = 8

    # ── Mode / data ──
    trading_mode: Literal["PAPER", "LIVE"] = "PAPER"
    # PAPER price feed: KITE = real Kite quotes (orders stay simulated). SIM = no market data: the validated agents
    # refuse every entry (no_kite_data); used for local runs without credentials.
    paper_data_source: Literal["SIM", "KITE"] = "SIM"
    # PAPER book persistence (paper_book.py): simulated orders / positions and the validated agents' books are
    # written to PAPER_BOOK_FILE and restored after a restart. Never in LIVE.
    persist_paper_book: bool = True
    paper_book_save_interval_s: float = 3.0
    squareoff_time: str = "15:15"       # MIS flatten time (the agents were validated with 15:15)
    market_extra_sessions: str = ""     # full weekend sessions "YYYY-MM-DD:Name,..." (adds to ist_clock.EXTRA_TRADING_DAYS)
    market_holidays: str = ""           # extra exchange holidays "YYYY-MM-DD:Name,..."
    auto_start_watchlist: str = ""      # price-feed warm watchlist "RELIANCE,TCS" (empty: the first 20 NIFTY 50 names)

    # ── Order limits (SEBI pre-order check) ──
    max_position_size: float = 150000.0         # per order ₹ (the SEBI check allows max(this, ₹10 L))
    max_futures_order_value: float = 2_500_000.0
    max_daily_loss: float = 5000.0              # disclosure document only
    max_open_positions: int = 8                 # disclosure document only
    total_capital: float = 500000.0             # paper account capital: the portfolio drawdown brake's base

    # ── Validated strategies (pre-registered lab protocol) ──
    # VALIDATED_ONLY: kept for compatibility; true = the validated agents may take entries (false blocks entries,
    # open positions still exit by rule)
    validated_only: bool = True
    validated_notional_scale: float = 1.0             # ₹1 L per stock trade × this (auction, attention, PEAD)
    validated_special_sessions: str = ""              # "YYYY-MM-DD,..." special/short sessions (Muhurat, DR drills)
    # AUC-FADE (agents/validated_auction.py)
    validated_auction_enabled: bool = True
    validated_auction_variant: Literal["3PCT", "2PCT"] = "3PCT"
    validated_auction_max_gross: float = 1_000_000.0
    validated_auction_daily_loss_stop: float = 16_000.0
    validated_auction_dd_kill: float = 37_000.0
    # SWING-ATTN-ON-3X (agents/validated_attention.py)
    validated_attention_enabled: bool = True
    validated_attention_max_gross: float = 1_000_000.0
    validated_attention_daily_loss_stop: float = 10_000.0
    validated_attention_dd_kill: float = 20_000.0
    # VRP-STRANGLE-016-WINGED (agents/validated_strangle.py)
    validated_strangle_enabled: bool = True
    validated_vrp_lots: int = 2
    validated_vrp_max_margin: float = 250_000.0
    validated_vrp_max_loss: float = 75_000.0
    validated_vrp_dd_kill: float = 195_000.0
    # EV-PEAD-LONG-H60 candidate (agents/validated_pead.py): PAPER only, OFF by default (no job is scheduled)
    validated_pead_enabled: bool = False
    validated_pead_max_positions: int = 100
    validated_pead_max_gross: float = 10_000_000.0
    validated_pead_daily_loss_stop: float = 240_000.0
    validated_pead_dd_kill: float = 1_680_000.0
    # Shadow-incubation registry (shadow_registry.py): observation only, off by default
    shadow_registry_enabled: bool = False
    # Forward lab (forward_lab.py): every lab strategy on each new session after the close, PAPER research only,
    # off by default (no job scheduled, no Kite call). Engine runs in its own process with these limits.
    forward_lab_enabled: bool = False
    forward_lab_max_mb: float = 300.0          # disk cap of the forward-lab directory (stop writing, never delete)
    forward_lab_jobs: int = 1                  # engine worker processes
    forward_lab_time_limit_s: int = 2700       # engine wall-clock limit per session
    forward_lab_mem_mb: int = 3072             # engine address-space limit per process
    forward_lab_telegram: bool = False         # at most one summary per day
    # Strategy-builder data layer (builder_data.py, builder_minute.py): deep 1-minute history from Kite (2016 ->) and
    # NSE end-of-day archives (2012 ->) on the volume, PAPER research only, read-only towards trading. OFF by default
    # (no job scheduled, no network call). Hard disk cap for the whole builder directory (stop writing, never delete).
    builder_data_enabled: bool = False
    builder_data_max_mb: float = 2600.0            # 1,800 before the NIFTY 200 minute universe (worst-case volume 4,524 MB)
    # minute store universe: nifty100 = the forward lab's 97 + 2 indices (as before); nifty200 = those, unchanged, plus
    # the added NIFTY 200 names (current list + historical NIFTY 100 members still on Kite; builder_universe.py)
    builder_minute_universe: str = "nifty200"
    builder_kite_max_calls_per_run: int = 1500     # Kite historical_data calls per minute-backfill run
    builder_nse_max_requests_per_run: int = 2500   # NSE requests per archive run (one at a time, >= 1.5 s apart)
    # Automatic strategy builder (builder.py, builder_rule.json): one frozen batch of 400 candidates a week from our own
    # data (Saturday 06:00 IST), evaluated in its own process; PAPER research only, OFF by default (no job scheduled).
    # Measured on the lab machine (4 vCPU): one real batch 82 s with 2 workers, ~1.6 GB peak RSS per process.
    builder_enabled: bool = False
    builder_max_mb: float = 400.0              # disk cap of the builder directory (stop writing, never delete)
    builder_jobs: int = 2                      # evaluation worker processes (capped at 2)
    builder_time_limit_s: int = 7200           # wall-clock limit of one batch evaluation (resumable)
    builder_mem_mb: int = 4096                 # address-space limit per evaluation process

    # ── Engine: risk budget, trading states, approvals, drift, execution (see REBUILD.md) ──
    # Risk budget (risk_budget.py). RISK_BUDGET_DAILY_VOL = the book's daily P&L standard deviation budget in ₹, split
    # equally (equal risk) over the enabled kept strategies and divided by each one's lab daily P&L std; 0 (default)
    # = every strategy at its lab sizing (scale 1.0, paper stays comparable with the lab).
    risk_budget_daily_vol: float = 0.0
    risk_budget_allow_upscale: bool = False     # the allocator may only scale DOWN unless this is true
    risk_budget_max_scale: float = 1.0          # ceiling of the per-strategy scale when upscaling is allowed
    # Portfolio drawdown brake: realised net P&L of the kept strategies (since the last brake reset) falls from its
    # running peak by reduce_pct % of RISK_CAPITAL → every strategy REDUCE_ONLY; by halt_pct % → HALTED. Latches until
    # reset through the v2 API. Defaults: ₹25 L ≈ capital in use of the three active strategies at lab size (attention
    # ₹10 L CNC gross, auction fade ₹10 L MIS gross, strangle ≈ ₹1.3 L margin, plus buffer); 16 % = ₹4 L and
    # 24 % = ₹6 L are above the lab's combined block-bootstrap p99 drawdown (₹3.25 L, lab_risk.json), so a lab-like
    # path never trips it: it is a "something is broken" brake.
    risk_capital: float = 2_500_000.0
    risk_brake_reduce_pct: float = 16.0
    risk_brake_halt_pct: float = 24.0
    # Semi-auto approval mode: each strategy's entries of the day need an approval via the API (default OFF)
    approval_mode: bool = False
    # Drift / fill-quality monitor: warnings only; at most one Telegram message per strategy per day when on
    drift_telegram: bool = False
    # LIVE order path: marketable LIMIT inside this band around the touch instead of a raw MARKET order
    order_protection_band_pct: float = 0.5
    # LIVE prerequisite (Kite, from 1 Apr 2026): the static outbound IP registered in the Kite developer console
    kite_static_ip: str = ""

    # ── Server ──
    host: str = "0.0.0.0"
    port: int = 8000
    allowed_origins: str = "http://localhost:3000,http://localhost:5173"

    @field_validator("squareoff_time")
    @classmethod
    def validate_squareoff_time(cls, v: str) -> str:
        if not re.match(r"^\d{2}:\d{2}$", v):
            raise ValueError("squareoff_time must be HH:MM format")
        h, m = int(v[:2]), int(v[3:])
        if not (9 <= h <= 15 and 0 <= m <= 59):
            raise ValueError("squareoff_time must be between 09:00 and 15:59")
        return v

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def uses_kite_market_data(self) -> bool:
        """True when prices come from Kite: LIVE mode, or PAPER mode with PAPER_DATA_SOURCE=KITE."""
        return self.trading_mode == "LIVE" or self.paper_data_source == "KITE"


settings = Settings()

# Keys saved from the dashboard live encrypted in the vault (see secret_store.py) and override .env.
from secret_store import apply_to as _apply_stored_secrets  # noqa: E402
_apply_stored_secrets(settings)
