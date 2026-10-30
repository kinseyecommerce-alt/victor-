import re
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Literal


class Settings(BaseSettings):
    # Zerodha
    kite_api_key: str = ""
    kite_api_secret: str = ""
    kite_access_token: str = ""
    # Kite auto-login (Playwright) — set these to enable morning auto-refresh
    kite_user_id: str = ""
    kite_password: str = ""
    kite_totp_secret: str = ""      # TOTP seed from Zerodha 2FA setup
    kite_redirect_url: str = ""     # e.g. https://yourdomain.com/auth/kite/callback

    # Anthropic
    anthropic_api_key: str = ""

    # TrueData
    truedata_username: str = ""
    truedata_password: str = ""
    use_truedata_websocket:   bool = False  # primary live tick source (replaces Kite WS)
    use_truedata_historical:  bool = False  # OHLCV history for backtesting / warm-up
    use_truedata_options:     bool = False  # options chain: IV rank, PCR, max pain

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # n8n webhook integration
    n8n_webhook_url:    str = ""  # e.g. https://your-n8n.com/webhook/algotrader
    n8n_webhook_secret: str = ""  # optional HMAC-SHA256 signing secret

    # Security — API access
    api_key: str = ""                   # X-API-Key header for mutating routes
    kill_switch_reset_secret: str = ""  # Separate secret required to reset kill switch

    # Security — App login (JWT)
    admin_username: str = "admin"
    admin_password: str = ""            # plain fallback for dev only
    admin_password_hash: str = ""       # bcrypt hash — takes precedence
    jwt_secret_key: str = ""
    jwt_expire_hours: int = 8

    # Trading
    trading_mode: Literal["PAPER", "LIVE"] = "PAPER"
    # PAPER-mode price feed: SIM = GBM simulator, KITE = real Kite quotes (orders stay simulated)
    paper_data_source: Literal["SIM", "KITE"] = "SIM"

    # Risk
    max_daily_loss: float = 5000.0
    max_position_size: float = 50000.0
    max_open_positions: int = 5
    stop_loss_pct: float = 1.5
    target_pct: float = 3.0
    squareoff_time: str = "15:10"

    # Backtest gate thresholds
    bt_min_win_rate: float = 55.0
    bt_min_sharpe: float = 1.0
    bt_max_drawdown_pct: float = 15.0
    bt_min_trades: int = 20
    bt_lookback_days: int = 180

    # Overtrade prevention (per strategy per day)
    max_trades_intraday: int = 8
    max_trades_fno: int = 4
    max_trades_swing: int = 3
    max_trades_scalping: int = 20
    cooldown_after_loss_sec: int = 300

    # Pre-learned system (set after running historical_learner.py)
    skip_startup_backtest: bool = False   # use pre-learned approved_symbols.json
    use_nifty100_watchlist: bool = False  # auto-use full Nifty 100 as watchlist

    # Intelligence layer
    use_claude_trade_gate: bool = True    # per-trade Claude assessment via Sonnet
    claude_gate_threshold: int = 65       # min confidence to enter (master raises/lowers dynamically)
    use_multi_timeframe: bool = True      # require 5m/15m alignment with entry direction
    mtf_min_alignment: int = 2            # how many of 3 TFs must agree (1, 2, or 3)
    use_kelly_sizing: bool = True         # apply Claude gate's size_factor to qty

    # Auto-start (set to enable fully-lights-out operation)
    # Comma-separated strategy names e.g. "intraday,scalping"
    auto_start_strategies: str = ""
    # Comma-separated symbols e.g. "RELIANCE,TCS" — empty = use symbol scanner
    auto_start_watchlist: str = ""

    # Autonomous AI trader (ai_trader.py) — every limit below is enforced in code,
    # not left to the model. It trades NSE equities intraday (MIS) only.
    ai_agent_model:               str   = "claude-opus-5-5"
    ai_agent_effort:              Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    ai_agent_interval_min:        int   = 15       # minutes between full portfolio reviews (0 = off)
    ai_agent_allow_live:          bool  = False    # must be True for the AI to trade when TRADING_MODE=LIVE
    ai_agent_max_trades_per_day:  int   = 5
    ai_agent_max_open_positions:  int   = 2
    ai_agent_max_position_value:  float = 50000.0  # ₹ per trade (qty × price)
    ai_agent_max_sl_pct:          float = 2.0      # stop-loss must sit within this % of price
    ai_agent_min_sl_pct:          float = 0.2      # …and at least this far away
    ai_agent_max_steps:           int   = 12       # model ↔ tool round trips per cycle / chat turn
    ai_agent_max_daily_cost_usd:  float = 20.0     # stop autonomous cycles once today's API spend reaches this
    # Real-time mode: code watches every tick and calls Claude the moment a trigger fires
    ai_agent_realtime:            bool  = True
    ai_agent_realtime_effort:     Literal["low", "medium", "high", "xhigh", "max"] = "low"  # fast decisions
    ai_agent_trigger_cooldown_s:  int   = 180      # same symbol + trigger type at most once per this
    ai_agent_min_call_gap_s:      int   = 3        # spacing between real-time Claude calls
    ai_agent_event_max_age_s:     int   = 20       # alerts older than this are dropped, not decided
    ai_agent_max_drift_pct:       float = 0.3      # reject an entry if price moved this far since the alert
    ai_agent_burst_pct:           float = 0.5      # 1-minute move that counts as a momentum burst
    # AI option BUYING on index options (CE/PE). Max loss per trade = premium paid.
    ai_agent_options_enabled:     bool  = True
    ai_agent_option_underlyings:  str   = "NIFTY,BANKNIFTY"   # of NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY
    ai_agent_max_option_premium:  float = 15000.0  # ₹ premium × quantity per trade
    ai_agent_max_option_lots:     int   = 2
    ai_agent_option_min_sl_pct:   float = 15.0     # stop as % of premium
    ai_agent_option_max_sl_pct:   float = 40.0
    ai_agent_option_max_spread_pct: float = 3.0    # bid/ask spread as % of premium
    ai_agent_option_min_premium:  float = 10.0     # ₹ — skip near-worthless far-OTM contracts
    ai_agent_option_max_iv_rank:  float = 72.0     # don't buy expensive volatility

    # Real-time tick feed
    use_kite_websocket: bool = True   # use KiteConnect WebSocket for ticks in LIVE mode

    # Daily capital allocation by trading type
    total_capital:          float = 500000.0   # total account capital (₹)
    intraday_capital_pct:   float = 40.0       # % for equity intraday MIS (intraday + scalping)
    swing_capital_pct:      float = 25.0       # % for equity delivery CNC (swing)
    options_capital_pct:    float = 25.0       # % for options premium NRML (fno)
    futures_capital_pct:    float = 10.0       # % for futures margin NRML (reserved)

    # Max concurrent positions per agent (capital divided per-symbol to avoid overrun)
    max_intraday_positions: int = 5
    max_scalping_positions: int = 5
    max_swing_positions:    int = 3

    # Server
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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def uses_kite_market_data(self) -> bool:
        """True when ticks come from Kite: LIVE mode, or PAPER mode with PAPER_DATA_SOURCE=KITE."""
        return self.trading_mode == "LIVE" or self.paper_data_source == "KITE"


settings = Settings()
