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
    # Primary exchange — MCX commodity futures (replaces NSE/BSE equity)
    exchange: str = "MCX"

    # Risk
    max_daily_loss: float = 5000.0
    max_position_size: float = 50000.0           # equity notional cap (legacy)
    mcx_max_margin_per_position: float = 300000.0 # MCX: max margin blocked per position
    max_open_positions: int = 5
    stop_loss_pct: float = 1.5
    target_pct: float = 3.0
    squareoff_time: str = "23:15"   # MCX intraday square-off (session closes 23:30)

    # ── MCX session (IST) ─────────────────────────────────────────────
    mcx_open_time:       str = "09:00"   # MCX opens 09:00 IST
    mcx_close_time:      str = "23:30"   # normal session close (bullion/energy/metals)
    mcx_agri_close_time: str = "21:00"   # agri commodities close earlier

    # ── Inter-agent communication ─────────────────────────────────────
    use_agent_bus:         bool  = True   # agents publish/read signals on the shared bus
    use_agent_coordinator: bool  = True   # coordinator arbitrates entries between agents
    coord_max_concurrent_positions: int   = 6        # global cap on coordinated positions
    coord_group_margin_cap:         float = 250000.0 # ₹ margin cap per correlated group

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
    # Per-trade Claude gate is OFF by default — it added network latency + cost to
    # the order hot path. Risk posture is instead applied off-path via the master
    # agent's periodic regime review (published to the bus, consumed by the coordinator).
    use_claude_trade_gate: bool = False   # per-trade Claude assessment via Sonnet (opt-in)
    claude_gate_threshold: int = 65       # min confidence to enter (master raises/lowers dynamically)

    # ── Claude Brain — central intelligence powering the MCX agents ───────────
    # A single async Claude (Anthropic API) "brain" the master agent consults on
    # its periodic regime review (OFF the order hot path). It returns a risk
    # posture (regime + size_factor + per-agent directives) that is published to
    # the bus (TOPIC_REGIME) and applied by the coordinator to every entry.
    # Degrades gracefully to a rule-based posture when no API key is configured,
    # so the app runs fully offline.
    use_claude_brain:    bool  = True            # consult Claude on the regime review
    claude_brain_model:  str   = "claude-opus-4-8"
    claude_brain_max_tokens: int = 1024
    claude_brain_timeout_sec: float = 12.0       # off hot path — generous budget
    claude_brain_min_interval_sec: int = 55      # throttle: don't call more than ~1/min

    # ── Execution & cost controls ─────────────────────────────────────
    # Entry order type: MARKET | LIMIT | MARKETABLE_LIMIT (cap slippage at N ticks)
    entry_order_type:        str = "MARKETABLE_LIMIT"
    entry_limit_cross_ticks: int = 2      # ticks to cross the book on a marketable limit
    use_cost_gate:           bool = True  # skip trades whose target can't beat round-trip cost

    # Transaction-cost model (MCX; verify against your contract note)
    brokerage_flat:    float = 20.0        # ₹ per order
    brokerage_pct:     float = 0.0003      # 0.03% of turnover (min vs flat)
    exchange_txn_pct:  float = 0.000026    # MCX transaction charge
    gst_pct:           float = 0.18        # on brokerage + exchange txn
    sebi_per_crore:    float = 10.0        # ₹ per ₹1 crore turnover
    stamp_pct_buy:     float = 0.00002     # buy-side stamp duty
    slippage_ticks:    float = 1.0         # assumed slippage per side (ticks)
    use_multi_timeframe: bool = True      # require 5m/15m alignment with entry direction
    mtf_min_alignment: int = 2            # how many of 3 TFs must agree (1, 2, or 3)
    use_kelly_sizing: bool = True         # apply Claude gate's size_factor to qty

    # Auto-start (set to enable fully-lights-out operation)
    # Comma-separated strategy names e.g. "intraday,scalping"
    auto_start_strategies: str = ""
    # Comma-separated symbols e.g. "RELIANCE,TCS" — empty = use symbol scanner
    auto_start_watchlist: str = ""

    # Real-time tick feed — market data always comes from the broker (Kite) in
    # both paper and live modes. PAPER/LIVE only governs order execution.
    use_kite_websocket: bool = True    # use KiteConnect WebSocket for broker ticks
    use_paper_simulator: bool = False  # offline dev only: GBM sim instead of broker feed

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

    @field_validator("squareoff_time", "mcx_open_time", "mcx_close_time", "mcx_agri_close_time")
    @classmethod
    def validate_hhmm(cls, v: str) -> str:
        if not re.match(r"^\d{2}:\d{2}$", v):
            raise ValueError("time must be HH:MM format")
        h, m = int(v[:2]), int(v[3:])
        # MCX trades 09:00–23:30 IST, so allow the full commodity session window
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError("time must be a valid 24h HH:MM")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]


settings = Settings()
