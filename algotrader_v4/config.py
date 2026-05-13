from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    # Zerodha
    kite_api_key: str = ""
    kite_api_secret: str = ""
    kite_access_token: str = ""

    # Anthropic
    anthropic_api_key: str = ""

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Trading
    trading_mode: Literal["PAPER", "LIVE"] = "PAPER"

    # Risk
    max_daily_loss: float = 5000.0
    max_position_size: float = 50000.0
    max_open_positions: int = 5
    stop_loss_pct: float = 1.5
    target_pct: float = 3.0
    squareoff_time: str = "15:10"

    # Backtest gate thresholds
    bt_min_win_rate: float = 55.0       # % — reject symbol if below
    bt_min_sharpe: float = 1.0
    bt_max_drawdown_pct: float = 15.0   # % — reject if drawdown exceeds
    bt_min_trades: int = 20             # reject if sample too small
    bt_lookback_days: int = 180         # historical window

    # Overtrade prevention (per strategy per day)
    max_trades_intraday: int = 8
    max_trades_fno: int = 4
    max_trades_swing: int = 3
    max_trades_scalping: int = 20
    cooldown_after_loss_sec: int = 300  # 5 min pause after a losing trade

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    allowed_origins: str = "http://localhost:3000,http://localhost:5173"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]


settings = Settings()