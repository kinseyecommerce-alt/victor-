export interface TickData {
  symbol: string
  ltp: number
  bid: number
  ask: number
  change_pct: number
  volume: number
  day_high: number
  day_low: number
  trend: 'UP' | 'DOWN' | 'NEUTRAL'
  momentum: string
  volatility: string
  rsi: number
  vwap: number
  ema9: number
  ema21: number
  macd_hist: number
  vol_ratio: number
  source: 'KITE' | 'NSE' | 'PAPER'
  ts: string
}

export interface Position {
  tradingsymbol: string
  exchange: string
  product: string
  quantity: number
  average_price: number
  last_price: number
  pnl: number
  buy_quantity?: number
  sell_quantity?: number
}

export interface Order {
  order_id: string
  tradingsymbol: string
  exchange: string
  transaction_type: 'BUY' | 'SELL'
  quantity: number
  order_type: string
  product: string
  price: number
  status: string
  placed_at?: string
  tag?: string
}

export interface Bracket {
  bracket_id: string
  strategy: string
  symbol: string
  side: string
  quantity: number
  entry_price: number
  stop_loss?: number
  target_1?: number
  target_2?: number
  status: 'PENDING' | 'ACTIVE' | 'SL_HIT' | 'TARGET_HIT' | 'CANCELLED' | 'FAILED'
  pnl?: number
}

export interface RiskStatus {
  daily_pnl: number
  max_daily_loss: number
  open_positions: number
  max_open_positions: number
  is_halted: boolean
  trades_today: number
}

export interface Agent {
  name: string
  running: boolean
  last_signal?: string
  trades_today?: number
  win_rate?: number
}

export interface BotStatus {
  master_running: boolean
  strategies: string[]
  watchlist: string[]
  performance?: {
    total_trades: number
    daily_pnl: number
  }
}

export interface HealthData {
  status: string
  version: string
  mode: 'PAPER' | 'LIVE'
  market_open: boolean
  master: string
  tick_engine: string
  ticker_source: 'KITE' | 'NSE' | 'PAPER'
  time: string
}

export type TabId = 'positions' | 'orders' | 'brackets' | 'risk' | 'agents' | 'sebi'
