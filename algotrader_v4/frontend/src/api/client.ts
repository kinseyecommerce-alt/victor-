import axios from 'axios'
import { useStore } from '../store'

const getConfig = () => {
  const { apiBase, apiKey } = useStore.getState()
  return { base: apiBase, key: apiKey }
}

const ax = () => {
  const { base, key } = getConfig()
  return axios.create({
    baseURL: base,
    headers: key ? { 'X-API-Key': key } : {},
    timeout: 10000,
  })
}

export const api = {
  health: () => ax().get('/health'),
  configValidate: () => ax().get('/config/validate'),

  // Bot
  botStart: (strategies: string[], watchlist?: { symbol: string; exchange: string }[]) =>
    ax().post('/bot/start', { strategies, watchlist }),
  botStop: () => ax().post('/bot/stop'),
  botStatus: () => ax().get('/bot/status'),

  // Market
  marketLive: () => ax().get('/market/live'),
  marketStatus: () => ax().get('/market/status'),

  // Orders
  placeOrder: (data: {
    symbol: string; exchange: string; transaction_type: string;
    quantity: number; order_type: string; product: string;
    price?: number; trigger_price?: number
  }) => ax().post('/orders/place', data),
  cancelOrder: (orderId: string) => ax().delete(`/orders/${orderId}`),
  squareoffAll: () => ax().post('/orders/squareoff'),

  // Portfolio
  positions: () => ax().get('/portfolio/positions'),
  orders: () => ax().get('/portfolio/orders'),

  // Risk
  riskStatus: () => ax().get('/risk/status'),

  // Agents
  agents: () => ax().get('/agents'),
  pauseAgent: (name: string) => ax().post(`/agents/${name}/pause`),
  resumeAgent: (name: string) => ax().post(`/agents/${name}/resume`),

  // SEBI
  sebiStatus: () => ax().get('/sebi/status'),
  sebiKillSwitch: (reason?: string) =>
    ax().post('/sebi/kill-switch', null, { params: { reason: reason || 'Manual' } }),
  sebiResume: () => ax().post('/sebi/resume'),
  sebiResetKillSwitch: (secret: string) =>
    ax().post('/sebi/reset-kill-switch', { secret }),
  sebiAuditLog: (date: string) => ax().get(`/sebi/audit-log?date=${date}`),

  // Brackets
  brackets: () => ax().get('/brackets'),
  manualBracket: (data: {
    strategy: string; symbol: string; exchange: string; side: string;
    quantity: number; signal_price: number; product?: string;
    stop_loss?: number; target_1?: number; target_2?: number
  }) => ax().post('/brackets/manual', data),

  // Regime
  regimeStatus: () => ax().get('/regime/status'),

  // Credentials
  updateCredentials: (data: {
    kite_api_key?: string
    kite_api_secret?: string
    anthropic_api_key?: string
    truedata_username?: string
    truedata_password?: string
  }) => ax().post('/settings/credentials', data),

  // App login
  updateAppPassword: (data: { username?: string; new_password: string }) =>
    ax().post('/settings/app-password', data),
}
