import axios, { AxiosInstance } from 'axios'
import { useStore } from '../store'

// Two ways to be authorised, either is enough:
//   • the admin login on /login, which stores a JWT in localStorage ('jwtToken') — the normal path
//   • an X-API-Key from Settings → Connection — for a machine or a second browser
export const getJwt = (): string => {
  try { return localStorage.getItem('jwtToken') || '' } catch { return '' }
}
export const getUsername = (): string => {
  try { return localStorage.getItem('username') || '' } catch { return '' }
}
export const signOut = () => {
  try { localStorage.removeItem('jwtToken'); localStorage.removeItem('username') } catch { /* private mode */ }
  window.location.href = '/login'
}

const authHeaders = (key: string): Record<string, string> => {
  const h: Record<string, string> = {}
  if (key) h['X-API-Key'] = key
  const jwt = getJwt()
  if (jwt) h['Authorization'] = `Bearer ${jwt}`
  return h
}

let redirecting = false
const withAuthHandling = (inst: AxiosInstance): AxiosInstance => {
  inst.interceptors.response.use(undefined, (err) => {
    // 401 with nothing valid to offer → the session expired or there never was one: go and log in.
    // (The Kite status call returns 401-free JSON, so only real auth failures land here.)
    if (err?.response?.status === 401 && !redirecting && !useStore.getState().apiKey) {
      redirecting = true
      try { localStorage.removeItem('jwtToken') } catch { /* ignore */ }
      window.location.href = '/login?next=/dashboard'
    }
    return Promise.reject(err)
  })
  return inst
}

const ax = () => {
  const { apiBase, apiKey } = useStore.getState()
  return withAuthHandling(axios.create({ baseURL: apiBase, headers: authHeaders(apiKey), timeout: 10000 }))
}

// Agent calls run a multi-step Claude loop — allow minutes, not seconds
const axLong = () => {
  const { apiBase, apiKey } = useStore.getState()
  return withAuthHandling(axios.create({ baseURL: apiBase, headers: authHeaders(apiKey), timeout: 300000 }))
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

  // Overview dashboard (one aggregate call per refresh)
  dashboardSummary: () => ax().get('/dashboard/summary'),
  opportunities: (top = 8) => ax().get(`/market/opportunities?top=${top}`),
  scoreboard: () => ax().get('/performance/scoreboard'),
  strategyStats: () => ax().get('/performance/strategies'),
  strategyEnable: (name: string) => ax().post(`/performance/strategies/${name}/enable`),
  strategyDisable: (name: string) => ax().post(`/performance/strategies/${name}/disable`),

  // AI trader (Claude)
  aiStatus: () => ax().get('/agent/status'),
  aiStart: () => ax().post('/agent/start', {}),
  aiStop: () => ax().post('/agent/stop'),
  aiCycle: () => axLong().post('/agent/cycle'),
  aiCloseAll: () => ax().post('/agent/close-all'),
  aiLog: (n = 100) => ax().get(`/agent/log?n=${n}`),
  aiChat: (message: string, conversation_id?: string | null) =>
    axLong().post('/agent/chat', { message, conversation_id }),

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
    kite_user_id?: string
    kite_password?: string
    kite_totp_secret?: string
    anthropic_api_key?: string
    openai_api_key?: string
    truedata_username?: string
    truedata_password?: string
    persist?: boolean
  }) => ax().post('/settings/credentials', data),
  credentialsStatus: () => ax().get('/settings/credentials/status'),
  connections: () => ax().get('/status/connections'),
  deleteCredentials: (fields: string[]) => ax().delete('/settings/credentials', { data: { fields } }),
  kiteLoginUrl: () => ax().get('/auth/login-url'),
  kiteStatus: () => ax().get('/auth/kite/status'),
  kiteAutoLoginNow: () => ax().post('/auth/kite/auto-login'),
  testAiProviders: () => ax().post('/ai/providers/test'),
  allocator: () => ax().get('/allocator'),
  allocatorRun: () => ax().post('/allocator/run?reason=manual'),

  // App login
  updateAppPassword: (data: { username?: string; new_password: string }) =>
    ax().post('/settings/app-password', data),
}
