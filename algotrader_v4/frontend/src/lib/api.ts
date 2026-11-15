// Thin fetch wrapper for the dashboard API (/api/v2 — see API_V2.md).
// Auth, either is enough (exactly what the backend's gate accepts):
//   • JWT from POST /auth/login, kept in localStorage 'jwtToken' (same key the old /login page uses)
//   • an API key, kept in localStorage 'apiKey', sent as X-API-Key
// Without API_KEY on the server (local dev) nothing is checked.

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

const ls = {
  get(k: string): string {
    try { return localStorage.getItem(k) || '' } catch { return '' }
  },
  set(k: string, v: string) {
    try { if (v) localStorage.setItem(k, v); else localStorage.removeItem(k) } catch { /* private mode */ }
  },
}

export const auth = {
  jwt: () => ls.get('jwtToken'),
  apiKey: () => ls.get('apiKey') || ls.get('api_key'),
  username: () => ls.get('username'),
  setJwt(token: string, user: string) { ls.set('jwtToken', token); ls.set('username', user) },
  setApiKey(key: string) { ls.set('apiKey', key); ls.set('api_key', '') },
  signOut() {
    ls.set('jwtToken', ''); ls.set('username', ''); ls.set('apiKey', ''); ls.set('api_key', '')
    window.dispatchEvent(new Event('auth-required'))
  },
  hasCredentials: () => Boolean(ls.get('jwtToken') || ls.get('apiKey') || ls.get('api_key')),
}

function headers(json: boolean): Record<string, string> {
  const h: Record<string, string> = { Accept: 'application/json' }
  if (json) h['Content-Type'] = 'application/json'
  const key = auth.apiKey()
  if (key) h['X-API-Key'] = key
  const jwt = auth.jwt()
  if (jwt) h['Authorization'] = `Bearer ${jwt}`
  return h
}

async function handle(res: Response): Promise<any> {
  let body: any = null
  const text = await res.text()
  if (text) {
    try { body = JSON.parse(text) } catch { body = { detail: text.slice(0, 200) } }
  }
  if (!res.ok) {
    if (res.status === 401) window.dispatchEvent(new Event('auth-required'))
    let msg = body?.detail
    if (Array.isArray(msg)) msg = msg.map((d: any) => d?.msg || JSON.stringify(d)).join('; ')
    throw new ApiError(res.status, typeof msg === 'string' && msg ? msg : `HTTP ${res.status}`)
  }
  return body
}

async function request(method: string, path: string, body?: unknown, timeoutMs = 15000): Promise<any> {
  const ctl = new AbortController()
  const t = setTimeout(() => ctl.abort(), timeoutMs)
  try {
    const res = await fetch(path, {
      method,
      headers: headers(body !== undefined),
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: ctl.signal,
      cache: 'no-store',
    })
    return await handle(res)
  } catch (e: any) {
    if (e instanceof ApiError) throw e
    if (e?.name === 'AbortError') throw new ApiError(0, 'The server did not answer in time')
    throw new ApiError(0, 'Cannot reach the server (offline?)')
  } finally {
    clearTimeout(t)
  }
}

export const api = {
  get: (path: string) => request('GET', path),
  post: (path: string, body: unknown = {}) => request('POST', path, body, 60000),
  async login(username: string, password: string): Promise<void> {
    const res = await fetch('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`,
    })
    const text = await res.text()
    let data: any = {}
    try { data = JSON.parse(text) } catch { /* ignore */ }
    if (!res.ok || !data.access_token) throw new ApiError(res.status, data.detail || 'Sign-in failed')
    auth.setJwt(data.access_token, username)
  },
}

export const V2 = '/api/v2'
