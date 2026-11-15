import { FormEvent, useState } from 'react'
import { KeyRound, LogIn } from 'lucide-react'
import { api, auth } from '../lib/api'
import { Button, Notice, Segmented } from './ui'

// Shown when the server answers 401. Two ways in, both supported by the backend:
// the dashboard password (POST /auth/login → JWT) or the server's API key (X-API-Key).
export function Login({ onDone }: { onDone: () => void }) {
  const [mode, setMode] = useState<'password' | 'key'>('password')
  const [user, setUser] = useState(auth.username() || 'admin')
  const [pass, setPass] = useState('')
  const [key, setKey] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setErr('')
    setBusy(true)
    try {
      if (mode === 'password') {
        await api.login(user.trim(), pass)
      } else {
        auth.setApiKey(key.trim())
        await api.get('/api/v2/overview')        // throws 401 when the key is wrong
      }
      onDone()
    } catch (e: any) {
      if (mode === 'key') auth.setApiKey('')
      setErr(e?.status === 401 ? (mode === 'key' ? 'That API key was not accepted.' : 'Wrong username or password.') : e?.message || 'Sign-in failed')
    } finally {
      setBusy(false)
    }
  }

  const input = 'mt-1 w-full rounded-xl border border-slate-300 bg-white px-3 py-2.5 text-base dark:border-slate-700 dark:bg-slate-800'
  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <form onSubmit={submit} className="w-full max-w-sm space-y-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm dark:border-slate-800 dark:bg-slate-900" data-testid="login-form">
        <div>
          <div className="text-xl font-bold">AlgoTrader Pro</div>
          <div className="text-sm text-slate-500 dark:text-slate-400">Paper-trading dashboard · sign in</div>
        </div>
        <Segmented options={[{ value: 'password', label: 'Password' }, { value: 'key', label: 'API key' }]} value={mode} onChange={setMode} size="sm" />
        {mode === 'password' ? (
          <>
            <label className="block text-sm">Username
              <input className={input} value={user} onChange={(e) => setUser(e.target.value)} autoComplete="username" name="username" />
            </label>
            <label className="block text-sm">Password
              <input className={input} type="password" value={pass} onChange={(e) => setPass(e.target.value)} autoComplete="current-password" name="password" />
            </label>
          </>
        ) : (
          <label className="block text-sm">API key (the server’s API_KEY)
            <input className={input} type="password" value={key} onChange={(e) => setKey(e.target.value)} autoComplete="off" name="apikey" />
            <span className="mt-1 block text-xs text-slate-500">Kept only in this browser.</span>
          </label>
        )}
        {err && <Notice tone="red">{err}</Notice>}
        <Button type="submit" tone="primary" className="w-full py-2.5" disabled={busy || (mode === 'password' ? !user || !pass : !key)} testid="login-submit">
          {mode === 'password' ? <LogIn size={16} /> : <KeyRound size={16} />} {busy ? 'Signing in…' : 'Sign in'}
        </Button>
      </form>
    </div>
  )
}
