import React, { useEffect, useState, useCallback } from 'react'
import { Activity, Wifi, WifiOff, Settings, Zap, ZapOff, Eye, EyeOff, Lock, ExternalLink } from 'lucide-react'
import { clsx } from 'clsx'
import { useStore } from '../../store'
import { api } from '../../api/client'
import { Badge, Btn, Modal, Input } from '../ui'

type SettingsTab = 'connection' | 'apikeys' | 'applogin'

export default function Header() {
  const { health, botStatus, wsConnected, apiKey, apiBase, setApiKey, setApiBase, setHealth, setBotStatus, addToast } = useStore()
  const [time, setTime] = useState(new Date())
  const [configOpen, setConfigOpen] = useState(false)
  const [tempKey, setTempKey] = useState(apiKey)
  const [tempBase, setTempBase] = useState(apiBase)
  const [botLoading, setBotLoading] = useState(false)

  // Settings modal
  const [settingsTab, setSettingsTab] = useState<SettingsTab>('connection')

  // API Keys tab
  const [credForm, setCredForm] = useState({
    kite_api_key: '', kite_api_secret: '',
    anthropic_api_key: '',
    truedata_username: '', truedata_password: '',
  })
  const [showFields, setShowFields] = useState<Record<string, boolean>>({})
  const [credStatus, setCredStatus] = useState<Record<string, boolean>>({})
  const [credSaving, setCredSaving] = useState(false)
  // Encrypted vault: which fields are stored (never the values) + "remember" toggle
  const [vault, setVault] = useState<Record<string, { stored: boolean; hint?: string; expired?: boolean }>>({})
  const [vaultError, setVaultError] = useState<string | null>(null)
  const [remember, setRemember] = useState(true)
  const [kite, setKite] = useState<{ connected: boolean; name?: string; account_id?: string } | null>(null)

  // App Login tab
  const [appLoginForm, setAppLoginForm] = useState({ username: '', new_password: '', confirm_password: '' })
  const [showAppFields, setShowAppFields] = useState<Record<string, boolean>>({})
  const [appSaving, setAppSaving] = useState(false)

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    const poll = () => {
      api.health().then(r => setHealth(r.data)).catch(() => {})
      api.botStatus().then(r => setBotStatus(r.data)).catch(() => {})
    }
    poll()
    const t = setInterval(poll, 5000)
    return () => clearInterval(t)
  }, [])

  const applyCredStatus = (c: Record<string, any>, v?: any) => {
    setCredStatus({
      kite_api_key:      c.kite_api_key      ?? false,
      kite_api_secret:   c.kite_api_secret   ?? false,
      anthropic_api_key: c.anthropic_api_key ?? false,
      truedata_username: c.truedata_username ?? false,
      truedata_password: c.truedata_password ?? false,
    })
    if (v) { setVault(v.fields ?? {}); setVaultError(v.error ?? null) }
  }

  const refreshCredentials = () => {
    api.credentialsStatus().then(r => applyCredStatus(r.data.credentials ?? {}, r.data.vault)).catch(() => {})
    api.kiteStatus().then(r => setKite(r.data)).catch(() => setKite(null))
  }

  useEffect(() => { if (settingsTab === 'apikeys') refreshCredentials() }, [settingsTab])

  const handleKiteLogin = async () => {
    try {
      const r = await api.kiteLoginUrl()
      window.open(r.data.login_url, '_blank', 'noopener')
      addToast('Log in on the Zerodha page, then come back and reopen Settings', 'info')
    } catch (e: any) {
      addToast(e.response?.data?.detail || 'Save the Kite API key first', 'error')
    }
  }

  const handleRemoveCredential = async (field: string) => {
    try {
      const r = await api.deleteCredentials([field])
      applyCredStatus(r.data.credentials ?? {}, r.data.vault)
      addToast('Removed from the vault and the running server', 'info')
    } catch (e: any) {
      addToast(e.response?.data?.detail || 'Remove failed', 'error')
    }
  }

  // Fetch /config/validate when App Login tab opens
  useEffect(() => {
    if (settingsTab === 'applogin') {
      api.configValidate().then(r => {
        setCredStatus({
          kite_api_key:      r.data.kite_api_key      ?? false,
          kite_api_secret:   r.data.kite_api_secret   ?? false,
          anthropic_api_key: r.data.anthropic_api_key ?? false,
          truedata_username: r.data.truedata_username ?? false,
          truedata_password: r.data.truedata_password ?? false,
        })
        setAppLoginForm(p => ({ ...p, username: r.data.admin_username ?? '' }))
      }).catch(() => {})
    }
  }, [settingsTab])

  const handleBotToggle = useCallback(async () => {
    setBotLoading(true)
    try {
      if (botStatus?.master_running) {
        await api.botStop()
        addToast('Bot stopped', 'info')
        setBotStatus(null)
      } else {
        const r = await api.botStart(['intraday', 'scalping'])
        addToast(`Bot started — ${r.data.watchlist?.length || 0} symbols`, 'buy')
        setBotStatus(r.data)
      }
    } catch (e: any) {
      addToast(e.response?.data?.detail || 'Bot toggle failed', 'error')
    } finally {
      setBotLoading(false)
    }
  }, [botStatus])

  const handleSaveCredentials = async () => {
    const payload: Record<string, any> = {}
    if (credForm.kite_api_key.trim())      payload.kite_api_key      = credForm.kite_api_key.trim()
    if (credForm.kite_api_secret.trim())   payload.kite_api_secret   = credForm.kite_api_secret.trim()
    if (credForm.anthropic_api_key.trim()) payload.anthropic_api_key = credForm.anthropic_api_key.trim()
    if (credForm.truedata_username.trim()) payload.truedata_username = credForm.truedata_username.trim()
    if (credForm.truedata_password.trim()) payload.truedata_password = credForm.truedata_password.trim()
    if (!Object.keys(payload).length) { addToast('No credentials entered', 'info'); return }
    setCredSaving(true)
    try {
      const r = await api.updateCredentials({ ...payload, persist: remember })
      applyCredStatus(r.data.credentials ?? {}, r.data.vault)
      setCredForm({ kite_api_key: '', kite_api_secret: '', anthropic_api_key: '', truedata_username: '', truedata_password: '' })
      setShowFields({})
      addToast(remember ? 'Saved — encrypted on the server' : 'Updated until the server restarts', 'buy')
    } catch (e: any) {
      addToast(e.response?.data?.detail || 'Failed to update credentials', 'error')
    } finally { setCredSaving(false) }
  }

  const handleSaveAppPassword = async () => {
    if (appLoginForm.new_password.length < 8) {
      addToast('Password must be at least 8 characters', 'error'); return
    }
    if (appLoginForm.new_password !== appLoginForm.confirm_password) {
      addToast('Passwords do not match', 'error'); return
    }
    const payload: { username?: string; new_password: string } = {
      new_password: appLoginForm.new_password,
    }
    if (appLoginForm.username.trim()) payload.username = appLoginForm.username.trim()
    setAppSaving(true)
    try {
      const r = await api.updateAppPassword(payload)
      setAppLoginForm(p => ({ ...p, username: r.data.admin_username, new_password: '', confirm_password: '' }))
      addToast('Login credentials updated', 'buy')
    } catch (e: any) {
      addToast(e.response?.data?.detail || 'Failed to update login credentials', 'error')
    } finally { setAppSaving(false) }
  }

  // Credential field: label + status badge + input + optional eye toggle
  const credField = (label: string, key: keyof typeof credForm, isSecret = true) => (
    <div key={key}>
      <div className="flex items-center justify-between mb-1">
        <label className="text-sm font-medium text-slate-700">{label}</label>
        <span className="flex items-center gap-2">
          {vault[key]?.stored ? (
            <>
              <Badge variant="buy"><Lock className="w-3 h-3 inline-block mr-1 -mt-0.5" />Saved {vault[key]?.hint}</Badge>
              <button type="button" onClick={() => handleRemoveCredential(key)}
                className="text-xs text-slate-400 hover:text-rose-600">Remove</button>
            </>
          ) : (
            <Badge variant={credStatus[key] ? 'buy' : 'warning'}>
              {credStatus[key] ? 'Set (.env)' : 'Not set'}
            </Badge>
          )}
        </span>
      </div>
      <div className="relative">
        <Input
          type={isSecret && !showFields[key] ? 'password' : 'text'}
          value={credForm[key]}
          onChange={e => setCredForm(p => ({ ...p, [key]: e.target.value }))}
          placeholder="Leave blank to keep current"
          className={isSecret ? 'pr-9' : ''}
        />
        {isSecret && (
          <button
            type="button"
            onClick={() => setShowFields(p => ({ ...p, [key]: !p[key] }))}
            className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
          >
            {showFields[key] ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          </button>
        )}
      </div>
    </div>
  )

  const mode = health?.mode || 'PAPER'
  const marketOpen = health?.market_open

  const TAB_LABELS: Record<SettingsTab, string> = {
    connection: 'Connection',
    apikeys: 'API Keys',
    applogin: 'App Login',
  }

  return (
    <header className="h-14 bg-white border-b border-slate-200 flex items-center px-4 gap-4 shrink-0 z-30">
      {/* Logo */}
      <div className="flex items-center gap-2 min-w-[160px]">
        <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
          <Activity className="w-4 h-4 text-white" />
        </div>
        <div>
          <div className="text-sm font-bold text-slate-900 leading-none">AlgoTrader Pro</div>
          <div className="text-xs text-slate-500 leading-none mt-0.5">Nirma Trade v4</div>
        </div>
      </div>

      {/* Mode badge */}
      <Badge variant={mode === 'LIVE' ? 'live' : 'paper'}>{mode}</Badge>

      {/* Ticker source */}
      {health?.ticker_source && (
        <span className="text-xs text-slate-500 font-mono hidden sm:block">
          Ticks: <span className="text-slate-700 font-medium">{health.ticker_source}</span>
        </span>
      )}

      {/* Market status */}
      <div className="flex items-center gap-1.5">
        <span className={`w-2 h-2 rounded-full ${marketOpen ? 'bg-green-500 animate-pulse' : 'bg-slate-400'}`} />
        <span className="text-xs text-slate-600 hidden sm:block">
          Market {marketOpen ? 'OPEN' : 'CLOSED'}
        </span>
      </div>

      <div className="flex-1" />

      {/* Clock */}
      <span className="font-mono text-sm text-slate-700 tabular-nums hidden md:block">
        {time.toLocaleTimeString('en-IN', { hour12: false, timeZone: 'Asia/Kolkata' })} IST
      </span>

      {/* WS indicator */}
      <div className="flex items-center gap-1">
        {wsConnected ? (
          <Wifi className="w-4 h-4 text-green-500" />
        ) : (
          <WifiOff className="w-4 h-4 text-slate-400" />
        )}
        <span className="text-xs text-slate-500 hidden sm:block">{wsConnected ? 'Live' : 'Offline'}</span>
      </div>

      {/* Bot toggle */}
      <Btn
        variant={botStatus?.master_running ? 'danger' : 'buy'}
        size="sm"
        onClick={handleBotToggle}
        disabled={botLoading}
      >
        {botStatus?.master_running ? (
          <><ZapOff className="w-3.5 h-3.5 mr-1 inline-block" />Stop Bot</>
        ) : (
          <><Zap className="w-3.5 h-3.5 mr-1 inline-block" />Start Bot</>
        )}
      </Btn>

      {/* Settings button */}
      <button
        onClick={() => {
          setTempKey(apiKey); setTempBase(apiBase)
          setSettingsTab('connection')
          setCredForm({ kite_api_key: '', kite_api_secret: '', anthropic_api_key: '', truedata_username: '', truedata_password: '' })
          setShowFields({})
          setAppLoginForm({ username: '', new_password: '', confirm_password: '' })
          setShowAppFields({})
          setConfigOpen(true)
        }}
        className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-500 hover:text-slate-700 transition-colors"
      >
        <Settings className="w-4 h-4" />
      </button>

      {/* Settings modal */}
      <Modal open={configOpen} onClose={() => setConfigOpen(false)} title="Settings">
        {/* Tab bar */}
        <div className="flex border-b border-slate-200 mb-4 -mt-1">
          {(['connection', 'apikeys', 'applogin'] as const).map(tab => (
            <button
              key={tab}
              className={clsx(
                'px-4 py-2 text-sm font-medium border-b-2 transition-colors',
                settingsTab === tab
                  ? 'border-indigo-600 text-indigo-600'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              )}
              onClick={() => setSettingsTab(tab)}
            >
              {TAB_LABELS[tab]}
            </button>
          ))}
        </div>

        {/* Connection tab */}
        {settingsTab === 'connection' && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">API Base URL</label>
              <Input
                value={tempBase}
                onChange={e => setTempBase(e.target.value)}
                placeholder="http://localhost:8000"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">X-API-Key</label>
              <Input
                type="password"
                value={tempKey}
                onChange={e => setTempKey(e.target.value)}
                placeholder="Leave empty if not set"
              />
            </div>
            <div className="flex gap-2 pt-2">
              <Btn onClick={() => { setApiKey(tempKey); setApiBase(tempBase); setConfigOpen(false) }}>
                Save & Reconnect
              </Btn>
              <Btn variant="outline" onClick={() => setConfigOpen(false)}>Cancel</Btn>
            </div>
          </div>
        )}

        {/* API Keys tab */}
        {settingsTab === 'apikeys' && (
          <div className="space-y-4">
            <div>
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">Kite (Zerodha)</p>
              <div className="space-y-3">
                {credField('API Key', 'kite_api_key')}
                {credField('API Secret', 'kite_api_secret')}
              </div>
              <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-3 space-y-2" data-testid="kite-session">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm text-slate-700">
                    Today's Kite login:{' '}
                    {kite?.connected
                      ? <span className="font-medium text-green-700">connected{kite.account_id ? ` (${kite.account_id})` : ''}</span>
                      : vault.kite_access_token?.stored && !vault.kite_access_token?.expired
                        ? <span className="font-medium text-green-700">saved for today</span>
                        : <span className="font-medium text-amber-700">not logged in</span>}
                  </span>
                  <Btn size="sm" variant="outline" className="shrink-0 whitespace-nowrap" onClick={handleKiteLogin} disabled={!credStatus.kite_api_key}>
                    <ExternalLink className="w-3.5 h-3.5 mr-1 inline-block" />Log in with Zerodha
                  </Btn>
                </div>
                <p className="text-xs text-slate-500">
                  Your Zerodha user ID, password and TOTP are typed on Zerodha's own page — this app never
                  sees or stores them. Kite requires this login once a day; the session is kept until it expires.
                </p>
              </div>
            </div>
            <div>
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">Claude AI (Anthropic)</p>
              {credField('Anthropic API Key', 'anthropic_api_key')}
            </div>
            <div>
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">TrueData</p>
              <div className="space-y-3">
                {credField('Username', 'truedata_username', false)}
                {credField('Password', 'truedata_password')}
              </div>
            </div>
            <label className="flex items-start gap-2 text-sm text-slate-700 cursor-pointer">
              <input type="checkbox" className="mt-0.5" checked={remember} onChange={e => setRemember(e.target.checked)} />
              <span>
                Remember on this server (encrypted)
                <span className="block text-xs text-slate-400">
                  {remember
                    ? 'Saved encrypted in .secrets/ on the server and loaded at startup. Never shown again, only the last 4 characters.'
                    : 'Kept in memory only — a restart reverts to .env.'}
                </span>
              </span>
            </label>
            {vaultError && <p className="text-xs text-rose-600">Vault: {vaultError}</p>}
            <div className="flex gap-2 pt-1">
              <Btn onClick={handleSaveCredentials} disabled={credSaving}>
                {credSaving ? 'Saving…' : 'Save Credentials'}
              </Btn>
              <Btn variant="outline" onClick={() => setConfigOpen(false)}>Cancel</Btn>
            </div>
          </div>
        )}

        {/* App Login tab */}
        {settingsTab === 'applogin' && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Admin Username</label>
              <Input
                value={appLoginForm.username}
                onChange={e => setAppLoginForm(p => ({ ...p, username: e.target.value }))}
                placeholder="admin"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">New Password</label>
              <div className="relative">
                <Input
                  type={showAppFields.new_password ? 'text' : 'password'}
                  value={appLoginForm.new_password}
                  onChange={e => setAppLoginForm(p => ({ ...p, new_password: e.target.value }))}
                  placeholder="Min 8 characters"
                  className="pr-9"
                />
                <button
                  type="button"
                  onClick={() => setShowAppFields(p => ({ ...p, new_password: !p.new_password }))}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                >
                  {showAppFields.new_password ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Confirm Password</label>
              <div className="relative">
                <Input
                  type={showAppFields.confirm_password ? 'text' : 'password'}
                  value={appLoginForm.confirm_password}
                  onChange={e => setAppLoginForm(p => ({ ...p, confirm_password: e.target.value }))}
                  placeholder="Re-enter password"
                  className="pr-9"
                />
                <button
                  type="button"
                  onClick={() => setShowAppFields(p => ({ ...p, confirm_password: !p.confirm_password }))}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                >
                  {showAppFields.confirm_password ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>
            <p className="text-xs text-slate-400">In-memory only — restart reverts to environment variables.</p>
            <div className="flex gap-2 pt-1">
              <Btn onClick={handleSaveAppPassword} disabled={appSaving}>
                {appSaving ? 'Saving…' : 'Update Login'}
              </Btn>
              <Btn variant="outline" onClick={() => setConfigOpen(false)}>Cancel</Btn>
            </div>
          </div>
        )}
      </Modal>
    </header>
  )
}
