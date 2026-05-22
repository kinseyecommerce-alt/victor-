import React, { useEffect, useState, useCallback } from 'react'
import { Activity, Wifi, WifiOff, Settings, Zap, ZapOff } from 'lucide-react'
import { useStore } from '../../store'
import { api } from '../../api/client'
import { Badge, Btn, Modal, Input } from '../ui'

export default function Header() {
  const { health, botStatus, wsConnected, apiKey, apiBase, setApiKey, setApiBase, setHealth, setBotStatus, addToast } = useStore()
  const [time, setTime] = useState(new Date())
  const [configOpen, setConfigOpen] = useState(false)
  const [tempKey, setTempKey] = useState(apiKey)
  const [tempBase, setTempBase] = useState(apiBase)
  const [botLoading, setBotLoading] = useState(false)

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

  const mode = health?.mode || 'PAPER'
  const marketOpen = health?.market_open

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

      {/* Config */}
      <button
        onClick={() => { setConfigOpen(true); setTempKey(apiKey); setTempBase(apiBase) }}
        className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-500 hover:text-slate-700 transition-colors"
      >
        <Settings className="w-4 h-4" />
      </button>

      {/* Config modal */}
      <Modal open={configOpen} onClose={() => setConfigOpen(false)} title="Connection Settings">
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
      </Modal>
    </header>
  )
}
