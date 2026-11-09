import React, { useEffect, useState } from 'react'
import { Brain, Plug, Radio, Wifi, WifiOff } from 'lucide-react'
import { clsx } from 'clsx'
import { api } from '../api/client'
import { useStore } from '../store'

// Kite session · Claude API · market data · live socket — one glance, hover for the reason.
// Polled every 10 s from /status/connections (which makes no external calls itself).

type Conn = {
  kite: { connected: boolean; keys: boolean; auto_login: boolean; account?: string; detail?: string }
  claude: { state: 'ok' | 'untested' | 'no_key' | 'bad_key' | 'billing' | 'rate_limited' | 'overloaded' | 'server' | 'connection' | 'not_found' | 'invalid_request' | 'error';
            detail: string; model: string; at?: string; active?: string | null; primary?: string; failovers?: number
            providers?: Record<string, { state: string; detail: string; model: string; role: string }> }
  data: { source: string; symbols: number; last_tick_age_s: number | null; streaming: boolean; market_open: boolean }
  ws_clients: number
  bot_running: boolean
  mode: string
}

type Tone = 'ok' | 'warn' | 'bad' | 'off'
const DOT: Record<Tone, string> = {
  ok: 'bg-emerald-500', warn: 'bg-amber-400', bad: 'bg-rose-500', off: 'bg-slate-400',
}

function kiteTone(c: Conn): Tone {
  if (c.kite.connected) return 'ok'
  if (!c.kite.keys) return 'off'
  return c.data.market_open ? 'bad' : 'warn'
}
function claudeTone(c: Conn): Tone {
  switch (c.claude.state) {
    case 'ok': return 'ok'
    case 'untested': return 'warn'
    case 'rate_limited': return 'warn'
    case 'overloaded': return 'warn'
    case 'no_key': return 'off'
    default: return 'bad'
  }
}
function claudeText(c: Conn): string {
  const p = c.claude.providers
  if (!p) return `${c.claude.detail} · ${c.claude.model}`
  const parts = Object.entries(p).filter(([, v]) => v.role !== 'off')
    .map(([n, v]) => `${n} (${v.role}): ${v.state === 'ok' ? 'ok' : v.state.replace('_', ' ')}`)
  return `${c.claude.detail} · ${parts.join(' · ')}`
}
function dataTone(c: Conn): Tone {
  if (c.data.streaming) return 'ok'
  if (!c.data.market_open) return 'off'
  return c.bot_running ? 'bad' : 'warn'
}

export default function ConnectionStatus({ dark = false, onClick }: { dark?: boolean; onClick?: () => void }) {
  const [c, setC] = useState<Conn | null>(null)
  const [open, setOpen] = useState(false)
  const wsConnected = useStore(s => s.wsConnected)

  useEffect(() => {
    const poll = () => api.connections().then(r => setC(r.data)).catch(() => {})
    poll()
    const t = setInterval(poll, 10000)
    return () => clearInterval(t)
  }, [])

  const items: { key: string; label: string; tone: Tone; icon: React.ReactNode; text: string }[] = c ? [
    { key: 'kite', label: 'Kite', tone: kiteTone(c), icon: <Plug size={14} />,
      text: c.kite.connected ? `Kite connected${c.kite.account ? ` (${c.kite.account})` : ''}${c.kite.auto_login ? ' · auto-login on' : ''}`
          : c.kite.detail || (c.kite.keys ? 'Not logged in to Kite today' : 'Kite API key not set') },
    { key: 'claude', label: 'AI', tone: claudeTone(c), icon: <Brain size={14} />, text: claudeText(c) },
    { key: 'data', label: 'Data', tone: dataTone(c), icon: <Radio size={14} />,
      text: c.data.streaming ? `Live prices · ${c.data.symbols} symbols · ${c.data.last_tick_age_s}s ago`
          : !c.data.market_open ? `Market closed · ${c.data.source}`
          : c.bot_running ? `No ticks for ${c.data.last_tick_age_s ?? '—'}s · ${c.data.source}` : `Bot stopped · ${c.data.source}` },
    { key: 'ws', label: 'Socket', tone: wsConnected ? 'ok' : 'bad', icon: wsConnected ? <Wifi size={14} /> : <WifiOff size={14} />,
      text: wsConnected ? 'Dashboard live feed connected' : 'Dashboard live feed disconnected — reconnecting' },
  ] : []

  const worst: Tone = items.some(i => i.tone === 'bad') ? 'bad' : items.some(i => i.tone === 'warn') ? 'warn' : items.length ? 'ok' : 'off'

  return (
    <div className="relative" data-testid="conn-status" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button type="button" onClick={onClick}
        className={clsx('flex items-center gap-1.5 rounded-lg px-2 py-1 text-xs border transition-colors',
          dark ? 'bg-white/5 border-white/10 text-slate-200 hover:bg-white/10'
               : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-50')}
        title={items.map(i => `${i.label}: ${i.text}`).join('\n')}>
        {items.length === 0 && <span className="text-slate-400">…</span>}
        {items.map(i => (
          <span key={i.key} className="flex items-center gap-1" data-testid={`conn-${i.key}`} data-tone={i.tone}>
            <span className={clsx('h-2 w-2 rounded-full', DOT[i.tone], i.tone === 'ok' && 'animate-pulse')} />
            <span className={clsx(i.tone === 'off' && 'opacity-50')}>{i.icon}</span>
          </span>
        ))}
        <span className={clsx('ml-1 hidden md:inline font-medium',
          worst === 'ok' ? 'text-emerald-500' : worst === 'warn' ? 'text-amber-500' : worst === 'bad' ? 'text-rose-500' : 'text-slate-400')}>
          {worst === 'ok' ? 'All connected' : worst === 'warn' ? 'Attention' : worst === 'bad' ? 'Problem' : ''}
        </span>
      </button>
      {open && items.length > 0 && (
        <div className={clsx('absolute right-0 top-full mt-1 z-50 w-80 rounded-xl border p-3 text-xs shadow-xl space-y-2',
          dark ? 'bg-slate-900 border-white/10 text-slate-200' : 'bg-white border-slate-200 text-slate-700')}>
          {items.map(i => (
            <div key={i.key} className="flex items-start gap-2">
              <span className={clsx('mt-1 h-2 w-2 shrink-0 rounded-full', DOT[i.tone])} />
              <div><div className="font-medium">{i.label}</div><div className="opacity-80">{i.text}</div></div>
            </div>
          ))}
          {onClick && <div className="pt-1 opacity-60">Click to open Settings → API Keys</div>}
        </div>
      )}
    </div>
  )
}
