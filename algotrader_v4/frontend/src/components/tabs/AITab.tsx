import React, { useEffect, useRef, useState } from 'react'
import { clsx } from 'clsx'
import { useStore } from '../../store'
import { api } from '../../api/client'
import { Btn, Badge } from '../ui'

interface AIStatus {
  enabled: boolean
  mode: string
  can_trade: boolean
  blocked_reason: string
  model: string
  effort: string
  interval_min: number
  trades_today: number
  cycles_today: number
  open_positions: number
  cost_today_usd: number
  last_cycle_at: string | null
  last_summary: string
  api_key_set: boolean
  realtime: boolean
  alerts_today: number
  decisions_today: number
  pending_alerts: number
  avg_decision_ms: number | null
  last_alert: { symbol: string; kind: string; reason: string; ltp: number; at: string } | null
  limits: {
    max_trades_per_day: number
    trades_left_today: number
    max_open_positions: number
    max_position_value: number
    stop_loss_pct_range: [number, number]
    daily_cost_cap_usd: number
  }
}

interface LogEntry { t: string; kind: string; text: string }
interface ChatMsg { role: 'user' | 'ai'; text: string; actions?: LogEntry[] }

const KIND_STYLE: Record<string, string> = {
  trade:    'text-green-700',
  alert:    'text-sky-700',
  decision: 'text-slate-800',
  close:    'text-indigo-700',
  rejected: 'text-amber-700',
  error:    'text-red-700',
  cycle:    'text-slate-700',
  system:   'text-slate-500',
  chat:     'text-slate-500',
}

const MODE_VARIANT: Record<string, 'paper' | 'live' | 'warning' | 'neutral'> = {
  'PAPER-SIM': 'paper', 'PAPER-KITE': 'paper', LIVE: 'live', 'LIVE-BLOCKED': 'warning',
}

export default function AITab() {
  const { addToast } = useStore()
  const [status, setStatus] = useState<AIStatus | null>(null)
  const [log, setLog] = useState<LogEntry[]>([])
  const [chat, setChat] = useState<ChatMsg[]>([])
  const [input, setInput] = useState('')
  const [convId, setConvId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const chatEnd = useRef<HTMLDivElement>(null)

  const refresh = () => {
    api.aiStatus().then(r => setStatus(r.data)).catch(() => {})
    api.aiLog(80).then(r => setLog(r.data.log || [])).catch(() => {})
  }

  useEffect(() => {
    refresh()
    const t = setInterval(refresh, 5000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => { chatEnd.current?.scrollIntoView({ behavior: 'smooth' }) }, [chat])

  const act = async (fn: () => Promise<any>, ok: string) => {
    try {
      await fn()
      addToast(ok, 'info')
      refresh()
    } catch (e: any) {
      addToast(e.response?.data?.detail || 'Request failed', 'error')
    }
  }

  const toggle = () => status?.enabled
    ? act(api.aiStop, 'AI trader stopped')
    : act(api.aiStart, 'AI trader started')

  const runCycle = async () => {
    setBusy(true)
    try {
      const r = await api.aiCycle()
      addToast(r.data.ran ? 'Cycle complete' : 'Cycle skipped (market closed, cap reached or busy)', 'info')
      refresh()
    } catch (e: any) {
      addToast(e.response?.data?.detail || 'Cycle failed', 'error')
    } finally {
      setBusy(false)
    }
  }

  const closeAll = () => {
    if (window.confirm('Close every open AI position at market?')) {
      act(api.aiCloseAll, 'AI positions closed')
    }
  }

  const send = async () => {
    const msg = input.trim()
    if (!msg || busy) return
    setInput('')
    setChat(c => [...c, { role: 'user', text: msg }])
    setBusy(true)
    try {
      const r = await api.aiChat(msg, convId)
      setConvId(r.data.conversation_id)
      setChat(c => [...c, { role: 'ai', text: r.data.reply, actions: r.data.actions }])
      if (r.data.actions?.length) refresh()
    } catch (e: any) {
      setChat(c => [...c, { role: 'ai', text: e.response?.data?.detail || 'Request failed' }])
    } finally {
      setBusy(false)
    }
  }

  const s = status
  return (
    <div className="h-full grid grid-cols-[15rem_1fr_1.3fr] gap-3 p-3 text-xs">
      {/* Status & controls */}
      <div className="flex flex-col gap-2 overflow-auto">
        <div className="flex items-center gap-2">
          <span className={clsx('w-2.5 h-2.5 rounded-full', s?.enabled ? 'bg-green-500' : 'bg-slate-300')} />
          <span className="font-bold text-slate-900">Claude trader</span>
          {s && <Badge variant={MODE_VARIANT[s.mode] || 'neutral'}>{s.mode}</Badge>}
        </div>
        {s && !s.api_key_set && (
          <div className="text-red-700">ANTHROPIC_API_KEY is not set on the server.</div>
        )}
        {s && !s.can_trade && <div className="text-amber-700">Entries blocked: {s.blocked_reason}</div>}
        <div className="flex gap-1.5">
          <Btn size="sm" variant={s?.enabled ? 'outline' : 'primary'} className="flex-1 text-xs"
               onClick={toggle} disabled={!s?.api_key_set}>
            {s?.enabled ? 'Stop' : 'Start'}
          </Btn>
          <Btn size="sm" variant="outline" className="flex-1 text-xs" onClick={runCycle}
               disabled={busy || !s?.api_key_set}>
            Run now
          </Btn>
        </div>
        <Btn size="sm" variant="danger" className="text-xs" onClick={closeAll}
             disabled={!s?.open_positions}>
          Close AI positions ({s?.open_positions ?? 0})
        </Btn>
        {s && (
          <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-slate-600">
            <span>Trades</span>
            <span className="font-mono text-right">{s.trades_today}/{s.limits.max_trades_per_day}</span>
            <span>Real-time</span>
            <span className="font-mono text-right">
              {s.realtime ? `${s.alerts_today} alerts → ${s.decisions_today}` : 'off'}
            </span>
            {s.realtime && (
              <>
                <span>Decision time</span>
                <span className="font-mono text-right">
                  {s.avg_decision_ms != null ? `${(s.avg_decision_ms / 1000).toFixed(1)}s avg` : '—'}
                  {s.pending_alerts > 0 && ` · ${s.pending_alerts} queued`}
                </span>
              </>
            )}
            <span>Reviews</span>
            <span className="font-mono text-right">
              {s.interval_min > 0 ? `${s.cycles_today} · every ${s.interval_min}m` : 'off'}
            </span>
            <span>API cost today</span>
            <span className="font-mono text-right">
              ${s.cost_today_usd.toFixed(2)}/{s.limits.daily_cost_cap_usd}
            </span>
            <span>Per trade max</span>
            <span className="font-mono text-right">₹{s.limits.max_position_value.toLocaleString('en-IN')}</span>
            <span>Stop range</span>
            <span className="font-mono text-right">
              {s.limits.stop_loss_pct_range[0]}–{s.limits.stop_loss_pct_range[1]}%
            </span>
            <span>Model</span>
            <span className="font-mono text-right truncate" title={s.model}>{s.model} · {s.effort}</span>
          </div>
        )}
      </div>

      {/* Decision log */}
      <div className="flex flex-col min-h-0 border-l border-slate-100 pl-3">
        <div className="font-semibold text-slate-600 uppercase tracking-wide mb-1">
          Decisions {s?.last_cycle_at && <span className="normal-case font-normal">· last cycle {s.last_cycle_at}</span>}
        </div>
        <div className="flex-1 overflow-auto space-y-1">
          {log.length === 0 && <div className="text-slate-400">No activity yet.</div>}
          {[...log].reverse().map((e, i) => (
            <div key={i} className={KIND_STYLE[e.kind] || 'text-slate-700'}>
              <span className="font-mono text-slate-400 mr-1">{e.t}</span>
              <span className="font-semibold uppercase mr-1">{e.kind}</span>
              {e.text}
            </div>
          ))}
        </div>
      </div>

      {/* Chat */}
      <div className="flex flex-col min-h-0 border-l border-slate-100 pl-3">
        <div className="flex items-center justify-between mb-1">
          <span className="font-semibold text-slate-600 uppercase tracking-wide">Chat</span>
          {chat.length > 0 && (
            <button className="text-slate-400 hover:text-slate-600" onClick={() => { setChat([]); setConvId(null) }}>
              New chat
            </button>
          )}
        </div>
        <div className="flex-1 overflow-auto space-y-2 pr-1">
          {chat.length === 0 && (
            <div className="text-slate-400">
              Ask about the market or its positions — e.g. “What are you watching?” or “Close RELIANCE.”
              It can place and close trades within its limits.
            </div>
          )}
          {chat.map((m, i) => (
            <div key={i} className={clsx('rounded-lg px-2.5 py-1.5 whitespace-pre-wrap',
              m.role === 'user' ? 'bg-indigo-50 text-indigo-900 ml-6' : 'bg-slate-100 text-slate-800 mr-6')}>
              {m.text}
              {m.actions?.map((a, j) => (
                <div key={j} className={clsx('mt-1 font-medium', KIND_STYLE[a.kind])}>
                  {a.kind === 'rejected' ? '✕' : '✓'} {a.text}
                </div>
              ))}
            </div>
          ))}
          {busy && <div className="text-slate-400">Claude is working…</div>}
          <div ref={chatEnd} />
        </div>
        <form className="flex gap-1.5 mt-1.5" onSubmit={e => { e.preventDefault(); send() }}>
          <input
            className="flex-1 border border-slate-300 rounded-lg px-2 py-1.5 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder={s?.api_key_set ? 'Message the AI trader…' : 'Set ANTHROPIC_API_KEY to chat'}
            disabled={busy || !s?.api_key_set}
            maxLength={4000}
          />
          <Btn size="sm" type="submit" className="text-xs" disabled={busy || !input.trim() || !s?.api_key_set}>
            Send
          </Btn>
        </form>
      </div>
    </div>
  )
}
