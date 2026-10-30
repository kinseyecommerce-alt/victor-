import React, { useEffect, useRef, useState } from 'react'
import { animate, motion, AnimatePresence, useReducedMotion } from 'framer-motion'
import { clsx } from 'clsx'
import { Area, AreaChart, ResponsiveContainer, Tooltip, YAxis, ReferenceLine } from 'recharts'

// ── Types (shape of GET /dashboard/summary) ─────────────────────────────────
export interface DashPosition {
  id: string; symbol: string; strategy: string; side: 'BUY' | 'SELL'; quantity: number
  entry: number; ltp: number; stop: number; target: number; pnl: number; pnl_pct: number
  r_multiple: number | null; exchange: string; trailing: 'AI' | 'ON' | 'WAIT'; sl_moves: number
}
export interface DashSummary {
  time_ist: string; date: string; demo: boolean; mode: string; market_open: boolean
  bot_running: boolean; regime: string; sebi_state: string
  kpis: {
    day_pnl: number; realised: number; unrealised: number; trades_closed: number
    wins: number; losses: number; win_rate: number | null; profit_factor: number | null
    open_positions: number; max_daily_loss: number; loss_used_pct: number
    is_halted: boolean; ai_cost_usd: number
  }
  positions: DashPosition[]
  agents: { name: string; running: boolean; enabled: boolean; trades_today: number }[]
  ai: { enabled: boolean; alerts_today: number; decisions_today: number
        avg_decision_ms: number | null; api_key_set: boolean; trades_today: number }
  watch: { symbol: string; ltp: number; change_pct: number; trend: string; rsi: number }[]
  activity: { t: string; kind: string; text: string; source: string }[]
}

export const inr = (v: number, digits = 0) =>
  `${v < 0 ? '-' : ''}₹${Math.abs(v).toLocaleString('en-IN', { maximumFractionDigits: digits, minimumFractionDigits: digits })}`

// ── Animated number (springs from the old value to the new one) ─────────────
export function AnimatedNumber({ value, format, className }: {
  value: number; format: (v: number) => string; className?: string
}) {
  const ref = useRef<HTMLSpanElement>(null)
  const prev = useRef(value)
  const reduce = useReducedMotion()
  useEffect(() => {
    const node = ref.current
    if (!node) return
    if (reduce) { node.textContent = format(value); prev.current = value; return }
    const controls = animate(prev.current, value, {
      duration: 0.8, ease: [0.16, 1, 0.3, 1],
      onUpdate: v => { node.textContent = format(v) },
    })
    prev.current = value
    return () => controls.stop()
  }, [value])
  return <span ref={ref} className={className}>{format(value)}</span>
}

// ── KPI card ────────────────────────────────────────────────────────────────
export function KpiCard({ label, children, tone = 'neutral', sub, index = 0, glow }: {
  label: string; children: React.ReactNode; tone?: 'up' | 'down' | 'neutral'
  sub?: React.ReactNode; index?: number; glow?: boolean
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className={clsx(
        'relative overflow-hidden rounded-2xl border bg-white/[0.04] backdrop-blur px-4 py-3.5',
        tone === 'up' ? 'border-emerald-400/25' : tone === 'down' ? 'border-rose-400/25' : 'border-white/10',
        glow && tone === 'up' && 'shadow-[0_0_40px_-12px_rgba(52,211,153,0.55)]',
        glow && tone === 'down' && 'shadow-[0_0_40px_-12px_rgba(251,113,133,0.55)]',
      )}
    >
      <div className="text-[11px] uppercase tracking-[0.14em] text-slate-400">{label}</div>
      <div className={clsx('mt-1 text-2xl font-semibold tabular-nums',
        tone === 'up' ? 'text-emerald-300' : tone === 'down' ? 'text-rose-300' : 'text-white')}>
        {children}
      </div>
      {sub && <div className="mt-0.5 text-xs text-slate-400">{sub}</div>}
    </motion.div>
  )
}

// ── Ticker tape (infinite marquee, cells flash on price change) ─────────────
function TickerCell({ w }: { w: DashSummary['watch'][number] }) {
  const prev = useRef(w.ltp)
  const [flash, setFlash] = useState<'up' | 'down' | null>(null)
  useEffect(() => {
    if (w.ltp !== prev.current) {
      setFlash(w.ltp > prev.current ? 'up' : 'down')
      prev.current = w.ltp
      const t = setTimeout(() => setFlash(null), 600)
      return () => clearTimeout(t)
    }
  }, [w.ltp])
  const up = (w.change_pct ?? 0) >= 0
  return (
    <span className={clsx('inline-flex items-center gap-2 px-4 py-1.5 rounded-lg transition-colors duration-500',
      flash === 'up' && 'bg-emerald-400/15', flash === 'down' && 'bg-rose-400/15')}>
      <span className="font-semibold text-slate-200">{w.symbol}</span>
      <span className="font-mono text-slate-100 tabular-nums">{w.ltp?.toFixed(2)}</span>
      <span className={clsx('font-mono text-xs tabular-nums', up ? 'text-emerald-300' : 'text-rose-300')}>
        {up ? '▲' : '▼'} {Math.abs(w.change_pct ?? 0).toFixed(2)}%
      </span>
    </span>
  )
}

export function TickerTape({ watch }: { watch: DashSummary['watch'] }) {
  if (!watch.length) {
    return <div className="h-10 flex items-center px-4 text-sm text-slate-500">No live prices yet — start the bot or the AI trader to stream symbols.</div>
  }
  const secs = Math.max(20, watch.length * 4)
  return (
    <div className="ticker-mask relative overflow-hidden h-10 flex items-center" data-testid="ticker">
      <div className="ticker-track flex whitespace-nowrap" style={{ animationDuration: `${secs}s` }}>
        {[...watch, ...watch].map((w, i) => <TickerCell key={`${w.symbol}-${i}`} w={w} />)}
      </div>
    </div>
  )
}

// ── Intraday P&L curve ──────────────────────────────────────────────────────
export function PnlChart({ data }: { data: { t: string; pnl: number }[] }) {
  const last = data.length ? data[data.length - 1].pnl : 0
  const up = last >= 0
  const color = up ? '#34d399' : '#fb7185'
  return (
    <div className="h-full w-full" data-testid="pnl-chart">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="pnlFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.45} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <YAxis hide domain={['auto', 'auto']} />
          <ReferenceLine y={0} stroke="rgba(148,163,184,0.35)" strokeDasharray="4 4" />
          <Tooltip
            contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, fontSize: 12 }}
            labelStyle={{ color: '#94a3b8' }}
            formatter={(v: number) => [inr(v, 2), 'Day P&L']}
            labelFormatter={(_, p) => (p?.[0]?.payload?.t ?? '')}
          />
          <Area type="monotone" dataKey="pnl" stroke={color} strokeWidth={2} fill="url(#pnlFill)"
                isAnimationActive={false} dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Position card: stop ── entry ── target rail with live price marker ──────
export function PositionCard({ p }: { p: DashPosition }) {
  const long = p.side === 'BUY'
  // Map prices onto a 0-100 rail, stop on the left and target on the right
  const lo = long ? p.stop : p.target, hi = long ? p.target : p.stop
  const span = Math.max(hi - lo, 1e-9)
  const pos = (v: number) => {
    const x = ((v - lo) / span) * 100
    return Math.min(100, Math.max(0, long ? x : 100 - x))
  }
  const up = p.pnl >= 0
  const prevMoves = useRef(p.sl_moves)
  const [trailed, setTrailed] = useState(false)
  useEffect(() => {
    if (p.sl_moves > prevMoves.current) {
      setTrailed(true)
      const t = setTimeout(() => setTrailed(false), 1500)
      prevMoves.current = p.sl_moves
      return () => clearTimeout(t)
    }
  }, [p.sl_moves])

  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.96, y: 10 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.96, x: 40 }}
      transition={{ type: 'spring', stiffness: 260, damping: 26 }}
      className={clsx('rounded-2xl border bg-white/[0.04] p-4',
        up ? 'border-emerald-400/20' : 'border-rose-400/20')}
      data-testid="position-card"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className={clsx('text-[10px] font-bold px-1.5 py-0.5 rounded',
              long ? 'bg-emerald-400/15 text-emerald-300' : 'bg-rose-400/15 text-rose-300')}>
              {long ? 'LONG' : 'SHORT'}
            </span>
            <span className="font-semibold text-white truncate">{p.symbol}</span>
          </div>
          <div className="mt-0.5 text-xs text-slate-400">
            {p.strategy} · {p.quantity} qty · entry {p.entry.toFixed(2)}
          </div>
        </div>
        <div className="text-right shrink-0">
          <AnimatedNumber value={p.pnl} format={v => inr(v)}
            className={clsx('block text-lg font-semibold tabular-nums', up ? 'text-emerald-300' : 'text-rose-300')} />
          <div className="text-xs text-slate-400 tabular-nums">
            {p.pnl_pct >= 0 ? '+' : ''}{p.pnl_pct.toFixed(2)}%{p.r_multiple != null && ` · ${p.r_multiple.toFixed(1)}R`}
          </div>
        </div>
      </div>

      {/* Rail */}
      <div className="relative mt-5 mb-6 h-2 rounded-full bg-gradient-to-r from-rose-500/40 via-slate-600/50 to-emerald-500/40">
        <div className="absolute -top-1 w-0.5 h-4 bg-slate-300/70" style={{ left: `${pos(p.entry)}%` }} />
        <motion.div
          className="absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white shadow-lg"
          style={{ background: up ? '#34d399' : '#fb7185' }}
          animate={{ left: `${pos(p.ltp)}%` }}
          transition={{ type: 'spring', stiffness: 120, damping: 18 }}
        />
        <div className="absolute -bottom-5 left-0 text-[10px] text-rose-300 font-mono">
          <motion.span animate={trailed ? { scale: [1, 1.25, 1] } : {}} className="inline-block">
            SL {p.stop.toFixed(2)}
          </motion.span>
        </div>
        <div className="absolute -bottom-5 right-0 text-[10px] text-emerald-300 font-mono">TGT {p.target.toFixed(2)}</div>
      </div>

      <div className="flex items-center justify-between text-[11px]">
        <span className="text-slate-400">LTP <span className="font-mono text-slate-200">{p.ltp.toFixed(2)}</span></span>
        <span className={clsx('px-2 py-0.5 rounded-full font-medium',
          p.trailing === 'AI' ? 'bg-violet-400/15 text-violet-300'
            : p.trailing === 'ON' ? 'bg-sky-400/15 text-sky-300' : 'bg-slate-500/15 text-slate-400',
          trailed && 'ring-2 ring-violet-300/60')}>
          {p.trailing === 'AI' ? '🧠 AI trailing' : p.trailing === 'ON' ? 'Trailing on' : 'Trail waiting'}
          {p.sl_moves > 0 && ` · ${p.sl_moves} moves`}
        </span>
      </div>
    </motion.div>
  )
}

// ── Agent orb ───────────────────────────────────────────────────────────────
const AGENT_META: Record<string, { label: string; hue: string; icon: string }> = {
  intraday: { label: 'Intraday', hue: '#818cf8', icon: '⚡' },
  scalping: { label: 'Scalping', hue: '#38bdf8', icon: '🎯' },
  swing:    { label: 'Swing',    hue: '#34d399', icon: '🌊' },
  fno:      { label: 'F&O',      hue: '#f59e0b', icon: '📊' },
  ai:       { label: 'AI Trader', hue: '#c084fc', icon: '🧠' },
}

export function AgentOrb({ a }: { a: DashSummary['agents'][number] }) {
  const meta = AGENT_META[a.name] ?? { label: a.name, hue: '#94a3b8', icon: '•' }
  const reduce = useReducedMotion()
  return (
    <div className="flex flex-col items-center gap-1.5" data-testid={`agent-${a.name}`}>
      <div className="relative h-14 w-14">
        {a.running && !reduce && (
          <motion.span className="absolute inset-0 rounded-full"
            style={{ border: `2px solid ${meta.hue}` }}
            animate={{ scale: [1, 1.45], opacity: [0.7, 0] }}
            transition={{ duration: 1.8, repeat: Infinity, ease: 'easeOut' }} />
        )}
        <div className="absolute inset-0 rounded-full flex items-center justify-center text-xl"
          style={{
            background: a.running ? `radial-gradient(circle at 35% 30%, ${meta.hue}66, ${meta.hue}22)` : 'rgba(148,163,184,0.08)',
            border: `1px solid ${a.running ? meta.hue : 'rgba(148,163,184,0.25)'}`,
            filter: a.running ? 'none' : 'grayscale(1)',
          }}>
          {meta.icon}
        </div>
      </div>
      <div className="text-xs font-medium text-slate-200">{meta.label}</div>
      <div className="text-[10px] text-slate-400">
        {a.running ? 'running' : a.enabled ? 'idle' : 'disabled'} · {a.trades_today} trades
      </div>
    </div>
  )
}

// ── Daily-loss gauge (semicircle) ───────────────────────────────────────────
export function LossGauge({ usedPct, maxLoss }: { usedPct: number; maxLoss: number }) {
  const r = 52, c = Math.PI * r
  const color = usedPct < 50 ? '#34d399' : usedPct < 80 ? '#f59e0b' : '#fb7185'
  return (
    <div className="flex flex-col items-center" data-testid="loss-gauge">
      <svg viewBox="0 0 128 72" className="w-40">
        <path d="M 12 66 A 52 52 0 0 1 116 66" fill="none" stroke="rgba(148,163,184,0.18)" strokeWidth="10" strokeLinecap="round" />
        <motion.path d="M 12 66 A 52 52 0 0 1 116 66" fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
          strokeDasharray={c}
          initial={{ strokeDashoffset: c }}
          animate={{ strokeDashoffset: c * (1 - Math.min(usedPct, 100) / 100) }}
          transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }} />
      </svg>
      <div className="-mt-8 text-xl font-semibold text-white tabular-nums">{usedPct.toFixed(0)}%</div>
      <div className="mt-1 text-xs text-slate-400">of ₹{maxLoss.toLocaleString('en-IN')} daily loss limit</div>
    </div>
  )
}

// ── Activity feed ───────────────────────────────────────────────────────────
const KIND_DOT: Record<string, string> = {
  trade: 'bg-emerald-400', close: 'bg-indigo-400', trail: 'bg-violet-400', alert: 'bg-sky-400',
  decision: 'bg-slate-300', rejected: 'bg-amber-400', error: 'bg-rose-400', order: 'bg-slate-500',
  system: 'bg-slate-600', cycle: 'bg-slate-400', chat: 'bg-slate-600',
}

export function ActivityFeed({ items }: { items: DashSummary['activity'] }) {
  if (!items.length) return <div className="text-sm text-slate-500 p-2">Nothing yet — orders, alerts and AI decisions appear here live.</div>
  return (
    <ul className="space-y-1.5" data-testid="activity">
      <AnimatePresence initial={false}>
        {items.map(a => (
          <motion.li key={`${a.t}-${a.kind}-${a.text}`} layout
            initial={{ opacity: 0, x: -16, height: 0 }}
            animate={{ opacity: 1, x: 0, height: 'auto' }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.35 }}
            className="flex gap-2.5 text-[13px] leading-snug">
            <span className={clsx('mt-1.5 h-2 w-2 shrink-0 rounded-full', KIND_DOT[a.kind] ?? 'bg-slate-500')} />
            <span className="font-mono text-[11px] text-slate-500 mt-0.5 shrink-0">{a.t}</span>
            <span className="text-slate-300 break-words min-w-0">
              <span className="uppercase text-[10px] tracking-wider text-slate-500 mr-1.5">{a.kind}</span>
              {a.text}
            </span>
          </motion.li>
        ))}
      </AnimatePresence>
    </ul>
  )
}
