import React, { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { clsx } from 'clsx'
import { Activity, Bot, Brain, CandlestickChart, Play, Square, ShieldCheck, Crosshair } from 'lucide-react'
import { api } from '../../api/client'
import { useStore } from '../../store'
import ConnectionStatus from '../ConnectionStatus'
import {
  ActivityFeed, AgentOrb, AnimatedNumber, DashSummary, KpiCard, LossGauge, PnlChart, PositionCard,
  ScoreboardPanel, StrategyTable, TickerTape, inr,
} from './parts'

const POLL_MS = 1500
const MAX_POINTS = 900

export default function Overview({ onOpenTrading }: { onOpenTrading: () => void }) {
  const { addToast } = useStore()
  const [data, setData] = useState<DashSummary | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [pnlSeries, setPnlSeries] = useState<{ t: string; pnl: number }[]>([])
  const [busy, setBusy] = useState(false)
  const lastT = useRef('')

  useEffect(() => {
    let alive = true
    const poll = () => api.dashboardSummary()
      .then(r => {
        if (!alive) return
        const d: DashSummary = r.data
        setData(d)
        setError(null)
        if (d.time_ist !== lastT.current) {
          lastT.current = d.time_ist
          setPnlSeries(s => [...s.slice(-MAX_POINTS + 1), { t: d.time_ist, pnl: d.kpis.day_pnl }])
        }
      })
      .catch(e => alive && setError(e.response?.status === 401
        ? 'The backend needs your API key — add it in Trading view → ⚙ Settings.'
        : 'Cannot reach the backend. Is the server running?'))
    poll()
    const t = setInterval(poll, POLL_MS)
    return () => { alive = false; clearInterval(t) }
  }, [])

  const run = async (fn: () => Promise<any>, ok: string) => {
    setBusy(true)
    try { await fn(); addToast(ok, 'info') }
    catch (e: any) { addToast(e.response?.data?.detail || 'Request failed', 'error') }
    finally { setBusy(false) }
  }
  const strategies = (data?.agents ?? []).filter(a => a.name !== 'ai' && a.enabled).map(a => a.name)

  const k = data?.kpis
  const tone = (v?: number) => (v == null || v === 0 ? 'neutral' : v > 0 ? 'up' : 'down') as 'up' | 'down' | 'neutral'

  return (
    <div className="overview-bg min-h-full text-slate-100" data-testid="overview">
      {/* Top bar */}
      <header className="sticky top-0 z-20 flex flex-wrap items-center gap-3 px-4 sm:px-6 py-3 border-b border-white/10 bg-slate-950/70 backdrop-blur">
        <div className="flex items-center gap-2.5">
          <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/30">
            <Activity size={18} />
          </div>
          <div className="leading-tight">
            <div className="font-semibold">AlgoTrader Pro</div>
            <div className="text-[11px] text-slate-400">{data?.date ?? '—'}</div>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <span className="px-2 py-1 rounded-lg bg-white/5 border border-white/10 font-mono">{data?.mode ?? '…'}</span>
          {data?.demo && <span className="px-2 py-1 rounded-lg bg-fuchsia-500/15 text-fuchsia-300 border border-fuchsia-400/30">DEMO</span>}
          <span className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-white/5 border border-white/10">
            <span className={clsx('h-2 w-2 rounded-full', data?.market_open ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500')} />
            {data?.market_open ? 'Market open' : 'Market closed'}
          </span>
          <span className="hidden md:inline px-2 py-1 rounded-lg bg-white/5 border border-white/10">Regime: {data?.regime ?? '—'}</span>
          <ConnectionStatus dark onClick={onOpenTrading} />
        </div>

        <div className="ml-auto flex items-center gap-2">
          <span className="font-mono text-lg tabular-nums text-slate-200 mr-1" data-testid="clock">{data?.time_ist ?? '--:--:--'}</span>
          {data?.bot_running
            ? <button disabled={busy} onClick={() => run(api.botStop, 'Bot stopped')}
                className="btn-dash border-rose-400/40 text-rose-200 hover:bg-rose-500/15"><Square size={14} /> Stop bot</button>
            : <button disabled={busy || !strategies.length} onClick={() => run(() => api.botStart(strategies), 'Bot started')}
                className="btn-dash border-emerald-400/40 text-emerald-200 hover:bg-emerald-500/15" data-testid="start-bot"><Play size={14} /> Start bot</button>}
          {data?.ai.api_key_set && (data.ai.enabled
            ? <button disabled={busy} onClick={() => run(api.aiStop, 'AI trader stopped')}
                className="btn-dash border-violet-400/40 text-violet-200 hover:bg-violet-500/15"><Brain size={14} /> Stop AI</button>
            : <button disabled={busy} onClick={() => run(api.aiStart, 'AI trader started')}
                className="btn-dash border-violet-400/40 text-violet-200 hover:bg-violet-500/15"><Brain size={14} /> Start AI</button>)}
          <button onClick={onOpenTrading} className="btn-dash border-white/15 text-slate-200 hover:bg-white/10" data-testid="open-trading">
            <CandlestickChart size={14} /> Trading view
          </button>
        </div>
      </header>

      <AnimatePresence>
        {error && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }}
            className="bg-rose-500/15 text-rose-200 text-sm px-6 py-2 border-b border-rose-400/20">{error}</motion.div>
        )}
        {k?.is_halted && (
          <motion.div initial={{ height: 0 }} animate={{ height: 'auto' }}
            className="bg-rose-600 text-white text-sm px-6 py-2">⛔ Trading halted — daily loss limit reached.</motion.div>
        )}
      </AnimatePresence>

      <div className="border-b border-white/5 bg-slate-950/40"><TickerTape watch={data?.watch ?? []} /></div>

      <main className="mx-auto max-w-[1500px] px-4 sm:px-6 py-5 space-y-5">
        {/* KPIs */}
        <section className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3" data-testid="kpis">
          <KpiCard label="Day P&L" tone={tone(k?.day_pnl)} glow index={0}
            sub={k && <>realised {inr(k.realised)} · open {inr(k.unrealised)}</>}>
            <AnimatedNumber value={k?.day_pnl ?? 0} format={v => inr(v)} />
          </KpiCard>
          <KpiCard label="Open positions" index={1} sub={`${data?.positions.length ?? 0} tracked with stops`}>
            <AnimatedNumber value={k?.open_positions ?? 0} format={v => v.toFixed(0)} />
          </KpiCard>
          <KpiCard label="Closed trades" index={2} sub={k ? `${k.wins} won · ${k.losses} lost` : undefined}>
            <AnimatedNumber value={k?.trades_closed ?? 0} format={v => v.toFixed(0)} />
          </KpiCard>
          <KpiCard label="Win rate" index={3} tone={k?.win_rate == null ? 'neutral' : k.win_rate >= 50 ? 'up' : 'down'}
            sub={k?.profit_factor != null ? `profit factor ${k.profit_factor}`
              : k?.trades_closed ? 'no losing trades yet' : 'no closed trades yet'}>
            {k?.win_rate == null ? '—' : <AnimatedNumber value={k.win_rate} format={v => `${v.toFixed(0)}%`} />}
          </KpiCard>
          <KpiCard label="AI decisions" index={4}
            sub={data?.ai.avg_decision_ms ? `${(data.ai.avg_decision_ms / 1000).toFixed(1)}s avg · ${data.ai.alerts_today} alerts` : `${data?.ai.alerts_today ?? 0} alerts`}>
            <AnimatedNumber value={data?.ai.decisions_today ?? 0} format={v => v.toFixed(0)} />
          </KpiCard>
          <KpiCard label="AI API cost" index={5} sub="today, estimated">
            <AnimatedNumber value={k?.ai_cost_usd ?? 0} format={v => `$${v.toFixed(2)}`} />
          </KpiCard>
        </section>

        {/* Chart + risk */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <div className="panel lg:col-span-2 h-72 flex flex-col">
            <div className="panel-title"><Activity size={14} /> Intraday P&L <span className="text-slate-500 font-normal">· today</span></div>
            <div className="flex-1 min-h-0">
              {(data?.pnl_history?.length ?? 0) > 1 || pnlSeries.length > 1
                ? <PnlChart data={(data?.pnl_history?.length ?? 0) > 1
                    ? [...data!.pnl_history!, { t: data!.time_ist, pnl: data!.kpis.day_pnl }]
                    : pnlSeries} />
                : <div className="h-full flex items-center justify-center text-sm text-slate-500">Collecting data…</div>}
            </div>
          </div>
          <div className="panel h-72 flex flex-col items-center justify-center gap-4">
            <div className="panel-title self-start"><ShieldCheck size={14} /> Risk</div>
            <LossGauge usedPct={k?.loss_used_pct ?? 0} maxLoss={k?.max_daily_loss ?? 0} />
            <div className="text-xs text-slate-400">SEBI kill switch: <span className={clsx(
              'font-semibold', data?.sebi_state === 'ACTIVE' ? 'text-emerald-300' : 'text-rose-300')}>{data?.sebi_state ?? '—'}</span></div>
          </div>
        </section>

        {/* Scoreboard vs index + strategy health */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <div className="panel">
            <div className="panel-title"><Activity size={14} /> Are we beating the index? <span className="text-slate-500 font-normal">· after costs</span></div>
            <ScoreboardPanel sb={data?.scoreboard} trend={data?.trend_filter} />
          </div>
          <div className="panel">
            <div className="panel-title"><Bot size={14} /> Strategy health <span className="text-slate-500 font-normal">· losers are pruned automatically</span></div>
            <StrategyTable rows={data?.strategy_stats} onToggle={s => run(
              () => (s.pruned || !s.enabled ? api.strategyEnable(s.strategy) : api.strategyDisable(s.strategy)),
              s.pruned || !s.enabled ? `${s.label} enabled` : `${s.label} disabled`)} />
          </div>
        </section>

        {/* Where the money is today — the AI's own scan */}
        <section className="panel" data-testid="opportunities">
          <div className="panel-title"><Crosshair size={14} /> Opportunities <span className="text-slate-500 font-normal">· the AI's scan of the NIFTY 100 + index, any direction</span></div>
          <OpportunitiesPanel />
        </section>

        {/* Agents */}
        <section className="panel">
          <div className="panel-title"><Bot size={14} /> Agents</div>
          <div className="flex flex-wrap justify-around gap-6 py-2">
            {(data?.agents ?? []).map(a => <AgentOrb key={a.name} a={a} />)}
          </div>
        </section>

        {/* Positions + activity */}
        <section className="grid grid-cols-1 xl:grid-cols-5 gap-5">
          <div className="panel xl:col-span-3">
            <div className="panel-title">Open positions</div>
            {data && data.positions.length === 0 ? (
              <div className="py-10 text-center text-sm text-slate-500">
                No open positions{data.watch.length ? ` — watching ${data.watch.length} symbols for setups` : ''}.
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3" data-testid="positions">
                <AnimatePresence>
                  {(data?.positions ?? []).map(p => <PositionCard key={p.id} p={p} />)}
                </AnimatePresence>
              </div>
            )}
          </div>
          <div className="panel xl:col-span-2 max-h-[520px] flex flex-col">
            <div className="panel-title">Live activity</div>
            <div className="flex-1 overflow-y-auto pr-1 dash-scroll">
              <ActivityFeed items={data?.activity ?? []} />
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}

type Opp = { symbol: string; play: string; direction: string; instrument: string; tool: string; score: number;
             why: string; ltp?: number; change_pct?: number; stop_idea?: number; target_idea?: number }
type OppScan = { at: string; regime: string; trend_filter?: string; nifty_change_pct: number | null; scanned: number;
                 universe: number; truncated: boolean; opportunities: Opp[]; total_found: number }

function OpportunitiesPanel() {
  const [scan, setScan] = useState<OppScan | null>(null)
  useEffect(() => {
    const poll = () => api.opportunities(8).then(r => setScan(r.data)).catch(() => {})
    poll()
    const t = setInterval(poll, 60000)
    return () => clearInterval(t)
  }, [])
  if (!scan) return <div className="text-sm text-slate-500">Scanning…</div>
  const dirTone = (d: string) => d.startsWith('LONG') ? 'text-emerald-300' : d.startsWith('SHORT') ? 'text-rose-300' : 'text-violet-300'
  return (
    <div className="space-y-2">
      <div className="text-[11px] text-slate-500">
        {scan.at} IST · regime {scan.regime} · NIFTY {scan.nifty_change_pct == null ? '—' : `${scan.nifty_change_pct > 0 ? '+' : ''}${scan.nifty_change_pct}%`}
        · {scan.scanned}/{scan.universe} scanned · {scan.total_found} set-ups
      </div>
      {scan.opportunities.length === 0
        ? <div className="text-sm text-slate-400">No set-up worth taking right now — the AI stays out.</div>
        : <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {scan.opportunities.map((o, i) => (
              <div key={o.symbol + o.play + i} className="rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2 text-xs" data-testid="opportunity-row">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="font-semibold text-slate-100">{o.symbol}</span>
                    <span className={clsx('font-medium', dirTone(o.direction))}>{o.direction}</span>
                    <span className="text-slate-400 truncate">{o.play.replace(/_/g, ' ')}</span>
                  </div>
                  <span className="shrink-0 rounded-md bg-white/10 px-1.5 py-0.5 font-mono text-slate-200">score {o.score}</span>
                </div>
                <div className="mt-1 text-slate-400 line-clamp-2">{o.why}</div>
                <div className="mt-1 flex flex-wrap gap-x-3 text-slate-500">
                  <span>{o.instrument}</span>
                  {o.ltp != null && <span>LTP {o.ltp}</span>}
                  {o.stop_idea != null && <span>stop ≈ {o.stop_idea}</span>}
                  {o.target_idea != null && <span>target ≈ {o.target_idea}</span>}
                </div>
              </div>
            ))}
          </div>}
    </div>
  )
}
