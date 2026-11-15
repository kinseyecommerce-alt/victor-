import { AlertTriangle, ChevronRight } from 'lucide-react'
import { V2 } from '../lib/api'
import { dateOnly, inr, lakh, num, pnlClass, plain, relative, hhmm, titleCase } from '../lib/format'
import { blockText, stateLabel, stateTone, STRATEGY_ORDER, strategyTitle } from '../lib/names'
import { Poll, usePoll } from '../lib/poll'
import { href } from '../lib/router'
import { Badge, Card, Dot, LoadState, Notice, PageHeader, Progress, Stat } from '../components/ui'

const JOB_LABEL: Record<string, string> = {
  validated_open: 'Opening-auction entries / overnight exits',
  validated_auc_exit: 'Auction-fade exit',
  validated_attn_entry: 'Attention overnight entries',
  validated_vrp_entry: 'NIFTY strangle entry',
  validated_vrp_settle: 'Strangle expiry settlement',
  validated_vrp_mark: 'Strangle mark-to-market',
  validated_preflight: 'Morning preflight (+ approval tickets)',
  validated_store: 'Save today’s price history',
  validated_catch_up: 'Restart catch-up',
  kite_refresh: 'Kite automatic login',
  kite_refresh_last: 'Kite login (last try)',
  daily_reset: 'Daily reset',
  squareoff: 'Intraday square-off (paper)',
  daily_scoreboard: 'Daily scoreboard',
  drift_monitor: 'Drift check vs lab',
  risk_brake: 'Drawdown brake check',
}
export const jobLabel = (id: string) => JOB_LABEL[id] || (id.startsWith('validated_pead') ? 'Post-earnings drift: ' + id.replace('validated_pead_', '').replace(/_/g, ' ')
  : id.startsWith('forward_') ? 'Forward test: ' + id.replace('forward_', '').replace(/_/g, ' ') : titleCase(id))

export function BrakeBar({ brake }: { brake: any }) {
  const cap = num(brake?.capital)
  const dd = num(brake?.drawdown) ?? 0
  const red = num(brake?.reduce_pct)
  const halt = num(brake?.halt_pct)
  if (!cap || !halt) return <div className="text-sm text-slate-500">Brake levels not available.</div>
  const haltAmt = (cap * halt) / 100
  const redAmt = red !== null ? (cap * red) / 100 : null
  const scale = haltAmt * 1.1
  const tone = brake?.state === 'HALTED' ? 'red' : brake?.state === 'REDUCE_ONLY' ? 'amber' : dd > 0 && redAmt && dd > redAmt * 0.6 ? 'amber' : 'green'
  return (
    <div className="space-y-2" data-testid="brake-bar">
      <div className="flex items-baseline justify-between">
        <span className="text-sm">Drawdown from peak <b className="tabular-nums">{inr(dd)}</b> <span className="text-slate-500">({plain(brake?.drawdown_pct, 2)} % of {lakh(cap)})</span></span>
        <Badge tone={stateTone(brake?.state)}>{brake?.state === 'ACTIVE' ? 'Brake off' : `Brake: ${stateLabel(brake?.state)}`}</Badge>
      </div>
      <Progress value={dd / scale} tone={tone} markers={[...(redAmt ? [{ at: redAmt / scale, label: 'Exit-only' }] : []), { at: haltAmt / scale, label: 'Stop' }]} />
      <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400">
        <span>₹0</span>
        {redAmt !== null && <span>Exit-only at {lakh(redAmt)} ({red} %)</span>}
        <span>Stop at {lakh(haltAmt)} ({halt} %)</span>
      </div>
    </div>
  )
}

export function Overview({ overview }: { overview: Poll }) {
  const trades = usePoll(`${V2}/trades?n=1`, 'slow')
  const risk = usePoll(`${V2}/risk-budget`, 'slow')
  const system = usePoll(`${V2}/system`, 'slow')
  const drift = usePoll(`${V2}/drift`, 'slow')
  const o = overview.data
  const p = o?.portfolio || {}
  const stats = trades.data?.stats || {}
  const keptStats = Object.values(stats).filter((s: any) => s?.kept !== false) as any[]
  const total = keptStats.length ? keptStats.reduce((a, s) => a + (num(s?.net_pnl) ?? 0), 0) : null
  const totalTrades = keptStats.reduce((a, s) => a + (num(s?.trades) ?? 0), 0)
  const today = (num(p.realised_today) ?? 0) + (num(p.unrealised) ?? 0)
  const strategies = o?.strategies || {}
  const driftList: { name: string; text: string; alert: boolean }[] = []
  for (const [n, d] of Object.entries<any>(drift.data?.strategies || {})) {
    for (const w of d?.alerts || []) driftList.push({ name: n, text: String(w), alert: true })
    for (const w of d?.warnings || []) driftList.push({ name: n, text: String(w), alert: false })
  }
  const jobs: any[] = (system.data?.jobs || []).filter((j: any) => !['risk_brake', 'validated_catch_up', 'validated_vrp_mark'].includes(j?.id)).slice(0, 6)

  return (
    <div className="space-y-4">
      <PageHeader title="Overview" subtitle="Paper trading — no real money is used." polls={[overview, trades, risk]} />
      <LoadState poll={overview}>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Card><Stat label="Mode" value={<span className="text-blue-600 dark:text-blue-400">{o?.mode || '—'}</span>} sub={o?.paper_data_source ? `Prices: ${o.paper_data_source}` : undefined} /></Card>
          <Card><Stat label="Kite (Zerodha)" value={<span className="flex items-center gap-2"><Dot tone={o?.kite?.connected ? 'green' : 'red'} />{o?.kite?.connected ? 'Connected' : 'Not connected'}</span>}
            sub={o?.kite?.connected ? o?.kite?.account : (o?.kite?.detail || o?.kite?.error || 'Log in from System')} /></Card>
          <Card><Stat label="Market" value={<span className="flex items-center gap-2"><Dot tone={o?.market?.open ? 'green' : 'slate'} />{o?.market?.open ? 'Open' : 'Closed'}</span>}
            sub={o?.market?.holiday ? `Holiday: ${o.market.holiday}` : o?.market?.open ? 'NSE 09:15–15:30' : `Next session ${dateOnly(o?.market?.next_session)}`} /></Card>
          <Card><Stat label="Whole-app trading state" value={<Badge tone={stateTone(o?.trading_state?.global)} className="text-sm">{stateLabel(o?.trading_state?.global)}</Badge>}
            sub={<>Engine {o?.bot_running ? 'running' : 'stopped'} · approvals {o?.approval_mode ? 'ON' : 'off'}</>} /></Card>
        </div>

        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <Card title="Paper profit & loss (after costs)">
            <div className="grid grid-cols-2 gap-3">
              <Stat label="Today" value={inr(today, { signed: true })} valueClass={pnlClass(today)}
                sub={`booked ${inr(p.realised_today, { signed: true })} · open ${inr(p.unrealised, { signed: true })}`} />
              <Stat label="All time (kept strategies)" value={total === null ? '—' : inr(total, { signed: true })} valueClass={pnlClass(total)}
                sub={`${plain(totalTrades)} trades · ${plain(p.open_positions)} open now`} />
            </div>
          </Card>
          <Card title="Portfolio drawdown vs the brake" right={<a href={href('risk')} className="text-xs text-blue-600 dark:text-blue-400">Risk →</a>}>
            <BrakeBar brake={risk.data?.brake || { drawdown: p.drawdown, drawdown_pct: p.drawdown_pct, capital: p.risk_capital, state: p.brake_level }} />
          </Card>
        </div>

        {(num(o?.drift_warnings) ?? 0) > 0 || driftList.length > 0 ? (
          <Notice tone={driftList.some((d) => d.alert) ? 'red' : 'amber'} title={<span className="flex items-center gap-1.5"><AlertTriangle size={16} /> Paper results drifting from the lab ({driftList.length || o?.drift_warnings})</span>}>
            <ul className="mt-1 list-disc space-y-0.5 pl-5">
              {driftList.slice(0, 6).map((d, i) => <li key={i}><b>{strategyTitle(d.name)}:</b> {d.text}</li>)}
            </ul>
            <div className="mt-1 text-xs">Warnings only — nothing is changed automatically.</div>
          </Notice>
        ) : o?.drift_warnings === null ? <Notice tone="amber">Drift monitor unavailable.</Notice> : null}

        <Card title="Strategies" right={<a href={href('strategies')} className="text-xs text-blue-600 dark:text-blue-400">All →</a>}>
          <ul className="divide-y divide-slate-100 dark:divide-slate-800">
            {STRATEGY_ORDER.filter((n) => strategies[n]).concat(Object.keys(strategies).filter((n) => !STRATEGY_ORDER.includes(n))).map((n) => {
              const s = strategies[n]
              const block = blockText(s?.entry_block)
              return (
                <li key={n}>
                  <a href={href('strategies', n)} className="flex items-center gap-3 py-2.5">
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium">{strategyTitle(n)}</div>
                      <div className="truncate text-xs text-slate-500 dark:text-slate-400">{s?.lab_id} · {block ? block : 'ready for new trades'}</div>
                    </div>
                    <div className="flex shrink-0 flex-col items-end gap-1">
                      <Badge tone={stateTone(s?.state)}>{stateLabel(s?.state)}</Badge>
                      <span className="text-xs text-slate-500">{plain(s?.positions)} open</span>
                    </div>
                    <ChevronRight size={16} className="text-slate-400" />
                  </a>
                </li>
              )
            })}
          </ul>
        </Card>

        <Card title="Next scheduled jobs (IST)">
          <LoadState poll={system}>
            <ul className="space-y-1.5 text-sm" data-testid="next-jobs">
              {jobs.map((j) => (
                <li key={j.id + j.next_run} className="flex justify-between gap-2">
                  <span className="truncate">{jobLabel(String(j.id))}</span>
                  <span className="shrink-0 tabular-nums text-slate-500 dark:text-slate-400">{hhmm(j.next_run)} · {relative(j.next_run)}</span>
                </li>
              ))}
              {!jobs.length && <li className="text-slate-500">No jobs scheduled.</li>}
            </ul>
          </LoadState>
        </Card>
      </LoadState>
    </div>
  )
}
