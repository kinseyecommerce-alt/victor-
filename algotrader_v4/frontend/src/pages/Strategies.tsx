import { ChevronRight } from 'lucide-react'
import { V2 } from '../lib/api'
import { fixed, inr, pnlClass, plain } from '../lib/format'
import { blockText, STRATEGY_INFO, strategyTitle } from '../lib/names'
import { usePoll } from '../lib/poll'
import { href } from '../lib/router'
import { Badge, Card, LoadState, Notice, PageHeader } from '../components/ui'
import { DriftBadge, ReadinessBar, StateSwitch, StatusBadge, VsLab } from '../components/strategy'

export function PositionLine({ p }: { p: any }) {
  return (
    <div className="flex items-center justify-between gap-2 text-sm">
      <span className="min-w-0 truncate"><Badge tone={p?.side === 'BUY' ? 'green' : 'red'}>{p?.side === 'BUY' ? 'Long' : p?.side === 'SELL' ? 'Short' : p?.side || '?'}</Badge> <b>{p?.symbol}</b> × {plain(p?.qty)} @ {fixed(p?.entry_px)}</span>
      <span className={`shrink-0 tabular-nums ${pnlClass(p?.unrealised)}`}>{inr(p?.unrealised, { signed: true })}</span>
    </div>
  )
}

function planSummary(plan: any): string | null {
  if (!plan || typeof plan !== 'object' || !Object.keys(plan).length) return null
  const bits: string[] = []
  if (Array.isArray(plan.candidates)) bits.push(`${plan.candidates.length} candidates`)
  if (Array.isArray(plan.planned)) bits.push(`${plan.planned.length} planned`)
  if (plan.entry_done !== undefined) bits.push(plan.entry_done ? 'entry done' : 'entry not done yet')
  if (!bits.length) bits.push(`${Object.keys(plan).length} items`)
  return bits.join(' · ')
}

export function StrategyCard({ c }: { c: any }) {
  const name = String(c?.name || '')
  const info = STRATEGY_INFO[name]
  const block = blockText(c?.entry_block)
  const positions: any[] = c?.today?.positions || []
  const tradesToday: any[] = c?.today?.trades || []
  const plan = planSummary(c?.today?.plan)
  const enabled = c?.enabled
  return (
    <Card className="space-y-3" id={`card-${name}`}>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h2 className="text-base font-semibold leading-tight">{strategyTitle(name, c?.label)}</h2>
          <div className="mt-0.5 font-mono text-xs text-slate-500 dark:text-slate-400">{c?.lab_id || name}</div>
        </div>
        <StatusBadge status={c?.status} />
      </div>
      {info && <p className="text-xs text-slate-600 dark:text-slate-400">{info.what} <span className="text-slate-500">({info.when})</span></p>}

      <StateSwitch name={name} label={c?.label} own={c?.trading_state?.own} effective={c?.trading_state?.effective} />
      <div className="flex flex-wrap gap-1.5 text-xs">
        {block ? <Badge tone="amber">No new trades: {block}</Badge> : <Badge tone="green">Ready for new trades</Badge>}
        {enabled && enabled.setting === false && <Badge tone="slate">Off on the server</Badge>}
        {enabled && enabled.toggle === false && <Badge tone="slate">Off on the dashboard</Badge>}
        <DriftBadge status={c?.drift?.status} />
        <Badge tone="slate">Size ×{fixed(c?.allocator_scale, 2)}</Badge>
      </div>

      <div className="rounded-xl border border-slate-100 p-2.5 dark:border-slate-800">
        <div className="mb-1 flex justify-between text-xs text-slate-500 dark:text-slate-400">
          <span>Today</span>
          <span className={pnlClass((c?.today?.realised ?? 0) + (c?.today?.unrealised ?? 0))}>
            booked {inr(c?.today?.realised, { signed: true })} · open {inr(c?.today?.unrealised, { signed: true })}
          </span>
        </div>
        <div className="space-y-1 text-sm">
          {plan && <div className="text-xs text-slate-600 dark:text-slate-300">Plan: {plan}</div>}
          {positions.slice(0, 4).map((p, i) => <PositionLine key={p?.oid || i} p={p} />)}
          {positions.length > 4 && <div className="text-xs text-slate-500">+{positions.length - 4} more positions</div>}
          {tradesToday.length > 0 && <div className="text-xs text-slate-600 dark:text-slate-300">{tradesToday.length} trades closed today</div>}
          {!plan && !positions.length && !tradesToday.length && <div className="text-xs text-slate-500">Nothing today yet.</div>}
          {c?.today?.approval && <div className="text-xs">Approval ticket: <b>{c.today.approval.status}</b></div>}
        </div>
      </div>

      <VsLab card={c} />
      <ReadinessBar readiness={c?.readiness} />

      <a href={href('strategies', name)} className="flex items-center justify-end gap-1 text-sm font-medium text-blue-600 dark:text-blue-400" data-testid={`details-${name}`}>
        Details & trade history <ChevronRight size={16} />
      </a>
    </Card>
  )
}

export function Strategies() {
  const poll = usePoll(`${V2}/strategies`)
  const cards: any[] = poll.data?.strategies || []
  return (
    <div>
      <PageHeader title="Strategies" subtitle="Four paper strategies from the lab. Switch one to Exit-only or Stopped at any time." polls={[poll]} />
      <LoadState poll={poll}>
        {cards.length === 0 ? <Notice>No strategies reported.</Notice> : (
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {cards.map((c) => <StrategyCard key={c?.name} c={c} />)}
          </div>
        )}
      </LoadState>
    </div>
  )
}
