import { V2 } from '../lib/api'
import { inr, num, pct, pnlClass, plain } from '../lib/format'
import { DRIFT_LABEL, STATE_HELP, STATE_LABEL, stateLabel, strategyTitle, TState } from '../lib/names'
import { useDialogs } from './Dialogs'
import { Badge, Progress, Segmented } from './ui'

export function StatusBadge({ status }: { status: unknown }) {
  return status === 'proven'
    ? <Badge tone="green">Proven on past unseen data</Badge>
    : <Badge tone="violet">Candidate</Badge>
}

export function DriftBadge({ status }: { status: unknown }) {
  const s = typeof status === 'string' ? status : ''
  if (!s) return <Badge tone="slate">Drift: n/a</Badge>
  const tone = s === 'ok' ? 'green' : s === 'warn' ? 'amber' : 'red'
  return <Badge tone={tone}>{DRIFT_LABEL[s] || s}</Badge>
}

const OPTIONS: { value: TState; label: string; tone: 'green' | 'amber' | 'red' }[] = [
  { value: 'ACTIVE', label: STATE_LABEL.ACTIVE, tone: 'green' },
  { value: 'REDUCE_ONLY', label: STATE_LABEL.REDUCE_ONLY, tone: 'amber' },
  { value: 'HALTED', label: STATE_LABEL.HALTED, tone: 'red' },
]

/** ACTIVE / EXIT-ONLY / STOPPED switch for one strategy (name) or the whole app (name = null). */
export function StateSwitch({ name, label, own, effective, size = 'md' }:
  { name: string | null; label?: string; own: unknown; effective?: unknown; size?: 'sm' | 'md' }) {
  const { act } = useDialogs()
  const who = name ? strategyTitle(name, label) : 'the whole app'
  const change = (state: TState) => act({
    title: `${name ? strategyTitle(name, label) : 'All strategies'}: switch to ${STATE_LABEL[state]}?`,
    body: (
      <div className="space-y-2">
        <p>{STATE_HELP[state]}</p>
        {name === null && state !== 'ACTIVE' && <p className="font-medium">This applies to every strategy at once.</p>}
        {state === 'ACTIVE' && <p>Entries resume only if nothing else blocks them (the whole-app state, the drawdown brake, server settings).</p>}
        <p className="text-xs text-slate-500">Changes {who} from {stateLabel(own)} to {STATE_LABEL[state]}. Logged with who and why.</p>
      </div>
    ),
    confirmLabel: `Switch to ${STATE_LABEL[state]}`,
    danger: state !== 'ACTIVE',
    reason: true,
    path: `${V2}/trading-state`,
    payload: (reason) => (name ? { scope: 'strategy', strategy: name, state, reason: reason || 'dashboard' } : { scope: 'global', state, reason: reason || 'dashboard' }),
    done: `${name ? strategyTitle(name, label) : 'Whole app'} is now ${STATE_LABEL[state]}`,
  })
  const ownS = typeof own === 'string' ? (own as TState) : null
  return (
    <div className="space-y-1">
      <Segmented options={OPTIONS} value={ownS} onChange={change} size={size} testid={`state-switch-${name || 'global'}`} />
      {Boolean(effective) && effective !== own && (
        <div className="text-xs text-amber-700 dark:text-amber-300">
          Actually {stateLabel(effective)} right now — a stricter whole-app state or the drawdown brake applies.
        </div>
      )}
    </div>
  )
}

export function ReadinessBar({ readiness }: { readiness: any }) {
  const trades = num(readiness?.trades) ?? 0
  const target = num(readiness?.target_trades)
  const progress = num(readiness?.progress) ?? (target ? trades / target : null)
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between text-xs">
        <span className="text-slate-500 dark:text-slate-400">Paper record toward its LIVE-readiness test</span>
        <span className="font-semibold tabular-nums">{target ? `${plain(trades)}/${plain(target)} trades` : `${plain(trades)} trades`}</span>
      </div>
      <Progress value={progress} tone={readiness?.ready ? 'green' : 'blue'} />
      {readiness?.ready && <div className="text-xs font-medium text-green-700 dark:text-green-400">Criteria met on paper (still never switches to LIVE by itself)</div>}
    </div>
  )
}

/** Paper result vs what the lab expects per trade. */
export function VsLab({ card }: { card: any }) {
  const live = card?.performance?.live || {}
  const ref = card?.performance?.lab_reference || {}
  const n = num(live.n ?? live.trades) ?? 0
  return (
    <div className="grid grid-cols-3 gap-2 text-center text-xs">
      <div className="rounded-xl bg-slate-50 p-2 dark:bg-slate-800/60">
        <div className="text-slate-500 dark:text-slate-400">Paper P&amp;L</div>
        <div className={`text-sm font-semibold tabular-nums ${pnlClass(live.net)}`}>{n ? inr(live.net, { signed: true }) : '₹0'}</div>
        <div className="text-slate-500">{plain(n)} trades</div>
      </div>
      <div className="rounded-xl bg-slate-50 p-2 dark:bg-slate-800/60">
        <div className="text-slate-500 dark:text-slate-400">Per trade</div>
        <div className={`text-sm font-semibold tabular-nums ${pnlClass(card?.performance?.live_mean_per_trade)}`}>{inr(card?.performance?.live_mean_per_trade, { signed: true })}</div>
        <div className="text-slate-500">lab {inr(ref.net_mean_per_trade, { signed: true })}</div>
      </div>
      <div className="rounded-xl bg-slate-50 p-2 dark:bg-slate-800/60">
        <div className="text-slate-500 dark:text-slate-400">Win rate</div>
        <div className="text-sm font-semibold tabular-nums">{n ? pct(live.win_pct, 0) : '—'}</div>
        <div className="text-slate-500">lab {pct(ref.win_pct, 0)}</div>
      </div>
    </div>
  )
}
