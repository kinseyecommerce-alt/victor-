import { ArrowLeft } from 'lucide-react'
import { V2 } from '../lib/api'
import { fixed, hhmm, inr, num, pct, plain, relative, titleCase } from '../lib/format'
import { blockText, PERIOD_LABEL, STRATEGY_INFO, strategyTitle } from '../lib/names'
import { usePoll } from '../lib/poll'
import { href } from '../lib/router'
import { useDialogs } from '../components/Dialogs'
import { Badge, Card, Collapse, Empty, KV, LoadState, Notice, PageHeader, Toggle, renderValue } from '../components/ui'
import { DriftBadge, ReadinessBar, StateSwitch, StatusBadge, VsLab } from '../components/strategy'
import { TradeList } from '../components/trades'
import { PositionLine } from './Strategies'
import { jobLabel } from './Overview'

const LAB_KEYS: Record<string, string> = {
  n: 'Trades', trades: 'Trades', net_bps: 'Net per trade (basis points)', net_per_trade: 'Net per trade (₹)', pf: 'Profit factor',
  win_pct: 'Win rate %', bps_of_margin: 'Net per trade (bps of margin)', credit_pts: 'Credit (NIFTY points)', worst: 'Worst trade (₹)', max_dd: 'Max drawdown (₹)',
}

function LabExpectation({ lab }: { lab: any }) {
  if (!lab || typeof lab !== 'object') return <Empty>No lab expectation reported.</Empty>
  const periods = ['DISC', 'FINAL'].filter((k) => lab[k] && typeof lab[k] === 'object')
  const rest = Object.fromEntries(Object.entries(lab).filter(([k]) => !periods.includes(k)))
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        {periods.map((k) => (
          <div key={k} className="rounded-xl border border-slate-100 p-3 dark:border-slate-800">
            <div className="mb-1 text-sm font-semibold">{PERIOD_LABEL[k] || k}</div>
            <KV data={lab[k]} labels={LAB_KEYS} />
          </div>
        ))}
      </div>
      {Object.keys(rest).length > 0 && <Collapse title="More lab detail"><KV data={rest} /></Collapse>}
    </div>
  )
}

function DriftDetail({ d }: { d: any }) {
  if (!d || typeof d !== 'object' || !Object.keys(d).length) return <Empty>No drift data.</Empty>
  if (d.error) return <Notice tone="red">{String(d.error)}</Notice>
  const fq = d.fill_quality || {}
  const ex = d.expectancy || {}
  const dd = d.drawdown || {}
  const fr = d.frequency || {}
  return (
    <div className="space-y-3 text-sm">
      {[...(d.alerts || []), ...(d.warnings || [])].map((w: any, i: number) => <Notice key={i} tone={i < (d.alerts || []).length ? 'red' : 'amber'}>{String(w)}</Notice>)}
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <div className="rounded-xl border border-slate-100 p-3 dark:border-slate-800">
          <div className="mb-1 font-semibold">How often it trades</div>
          <KV data={{ 'Sessions on paper': fr.live_sessions, 'Trades per session (paper)': fr.live_per_session, 'Trades per session (lab)': fr.lab_per_session, 'Ratio paper / lab': fr.ratio }} />
        </div>
        <div className="rounded-xl border border-slate-100 p-3 dark:border-slate-800">
          <div className="mb-1 font-semibold">Fill quality</div>
          <KV data={{ 'Lab fill model': fq.lab_fill_model, 'Lab slippage (bps)': fq.lab_slippage_bps, 'Paper entry slippage (bps, mean)': fq.entry_slip_bps_mean, 'Fills measured': fq.entry_slip_n, 'Quote vs fill (bps, mean)': fq.quote_vs_fill_bps_mean }} />
        </div>
        <div className="rounded-xl border border-slate-100 p-3 dark:border-slate-800">
          <div className="mb-1 font-semibold">Last 20 trades vs lab range</div>
          <KV data={{ 'Paper mean (last 20)': ex.last20_mean !== null && ex.last20_mean !== undefined ? inr(ex.last20_mean, { signed: true }) : null, 'Lab mean': inr(ex.lab_mean, { signed: true }),
            'Lab range p5 … p95': ex.lab_band ? `${inr(ex.lab_band.p5, { signed: true })} … ${inr(ex.lab_band.p95, { signed: true })}` : null }} />
        </div>
        <div className="rounded-xl border border-slate-100 p-3 dark:border-slate-800">
          <div className="mb-1 font-semibold">Drawdown vs lab</div>
          <KV data={{ 'Paper drawdown': inr(dd.live), 'Lab worst seen': inr(dd.lab_observed_max_dd), 'Lab p95 / p99 (bootstrap)': dd.lab_block_dd ? `${inr(dd.lab_block_dd.p95)} / ${inr(dd.lab_block_dd.p99)}` : null }} />
        </div>
      </div>
    </div>
  )
}

export function StrategyDetail({ name }: { name: string }) {
  const poll = usePoll(`${V2}/strategies/${encodeURIComponent(name)}`)
  const hist = usePoll(`${V2}/trades?strategy=${encodeURIComponent(name)}&n=500`, 'slow')
  const { act } = useDialogs()
  const c = poll.data
  const info = STRATEGY_INFO[name]
  const block = blockText(c?.entry_block)
  const st = hist.data?.stats?.[name]
  const toggleOn = c?.enabled?.toggle !== false

  const setEnabled = (on: boolean) => act({
    title: on ? 'Allow new trades again?' : 'Switch this strategy off on the dashboard?',
    body: on ? 'The dashboard switch goes back on. New trades still need the server setting, the trading state and the brake to allow them.'
      : 'It stops opening new trades (open trades still exit by the rule). This switch stays off until you turn it back on.',
    confirmLabel: on ? 'Switch on' : 'Switch off', danger: !on,
    path: `${V2}/strategies/${encodeURIComponent(name)}/enable`, payload: () => ({ enabled: on }),
    done: `${strategyTitle(name)} switched ${on ? 'on' : 'off'}`,
  })

  return (
    <div className="space-y-4">
      <a href={href('strategies')} className="inline-flex items-center gap-1 text-sm text-blue-600 dark:text-blue-400"><ArrowLeft size={16} /> Strategies</a>
      <PageHeader title={strategyTitle(name, c?.label)} subtitle={<span className="font-mono">{c?.lab_id || name}{c?.version ? ` · ${c.version}` : ''}</span>} polls={[poll, hist]} right={c ? <StatusBadge status={c.status} /> : null} />
      {poll.error?.status === 404 ? <Notice tone="red">Unknown strategy “{name}”.</Notice> : (
        <LoadState poll={poll}>
          {info && <p className="text-sm text-slate-600 dark:text-slate-300">{info.what} <span className="text-slate-500">({info.when})</span></p>}
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Card title="Controls">
              <div className="space-y-3">
                <StateSwitch name={name} label={c?.label} own={c?.trading_state?.own} effective={c?.trading_state?.effective} />
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm">
                    <div className="font-medium">Dashboard on/off switch</div>
                    <div className="text-xs text-slate-500">Server setting: {c?.enabled?.setting ? 'on' : 'off (cannot be changed here)'}</div>
                  </div>
                  <Toggle on={toggleOn} onChange={setEnabled} label="Dashboard on/off switch" testid={`enable-${name}`} />
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {block ? <Badge tone="amber">No new trades: {block}</Badge> : <Badge tone="green">Ready for new trades</Badge>}
                  <Badge tone="slate">Engine {c?.running ? 'running' : 'not running'}</Badge>
                  <Badge tone="slate">Size ×{fixed(c?.allocator_scale, 2)}</Badge>
                  <DriftBadge status={c?.drift?.status} />
                </div>
              </div>
            </Card>
            <Card title="Paper result vs lab">
              <div className="space-y-3">
                <VsLab card={c} />
                <ReadinessBar readiness={c?.readiness} />
                {c?.readiness?.criterion && <p className="text-xs text-slate-500 dark:text-slate-400">Test: {String(c.readiness.criterion)}</p>}
                {c?.readiness?.detail?.checks && (
                  <ul className="grid grid-cols-1 gap-1 text-xs sm:grid-cols-2">
                    {Object.entries<any>(c.readiness.detail.checks).map(([k, v]) => (
                      <li key={k} className="flex items-center gap-1.5"><span className={v ? 'text-green-600' : 'text-slate-400'}>{v ? '✓' : '○'}</span> {k.replace(/_/g, ' ')}</li>
                    ))}
                  </ul>
                )}
              </div>
            </Card>
          </div>

          <Card title="Today">
            <div className="space-y-3">
              <div>
                <div className="mb-1 text-xs font-semibold uppercase text-slate-500">Open positions</div>
                {(c?.today?.positions || []).length ? <div className="space-y-1">{c.today.positions.map((p: any, i: number) => <PositionLine key={p?.oid || i} p={p} />)}</div> : <Empty>None.</Empty>}
              </div>
              <div>
                <div className="mb-1 text-xs font-semibold uppercase text-slate-500">Trades closed today</div>
                <TradeList trades={c?.today?.trades || []} showStrategy={false} />
              </div>
              {c?.today?.approval && <div className="text-sm">Approval ticket: <b>{c.today.approval.status}</b> (deadline {hhmm(c.today.approval.deadline)})</div>}
              <Collapse title="Today’s plan (the strategy’s working notes)"><KV data={c?.today?.plan} /></Collapse>
              <Collapse title={`Decision log (${(c?.day_log || []).length})`}>
                {(c?.day_log || []).length ? (
                  <ul className="space-y-1 font-mono text-xs">
                    {c.day_log.map((l: any, i: number) => <li key={i} className="break-words">{typeof l === 'string' ? l : JSON.stringify(l)}</li>)}
                  </ul>
                ) : <Empty>Nothing logged today.</Empty>}
              </Collapse>
            </div>
          </Card>

          <Card title="Trade history (paper, after costs)">
            {st && (
              <div className="mb-3 grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
                <div><div className="text-xs text-slate-500">Trades</div><b>{plain(st.trades)}</b> <span className="text-xs text-slate-500">({plain(st.wins)} won)</span></div>
                <div><div className="text-xs text-slate-500">Net P&amp;L</div><b>{inr(st.net_pnl, { signed: true })}</b></div>
                <div><div className="text-xs text-slate-500">Profit factor</div><b>{fixed(st.profit_factor)}</b></div>
                <div><div className="text-xs text-slate-500">Max drawdown</div><b>{inr(st.max_drawdown)}</b></div>
              </div>
            )}
            <LoadState poll={hist}><TradeList trades={hist.data?.trades || []} showStrategy={false} /></LoadState>
          </Card>

          <Card title="What the lab expects"><LabExpectation lab={c?.performance?.lab_expectation} /></Card>
          <Card title="Drift vs the lab (warnings only)"><DriftDetail d={c?.drift} /></Card>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Card title="Own safety limits"><KV data={c?.limits} labels={{ daily_loss_stop: 'Daily loss stop (₹, 0 = none)', dd_kill: 'Drawdown kill (₹)', max_gross: 'Max gross exposure (₹)', max_margin: 'Max margin (₹)', max_loss: 'Max loss per structure (₹)' }} /></Card>
            <Card title="Next jobs">
              <ul className="space-y-1 text-sm">
                {(c?.next_jobs || []).map((j: any) => <li key={j.id} className="flex justify-between gap-2"><span>{jobLabel(String(j.id))}</span><span className="text-slate-500">{hhmm(j.next_run)} · {relative(j.next_run)}</span></li>)}
                {!(c?.next_jobs || []).length && <Empty>None.</Empty>}
              </ul>
            </Card>
          </div>
          <Collapse title="Skipped signals, deviations from the lab, errors">
            <div className="space-y-2 text-sm">
              <div><b>Skipped:</b> {renderValue(c?.skip_counts && Object.keys(c.skip_counts).length ? Object.fromEntries(Object.entries(c.skip_counts).map(([k, v]) => [titleCase(k), v])) : 'none')}</div>
              <div><b>Deviations:</b> {renderValue(c?.deviations && Object.keys(c.deviations).length ? c.deviations : 'none')}</div>
              <div><b>Flags:</b> {renderValue(c?.flags && Object.keys(c.flags).length ? c.flags : 'none')}</div>
              <div><b>Errors:</b> {renderValue((c?.errors || []).length ? c.errors : 'none')}</div>
              <div><b>Drawdown so far:</b> {inr(c?.performance?.drawdown)} · <b>Live win rate:</b> {pct(num(c?.performance?.live?.win_pct), 0)}</div>
            </div>
          </Collapse>
        </LoadState>
      )}
    </div>
  )
}
