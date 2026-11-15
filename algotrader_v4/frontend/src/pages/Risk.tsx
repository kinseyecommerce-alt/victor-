import { Check, X } from 'lucide-react'
import { V2 } from '../lib/api'
import { fixed, hhmm, inr, lakh, plain, when } from '../lib/format'
import { stateLabel, stateTone, STRATEGY_ORDER, strategyTitle } from '../lib/names'
import { usePoll } from '../lib/poll'
import { useDialogs } from '../components/Dialogs'
import { Badge, Button, Card, Collapse, Empty, LoadState, Notice, PageHeader, Table, Toggle } from '../components/ui'
import { StateSwitch } from '../components/strategy'
import { BrakeBar } from './Overview'

const TICKET_TONE: Record<string, string> = { pending: 'amber', approved: 'green', rejected: 'red', expired: 'slate', cancelled: 'slate' }

export function Risk() {
  const ts = usePoll(`${V2}/trading-state`)
  const rb = usePoll(`${V2}/risk-budget`)
  const ap = usePoll(`${V2}/approvals`)
  const { act } = useDialogs()

  const brake = rb.data?.brake
  const approvals = ap.data
  const pending: any[] = approvals?.pending || []
  const recent: any[] = approvals?.recent || []
  const modeOn = Boolean(approvals?.mode)

  const resetBrake = () => act({
    title: 'Reset the drawdown brake?',
    body: (
      <div className="space-y-2">
        <p>The brake goes back to <b>off</b> and the drawdown is measured afresh from now (today’s total becomes the new peak).</p>
        <p>Only do this when you have reviewed why it tripped. It latches on purpose.</p>
      </div>
    ),
    confirmLabel: 'Reset brake', danger: true, reason: true, reasonPlaceholder: 'manual reset after review',
    path: `${V2}/risk/brake/reset`, payload: (r) => ({ reason: r || 'manual reset' }), done: 'Drawdown brake reset',
  })

  const setMode = (on: boolean) => act({
    title: on ? 'Turn semi-auto approvals ON?' : 'Turn approvals off?',
    body: on
      ? 'Each strategy then needs your approval every day before it may open new trades (one ticket per strategy per day, before its entry time). Exits never need approval. If you do not approve in time, that day’s entries are skipped.'
      : 'Strategies go back to opening trades automatically by their rules.',
    confirmLabel: on ? 'Turn on' : 'Turn off', danger: false,
    path: `${V2}/approvals/mode`, payload: () => ({ on }), done: `Approvals ${on ? 'ON' : 'off'}`,
  })

  const decide = (t: any, approve: boolean) => act({
    title: `${approve ? 'Approve' : 'Reject'} today’s entries for ${strategyTitle(String(t?.strategy || ''))}?`,
    body: approve ? `It may then open its trades for ${t?.day} (${t?.stage || 'entries'}) by its rule.` : `It will open no new trades on ${t?.day}. Open trades still exit by the rule.`,
    confirmLabel: approve ? 'Approve' : 'Reject', danger: !approve, reason: true, reasonPlaceholder: 'optional note',
    path: `${V2}/approvals/${encodeURIComponent(String(t?.id))}/${approve ? 'approve' : 'reject'}`, payload: (r) => ({ note: r }),
    done: approve ? 'Approved' : 'Rejected',
  })

  const openTicket = (s: string) => act({
    title: `Ask for today’s approval: ${strategyTitle(s)}?`,
    body: 'Opens (or shows) today’s approval ticket for this strategy. You can then approve or reject it below.',
    confirmLabel: 'Open ticket', path: `${V2}/approvals`, payload: () => ({ strategy: s }), done: 'Ticket opened',
  })

  const caps = rb.data?.strategies || {}

  return (
    <div className="space-y-4">
      <PageHeader title="Risk & controls" subtitle="Whole-app switch, drawdown brake, approvals and the risk budget." polls={[ts, rb, ap]} />

      <Card title="Whole-app trading state">
        <LoadState poll={ts}>
          <div className="space-y-3">
            <p className="text-sm text-slate-600 dark:text-slate-300">Applies to every strategy. Each strategy uses the strictest of this, its own state and the drawdown brake.</p>
            <StateSwitch name={null} own={ts.data?.global?.state} />
            {ts.data?.global?.reason && <div className="text-xs text-slate-500">Last change: “{ts.data.global.reason}” by {ts.data.global.by || '?'} {when(ts.data.global.at)}</div>}
            <div className="space-y-2 pt-2">
              <div className="text-xs font-semibold uppercase text-slate-500">Per strategy</div>
              {STRATEGY_ORDER.filter((s) => ts.data?.strategies?.[s]).map((s) => {
                const x = ts.data.strategies[s]
                return (
                  <div key={s} className="rounded-xl border border-slate-100 p-2.5 dark:border-slate-800">
                    <div className="mb-1.5 flex items-center justify-between gap-2 text-sm">
                      <span className="font-medium">{strategyTitle(s)}</span>
                      <Badge tone={stateTone(x?.effective)}>now {stateLabel(x?.effective)}</Badge>
                    </div>
                    <StateSwitch name={s} own={x?.state} size="sm" />
                    {x?.reason && <div className="mt-1 text-xs text-slate-500">“{x.reason}” · {x.by} · {when(x.at)}</div>}
                  </div>
                )
              })}
            </div>
            <Collapse title={`Change log (${(ts.data?.history || []).length})`}>
              <Table head={['When', 'What', 'From → to', 'Why', 'Who']} empty="No changes yet."
                rows={[...(ts.data?.history || [])].reverse().slice(0, 50).map((h: any) => [when(h?.at), String(h?.scope || '').replace('strategy:', ''), `${stateLabel(h?.from)} → ${stateLabel(h?.to)}`, h?.reason || '—', h?.by || '—'])} />
            </Collapse>
          </div>
        </LoadState>
      </Card>

      <Card title="Drawdown brake" right={brake ? <Badge tone={stateTone(brake.state)}>{brake.state === 'ACTIVE' ? 'Off' : stateLabel(brake.state)}</Badge> : null}>
        <LoadState poll={rb}>
          <div className="space-y-3">
            <p className="text-sm text-slate-600 dark:text-slate-300">
              Watches the booked profit and loss of all strategies together. If it falls {fixed(brake?.reduce_pct, 0)} % of {lakh(brake?.capital)} below its peak,
              everything goes Exit-only; at {fixed(brake?.halt_pct, 0)} % everything stops. It stays tripped until you reset it.
            </p>
            <BrakeBar brake={brake} />
            <div className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
              <div><div className="text-xs text-slate-500">Booked since reset</div><b>{inr(brake?.net, { signed: true })}</b></div>
              <div><div className="text-xs text-slate-500">Peak</div><b>{inr(brake?.peak, { signed: true })}</b></div>
              <div><div className="text-xs text-slate-500">Would be now</div><b>{stateLabel(brake?.would_be)}</b></div>
              <div><div className="text-xs text-slate-500">Measured from</div><b>{brake?.since ? when(brake.since) : 'start of the record'}</b></div>
            </div>
            {brake?.state && brake.state !== 'ACTIVE'
              ? <Button tone="danger" onClick={resetBrake} testid="brake-reset">Reset brake…</Button>
              : <div className="flex flex-wrap items-center gap-2"><Button disabled testid="brake-reset">Reset brake…</Button><span className="text-xs text-slate-500">Only needed after the brake has tripped.</span></div>}
          </div>
        </LoadState>
      </Card>

      <Card title="Semi-auto approvals" right={<Toggle on={modeOn} onChange={setMode} label="Approval mode" disabled={!approvals} testid="approval-mode" />}>
        <LoadState poll={ap}>
          <div className="space-y-3">
            <p className="text-sm text-slate-600 dark:text-slate-300">
              {modeOn ? <><b>ON:</b> each strategy needs your approval every day before it opens trades.</> : <><b>Off:</b> strategies open trades automatically by their rules.</>}
              {' '}Exits never need approval.
            </p>
            {approvals?.deadlines && (
              <div className="flex flex-wrap gap-1.5 text-xs">
                {Object.entries<any>(approvals.deadlines).map(([s, d]) => <Badge key={s}>{strategyTitle(s)}: by {d}</Badge>)}
              </div>
            )}
            <div>
              <div className="mb-1 text-xs font-semibold uppercase text-slate-500">Waiting for you ({pending.length})</div>
              {pending.length === 0 ? <Empty>No pending approvals.</Empty> : (
                <ul className="space-y-2" data-testid="pending-approvals">
                  {pending.map((t) => (
                    <li key={t.id} className="rounded-xl border border-amber-200 bg-amber-50 p-3 dark:border-amber-900 dark:bg-amber-950/40">
                      <div className="text-sm font-medium">{strategyTitle(String(t?.strategy || ''))} — {t?.stage || 'entries'} on {t?.day}</div>
                      <div className="text-xs text-slate-600 dark:text-slate-400">Decide by {hhmm(t?.deadline)} · requested by {t?.requested_by || '?'}{t?.note ? ` · ${t.note}` : ''}</div>
                      <div className="mt-2 flex gap-2">
                        <Button tone="success" className="flex-1" onClick={() => decide(t, true)} testid={`approve-${t.id}`}><Check size={16} /> Approve</Button>
                        <Button tone="danger" className="flex-1" onClick={() => decide(t, false)} testid={`reject-${t.id}`}><X size={16} /> Reject</Button>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
            {modeOn && (
              <div className="flex flex-wrap gap-2">
                {STRATEGY_ORDER.map((s) => <Button key={s} onClick={() => openTicket(s)} className="text-xs" testid={`open-ticket-${s}`}>Ask approval: {strategyTitle(s)}</Button>)}
              </div>
            )}
            <Collapse title={`Recent decisions (${recent.length})`}>
              <Table head={['Day', 'Strategy', 'Status', 'Decided', 'By']} empty="None yet."
                rows={recent.slice(0, 30).map((t: any) => [t?.day, strategyTitle(String(t?.strategy || '')), <Badge tone={TICKET_TONE[t?.status] || 'slate'}>{t?.status}</Badge>, when(t?.decided_at), t?.decided_by || '—'])} />
            </Collapse>
          </div>
        </LoadState>
      </Card>

      <Card title="Risk budget per strategy">
        <LoadState poll={rb}>
          <p className="mb-2 text-sm text-slate-600 dark:text-slate-300">Sizing: <b>{rb.data?.mode || '—'}</b>. A size of ×1.00 means exactly the lab’s size. Built from lab data of {rb.data?.lab_built || '—'}.</p>
          <Table testid="risk-budget-table" head={['Strategy', 'Size', 'Lab daily swing', 'Lab worst drawdown', 'Lab p99 drawdown', 'Daily loss stop', 'Drawdown kill']}
            rows={STRATEGY_ORDER.filter((s) => caps[s]).map((s) => {
              const c = caps[s]
              return [<span>{strategyTitle(s)}{c?.active === false && <Badge className="ml-1">off</Badge>}</span>, `×${fixed(c?.scale, 2)}`, inr(c?.lab_daily_std), inr(c?.lab_max_dd), inr(c?.lab_block_dd_p99),
                c?.hard_caps?.daily_loss_stop ? inr(c.hard_caps.daily_loss_stop) : 'none', inr(c?.hard_caps?.dd_kill)]
            })} />
          {rb.data?.lab_combined && (
            <Notice tone="blue">
              All proven + strangle strategies together (lab): daily swing {inr(rb.data.lab_combined.daily_std)}, worst drawdown seen {lakh(rb.data.lab_combined.observed_max_dd)},
              99th-percentile drawdown {lakh(rb.data.lab_combined.block_dd?.p99)} — below the brake’s Exit-only level of {lakh(brake ? (brake.capital * brake.reduce_pct) / 100 : null)}, so a lab-like path never trips it.
              {' '}({plain((rb.data.lab_combined.strategies || []).length)} strategies)
            </Notice>
          )}
        </LoadState>
      </Card>
    </div>
  )
}
