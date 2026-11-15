import { useMemo, useState } from 'react'
import { ArrowLeft, ChevronDown, ChevronRight, Hammer } from 'lucide-react'
import { V2 } from '../lib/api'
import { dateOnly, fixed, inr, num, pnlClass, plain } from '../lib/format'
import { usePoll } from '../lib/poll'
import { href } from '../lib/router'
import { Badge, Card, LoadState, Notice, PageHeader, Stat } from '../components/ui'

const AGENT_LABEL: Record<string, string> = {
  intraday: 'Intraday', scalping: 'Scalping', swing: 'Swing', options: 'Options', futures: 'Futures',
}
const MT_LABEL: Record<string, string> = {
  RISING: 'Rising', FALLING: 'Falling', SIDEWAYS: 'Sideways', HIGH_VOL: 'High volatility',
}
const MT_SHORT: Record<string, string> = { RISING: 'Rising', FALLING: 'Falling', SIDEWAYS: 'Sideways', HIGH_VOL: 'High vol' }
const STAGE: Record<string, { label: string; tone: string }> = {
  frozen: { label: 'Frozen, not evaluated', tone: 'slate' },
  discovery: { label: 'Failed discovery', tone: 'slate' },
  discovery_passed: { label: 'Passed discovery', tone: 'blue' },
  validation_failed: { label: 'Failed validation', tone: 'amber' },
  final_rejected: { label: 'Rejected on final', tone: 'amber' },
  final_confirmed: { label: 'Final confirmed, not proven', tone: 'violet' },
  proven: { label: 'Proven', tone: 'green' },
}
const ORDER = ['proven', 'final_confirmed', 'final_rejected', 'validation_failed', 'discovery_passed', 'discovery', 'frozen']

function dsr(x: unknown): string {
  const v = num(x)
  return v === null ? '—' : v < 0.001 && v > 0 ? '<0.001' : v.toFixed(3)
}

function Grid({ grid, agents, mtypes }: { grid: any[]; agents: string[]; mtypes: string[] }) {
  const cell = (a: string, m: string) => grid.find((g) => g?.agent === a && g?.mtype === m) || {}
  return (
    <div className="overflow-x-auto" data-testid="builder-grid">
      <table className="w-full table-fixed border-separate border-spacing-0.5 text-sm sm:border-spacing-1">
        <thead>
          <tr>
            <th className="w-[4.2rem] text-left text-[11px] font-medium text-slate-500 sm:w-24 sm:text-xs">Agent</th>
            {mtypes.map((m) => <th key={m} className="text-left text-[11px] font-medium text-slate-500 sm:text-xs"><span className="sm:hidden">{MT_SHORT[m] || m}</span><span className="hidden sm:inline">{MT_LABEL[m] || m}</span></th>)}
          </tr>
        </thead>
        <tbody>
          {agents.map((a) => (
            <tr key={a}>
              <td className="pr-0.5 text-[11px] font-semibold sm:text-xs">{AGENT_LABEL[a] || a}</td>
              {mtypes.map((m) => {
                const c = cell(a, m)
                const n = num(c.proven) ?? 0
                return (
                  <td key={m} className={`rounded-lg px-1.5 py-1.5 align-top sm:rounded-xl sm:px-2 sm:py-2 ${n > 0 ? 'bg-green-50 dark:bg-green-950/40' : 'bg-slate-50 dark:bg-slate-800/50'}`} data-testid={`cell-${a}-${m}`}>
                    {n > 0 ? (
                      <div className="space-y-0.5">
                        <div className="text-sm font-semibold text-green-700 sm:text-base dark:text-green-400">{plain(n)} proven</div>
                        <div className="text-[10px] leading-tight text-slate-500 sm:text-[11px]">{plain(c.in_forward_lab)} in forward lab · {plain(c.qualified)} qualified{num(c.ready) ? ` · ${plain(c.ready)} ready` : ''}</div>
                      </div>
                    ) : (
                      <div className="text-[10px] leading-tight text-slate-400 sm:text-xs">none proven yet</div>
                    )}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Builder() {
  const poll = usePoll(`${V2}/builder`, 'slow')
  const d = poll.data
  const agents: string[] = Array.isArray(d?.agents) ? d.agents : Object.keys(AGENT_LABEL)
  const mtypes: string[] = Array.isArray(d?.market_types) ? d.market_types : Object.keys(MT_LABEL)
  const batches: any[] = Array.isArray(d?.batches) ? d.batches : []
  const grid: any[] = Array.isArray(d?.grid) ? d.grid : []
  return (
    <div className="space-y-4">
      <PageHeader title="Strategy builder" subtitle="Builds strategies from our own data every week — and counts every try against luck" polls={[poll]} />
      <LoadState poll={poll}>
        {d?.rule_error && <Notice tone="red" title="Rule file changed">{String(d.rule_error)}</Notice>}
        {d && !d.enabled && (
          <Notice tone="blue" title="Weekly runs are off on this server">
            The builder runs one batch every Saturday at 06:00 once it is switched on (BUILDER_ENABLED). Batches already run are shown below.
          </Notice>
        )}
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Card><Stat label="Batches" value={plain(batches.length)} sub={batches[0] ? `last ${dateOnly(batches[0].created)}` : 'none yet'} /></Card>
          <Card><Stat label="Candidates tried" value={plain(d?.evaluated)} sub={`${plain(d?.final_looks)} final looks`} /></Card>
          <Card><Stat label="Proven" value={plain(d?.promoted)} valueClass="text-green-600 dark:text-green-400" /></Card>
          <Card><Stat label="Luck correction N" value={plain(d?.n_used)} sub="every try ever counted" /></Card>
        </div>

        <Card title="Proven strategies by agent and market type">
          <Grid grid={grid} agents={agents} mtypes={mtypes} />
          <p className="mt-2 text-xs text-slate-500">
            A strategy is proven only if it passes discovery, validation and one final look, beats random dates, loses its edge when its signal is shuffled,
            and its Deflated Sharpe Ratio is at least 0.95 after counting all {plain(d?.n_used)} tries. Proven strategies go to the forward lab first; paper trading is a human decision.
          </p>
        </Card>

        <Card title={`Batches (${batches.length})`}>
          <ul className="divide-y divide-slate-100 dark:divide-slate-800" data-testid="builder-batches">
            {batches.map((b) => (
              <li key={b?.id}>
                <a href={href('builder', String(b?.id))} className="flex items-center gap-2 py-2.5">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold">Batch {b?.id}</span>
                      <Badge tone={b?.status === 'evaluated' ? 'blue' : 'slate'}>{b?.status}</Badge>
                      {num(b?.errata) ? <Badge tone="amber">{plain(b.errata)} erratum</Badge> : null}
                    </div>
                    <div className="mt-0.5 text-xs text-slate-500">
                      {dateOnly(b?.created)} · {plain(b?.candidates)} candidates · discovery {plain(b?.discovery_passers)} (shuffle expects {plain(b?.shuffle_expected)})
                      {' '}· validation {plain(b?.validation_passers)} · final {plain(b?.final_confirmed)} · <b>{plain(b?.proven)} proven</b>
                    </div>
                    <div className="truncate font-mono text-[10px] text-slate-400">hash {String(b?.hash || '').slice(0, 16)}…</div>
                  </div>
                  <ChevronRight size={16} className="text-slate-400" />
                </a>
              </li>
            ))}
            {!batches.length && <li className="py-3 text-sm text-slate-500">No batch yet.</li>}
          </ul>
        </Card>
        <div className="text-xs text-slate-500">Rule {d?.rule_version || '—'} · sha256 <span className="font-mono">{String(d?.rule_sha256 || '').slice(0, 12)}</span> · lab trials counted {plain(d?.lab_trials_count)}{d?.chain_ok === false ? ' · REGISTRY CHAIN BROKEN' : ''}</div>
      </LoadState>
    </div>
  )
}

function CandidateRow({ c }: { c: any }) {
  const [open, setOpen] = useState(false)
  const st = STAGE[c?.stage] || { label: c?.stage, tone: 'slate' }
  const disc = c?.discovery || {}
  return (
    <li className="py-2" data-testid="builder-candidate">
      <button className="flex w-full items-start gap-2 text-left" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        {open ? <ChevronDown size={16} className="mt-0.5 shrink-0 text-slate-400" /> : <ChevronRight size={16} className="mt-0.5 shrink-0 text-slate-400" />}
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1 text-xs">
            <Badge tone={st.tone}>{st.label}</Badge>
            <span className="text-slate-500">{AGENT_LABEL[c?.agent] || c?.agent} · {MT_LABEL[c?.mtype] || c?.mtype}</span>
          </div>
          <div className="mt-1 text-sm">{c?.text}</div>
        </div>
      </button>
      {open && (
        <div className="ml-6 mt-2 space-y-2 text-xs" data-testid="builder-candidate-detail">
          <div className="font-mono text-[11px] text-slate-500">{c?.id}</div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-4">
            <div>Discovery trades <b>{plain(disc.trades)}</b></div>
            <div>PF <b>{fixed(disc.pf)}</b></div>
            <div>Net <b className={pnlClass(disc.net)}>{inr(disc.net, { signed: true })}</b></div>
            <div>DSR <b>{dsr(disc.dsr)}</b></div>
            <div>First half <span className={pnlClass(disc.h1)}>{inr(disc.h1, { signed: true })}</span></div>
            <div>Second half <span className={pnlClass(disc.h2)}>{inr(disc.h2, { signed: true })}</span></div>
            <div>Shuffled signal {disc.shuffle_passed ? <b className="text-amber-600">also passed</b> : 'failed'}</div>
          </div>
          {Array.isArray(disc.why) && disc.why.length > 0 && <div className="text-slate-500">Why not: {disc.why.join('; ')}</div>}
          {c?.validation && (
            <div>Validation: <b>{c.validation.passed ? 'passed' : 'failed'}</b> — {plain(c.validation.trades)} trades, PF {fixed(c.validation.pf)}, net <span className={pnlClass(c.validation.net)}>{inr(c.validation.net, { signed: true })}</span>
              {Array.isArray(c.validation.reasons) && c.validation.reasons.length > 0 ? ` (${c.validation.reasons.join('; ')})` : ''}</div>
          )}
          {c?.final && (
            <div>Final (one look): <b>{c.final.confirmed ? 'confirmed' : 'rejected'}</b> — {plain(c.final.trades)} trades, PF {fixed(c.final.pf)}, net <span className={pnlClass(c.final.net)}>{inr(c.final.net, { signed: true })}</span></div>
          )}
          {c?.proof && (
            <div>Proof: DSR {dsr(c.proof.dsr)} · {Object.entries(c.proof.gates || {}).map(([k, v]) => `${k} ${v ? 'yes' : 'no'}`).join(' · ')}</div>
          )}
        </div>
      )}
    </li>
  )
}

export function BuilderBatch({ id }: { id: string }) {
  const poll = usePoll(`${V2}/builder/batches/${encodeURIComponent(id)}`, 'slow')
  const d = poll.data
  const [filter, setFilter] = useState<string>('all')
  const [limit, setLimit] = useState(40)
  const cands: any[] = Array.isArray(d?.candidates) ? d.candidates : []
  const shown = useMemo(() => {
    const f = filter === 'all' ? cands : filter === 'passed' ? cands.filter((c) => c?.stage !== 'discovery' && c?.stage !== 'frozen') : cands.filter((c) => c?.agent === filter)
    return [...f].sort((a, b) => ORDER.indexOf(a?.stage) - ORDER.indexOf(b?.stage) || (num(b?.discovery?.dsr) ?? -1) - (num(a?.discovery?.dsr) ?? -1))
  }, [cands, filter])
  const errata: any[] = Array.isArray(d?.errata) ? d.errata : []
  const planted = d?.planted || {}
  return (
    <div className="space-y-4">
      <a href={href('builder')} className="inline-flex items-center gap-1 text-sm text-blue-600 dark:text-blue-400"><ArrowLeft size={16} /> Strategy builder</a>
      <PageHeader title={`Batch ${id}`} subtitle={d?.created ? `frozen ${dateOnly(d.created)} · data to ${d?.snapshot_end ?? '—'}` : undefined} polls={[poll]} />
      {poll.error?.status === 404 ? (
        <Card><div className="flex flex-col items-center gap-2 py-8 text-center"><Hammer size={32} className="text-slate-400" /><div className="font-semibold">No such batch</div></div></Card>
      ) : (
        <LoadState poll={poll}>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
            <Card><Stat label="Candidates" value={plain(cands.length)} /></Card>
            <Card><Stat label="Passed discovery" value={plain(d?.discovery_passers)} sub={`shuffled signals: ${plain(d?.shuffle_passers)}`} /></Card>
            <Card><Stat label="Passed validation" value={plain(d?.validation_passers)} sub={`${plain(d?.validation_tested)} tested`} /></Card>
            <Card><Stat label="Final confirmed" value={plain(d?.final_confirmed)} /></Card>
            <Card><Stat label="Proven" value={plain(Array.isArray(d?.proven) ? d.proven.length : d?.proven)} valueClass="text-green-600 dark:text-green-400" sub={`N = ${plain(d?.n_proof)}`} /></Card>
          </div>
          {errata.map((e, i) => (
            <Notice key={i} tone="amber" title={`Erratum (${e?.agent || 'batch'})`}>
              <div className="space-y-1 text-xs"><div>{e?.issue}</div><div><b>Effect:</b> {e?.effect}</div><div><b>Action:</b> {e?.action}</div></div>
            </Notice>
          ))}
          <Card title="Controls">
            <div className="space-y-1 text-xs">
              <div>Shuffled-signal control: <b>{plain(d?.shuffle_passers)}</b> shuffled versions passed discovery, against <b>{plain(d?.discovery_passers)}</b> real candidates.</div>
              <div className="flex flex-wrap gap-1">Planted-edge check per agent:{' '}
                {Object.keys(AGENT_LABEL).map((a) => planted[`gate_${a}`] === undefined ? null : (
                  <Badge key={a} tone={planted[`gate_${a}`] ? 'green' : 'red'}>{AGENT_LABEL[a]} {planted[`gate_${a}`] ? 'detected' : 'not detected'}</Badge>
                ))}
              </div>
              <div className="text-slate-500">Windows: discovery to {d?.windows?.discovery?.[1] ?? '—'}, validation {d?.windows?.validation?.join('–') ?? '—'}, final {d?.windows?.final?.join('–') ?? '—'}. Hash <span className="font-mono">{String(d?.hash || '').slice(0, 16)}</span></div>
            </div>
          </Card>
          <Card title={`Candidates (${shown.length})`} right={
            <select className="rounded-lg border border-slate-300 bg-white px-2 py-1 text-sm dark:border-slate-700 dark:bg-slate-800" value={filter} onChange={(e) => { setFilter(e.target.value); setLimit(40) }} aria-label="Filter" data-testid="builder-filter">
              <option value="all">All</option>
              <option value="passed">Passed discovery</option>
              {Object.keys(AGENT_LABEL).map((a) => <option key={a} value={a}>{AGENT_LABEL[a]}</option>)}
            </select>
          }>
            <ul className="divide-y divide-slate-100 dark:divide-slate-800">
              {shown.slice(0, limit).map((c) => <CandidateRow key={c?.id} c={c} />)}
              {!shown.length && <li className="py-3 text-sm text-slate-500">No candidates.</li>}
            </ul>
            {shown.length > limit && (
              <button className="mt-2 w-full rounded-xl border border-slate-300 py-2 text-sm font-medium dark:border-slate-700" onClick={() => setLimit((l) => l + 40)} data-testid="builder-more">
                Show {Math.min(40, shown.length - limit)} more of {shown.length - limit}
              </button>
            )}
          </Card>
        </LoadState>
      )}
    </div>
  )
}
