import { useMemo, useState } from 'react'
import { ArrowLeft, ArrowDown, ArrowUp, ChevronRight, FlaskConical } from 'lucide-react'
import { V2 } from '../lib/api'
import { dateOnly, fixed, inr, num, pnlClass, plain } from '../lib/format'
import { Poll, usePoll } from '../lib/poll'
import { href } from '../lib/router'
import { Badge, Card, KV, LoadState, Notice, PageHeader, Stat } from '../components/ui'

const STATUS: Record<string, { label: string; tone: string }> = {
  watching: { label: 'Watching', tone: 'slate' },
  qualified: { label: 'Cleared the bar', tone: 'green' },
  demoted: { label: 'Demoted', tone: 'red' },
  consistent: { label: 'Consistent with lab', tone: 'blue' },
  inconsistent: { label: 'Inconsistent with lab', tone: 'amber' },
}
const GROUP: Record<string, { label: string; tone: string }> = {
  proven: { label: 'Proven', tone: 'green' },
  candidate: { label: 'Candidate', tone: 'violet' },
  forward: { label: 'Forward only', tone: 'slate' },
}

function notEnabled(poll: Poll): boolean {
  return poll.error?.status === 404 || (poll.data !== null && poll.data?.enabled === false)
}

function NotEnabled() {
  return (
    <Card>
      <div className="flex flex-col items-center gap-2 py-8 text-center" data-testid="forward-not-enabled">
        <FlaskConical size={36} className="text-slate-400" />
        <div className="text-lg font-semibold">Not enabled yet</div>
        <p className="max-w-md text-sm text-slate-500 dark:text-slate-400">
          The forward lab will follow many strategies on live data every day and say which ones really earn money,
          after correcting for how many are being tested. It is not switched on on this server yet.
        </p>
      </div>
    </Card>
  )
}

type SortKey = 'id' | 'family' | 'group' | 'trades' | 'days' | 'net' | 'pf' | 'z' | 'status'

export function ForwardLab() {
  const poll = usePoll(`${V2}/forward-lab`, 'slow')
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: 'z', dir: -1 })
  const d = poll.data
  const rows: any[] = Array.isArray(d?.strategies) ? d.strategies : []
  const sorted = useMemo(() => [...rows].sort((a, b) => {
    const x = a?.[sort.key], y = b?.[sort.key]
    const nx = num(x), ny = num(y)
    let c: number
    if (nx !== null || ny !== null) c = (nx ?? -Infinity) - (ny ?? -Infinity)
    else c = String(x ?? '').localeCompare(String(y ?? ''))
    return c * sort.dir
  }), [rows, sort])
  const rule = d?.rule || {}
  const z = num(rule.z_required)
  const N = num(rule.family_size) ?? num(d?.summary?.n_strategies)

  const th = (key: SortKey, label: string, right = false) => (
    <th className={`cursor-pointer select-none whitespace-nowrap py-2 pr-3 font-medium ${right ? 'text-right' : ''}`}
      onClick={() => setSort((s) => ({ key, dir: s.key === key ? (s.dir === 1 ? -1 : 1) : key === 'id' || key === 'family' ? 1 : -1 }))}>
      {label}{sort.key === key && (sort.dir === 1 ? <ArrowUp size={12} className="ml-0.5 inline" /> : <ArrowDown size={12} className="ml-0.5 inline" />)}
    </th>
  )

  return (
    <div className="space-y-4">
      <PageHeader title="Forward lab" subtitle="Many strategies followed on live data — which ones really work?" polls={[poll]} />
      {notEnabled(poll) ? <NotEnabled /> : (
        <LoadState poll={poll}>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Card><Stat label="Strategies watched" value={plain(d?.summary?.n_strategies ?? rows.length)} sub={d?.as_of ? `as of ${dateOnly(d.as_of)}` : undefined} /></Card>
            <Card><Stat label="Cleared the bar" value={plain(d?.summary?.qualified)} valueClass="text-green-600 dark:text-green-400" /></Card>
            <Card><Stat label="Demoted" value={plain(d?.summary?.demoted)} /></Card>
            <Card><Stat label="Lab-consistency fails" value={plain(d?.summary?.consistency_fail)} /></Card>
          </div>

          <Card title="The bar a strategy must clear" id="forward-bar">
            <div className="space-y-2 text-sm">
              <div className="rounded-xl bg-slate-900 px-4 py-3 text-center text-base font-semibold text-white dark:bg-slate-800" data-testid="forward-maths">
                needs z ≥ {z === null ? '—' : z.toFixed(2)} counting all {N === null ? '—' : plain(N)} strategies
              </div>
              <p className="text-slate-600 dark:text-slate-300">
                When many strategies are tested at once, some look profitable by pure luck. The bar is raised for every strategy being watched
                (family-wise error {rule.alpha !== undefined ? `${(Number(rule.alpha) * 100).toFixed(0)} %` : '—'}), and a strategy also needs at least {plain(rule.min_trades)} trades over {plain(rule.min_days)} days.
              </p>
              <Notice tone="amber"><b>A strategy only counts as profitable when it clears this bar.</b> A positive P&amp;L below the bar is not evidence.</Notice>
              <div className="text-xs text-slate-500">Rule {rule.version || '—'}{d?.last_run ? ` · last run ${dateOnly(d.last_run.date)}: ${d.last_run.status || '?'}, ${plain(d.last_run.strategies_run)} run, ${plain(d.last_run.errors)} errors` : ''}</div>
            </div>
          </Card>

          <Card title={`All strategies (${rows.length})`}>
            {/* phones: cards + sort picker */}
            <div className="mb-2 md:hidden">
              <select className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-base dark:border-slate-700 dark:bg-slate-800" value={`${sort.key}:${sort.dir}`}
                onChange={(e) => { const [k, dir] = e.target.value.split(':'); setSort({ key: k as SortKey, dir: Number(dir) as 1 | -1 }) }} aria-label="Sort">
                <option value="z:-1">Sort: z (highest first)</option>
                <option value="net:-1">Sort: net P&amp;L (highest first)</option>
                <option value="trades:-1">Sort: trades (most first)</option>
                <option value="pf:-1">Sort: profit factor</option>
                <option value="id:1">Sort: name</option>
                <option value="status:1">Sort: status</option>
              </select>
            </div>
            <ul className="divide-y divide-slate-100 md:hidden dark:divide-slate-800">
              {sorted.map((r) => (
                <li key={r?.id}>
                  <a href={href('forward', String(r?.id))} className="flex items-center gap-2 py-2.5">
                    <div className="min-w-0 flex-1">
                      <div className="truncate font-mono text-sm">{r?.id}</div>
                      <div className="mt-0.5 flex flex-wrap items-center gap-1 text-xs text-slate-500">
                        <Badge tone={STATUS[r?.status]?.tone || 'slate'}>{STATUS[r?.status]?.label || r?.status}</Badge>
                        <Badge tone={GROUP[r?.group]?.tone || 'slate'}>{GROUP[r?.group]?.label || r?.group}</Badge>
                        <span>{plain(r?.trades)} trades · {plain(r?.days)} days</span>
                      </div>
                    </div>
                    <div className="shrink-0 text-right">
                      <div className={`text-sm font-semibold ${pnlClass(r?.net)}`}>{inr(r?.net, { signed: true })}</div>
                      <div className={`text-xs ${z !== null && num(r?.z) !== null && Number(r.z) >= z ? 'font-semibold text-green-600' : 'text-slate-500'}`}>z {fixed(r?.z)}</div>
                    </div>
                    <ChevronRight size={16} className="text-slate-400" />
                  </a>
                </li>
              ))}
            </ul>
            <div className="hidden overflow-x-auto md:block">
              <table className="w-full text-left text-sm" data-testid="forward-table">
                <thead><tr className="border-b border-slate-200 text-xs text-slate-500 dark:border-slate-800">
                  {th('id', 'Strategy')}{th('family', 'Family')}{th('group', 'Group')}{th('trades', 'Trades', true)}{th('days', 'Days', true)}{th('net', 'Net P&L', true)}{th('pf', 'PF', true)}{th('z', 'z', true)}{th('status', 'Status')}
                </tr></thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                  {sorted.map((r) => (
                    <tr key={r?.id} className="cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50" onClick={() => { window.location.hash = href('forward', String(r?.id)).slice(1) }}>
                      <td className="py-1.5 pr-3 font-mono text-xs">{r?.id}</td>
                      <td className="py-1.5 pr-3">{r?.family}</td>
                      <td className="py-1.5 pr-3"><Badge tone={GROUP[r?.group]?.tone || 'slate'}>{GROUP[r?.group]?.label || r?.group}</Badge></td>
                      <td className="py-1.5 pr-3 text-right tabular-nums">{plain(r?.trades)}</td>
                      <td className="py-1.5 pr-3 text-right tabular-nums">{plain(r?.days)}</td>
                      <td className={`py-1.5 pr-3 text-right tabular-nums ${pnlClass(r?.net)}`}>{inr(r?.net, { signed: true })}</td>
                      <td className="py-1.5 pr-3 text-right tabular-nums">{fixed(r?.pf)}</td>
                      <td className={`py-1.5 pr-3 text-right tabular-nums ${z !== null && num(r?.z) !== null && Number(r.z) >= z ? 'font-semibold text-green-600' : ''}`}>{fixed(r?.z)}</td>
                      <td className="py-1.5 pr-3"><Badge tone={STATUS[r?.status]?.tone || 'slate'}>{STATUS[r?.status]?.label || r?.status}</Badge></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {!rows.length && <div className="py-3 text-sm text-slate-500">No strategies yet.</div>}
          </Card>
        </LoadState>
      )}
    </div>
  )
}

function CumChart({ daily }: { daily: any[] }) {
  const pts = daily.reduce<number[]>((acc, d) => { acc.push((acc[acc.length - 1] ?? 0) + (num(d?.net) ?? 0)); return acc }, [])
  if (pts.length < 2) return null
  const w = 600, h = 140
  const min = Math.min(0, ...pts), max = Math.max(0, ...pts)
  const y = (v: number) => h - 8 - ((v - min) / (max - min || 1)) * (h - 16)
  const x = (i: number) => (i / (pts.length - 1)) * w
  const path = pts.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
  const last = pts[pts.length - 1]
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-36 w-full" role="img" aria-label="Cumulative net P&L">
      <line x1="0" x2={w} y1={y(0)} y2={y(0)} className="stroke-slate-300 dark:stroke-slate-700" strokeDasharray="4 4" />
      <path d={path} fill="none" strokeWidth="2.5" className={last >= 0 ? 'stroke-green-600' : 'stroke-red-600'} />
    </svg>
  )
}

export function ForwardDetail({ id }: { id: string }) {
  const poll = usePoll(`${V2}/forward-lab/${encodeURIComponent(id)}`, 'slow')
  const d = poll.data
  const daily: any[] = Array.isArray(d?.daily) ? d.daily : []
  return (
    <div className="space-y-4">
      <a href={href('forward')} className="inline-flex items-center gap-1 text-sm text-blue-600 dark:text-blue-400"><ArrowLeft size={16} /> Forward lab</a>
      <PageHeader title={<span className="font-mono">{id}</span>} polls={[poll]} />
      {poll.error?.status === 404 ? <NotEnabled /> : (
        <LoadState poll={poll}>
          <Card title="Cumulative net P&L (paper, after costs)"><CumChart daily={daily} />{daily.length < 2 && <div className="text-sm text-slate-500">Not enough days yet.</div>}</Card>
          <Card title="Rule status"><KV data={d?.rule_status} /></Card>
          <Card title={`Daily results (${daily.length})`}>
            <ul className="divide-y divide-slate-100 text-sm dark:divide-slate-800">
              {[...daily].reverse().map((r, i) => (
                <li key={i} className="flex justify-between py-1.5"><span>{dateOnly(r?.date)} · {plain(r?.trades)} trades</span><span className={`tabular-nums ${pnlClass(r?.net)}`}>{inr(r?.net, { signed: true })}</span></li>
              ))}
              {!daily.length && <li className="py-2 text-slate-500">No days yet.</li>}
            </ul>
          </Card>
        </LoadState>
      )}
    </div>
  )
}
