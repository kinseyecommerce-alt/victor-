import { V2 } from '../lib/api'
import { dateOnly, inr, num, pnlClass, plain, titleCase } from '../lib/format'
import { usePoll } from '../lib/poll'
import { Badge, Card, Collapse, KV, LoadState, Notice, PageHeader, Stat } from '../components/ui'

const STATUS: Record<string, { label: string; tone: string }> = {
  incubating: { label: 'Collecting trades', tone: 'slate' },
  active: { label: 'Active', tone: 'blue' },
  promotion_eligible: { label: 'Promotion eligible', tone: 'green' },
  demoted: { label: 'Demoted', tone: 'red' },
  retired: { label: 'Retired', tone: 'slate' },
  config_drift: { label: 'Config changed', tone: 'amber' },
}

export function Shadows() {
  const poll = usePoll(`${V2}/shadow-registry`, 'slow')
  const legacy = usePoll('/api/shadow-registry', 'slow')        // carries the pre-registered rule text
  const d = poll.data
  const rule = legacy.data?.rule || {}
  const shadows = Object.entries<any>(d?.shadows || {})
  const looks: number[] = Array.isArray(rule.looks_trades) ? rule.looks_trades : []
  const nFam = num(d?.n_family)
  const alpha = num(rule.family_alpha)
  const thr = alpha !== null && looks.length ? alpha / (Math.max(1, nFam ?? 0) * looks.length) : null

  return (
    <div className="space-y-4">
      <PageHeader title="Shadow registry" subtitle="Tamper-evident forward record of every strategy, with a fixed promotion rule." polls={[poll]} />
      <LoadState poll={poll}>
        {d?.error && <Notice tone="red">{String(d.error)}</Notice>}
        {d?.enabled === false && <Notice tone="blue">The registry is switched off on this server (SHADOW_REGISTRY_ENABLED). Existing records are shown read-only.</Notice>}
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Card><Stat label="Active shadows" value={plain(d?.n_active)} /></Card>
          <Card><Stat label="Ever registered (family)" value={plain(d?.n_family)} sub="never decreases" /></Card>
          <Card><Stat label="Ledger chain check" value={d?.chain_ok === true ? 'OK' : d?.chain_ok === false ? 'BROKEN' : '—'} valueClass={d?.chain_ok === false ? 'text-red-600' : d?.chain_ok ? 'text-green-600 dark:text-green-400' : ''} sub="tamper detection" /></Card>
          <Card><Stat label="Promotion threshold" value={thr === null ? '—' : `p ≤ ${thr.toPrecision(2)}`} sub={looks.length ? `looks at ${looks.join(' / ')} trades` : undefined} /></Card>
        </div>
        <Notice tone="amber">“Promotion eligible” is only a recommendation recorded in the registry. It never switches anything on by itself.</Notice>
        <Card title={`Shadows (${shadows.length})`}>
          {shadows.length === 0 ? <div className="py-2 text-sm text-slate-500">No shadows registered yet.</div> : (
            <ul className="divide-y divide-slate-100 dark:divide-slate-800" data-testid="shadow-list">
              {shadows.map(([id, s]) => {
                const st = STATUS[s?.status] || { label: titleCase(String(s?.status || '?')), tone: 'slate' }
                const lk = s?.looks && typeof s.looks === 'object' ? Object.keys(s.looks).length : 0
                return (
                  <li key={id} className="py-2.5">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <div className="truncate font-mono text-sm">{id}</div>
                        <div className="text-xs text-slate-500">{s?.family} · from {dateOnly(s?.forward_from)} · {s?.source}</div>
                      </div>
                      <Badge tone={st.tone}>{st.label}</Badge>
                    </div>
                    <div className="mt-1 flex gap-4 text-xs">
                      <span>{plain(s?.trades)} trades</span>
                      <span className={pnlClass(s?.net)}>{inr(s?.net, { signed: true })}</span>
                      <span>{lk}/{looks.length || '?'} looks done</span>
                    </div>
                    {s?.status_reason && <div className="mt-0.5 text-xs text-slate-500">{String(s.status_reason)}</div>}
                    {lk > 0 && <div className="mt-1"><Collapse title="Look results"><KV data={s.looks} /></Collapse></div>}
                  </li>
                )
              })}
            </ul>
          )}
        </Card>
        {Object.keys(rule).length > 0 && (
          <Collapse title={`Pre-registered rule ${rule.version || ''}`}>
            <KV data={rule} />
          </Collapse>
        )}
        <div className="break-all text-xs text-slate-500">Rule hash {d?.rule_hash || '—'}</div>
      </LoadState>
    </div>
  )
}
