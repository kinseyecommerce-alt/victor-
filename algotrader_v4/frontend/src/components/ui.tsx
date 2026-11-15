import { ReactNode, useEffect, useState } from 'react'
import clsx from 'clsx'
import { AlertTriangle, ChevronDown, RefreshCw, WifiOff } from 'lucide-react'
import type { Poll } from '../lib/poll'
import { DASH, hhmm, titleCase } from '../lib/format'

export function Card({ children, className, title, right, id }:
  { children: ReactNode; className?: string; title?: ReactNode; right?: ReactNode; id?: string }) {
  return (
    <section id={id} className={clsx('min-w-0 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900', className)}>
      {(title || right) && (
        <div className="mb-3 flex items-start justify-between gap-2">
          {title && <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-200">{title}</h2>}
          {right}
        </div>
      )}
      {children}
    </section>
  )
}

const TONES: Record<string, string> = {
  green: 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300',
  amber: 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300',
  red: 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300',
  blue: 'bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300',
  violet: 'bg-violet-100 text-violet-800 dark:bg-violet-900/40 dark:text-violet-300',
  slate: 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300',
}

export type Tone = keyof typeof TONES

export function Badge({ tone = 'slate', children, className }: { tone?: Tone | string; children: ReactNode; className?: string }) {
  return (
    <span className={clsx('inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium', TONES[tone] || TONES.slate, className)}>
      {children}
    </span>
  )
}

export function Dot({ tone }: { tone: 'green' | 'red' | 'amber' | 'slate' }) {
  const c = { green: 'bg-green-500', red: 'bg-red-500', amber: 'bg-amber-500', slate: 'bg-slate-400' }[tone]
  return <span className={clsx('inline-block h-2 w-2 shrink-0 rounded-full', c)} />
}

export function Stat({ label, value, sub, valueClass }: { label: ReactNode; value: ReactNode; sub?: ReactNode; valueClass?: string }) {
  return (
    <div className="min-w-0 [overflow-wrap:anywhere]">
      <div className="text-xs text-slate-500 dark:text-slate-400">{label}</div>
      <div className={clsx('break-words text-base font-semibold leading-snug tabular-nums sm:text-lg', valueClass)}>{value}</div>
      {sub && <div className="text-xs text-slate-500 dark:text-slate-400">{sub}</div>}
    </div>
  )
}

export function Progress({ value, tone = 'blue', markers }: { value: number | null; tone?: 'blue' | 'green' | 'amber' | 'red'; markers?: { at: number; label: string }[] }) {
  const v = Math.max(0, Math.min(1, value ?? 0))
  const c = { blue: 'bg-blue-500', green: 'bg-green-500', amber: 'bg-amber-500', red: 'bg-red-500' }[tone]
  return (
    <div className="relative h-2.5 w-full overflow-visible rounded-full bg-slate-200 dark:bg-slate-800">
      <div className={clsx('h-full rounded-full transition-all', c)} style={{ width: `${v * 100}%` }} />
      {markers?.map((m) => (
        <div key={m.label} className="absolute w-0.5 bg-slate-500 dark:bg-slate-400" style={{ left: `${Math.min(100, m.at * 100)}%`, top: -4, height: 18 }} title={m.label} />
      ))}
    </div>
  )
}

/** Key / value list for objects whose shape differs per strategy. */
export function KV({ data, labels, skip }: { data: any; labels?: Record<string, string>; skip?: string[] }) {
  if (!data || typeof data !== 'object' || Object.keys(data).length === 0) return <Empty>Nothing yet.</Empty>
  const entries = Object.entries(data).filter(([k]) => !skip?.includes(k))
  return (
    <dl className="divide-y divide-slate-100 text-sm dark:divide-slate-800">
      {entries.map(([k, v]) => (
        <div key={k} className="flex flex-col gap-0.5 py-1.5 sm:flex-row sm:gap-3">
          <dt className="shrink-0 text-slate-500 sm:w-48 dark:text-slate-400">{labels?.[k] || titleCase(k)}</dt>
          <dd className="min-w-0 break-words font-medium tabular-nums">{renderValue(v)}</dd>
        </div>
      ))}
    </dl>
  )
}

export function renderValue(v: any): ReactNode {
  if (v === null || v === undefined || v === '') return DASH
  if (typeof v === 'boolean') return v ? 'Yes' : 'No'
  if (typeof v === 'number') return Number.isInteger(v) ? v.toLocaleString('en-IN') : v.toLocaleString('en-IN', { maximumFractionDigits: 4 })
  if (typeof v === 'string') return v
  if (Array.isArray(v)) {
    if (v.length === 0) return 'none'
    if (v.every((x) => typeof x !== 'object' || x === null)) return v.join(', ')
    return <code className="block whitespace-pre-wrap break-all text-xs font-normal text-slate-600 dark:text-slate-300">{v.map((x) => JSON.stringify(x)).join('\n')}</code>
  }
  if (typeof v === 'object') {
    const keys = Object.keys(v)
    if (keys.length === 0) return 'none'
    return (
      <span className="block space-y-0.5">
        {keys.map((k) => (
          <span key={k} className="block text-xs font-normal">
            <span className="text-slate-500 dark:text-slate-400">{titleCase(k)}: </span>
            {typeof v[k] === 'object' && v[k] !== null ? JSON.stringify(v[k]) : renderValue(v[k])}
          </span>
        ))}
      </span>
    )
  }
  return String(v)
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="py-3 text-sm text-slate-500 dark:text-slate-400">{children}</div>
}

export function Notice({ tone = 'amber', children, title }: { tone?: 'amber' | 'red' | 'blue' | 'green'; children: ReactNode; title?: ReactNode }) {
  const c = {
    amber: 'border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950/50 dark:text-amber-200',
    red: 'border-red-300 bg-red-50 text-red-900 dark:border-red-800 dark:bg-red-950/50 dark:text-red-200',
    blue: 'border-blue-300 bg-blue-50 text-blue-900 dark:border-blue-800 dark:bg-blue-950/50 dark:text-blue-200',
    green: 'border-green-300 bg-green-50 text-green-900 dark:border-green-800 dark:bg-green-950/50 dark:text-green-200',
  }[tone]
  return (
    <div className={clsx('rounded-xl border px-3 py-2 text-sm', c)}>
      {title && <div className="font-semibold">{title}</div>}
      <div>{children}</div>
    </div>
  )
}

/** "Updated 09:20:05 · refreshes every 15 s" + error / offline state for one poll. */
export function Freshness({ polls }: { polls: Poll[] }) {
  const [online, setOnline] = useState(typeof navigator === 'undefined' ? true : navigator.onLine)
  useEffect(() => {
    const on = () => setOnline(true)
    const off = () => setOnline(false)
    window.addEventListener('online', on)
    window.addEventListener('offline', off)
    return () => { window.removeEventListener('online', on); window.removeEventListener('offline', off) }
  }, [])
  const times = polls.map((p) => p.updatedAt).filter((t): t is number => t !== null)
  const last = times.length ? Math.min(...times) : null
  const err = polls.find((p) => p.error && p.error.status !== 404 && p.error.status !== 401)?.error
  const busy = polls.some((p) => p.loading)
  const every = Math.round((polls[0]?.intervalMs || 60000) / 1000)
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500 dark:text-slate-400" data-testid="freshness">
      {!online && <Badge tone="red"><WifiOff size={12} /> Offline</Badge>}
      {err && <Badge tone="red"><AlertTriangle size={12} /> {err.message}</Badge>}
      <span>{last ? `Updated ${hhmm(new Date(last).toISOString(), true)}` : busy ? 'Loading…' : 'Not loaded'}{` · refreshes every ${every} s`}</span>
      <button className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 hover:bg-slate-200 dark:hover:bg-slate-800"
        onClick={() => polls.forEach((p) => p.refresh())} aria-label="Refresh now">
        <RefreshCw size={12} className={busy ? 'animate-spin' : ''} /> Refresh
      </button>
    </div>
  )
}

export function PageHeader({ title, subtitle, polls, right }: { title: ReactNode; subtitle?: ReactNode; polls?: Poll[]; right?: ReactNode }) {
  return (
    <div className="mb-4 space-y-1">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <h1 className="text-xl font-bold tracking-tight">{title}</h1>
          {subtitle && <p className="break-words text-sm text-slate-500 dark:text-slate-400">{subtitle}</p>}
        </div>
        {right}
      </div>
      {polls && <Freshness polls={polls} />}
    </div>
  )
}

export function LoadState({ poll, children, empty }: { poll: Poll; children: ReactNode; empty?: ReactNode }) {
  if (poll.data) return <>{children}</>
  if (poll.loading) return <div className="animate-pulse py-6 text-sm text-slate-400">Loading…</div>
  if (poll.error) return <Notice tone="red" title="Could not load">{poll.error.message}</Notice>
  return <>{empty ?? <Empty>No data.</Empty>}</>
}

export function Segmented<T extends string>({ options, value, onChange, disabled, size = 'md', testid }:
  { options: { value: T; label: string; tone?: 'green' | 'amber' | 'red' }[]; value: T | null | undefined; onChange: (v: T) => void; disabled?: boolean; size?: 'sm' | 'md'; testid?: string }) {
  return (
    <div className="inline-flex w-full rounded-xl bg-slate-100 p-1 dark:bg-slate-800" role="radiogroup" data-testid={testid}>
      {options.map((o) => {
        const active = o.value === value
        const tone = { green: 'bg-green-600 text-white', amber: 'bg-amber-500 text-white', red: 'bg-red-600 text-white' }[o.tone || 'green']
        return (
          <button key={o.value} role="radio" aria-checked={active} disabled={disabled}
            onClick={() => !active && onChange(o.value)}
            className={clsx('flex-1 rounded-lg font-medium transition', size === 'sm' ? 'px-2 py-1 text-xs' : 'px-2 py-2 text-sm',
              active ? tone + ' shadow-sm' : 'text-slate-600 hover:bg-white/70 dark:text-slate-300 dark:hover:bg-slate-700',
              disabled && 'cursor-not-allowed opacity-60')}>
            {o.label}
          </button>
        )
      })}
    </div>
  )
}

export function Toggle({ on, onChange, label, disabled, testid }: { on: boolean; onChange: (v: boolean) => void; label: string; disabled?: boolean; testid?: string }) {
  return (
    <button role="switch" aria-checked={on} aria-label={label} disabled={disabled} data-testid={testid}
      onClick={() => onChange(!on)}
      className={clsx('relative inline-flex h-7 w-12 shrink-0 items-center rounded-full transition',
        on ? 'bg-green-600' : 'bg-slate-300 dark:bg-slate-700', disabled && 'opacity-50')}>
      <span className={clsx('inline-block h-5 w-5 transform rounded-full bg-white shadow transition', on ? 'translate-x-6' : 'translate-x-1')} />
    </button>
  )
}

export function Button({ children, onClick, tone = 'default', disabled, className, testid, type = 'button' }:
  { children: ReactNode; onClick?: () => void; tone?: 'default' | 'primary' | 'danger' | 'success'; disabled?: boolean; className?: string; testid?: string; type?: 'button' | 'submit' }) {
  const c = {
    default: 'border border-slate-300 bg-white text-slate-800 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100 dark:hover:bg-slate-700',
    primary: 'bg-blue-600 text-white hover:bg-blue-700',
    danger: 'bg-red-600 text-white hover:bg-red-700',
    success: 'bg-green-600 text-white hover:bg-green-700',
  }[tone]
  return (
    <button type={type} onClick={onClick} disabled={disabled} data-testid={testid}
      className={clsx('inline-flex items-center justify-center gap-1.5 rounded-xl px-3 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50', c, className)}>
      {children}
    </button>
  )
}

export function Collapse({ title, children, defaultOpen = false }: { title: ReactNode; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-800">
      <button className="flex w-full items-center justify-between px-3 py-2 text-left text-sm font-medium" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span>{title}</span>
        <ChevronDown size={16} className={clsx('transition', open && 'rotate-180')} />
      </button>
      {open && <div className="border-t border-slate-200 px-3 py-2 dark:border-slate-800">{children}</div>}
    </div>
  )
}

/** Responsive table: horizontal scroll on phones. */
export function Table({ head, rows, empty = 'Nothing yet.', testid }: { head: ReactNode[]; rows: ReactNode[][]; empty?: string; testid?: string }) {
  if (!rows.length) return <Empty>{empty}</Empty>
  return (
    <div className="-mx-4 overflow-x-auto px-4" data-testid={testid}>
      <table className="w-full min-w-max text-left text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
            {head.map((h, i) => <th key={i} className="whitespace-nowrap py-2 pr-4 font-medium">{h}</th>)}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
          {rows.map((r, i) => (
            <tr key={i}>{r.map((c, j) => <td key={j} className="whitespace-nowrap py-2 pr-4 tabular-nums">{c}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
