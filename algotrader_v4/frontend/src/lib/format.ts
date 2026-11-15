// Indian number / money / time formatting. Every function accepts null / undefined / garbage and returns '—'.

export const DASH = '—'

export function num(x: unknown): number | null {
  if (x === null || x === undefined || x === '') return null
  const n = typeof x === 'number' ? x : Number(x)
  return Number.isFinite(n) ? n : null
}

const inr0 = new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 })
const inr2 = new Intl.NumberFormat('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })

/** ₹1,23,456 (no decimals by default). signed=true adds + for gains. */
export function inr(x: unknown, opts: { signed?: boolean; decimals?: boolean } = {}): string {
  const n = num(x)
  if (n === null) return DASH
  const body = (opts.decimals ? inr2 : inr0).format(Math.abs(n))
  const sign = n < 0 ? '−' : opts.signed && n > 0 ? '+' : ''
  return `${sign}₹${body}`
}

/** Compact lakh / crore: ₹4.00 L, ₹1.12 Cr, ₹45,000 below a lakh. */
export function lakh(x: unknown, opts: { signed?: boolean } = {}): string {
  const n = num(x)
  if (n === null) return DASH
  const a = Math.abs(n)
  const sign = n < 0 ? '−' : opts.signed && n > 0 ? '+' : ''
  if (a >= 1e7) return `${sign}₹${(a / 1e7).toFixed(2)} Cr`
  if (a >= 1e5) return `${sign}₹${(a / 1e5).toFixed(2)} L`
  return `${sign}₹${inr0.format(a)}`
}

export function plain(x: unknown, digits = 0): string {
  const n = num(x)
  if (n === null) return DASH
  return new Intl.NumberFormat('en-IN', { maximumFractionDigits: digits, minimumFractionDigits: 0 }).format(n)
}

export function fixed(x: unknown, digits = 2): string {
  const n = num(x)
  return n === null ? DASH : n.toFixed(digits)
}

export function pct(x: unknown, digits = 1): string {
  const n = num(x)
  return n === null ? DASH : `${n.toFixed(digits)} %`
}

export function pnlClass(x: unknown): string {
  const n = num(x)
  if (n === null || n === 0) return ''
  return n > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
}

function parse(t: unknown): Date | null {
  if (typeof t !== 'string' || !t) return null
  const d = new Date(t.length === 10 ? `${t}T00:00:00+05:30` : t)
  return isNaN(d.getTime()) ? null : d
}

const tz = 'Asia/Kolkata'

/** 09:20 (IST) */
export function hhmm(t: unknown, seconds = false): string {
  const d = parse(t)
  if (!d) return DASH
  return d.toLocaleTimeString('en-IN', { timeZone: tz, hour: '2-digit', minute: '2-digit',
    second: seconds ? '2-digit' : undefined, hour12: false })
}

/** 16 Nov, 09:20 — or just "09:20" when it is today */
export function when(t: unknown): string {
  const d = parse(t)
  if (!d) return DASH
  const day = d.toLocaleDateString('en-IN', { timeZone: tz, day: 'numeric', month: 'short' })
  const today = new Date().toLocaleDateString('en-IN', { timeZone: tz, day: 'numeric', month: 'short' })
  if (typeof t === 'string' && t.length === 10) return d.toLocaleDateString('en-IN', { timeZone: tz, day: 'numeric', month: 'short', year: 'numeric' })
  return day === today ? `Today ${hhmm(t)}` : `${day}, ${hhmm(t)}`
}

export function dateOnly(t: unknown): string {
  const d = parse(t)
  if (!d) return DASH
  return d.toLocaleDateString('en-IN', { timeZone: tz, day: 'numeric', month: 'short', year: 'numeric' })
}

/** "in 5 min", "in 2 h 10 min", "tomorrow 09:15" */
export function relative(t: unknown): string {
  const d = parse(t)
  if (!d) return DASH
  const mins = Math.round((d.getTime() - Date.now()) / 60000)
  if (mins < -1) return when(t)
  if (mins <= 0) return 'now'
  if (mins < 60) return `in ${mins} min`
  if (mins < 24 * 60) return `in ${Math.floor(mins / 60)} h ${mins % 60} min`
  return when(t)
}

export function titleCase(s: string): string {
  return s.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}
