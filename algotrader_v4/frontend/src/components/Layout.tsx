import { ReactNode, useState } from 'react'
import clsx from 'clsx'
import { Activity, FlaskConical, Hammer, Home, Layers, ListOrdered, Menu, Moon, ShieldAlert, Sun, Ghost, Server } from 'lucide-react'
import { href } from '../lib/router'
import { stateLabel, stateTone } from '../lib/names'
import { Badge, Dot } from './ui'

export const NAV = [
  { id: '', label: 'Overview', icon: Home },
  { id: 'strategies', label: 'Strategies', icon: Layers },
  { id: 'trades', label: 'Positions & trades', short: 'Trades', icon: ListOrdered },
  { id: 'risk', label: 'Risk & controls', short: 'Risk', icon: ShieldAlert },
  { id: 'forward', label: 'Forward lab', icon: FlaskConical },
  { id: 'builder', label: 'Builder', icon: Hammer },
  { id: 'shadows', label: 'Shadow registry', icon: Ghost },
  { id: 'system', label: 'System', icon: Server },
]
const PHONE_TABS = ['', 'strategies', 'trades', 'risk']

export function useTheme(): [boolean, () => void] {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'))
  const toggle = () => {
    const d = !dark
    document.documentElement.classList.toggle('dark', d)
    try { localStorage.setItem('theme', d ? 'dark' : 'light') } catch { /* ignore */ }
    setDark(d)
  }
  return [dark, toggle]
}

export function Layout({ route, overview, children }: { route: string[]; overview: any; children: ReactNode }) {
  const [dark, toggleTheme] = useTheme()
  const section = route[0] || ''
  const inMore = !PHONE_TABS.includes(section) || section === 'more'
  const kite = overview?.kite
  const market = overview?.market
  const g = overview?.trading_state?.global
  const brake = overview?.trading_state?.brake
  const worst = g === 'HALTED' || brake === 'HALTED' ? 'HALTED' : g === 'REDUCE_ONLY' || brake === 'REDUCE_ONLY' ? 'REDUCE_ONLY' : g

  return (
    <div className="min-h-screen md:flex">
      {/* desktop sidebar */}
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r border-slate-200 bg-white p-3 md:flex dark:border-slate-800 dark:bg-slate-900">
        <a href={href()} className="mb-4 flex items-center gap-2 px-2 py-1 text-lg font-bold">
          <Activity className="text-green-600" size={22} /> AlgoTrader Pro
        </a>
        <nav className="flex-1 space-y-0.5">
          {NAV.map((n) => (
            <a key={n.id} href={href(...(n.id ? [n.id] : []))}
              className={clsx('flex items-center gap-2.5 rounded-xl px-3 py-2 text-sm font-medium',
                section === n.id ? 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900' : 'text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800')}>
              <n.icon size={17} /> {n.label}
            </a>
          ))}
        </nav>
        <button onClick={toggleTheme} className="flex items-center gap-2 rounded-xl px-3 py-2 text-sm text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800" data-testid="theme-toggle-desktop">
          {dark ? <Sun size={16} /> : <Moon size={16} />} {dark ? 'Light mode' : 'Dark mode'}
        </button>
      </aside>

      <div className="min-w-0 flex-1">
        {/* status bar */}
        <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/90 backdrop-blur dark:border-slate-800 dark:bg-slate-950/90" style={{ paddingTop: 'env(safe-area-inset-top)' }}>
          <div className="mx-auto flex max-w-6xl items-center gap-2 px-4 py-2 text-xs sm:gap-3">
            <a href={href()} className="flex items-center md:hidden" aria-label="Overview"><Activity className="text-green-600" size={20} /></a>
            <Badge tone="blue" className="font-semibold">{overview?.mode || 'PAPER'}</Badge>
            <span className="flex items-center gap-1 whitespace-nowrap" title={kite?.detail || kite?.error || ''} data-testid="hdr-kite">
              <Dot tone={kite?.connected ? 'green' : 'red'} /> Kite<span className="hidden sm:inline">{kite ? (kite.connected ? ' connected' : ' off') : ''}</span>
            </span>
            <span className="flex items-center gap-1 whitespace-nowrap">
              <Dot tone={market?.open ? 'green' : 'slate'} /> {market ? (market.open ? 'Open' : 'Closed') : '…'}<span className="hidden sm:inline">&nbsp;market</span>
            </span>
            {worst && <Badge tone={stateTone(worst)}>{stateLabel(worst)}</Badge>}
            <button onClick={toggleTheme} className="ml-auto rounded-lg p-1.5 hover:bg-slate-100 md:hidden dark:hover:bg-slate-800" aria-label="Toggle dark mode" data-testid="theme-toggle">
              {dark ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-4 pb-28 pt-4 md:pb-10">{children}</main>
      </div>

      {/* phone bottom tabs */}
      <nav data-testid="phone-nav" className="fixed inset-x-0 bottom-0 z-40 grid grid-cols-5 border-t border-slate-200 bg-white/95 backdrop-blur md:hidden dark:border-slate-800 dark:bg-slate-900/95" style={{ paddingBottom: 'env(safe-area-inset-bottom)' }}>
        {NAV.filter((n) => PHONE_TABS.includes(n.id)).map((n) => (
          <a key={n.id} href={href(...(n.id ? [n.id] : []))} data-testid={`tab-${n.id || 'overview'}`}
            className={clsx('flex flex-col items-center gap-0.5 py-2 text-[11px] font-medium', section === n.id ? 'text-blue-600 dark:text-blue-400' : 'text-slate-500 dark:text-slate-400')}>
            <n.icon size={20} /> {n.short || n.label}
          </a>
        ))}
        <a href={href('more')} data-testid="tab-more"
          className={clsx('flex flex-col items-center gap-0.5 py-2 text-[11px] font-medium', inMore ? 'text-blue-600 dark:text-blue-400' : 'text-slate-500 dark:text-slate-400')}>
          <Menu size={20} /> More
        </a>
      </nav>
    </div>
  )
}
