import { useState } from 'react'
import { fixed, inr, pnlClass, plain, when } from '../lib/format'
import { strategyTitle } from '../lib/names'
import { Badge, Empty } from './ui'

function sideBadge(side: unknown) {
  return <Badge tone={side === 'BUY' ? 'green' : side === 'SELL' ? 'red' : 'slate'}>{side === 'BUY' ? 'Long' : side === 'SELL' ? 'Short' : String(side ?? '?')}</Badge>
}

/** Closed trades: cards on phones, a table on wider screens. */
export function TradeList({ trades, showStrategy = true, limit = 25 }: { trades: any[]; showStrategy?: boolean; limit?: number }) {
  const [shown, setShown] = useState(limit)
  const rows = trades.slice(0, shown)
  if (!rows.length) return <Empty>No trades yet.</Empty>
  const more = trades.length - rows.length
  return (
    <>
      <ul className="divide-y divide-slate-100 md:hidden dark:divide-slate-800" data-testid="trade-cards">
        {rows.map((t, i) => (
          <li key={(t?.oid || '') + i} className="py-2">
            <div className="flex items-center justify-between gap-2">
              <span className="min-w-0 truncate text-sm">{sideBadge(t?.side)} <b>{t?.symbol}</b> × {plain(t?.qty)}</span>
              <span className={`shrink-0 text-sm font-semibold tabular-nums ${pnlClass(t?.net)}`}>{inr(t?.net, { signed: true })}</span>
            </div>
            <div className="mt-0.5 flex justify-between gap-2 text-xs text-slate-500 dark:text-slate-400">
              <span className="truncate">{when(t?.t)}{showStrategy ? ` · ${strategyTitle(String(t?.strategy || ''))}` : ''}</span>
              <span className="shrink-0">{fixed(t?.entry)} → {fixed(t?.exit)} · cost {inr(t?.cost)}</span>
            </div>
            {(t?.note || (t?.exit_kind && t.exit_kind !== 'normal')) && <div className="text-xs text-slate-500">{[t?.exit_kind !== 'normal' ? t?.exit_kind : null, t?.note].filter(Boolean).join(' · ')}</div>}
          </li>
        ))}
      </ul>
      <div className="hidden overflow-x-auto md:block">
        <table className="w-full text-left text-sm" data-testid="trade-table">
          <thead>
            <tr className="border-b border-slate-200 text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
              <th className="py-2 pr-3 font-medium">Closed</th>
              {showStrategy && <th className="py-2 pr-3 font-medium">Strategy</th>}
              <th className="py-2 pr-3 font-medium">Stock / contract</th>
              <th className="py-2 pr-3 font-medium">Side</th>
              <th className="py-2 pr-3 text-right font-medium">Qty</th>
              <th className="py-2 pr-3 text-right font-medium">Entry → exit</th>
              <th className="py-2 pr-3 text-right font-medium">Costs</th>
              <th className="py-2 pr-3 text-right font-medium">Net P&amp;L</th>
              <th className="py-2 pr-3 font-medium">Note</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
            {rows.map((t, i) => (
              <tr key={(t?.oid || '') + i}>
                <td className="whitespace-nowrap py-1.5 pr-3">{when(t?.t)}</td>
                {showStrategy && <td className="whitespace-nowrap py-1.5 pr-3">{strategyTitle(String(t?.strategy || ''))}</td>}
                <td className="py-1.5 pr-3 font-medium">{t?.symbol}</td>
                <td className="py-1.5 pr-3">{sideBadge(t?.side)}</td>
                <td className="py-1.5 pr-3 text-right tabular-nums">{plain(t?.qty)}</td>
                <td className="whitespace-nowrap py-1.5 pr-3 text-right tabular-nums">{fixed(t?.entry)} → {fixed(t?.exit)}</td>
                <td className="py-1.5 pr-3 text-right tabular-nums">{inr(t?.cost)}</td>
                <td className={`py-1.5 pr-3 text-right font-semibold tabular-nums ${pnlClass(t?.net)}`}>{inr(t?.net, { signed: true })}</td>
                <td className="max-w-[16rem] truncate py-1.5 pr-3 text-xs text-slate-500">{[t?.exit_kind !== 'normal' ? t?.exit_kind : null, t?.note].filter(Boolean).join(' · ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {more > 0 && (
        <button className="mt-2 w-full rounded-xl border border-slate-200 py-2 text-sm font-medium text-blue-600 dark:border-slate-800 dark:text-blue-400"
          onClick={() => setShown(shown + 50)} data-testid="trades-more">
          Show {Math.min(50, more)} more ({more} hidden)
        </button>
      )}
    </>
  )
}
