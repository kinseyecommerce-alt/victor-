import React, { useEffect, memo } from 'react'
import { TrendingUp, TrendingDown } from 'lucide-react'
import { useStore } from '../../store'
import { api } from '../../api/client'
import { clsx } from 'clsx'
import type { TickData } from '../../types'

// Ticks arrive for every symbol about once a second. Each row is memoised on its own tick and
// sparkline so a tick for RELIANCE re-renders only the RELIANCE row, and the sparkline is a
// plain SVG polyline: a charting library here cost ~1 s of CPU per second with 10 symbols.
function Sparkline({ values, up }: { values: number[]; up: boolean }) {
  if (values.length < 4) return null
  const w = 64, h = 32
  const min = Math.min(...values), max = Math.max(...values)
  const span = max - min || 1
  const pts = values.map((v, i) =>
    `${(i / (values.length - 1) * w).toFixed(1)},${(h - 2 - (v - min) / span * (h - 4)).toFixed(1)}`).join(' ')
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="shrink-0" aria-hidden="true">
      <polyline points={pts} fill="none" stroke={up ? '#16A34A' : '#DC2626'} strokeWidth={1.5}
                strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  )
}

const Row = memo(function Row({ sym, tick, spark, selected, onSelect }:
  { sym: string; tick: TickData; spark: number[]; selected: boolean; onSelect: (s: string) => void }) {
  const up = tick.change_pct >= 0
  return (
    <button
      onClick={() => onSelect(sym)}
      className={clsx(
        'w-full flex flex-col px-3 py-2.5 border-b border-slate-100 transition-colors text-left',
        selected ? 'bg-indigo-50 border-l-2 border-l-indigo-500' : 'hover:bg-slate-50'
      )}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          {up
            ? <TrendingUp className="w-3 h-3 text-green-500" />
            : <TrendingDown className="w-3 h-3 text-red-500" />}
          <span className="text-xs font-semibold text-slate-900">{sym}</span>
        </div>
        <span className={clsx('text-xs font-mono font-semibold tabular-nums', up ? 'text-green-600' : 'text-red-600')}>
          {up ? '+' : ''}{tick.change_pct.toFixed(2)}%
        </span>
      </div>

      <div className="flex items-end justify-between mt-1">
        <div>
          <div className="text-sm font-bold font-mono text-slate-900 tabular-nums">
            ₹{tick.ltp.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
          <div className="text-xs text-slate-400 mt-0.5">
            H: {tick.day_high.toFixed(1)} L: {tick.day_low.toFixed(1)}
          </div>
        </div>
        <Sparkline values={spark} up={up} />
      </div>

      <div className="flex gap-2 mt-1">
        <span className={clsx(
          'text-xs px-1.5 rounded font-medium',
          tick.trend === 'UP' ? 'bg-green-100 text-green-700' :
          tick.trend === 'DOWN' ? 'bg-red-100 text-red-700' :
          'bg-slate-100 text-slate-600'
        )}>{tick.trend}</span>
        <span className="text-xs text-slate-400 font-mono">RSI {(tick.rsi ?? 50).toFixed(0)}</span>
      </div>
    </button>
  )
})

export default function Watchlist() {
  const { ticks, sparklines, selectedSymbol, setSelectedSymbol, setTick } = useStore()

  useEffect(() => {
    // Initial fetch
    api.marketLive().then(r => {
      const data = r.data as Record<string, any>
      Object.entries(data).forEach(([sym, info]: [string, any]) => {
        // /market/live names RSI `rsi_14`; WebSocket ticks (TickData) call it `rsi`
        setTick({ symbol: sym, ...info, rsi: info.rsi ?? info.rsi_14 } as any)
      })
      if (!selectedSymbol && Object.keys(data).length > 0) {
        setSelectedSymbol(Object.keys(data)[0])
      }
    }).catch(() => {})
  }, [])

  const symbols = Object.keys(ticks).sort()

  if (symbols.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-400 p-4 text-center">
        <div className="text-4xl mb-2">📊</div>
        <div className="text-sm font-medium">No symbols</div>
        <div className="text-xs mt-1">Connect Kite (Settings → API Keys) or start the bot</div>
      </div>
    )
  }

  return (
    <div className="overflow-y-auto h-full">
      {symbols.map(sym => (
        <Row key={sym} sym={sym} tick={ticks[sym]} spark={sparklines[sym] || EMPTY}
             selected={sym === selectedSymbol} onSelect={setSelectedSymbol} />
      ))}
    </div>
  )
}

const EMPTY: number[] = []
