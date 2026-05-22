import React, { useEffect } from 'react'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { useStore } from '../../store'
import { api } from '../../api/client'
import { clsx } from 'clsx'
import { LineChart, Line, ResponsiveContainer } from 'recharts'

export default function Watchlist() {
  const { ticks, sparklines, selectedSymbol, setSelectedSymbol, setTick } = useStore()

  useEffect(() => {
    // Initial fetch
    api.marketLive().then(r => {
      const data = r.data as Record<string, any>
      Object.entries(data).forEach(([sym, info]: [string, any]) => {
        setTick({ symbol: sym, ...info } as any)
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
        <div className="text-xs mt-1">Start the bot to subscribe symbols</div>
      </div>
    )
  }

  return (
    <div className="overflow-y-auto h-full">
      {symbols.map(sym => {
        const tick = ticks[sym]
        const spark = sparklines[sym] || []
        const up = tick.change_pct >= 0
        const selected = sym === selectedSymbol
        const sparkData = spark.map(v => ({ v }))

        return (
          <button
            key={sym}
            onClick={() => setSelectedSymbol(sym)}
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
              <span className={clsx(
                'text-xs font-mono font-semibold tabular-nums',
                up ? 'text-green-600' : 'text-red-600'
              )}>
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

              {sparkData.length > 3 && (
                <div className="w-16 h-8">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={sparkData}>
                      <Line
                        type="monotone" dataKey="v"
                        stroke={up ? '#16A34A' : '#DC2626'}
                        strokeWidth={1.5} dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            <div className="flex gap-2 mt-1">
              <span className={clsx(
                'text-xs px-1.5 rounded font-medium',
                tick.trend === 'UP' ? 'bg-green-100 text-green-700' :
                tick.trend === 'DOWN' ? 'bg-red-100 text-red-700' :
                'bg-slate-100 text-slate-600'
              )}>{tick.trend}</span>
              <span className="text-xs text-slate-400 font-mono">RSI {tick.rsi.toFixed(0)}</span>
            </div>
          </button>
        )
      })}
    </div>
  )
}
