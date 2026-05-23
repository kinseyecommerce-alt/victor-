import React, { useEffect, useRef, useCallback } from 'react'
import { createChart, IChartApi, ISeriesApi, ColorType } from 'lightweight-charts'
import { useStore } from '../../store'
import { api } from '../../api/client'

const CHART_OPTS = {
  layout: {
    background: { type: ColorType.Solid, color: '#FFFFFF' },
    textColor: '#64748B',
    fontSize: 11,
  },
  grid: {
    vertLines: { color: '#F1F5F9' },
    horzLines: { color: '#F1F5F9' },
  },
  crosshair: { mode: 1 },
  timeScale: {
    borderColor: '#E2E8F0',
    timeVisible: true,
    secondsVisible: false,
  },
  rightPriceScale: { borderColor: '#E2E8F0' },
  handleScroll: true,
  handleScale: true,
}

export default function MainChart() {
  const { selectedSymbol, ticks } = useStore()
  const chartRef = useRef<HTMLDivElement>(null)
  const chart    = useRef<IChartApi | null>(null)
  const candles  = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const ema9Line = useRef<ISeriesApi<'Line'> | null>(null)
  const ema21Line= useRef<ISeriesApi<'Line'> | null>(null)
  const volSeries= useRef<ISeriesApi<'Histogram'> | null>(null)

  // Init chart
  useEffect(() => {
    if (!chartRef.current) return
    const c = createChart(chartRef.current, { ...CHART_OPTS, width: chartRef.current.clientWidth, height: 340 })
    chart.current = c

    candles.current = c.addCandlestickSeries({
      upColor:   '#16A34A',
      downColor: '#DC2626',
      borderUpColor:   '#16A34A',
      borderDownColor: '#DC2626',
      wickUpColor:   '#16A34A',
      wickDownColor: '#DC2626',
    })

    ema9Line.current  = c.addLineSeries({ color: '#4F46E5', lineWidth: 1, priceLineVisible: false, lastValueVisible: false })
    ema21Line.current = c.addLineSeries({ color: '#F59E0B', lineWidth: 1, priceLineVisible: false, lastValueVisible: false })

    const ro = new ResizeObserver(entries => {
      for (const e of entries) c.applyOptions({ width: e.contentRect.width })
    })
    ro.observe(chartRef.current)
    return () => { ro.disconnect(); c.remove() }
  }, [])

  // Load candles when symbol changes
  useEffect(() => {
    if (!selectedSymbol || !candles.current) return
    // Clear
    candles.current.setData([])
    ema9Line.current?.setData([])
    ema21Line.current?.setData([])
    // Load from market/live endpoint (uses current tick + history from buffer)
    api.marketLive().then(r => {
      const sym = r.data[selectedSymbol]
      if (sym) {
        chart.current?.applyOptions({ watermark: { visible: true, text: selectedSymbol, color: '#E2E8F0', fontSize: 48 } })
      }
    }).catch(() => {})
  }, [selectedSymbol])

  // Update last price on new tick
  useEffect(() => {
    if (!selectedSymbol) return
    const tick = ticks[selectedSymbol]
    if (!tick || !candles.current) return

    const now = Math.floor(Date.now() / 1000)
    const barTs = Math.floor(now / 60) * 60

    candles.current.update({
      time: barTs as any,
      open:  tick.ltp,
      high:  tick.day_high,
      low:   tick.day_low,
      close: tick.ltp,
    })

    if (tick.ema9) ema9Line.current?.update({ time: barTs as any, value: tick.ema9 })
    if (tick.ema21) ema21Line.current?.update({ time: barTs as any, value: tick.ema21 })
  }, [ticks, selectedSymbol])

  const tick = selectedSymbol ? ticks[selectedSymbol] : null

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Symbol header */}
      <div className="flex items-center gap-4 px-4 py-2 border-b border-slate-100">
        {tick ? (
          <>
            <span className="text-lg font-bold text-slate-900">{selectedSymbol}</span>
            <span className={`text-xl font-mono font-bold ${tick.change_pct >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              ₹{tick.ltp.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
            </span>
            <span className={`text-sm font-mono ${tick.change_pct >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {tick.change_pct >= 0 ? '+' : ''}{tick.change_pct.toFixed(2)}%
            </span>
            <div className="flex gap-3 ml-2 text-xs text-slate-500">
              <span>H: <span className="text-slate-700 font-mono">{tick.day_high.toFixed(2)}</span></span>
              <span>L: <span className="text-slate-700 font-mono">{tick.day_low.toFixed(2)}</span></span>
              <span>VWAP: <span className="text-slate-700 font-mono">{tick.vwap.toFixed(2)}</span></span>
              <span>RSI: <span className={`font-mono font-medium ${tick.rsi > 70 ? 'text-red-600' : tick.rsi < 30 ? 'text-green-600' : 'text-slate-700'}`}>{tick.rsi.toFixed(1)}</span></span>
            </div>
            <div className="ml-auto flex gap-2 text-xs">
              <span className={`px-1.5 py-0.5 rounded font-medium ${tick.trend === 'UP' ? 'bg-green-100 text-green-700' : tick.trend === 'DOWN' ? 'bg-red-100 text-red-700' : 'bg-slate-100 text-slate-600'}`}>
                {tick.trend}
              </span>
              <span className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 font-medium">{tick.volatility}</span>
              <span className="text-slate-400 font-mono">{tick.source}</span>
            </div>
          </>
        ) : (
          <span className="text-slate-400 text-sm">Select a symbol from the watchlist</span>
        )}
      </div>

      {/* Chart */}
      <div ref={chartRef} className="flex-1 min-h-0" />

      {/* Indicator row */}
      {tick && (
        <div className="flex gap-4 px-4 py-2 bg-slate-50 border-t border-slate-100 text-xs font-mono text-slate-600">
          <span>EMA9: <span className="text-indigo-600 font-medium">{tick.ema9.toFixed(2)}</span></span>
          <span>EMA21: <span className="text-amber-600 font-medium">{tick.ema21.toFixed(2)}</span></span>
          <span>MACD: <span className={`font-medium ${tick.macd_hist >= 0 ? 'text-green-600' : 'text-red-600'}`}>{tick.macd_hist.toFixed(4)}</span></span>
          <span>Vol×: <span className="text-slate-700">{tick.vol_ratio.toFixed(2)}</span></span>
          <span>Spread: <span className="text-slate-700">{(tick.ask - tick.bid).toFixed(2)}</span></span>
        </div>
      )}
    </div>
  )
}
