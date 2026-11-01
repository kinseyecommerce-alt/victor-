import React, { useEffect } from 'react'
import { useStore } from '../../store'
import { api } from '../../api/client'
import { Badge } from '../ui'
import type { Bracket } from '../../types'

const statusColor = (s: Bracket['status']) => {
  const m: Record<string, string> = {
    ACTIVE: 'bg-green-100 text-green-800',
    SL_HIT: 'bg-red-100 text-red-800',
    TARGET_HIT: 'bg-blue-100 text-blue-800',
    PENDING: 'bg-amber-100 text-amber-800',
    CANCELLED: 'bg-slate-100 text-slate-600',
    FAILED: 'bg-red-100 text-red-800',
  }
  return m[s] || 'bg-slate-100 text-slate-600'
}

export default function BracketsTab() {
  const { brackets, setBrackets } = useStore()

  useEffect(() => {
    api.brackets().then(r => setBrackets(r.data.brackets || [])).catch(() => {})
    const t = setInterval(() => api.brackets().then(r => setBrackets(r.data.brackets || [])).catch(() => {}), 5000)
    return () => clearInterval(t)
  }, [])

  if (brackets.length === 0) {
    return <div className="flex items-center justify-center h-full text-slate-400 text-sm">No brackets yet</div>
  }

  return (
    <div className="h-full overflow-auto p-4 space-y-3">
      {brackets.map(b => {
        const range = b.target_1 && b.stop_loss ? b.target_1 - b.stop_loss : 0
        const progress = range > 0 && b.entry_price
          ? Math.max(0, Math.min(100, ((b.entry_price - b.stop_loss!) / range) * 100))
          : 50

        return (
          <div key={b.bracket_id} className="bg-white border border-slate-200 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <div>
                <span className="font-bold text-slate-900">{b.symbol}</span>
                <span className="ml-2 text-xs text-slate-500">{b.strategy} · {b.side}</span>
              </div>
              <span className={`text-xs px-2 py-0.5 rounded-full font-semibold ${statusColor(b.status)}`}>
                {b.status}
              </span>
            </div>
            <div className="grid grid-cols-4 gap-3 text-xs mb-3">
              <div>
                <div className="text-slate-400">Entry</div>
                <div className="font-mono font-medium text-slate-900">₹{b.entry_price.toFixed(2)}</div>
              </div>
              <div>
                <div className="text-slate-400">Stop Loss</div>
                <div className="font-mono font-medium text-red-600">{b.stop_loss ? `₹${b.stop_loss.toFixed(2)}` : '—'}</div>
              </div>
              <div>
                <div className="text-slate-400">Target 1</div>
                <div className="font-mono font-medium text-green-600">{b.target_1 ? `₹${b.target_1.toFixed(2)}` : '—'}</div>
              </div>
              <div>
                <div className="text-slate-400">Qty</div>
                <div className="font-mono font-medium text-slate-900">{b.quantity}</div>
              </div>
            </div>
            {range > 0 && (
              <div className="h-1.5 rounded-full bg-slate-100 overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-red-400 via-amber-400 to-green-500 transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
