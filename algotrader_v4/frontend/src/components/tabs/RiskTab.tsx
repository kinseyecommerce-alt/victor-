import React, { useEffect } from 'react'
import { useStore } from '../../store'
import { api } from '../../api/client'
import { Pnl } from '../ui'
import { RadialBarChart, RadialBar, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'

export default function RiskTab() {
  const { riskStatus, setRiskStatus } = useStore()

  useEffect(() => {
    api.riskStatus().then(r => setRiskStatus(r.data)).catch(() => {})
    const t = setInterval(() => api.riskStatus().then(r => setRiskStatus(r.data)).catch(() => {}), 5000)
    return () => clearInterval(t)
  }, [])

  if (!riskStatus) {
    return <div className="flex items-center justify-center h-full text-slate-400 text-sm">Loading risk data…</div>
  }

  const lossUsed = Math.min(100, Math.abs(Math.min(0, riskStatus.daily_pnl)) / riskStatus.max_daily_loss * 100)
  const posUsed  = (riskStatus.open_positions / riskStatus.max_open_positions) * 100
  const pnlColor = riskStatus.daily_pnl >= 0 ? '#16A34A' : '#DC2626'
  const limitColor = lossUsed > 80 ? '#DC2626' : lossUsed > 50 ? '#F59E0B' : '#16A34A'

  const gaugeData = [{ value: lossUsed, fill: limitColor }, { value: 100 - lossUsed, fill: '#F1F5F9' }]

  return (
    <div className="h-full overflow-auto p-4">
      {riskStatus.is_halted && (
        <div className="mb-4 bg-red-50 border border-red-200 rounded-xl p-3 flex items-center gap-2">
          <span className="text-2xl">🛑</span>
          <div>
            <div className="text-sm font-bold text-red-800">Trading HALTED</div>
            <div className="text-xs text-red-600">Daily loss limit reached. Resume via SEBI tab.</div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-4 mb-4">
        {/* Daily P&L */}
        <div className="bg-white border border-slate-200 rounded-xl p-4 text-center">
          <div className="text-xs text-slate-500 mb-1">Daily P&L</div>
          <div className="text-2xl font-bold font-mono" style={{ color: pnlColor }}>
            {riskStatus.daily_pnl >= 0 ? '+' : ''}₹{Math.abs(riskStatus.daily_pnl).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
          </div>
          <div className="text-xs text-slate-400 mt-1">Limit: ₹{riskStatus.max_daily_loss.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</div>
        </div>

        {/* Open Positions */}
        <div className="bg-white border border-slate-200 rounded-xl p-4 text-center">
          <div className="text-xs text-slate-500 mb-1">Open Positions</div>
          <div className="text-2xl font-bold font-mono text-slate-900">
            {riskStatus.open_positions} / {riskStatus.max_open_positions}
          </div>
          <div className="mt-2 h-1.5 rounded-full bg-slate-100 overflow-hidden">
            <div className="h-full rounded-full bg-indigo-500 transition-all" style={{ width: `${posUsed}%` }} />
          </div>
        </div>
      </div>

      {/* Loss limit gauge */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 mb-4">
        <div className="text-xs font-medium text-slate-700 mb-3">Loss Limit Usage</div>
        <div className="flex items-center gap-4">
          <div className="w-32 h-16">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={gaugeData} startAngle={180} endAngle={0}
                  innerRadius="55%" outerRadius="80%" dataKey="value" strokeWidth={0}>
                  {gaugeData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div>
            <div className="text-3xl font-bold font-mono" style={{ color: limitColor }}>
              {lossUsed.toFixed(0)}%
            </div>
            <div className="text-xs text-slate-500">
              ₹{Math.abs(Math.min(0, riskStatus.daily_pnl)).toLocaleString('en-IN', { maximumFractionDigits: 0 })} of
              ₹{riskStatus.max_daily_loss.toLocaleString('en-IN', { maximumFractionDigits: 0 })} used
            </div>
            {lossUsed > 80 && (
              <div className="text-xs text-red-600 font-medium mt-1">⚠ Approaching limit</div>
            )}
          </div>
        </div>
      </div>

      {/* Trades today */}
      <div className="bg-white border border-slate-200 rounded-xl p-4">
        <div className="text-xs font-medium text-slate-700 mb-2">Trades Today</div>
        <div className="text-2xl font-bold font-mono text-slate-900">{riskStatus.trades_today ?? '—'}</div>
      </div>
    </div>
  )
}
