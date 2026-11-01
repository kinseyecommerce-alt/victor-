import React, { useEffect } from 'react'
import { useStore } from '../../store'
import { api } from '../../api/client'
import { Btn, Badge } from '../ui'

const AGENT_LABELS: Record<string, { emoji: string; desc: string }> = {
  intraday: { emoji: '⚡', desc: 'VWAP+EMA momentum, MIS, 8 trades/day' },
  fno:      { emoji: '📊', desc: 'Options CE/PE, NRML, 4 trades/day' },
  swing:    { emoji: '🌊', desc: 'EMA200 trend, CNC, 3 trades/day' },
  scalping: { emoji: '🎯', desc: 'Micro EMA cross, MIS, 20 trades/day' },
}

export default function AgentsTab() {
  const { agents, setAgents, addToast } = useStore()

  useEffect(() => {
    const fetch = () => api.agents().then(r => setAgents(r.data)).catch(() => {})
    fetch()
    const t = setInterval(fetch, 5000)
    return () => clearInterval(t)
  }, [])

  const handlePause = async (name: string) => {
    try {
      await api.pauseAgent(name)
      addToast(`Agent ${name} paused`, 'info')
    } catch (e: any) {
      addToast(e.response?.data?.detail || 'Pause failed', 'error')
    }
  }

  const handleResume = async (name: string) => {
    try {
      await api.resumeAgent(name)
      addToast(`Agent ${name} resumed`, 'buy')
    } catch (e: any) {
      addToast(e.response?.data?.detail || 'Resume failed', 'error')
    }
  }

  const agentNames = Object.keys(AGENT_LABELS)

  return (
    <div className="h-full overflow-auto p-4 grid grid-cols-2 gap-4 auto-rows-min">
      {agentNames.map(name => {
        const agent = agents[name]
        const running = agent?.running ?? false
        const meta = AGENT_LABELS[name]

        return (
          <div key={name} className={`bg-white border rounded-xl p-4 transition-all ${
            running ? 'border-green-200 shadow-sm' : 'border-slate-200'
          }`}>
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-2">
                <span className="text-2xl">{meta.emoji}</span>
                <div>
                  <div className="text-sm font-bold text-slate-900 capitalize">{name}</div>
                  <div className="text-xs text-slate-500">{meta.desc}</div>
                </div>
              </div>
              <span className={`w-2.5 h-2.5 rounded-full mt-1 ${running ? 'bg-green-500' : 'bg-slate-300'}`} />
            </div>

            {agent && (
              <div className="grid grid-cols-2 gap-2 mb-3 text-xs">
                {agent.trades_today !== undefined && (
                  <div>
                    <div className="text-slate-400">Trades</div>
                    <div className="font-mono font-medium text-slate-800">{agent.trades_today}</div>
                  </div>
                )}
                {agent.win_rate !== undefined && (
                  <div>
                    <div className="text-slate-400">Win Rate</div>
                    <div className={`font-mono font-medium ${agent.win_rate >= 55 ? 'text-green-600' : 'text-red-600'}`}>
                      {agent.win_rate.toFixed(1)}%
                    </div>
                  </div>
                )}
                {agent.last_signal && (
                  <div className="col-span-2">
                    <div className="text-slate-400">Last Signal</div>
                    <div className="font-medium text-slate-700 truncate">{agent.last_signal}</div>
                  </div>
                )}
              </div>
            )}

            <div className="flex gap-2">
              {running ? (
                <Btn variant="outline" size="sm" className="flex-1 text-xs" onClick={() => handlePause(name)}>
                  Pause
                </Btn>
              ) : (
                <Btn variant="primary" size="sm" className="flex-1 text-xs" onClick={() => handleResume(name)}>
                  Resume
                </Btn>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
