import React, { useEffect, useState } from 'react'
import { clsx } from 'clsx'
import Header from './components/Header'
import Watchlist from './components/Watchlist'
import MainChart from './components/MainChart'
import OrderPanel from './components/OrderPanel'
import PositionsTab from './components/tabs/PositionsTab'
import OrdersTab from './components/tabs/OrdersTab'
import BracketsTab from './components/tabs/BracketsTab'
import RiskTab from './components/tabs/RiskTab'
import AgentsTab from './components/tabs/AgentsTab'
import SebiTab from './components/tabs/SebiTab'
import { connectWS } from './ws/websocket'
import { useStore } from './store'
import type { TabId } from './types'

const TABS: { id: TabId; label: string; emoji: string }[] = [
  { id: 'positions', label: 'Positions', emoji: '📋' },
  { id: 'orders',   label: 'Orders',    emoji: '📝' },
  { id: 'brackets', label: 'Brackets',  emoji: '🎯' },
  { id: 'risk',     label: 'Risk',      emoji: '⚠️' },
  { id: 'agents',   label: 'Agents',    emoji: '🤖' },
  { id: 'sebi',     label: 'SEBI',      emoji: '🛡️' },
]

const TAB_COMPONENTS: Record<TabId, React.ComponentType> = {
  positions: PositionsTab,
  orders:    OrdersTab,
  brackets:  BracketsTab,
  risk:      RiskTab,
  agents:    AgentsTab,
  sebi:      SebiTab,
}

// Toast notifications
function Toasts() {
  const { toasts, removeToast } = useStore()
  const colors = { buy: 'bg-green-600', sell: 'bg-red-600', info: 'bg-indigo-600', error: 'bg-red-700' }
  return (
    <div className="fixed bottom-4 right-4 flex flex-col gap-2 z-50">
      {toasts.map(t => (
        <div
          key={t.id}
          onClick={() => removeToast(t.id)}
          className={clsx(
            'flex items-center gap-3 px-4 py-3 rounded-xl text-white text-sm font-medium shadow-lg cursor-pointer',
            'animate-in slide-in-from-right-4 duration-200',
            colors[t.type],
          )}
        >
          {t.msg}
        </div>
      ))}
    </div>
  )
}

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('positions')
  const { riskStatus, positions } = useStore()
  const TabComponent = TAB_COMPONENTS[activeTab]

  useEffect(() => {
    connectWS()
    const ping = setInterval(() => {
      import('./ws/websocket').then(m => m.sendPing())
    }, 30000)
    return () => clearInterval(ping)
  }, [])

  const openPositionCount = positions.filter(p => p.quantity !== 0).length
  const isHalted = riskStatus?.is_halted

  return (
    <div className="flex flex-col h-screen bg-slate-50 overflow-hidden">
      <Header />

      {isHalted && (
        <div className="bg-red-600 text-white text-xs text-center py-1.5 font-medium shrink-0">
          ⛔ Trading HALTED — daily loss limit reached. Go to SEBI tab to resume.
        </div>
      )}

      <div className="flex flex-1 min-h-0">
        {/* Watchlist sidebar */}
        <div className="w-52 shrink-0 bg-slate-50 border-r border-slate-200 flex flex-col">
          <div className="px-3 py-2 border-b border-slate-200">
            <span className="text-xs font-semibold text-slate-600 uppercase tracking-wide">Watchlist</span>
          </div>
          <div className="flex-1 min-h-0">
            <Watchlist />
          </div>
        </div>

        {/* Main content area */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Chart area */}
          <div className="flex flex-1 min-h-0">
            <div className="flex-1 border-b border-slate-200 min-w-0">
              <MainChart />
            </div>

            {/* Order panel */}
            <div className="w-64 shrink-0 border-l border-slate-200 bg-white overflow-hidden">
              <OrderPanel />
            </div>
          </div>

          {/* Bottom tabs */}
          <div className="h-56 shrink-0 flex flex-col border-t border-slate-200 bg-white">
            <div className="flex border-b border-slate-100 overflow-x-auto shrink-0">
              {TABS.map(tab => {
                const badge = tab.id === 'positions' && openPositionCount > 0 ? openPositionCount : undefined
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={clsx(
                      'flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium whitespace-nowrap transition-colors border-b-2',
                      activeTab === tab.id
                        ? 'border-indigo-600 text-indigo-600 bg-indigo-50/50'
                        : 'border-transparent text-slate-500 hover:text-slate-700 hover:bg-slate-50',
                    )}
                  >
                    <span>{tab.emoji}</span>
                    {tab.label}
                    {badge !== undefined && (
                      <span className="bg-indigo-600 text-white rounded-full text-xs w-4 h-4 flex items-center justify-center">
                        {badge}
                      </span>
                    )}
                  </button>
                )
              })}
            </div>
            <div className="flex-1 min-h-0">
              <TabComponent />
            </div>
          </div>
        </div>
      </div>

      <Toasts />
    </div>
  )
}
