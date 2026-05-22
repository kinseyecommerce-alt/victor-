import { create } from 'zustand'
import type { TickData, Position, Order, Bracket, RiskStatus, Agent, BotStatus, HealthData } from '../types'

interface AppStore {
  // Connection
  apiKey: string
  apiBase: string
  wsConnected: boolean
  setApiKey: (k: string) => void
  setApiBase: (b: string) => void
  setWsConnected: (v: boolean) => void

  // Health
  health: HealthData | null
  setHealth: (h: HealthData) => void

  // Ticks
  ticks: Record<string, TickData>
  sparklines: Record<string, number[]>   // 60-point rolling LTP
  setTick: (t: TickData) => void

  // Selected symbol for chart
  selectedSymbol: string
  setSelectedSymbol: (s: string) => void

  // Bot
  botStatus: BotStatus | null
  setBotStatus: (b: BotStatus | null) => void

  // Portfolio
  positions: Position[]
  orders: Order[]
  setPositions: (p: Position[]) => void
  setOrders: (o: Order[]) => void

  // Brackets
  brackets: Bracket[]
  setBrackets: (b: Bracket[]) => void

  // Risk
  riskStatus: RiskStatus | null
  setRiskStatus: (r: RiskStatus) => void

  // Agents
  agents: Record<string, Agent>
  setAgents: (a: Record<string, Agent>) => void

  // Toasts
  toasts: { id: string; msg: string; type: 'buy' | 'sell' | 'info' | 'error' }[]
  addToast: (msg: string, type?: 'buy' | 'sell' | 'info' | 'error') => void
  removeToast: (id: string) => void
}

export const useStore = create<AppStore>((set, get) => ({
  apiKey: localStorage.getItem('api_key') || '',
  // Priority: localStorage override → build-time env var → localhost fallback
  apiBase: localStorage.getItem('api_base') || import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  wsConnected: false,
  setApiKey: (k) => { localStorage.setItem('api_key', k); set({ apiKey: k }) },
  setApiBase: (b) => { localStorage.setItem('api_base', b); set({ apiBase: b }) },
  setWsConnected: (v) => set({ wsConnected: v }),

  health: null,
  setHealth: (h) => set({ health: h }),

  ticks: {},
  sparklines: {},
  setTick: (t) => set((state) => {
    const prev = state.sparklines[t.symbol] || []
    const spark = [...prev, t.ltp].slice(-60)
    return {
      ticks: { ...state.ticks, [t.symbol]: t },
      sparklines: { ...state.sparklines, [t.symbol]: spark },
    }
  }),

  selectedSymbol: '',
  setSelectedSymbol: (s) => set({ selectedSymbol: s }),

  botStatus: null,
  setBotStatus: (b) => set({ botStatus: b }),

  positions: [],
  orders: [],
  setPositions: (p) => set({ positions: p }),
  setOrders: (o) => set({ orders: o }),

  brackets: [],
  setBrackets: (b) => set({ brackets: b }),

  riskStatus: null,
  setRiskStatus: (r) => set({ riskStatus: r }),

  agents: {},
  setAgents: (a) => set({ agents: a }),

  toasts: [],
  addToast: (msg, type = 'info') => {
    const id = Math.random().toString(36).slice(2)
    set((s) => ({ toasts: [...s.toasts.slice(-4), { id, msg, type }] }))
    setTimeout(() => get().removeToast(id), 4000)
  },
  removeToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
}))
