import { useStore } from '../store'
import type { TickData } from '../types'

let ws: WebSocket | null = null
let reconnectTimer: ReturnType<typeof setTimeout> | null = null

export function connectWS() {
  const { apiBase, apiKey, setWsConnected, setTick, addToast } = useStore.getState()
  if (ws && ws.readyState === WebSocket.OPEN) return

  const wsBase = apiBase.replace(/^http/, 'ws')
  const url = `${wsBase}/ws${apiKey ? `?token=${apiKey}` : ''}`

  ws = new WebSocket(url)

  ws.onopen = () => {
    setWsConnected(true)
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null }
    ws?.send('ping')
  }

  ws.onclose = () => {
    setWsConnected(false)
    reconnectTimer = setTimeout(connectWS, 3000)
  }

  ws.onerror = () => {
    setWsConnected(false)
  }

  ws.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data)
      if (data.event === 'tick') {
        setTick(data as TickData)
      } else if (data.event === 'order_placed') {
        addToast(`Order placed: ${data.order_id} (${data.symbol})`, 'info')
        // Refresh portfolio
        import('../api/client').then(({ api }) => {
          api.positions().then(r => useStore.getState().setPositions(r.data.net || []))
          api.orders().then(r => useStore.getState().setOrders(r.data || []))
        })
      } else if (data.event === 'signal') {
        addToast(`Signal: ${data.signal?.action || data.signal} on ${data.symbol}`, 'info')
      }
    } catch { /* ignore parse errors */ }
  }
}

export function disconnectWS() {
  if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null }
  ws?.close()
  ws = null
}

export function sendPing() {
  if (ws?.readyState === WebSocket.OPEN) ws.send('ping')
}
