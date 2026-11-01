import React, { useEffect } from 'react'
import { useStore } from '../../store'
import { api } from '../../api/client'
import { Badge, Btn } from '../ui'

const statusVariant = (s: string): 'buy' | 'sell' | 'neutral' | 'warning' => {
  if (s === 'COMPLETE') return 'buy'
  if (s === 'CANCELLED' || s === 'REJECTED') return 'sell'
  if (s === 'OPEN' || s === 'TRIGGER PENDING') return 'warning'
  return 'neutral'
}

export default function OrdersTab() {
  const { orders, setOrders, addToast } = useStore()

  useEffect(() => {
    api.orders().then(r => setOrders(r.data || [])).catch(() => {})
    const t = setInterval(() => api.orders().then(r => setOrders(r.data || [])).catch(() => {}), 5000)
    return () => clearInterval(t)
  }, [])

  const handleCancel = async (orderId: string) => {
    try {
      await api.cancelOrder(orderId)
      addToast(`Order ${orderId} cancelled`, 'info')
      api.orders().then(r => setOrders(r.data || [])).catch(() => {})
    } catch (e: any) {
      addToast(e.response?.data?.detail || 'Cancel failed', 'error')
    }
  }

  if (orders.length === 0) {
    return <div className="flex items-center justify-center h-full text-slate-400 text-sm">No orders today</div>
  }

  return (
    <div className="h-full overflow-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-100 bg-slate-50 sticky top-0">
            {['Order ID', 'Symbol', 'Side', 'Qty', 'Type', 'Price', 'Status', ''].map(h => (
              <th key={h} className="text-left px-4 py-2 text-xs font-medium text-slate-500">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {[...orders].reverse().map(o => (
            <tr key={o.order_id} className="border-b border-slate-50 hover:bg-slate-50 transition-colors">
              <td className="px-4 py-2.5 font-mono text-xs text-slate-500">{o.order_id.slice(-8)}</td>
              <td className="px-4 py-2.5 font-semibold text-slate-900">{o.tradingsymbol}</td>
              <td className="px-4 py-2.5">
                <Badge variant={o.transaction_type === 'BUY' ? 'buy' : 'sell'}>{o.transaction_type}</Badge>
              </td>
              <td className="px-4 py-2.5 font-mono text-slate-700">{o.quantity}</td>
              <td className="px-4 py-2.5 text-slate-600 text-xs">{o.order_type}</td>
              <td className="px-4 py-2.5 font-mono text-slate-700">
                {o.price > 0 ? `₹${o.price.toFixed(2)}` : 'MKT'}
              </td>
              <td className="px-4 py-2.5">
                <Badge variant={statusVariant(o.status)}>{o.status}</Badge>
              </td>
              <td className="px-4 py-2.5">
                {(o.status === 'OPEN' || o.status === 'TRIGGER PENDING') && (
                  <Btn variant="ghost" size="sm" onClick={() => handleCancel(o.order_id)}
                    className="text-red-600 hover:bg-red-50 text-xs py-1">Cancel</Btn>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
