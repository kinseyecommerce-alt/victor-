import React, { useState, useEffect, useCallback } from 'react'
import { ShoppingCart, TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react'
import { useStore } from '../../store'
import { api } from '../../api/client'
import { Btn, Input, Select, Badge } from '../ui'

type Side = 'BUY' | 'SELL'
type OrderType = 'MARKET' | 'LIMIT' | 'SL' | 'SL-M'
type Product = 'MIS' | 'CNC' | 'NRML'

export default function OrderPanel() {
  const { selectedSymbol, ticks, addToast, setPositions, setOrders } = useStore()
  const [side, setSide] = useState<Side>('BUY')
  const [qty, setQty] = useState('1')
  const [price, setPrice] = useState('0')
  const [orderType, setOrderType] = useState<OrderType>('MARKET')
  const [product, setProduct] = useState<Product>('MIS')
  const [loading, setLoading] = useState(false)
  const [confirmSqOff, setConfirmSqOff] = useState(false)

  const tick = selectedSymbol ? ticks[selectedSymbol] : null

  useEffect(() => {
    if (tick && orderType === 'LIMIT') setPrice(tick.ltp.toFixed(2))
  }, [tick?.ltp, orderType])

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return
      if (e.key === 'b' || e.key === 'B') setSide('BUY')
      if (e.key === 's' || e.key === 'S') setSide('SELL')
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  const refresh = useCallback(() => {
    api.positions().then(r => setPositions(r.data.net || [])).catch(() => {})
    api.orders().then(r => setOrders(r.data || [])).catch(() => {})
  }, [])

  const handlePlace = async () => {
    if (!selectedSymbol) { addToast('Select a symbol first', 'error'); return }
    const q = parseInt(qty)
    if (!q || q <= 0) { addToast('Invalid quantity', 'error'); return }

    setLoading(true)
    try {
      const r = await api.placeOrder({
        symbol: selectedSymbol,
        exchange: 'NSE',
        transaction_type: side,
        quantity: q,
        order_type: orderType,
        product,
        price: orderType !== 'MARKET' ? parseFloat(price) : 0,
      })
      addToast(`${side} ${q} ${selectedSymbol} → ${r.data.order_id}`, side === 'BUY' ? 'buy' : 'sell')
      refresh()
    } catch (e: any) {
      const msg = e.response?.data?.detail || 'Order failed'
      addToast(msg, 'error')
    } finally {
      setLoading(false)
    }
  }

  const handleSquareOff = async () => {
    setLoading(true)
    try {
      const r = await api.squareoffAll()
      addToast(`Squared off ${r.data.squared_off} positions`, 'info')
      refresh()
    } catch (e: any) {
      addToast(e.response?.data?.detail || 'Squareoff failed', 'error')
    } finally {
      setLoading(false)
      setConfirmSqOff(false)
    }
  }

  const estimatedValue = tick ? parseInt(qty || '0') * tick.ltp : 0

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <ShoppingCart className="w-4 h-4 text-slate-500" />
          <span className="text-sm font-semibold text-slate-900">Order Panel</span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Symbol display */}
        <div className="text-center">
          {tick ? (
            <div>
              <div className="text-lg font-bold text-slate-900">{selectedSymbol}</div>
              <div className={`text-2xl font-mono font-bold ${tick.change_pct >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                ₹{tick.ltp.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                Bid: {tick.bid.toFixed(2)} · Ask: {tick.ask.toFixed(2)}
              </div>
            </div>
          ) : (
            <div className="text-slate-400 text-sm">No symbol selected</div>
          )}
        </div>

        {/* BUY / SELL toggle */}
        <div className="grid grid-cols-2 gap-1 p-1 bg-slate-100 rounded-xl">
          <button
            onClick={() => setSide('BUY')}
            className={`py-2 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-1.5 ${
              side === 'BUY' ? 'bg-green-600 text-white shadow-sm' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <TrendingUp className="w-3.5 h-3.5" />BUY <span className="text-xs font-normal opacity-70">(B)</span>
          </button>
          <button
            onClick={() => setSide('SELL')}
            className={`py-2 rounded-lg text-sm font-bold transition-all flex items-center justify-center gap-1.5 ${
              side === 'SELL' ? 'bg-red-600 text-white shadow-sm' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <TrendingDown className="w-3.5 h-3.5" />SELL <span className="text-xs font-normal opacity-70">(S)</span>
          </button>
        </div>

        {/* Form fields */}
        <div className="space-y-3">
          <div>
            <label className="block text-xs font-medium text-slate-600 mb-1">Quantity</label>
            <Input
              type="number"
              min="1"
              value={qty}
              onChange={e => setQty(e.target.value)}
              placeholder="1"
            />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Order Type</label>
              <Select value={orderType} onChange={e => setOrderType(e.target.value as OrderType)}>
                <option value="MARKET">MARKET</option>
                <option value="LIMIT">LIMIT</option>
                <option value="SL">SL</option>
                <option value="SL-M">SL-M</option>
              </Select>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Product</label>
              <Select value={product} onChange={e => setProduct(e.target.value as Product)}>
                <option value="MIS">MIS</option>
                <option value="CNC">CNC</option>
                <option value="NRML">NRML</option>
              </Select>
            </div>
          </div>

          {orderType !== 'MARKET' && (
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Price</label>
              <Input type="number" step="0.05" value={price} onChange={e => setPrice(e.target.value)} />
            </div>
          )}

          {estimatedValue > 0 && (
            <div className="text-xs text-slate-500 text-center">
              Est. value: <span className="font-mono font-medium text-slate-700">
                ₹{estimatedValue.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
              </span>
            </div>
          )}
        </div>

        {/* Place order button */}
        <Btn
          variant={side === 'BUY' ? 'buy' : 'sell'}
          size="lg"
          className="w-full font-bold text-base"
          onClick={handlePlace}
          disabled={loading || !selectedSymbol}
        >
          {loading ? 'Placing...' : `${side} ${qty || 0} ${selectedSymbol || '...'}`}
        </Btn>

        {/* Square off */}
        <div className="pt-2 border-t border-slate-100">
          {confirmSqOff ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-amber-700 text-xs bg-amber-50 rounded-lg p-2">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                Square off ALL positions at market price?
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Btn variant="danger" size="sm" onClick={handleSquareOff} disabled={loading}>
                  {loading ? '...' : 'Confirm'}
                </Btn>
                <Btn variant="outline" size="sm" onClick={() => setConfirmSqOff(false)}>Cancel</Btn>
              </div>
            </div>
          ) : (
            <Btn variant="outline" size="sm" className="w-full text-red-600 border-red-200 hover:bg-red-50" onClick={() => setConfirmSqOff(true)}>
              Square Off All Positions
            </Btn>
          )}
        </div>
      </div>
    </div>
  )
}
