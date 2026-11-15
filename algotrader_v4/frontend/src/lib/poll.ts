import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import { api, ApiError } from './api'

// Polling only (the backend has no websocket). Interval: 15 s while the market is open, 60 s otherwise.
// Polling pauses while the tab is hidden and refreshes immediately when it becomes visible again.

export const MarketContext = createContext<{ open: boolean }>({ open: false })

export function useInterval(kind: 'fast' | 'slow' = 'fast'): number {
  const { open } = useContext(MarketContext)
  if (kind === 'slow') return open ? 30000 : 60000
  return open ? 15000 : 60000
}

export interface Poll<T = any> {
  data: T | null
  error: ApiError | null
  loading: boolean
  updatedAt: number | null
  intervalMs: number
  refresh: () => Promise<void>
}

export function usePoll<T = any>(path: string | null, kind: 'fast' | 'slow' = 'fast'): Poll<T> {
  const intervalMs = useInterval(kind)
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [loading, setLoading] = useState(true)
  const [updatedAt, setUpdatedAt] = useState<number | null>(null)
  const pathRef = useRef(path)
  pathRef.current = path

  const refresh = useCallback(async () => {
    const p = pathRef.current
    if (!p) return
    try {
      const d = await api.get(p)
      if (pathRef.current !== p) return
      setData(d)
      setError(null)
      setUpdatedAt(Date.now())
    } catch (e: any) {
      if (pathRef.current !== p) return
      setError(e instanceof ApiError ? e : new ApiError(0, String(e?.message || e)))
    } finally {
      if (pathRef.current === p) setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!path) return
    setLoading(true)
    setData(null)
    setError(null)
    refresh()
  }, [path, refresh])

  useEffect(() => {
    if (!path) return
    const id = window.setInterval(() => {
      if (document.visibilityState === 'visible') refresh()
    }, intervalMs)
    const onVis = () => { if (document.visibilityState === 'visible') refresh() }
    const onOnline = () => refresh()
    document.addEventListener('visibilitychange', onVis)
    window.addEventListener('online', onOnline)
    window.addEventListener('dashboard-refresh', onOnline)
    return () => {
      window.clearInterval(id)
      document.removeEventListener('visibilitychange', onVis)
      window.removeEventListener('online', onOnline)
      window.removeEventListener('dashboard-refresh', onOnline)
    }
  }, [path, intervalMs, refresh])

  return { data, error, loading, updatedAt, intervalMs, refresh }
}

/** Ask every mounted poll to refresh now (after a write action). */
export function refreshAll() {
  window.dispatchEvent(new Event('dashboard-refresh'))
}
