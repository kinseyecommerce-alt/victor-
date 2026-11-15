import { createContext, ReactNode, useCallback, useContext, useEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import { CheckCircle2, XCircle } from 'lucide-react'
import { api } from '../lib/api'
import { refreshAll } from '../lib/poll'
import { Button } from './ui'

// Every write action goes through a confirmation dialog, then the toast shows what the server said.

interface AskOpts {
  title: string
  body: ReactNode
  confirmLabel: string
  danger?: boolean
  reason?: boolean            // show an optional "reason" text field
  reasonPlaceholder?: string
}

interface Ctx {
  ask: (o: AskOpts) => Promise<{ ok: boolean; reason: string }>
  toast: (msg: string, ok?: boolean) => void
  /** ask, then POST; returns the response or null when cancelled / failed */
  act: (o: AskOpts & { path: string; payload: (reason: string) => unknown; done: string }) => Promise<any | null>
}

const DialogCtx = createContext<Ctx | null>(null)

export function useDialogs(): Ctx {
  const c = useContext(DialogCtx)
  if (!c) throw new Error('DialogProvider missing')
  return c
}

export function DialogProvider({ children }: { children: ReactNode }) {
  const [pending, setPending] = useState<(AskOpts & { resolve: (r: { ok: boolean; reason: string }) => void }) | null>(null)
  const [reason, setReason] = useState('')
  const [toasts, setToasts] = useState<{ id: number; msg: string; ok: boolean }[]>([])
  const idRef = useRef(0)
  const confirmRef = useRef<HTMLButtonElement>(null)

  const ask = useCallback((o: AskOpts) => new Promise<{ ok: boolean; reason: string }>((resolve) => {
    setReason('')
    setPending({ ...o, resolve })
  }), [])

  const toast = useCallback((msg: string, ok = true) => {
    const id = ++idRef.current
    setToasts((t) => [...t, { id, msg, ok }])
    window.setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), ok ? 3500 : 7000)
  }, [])

  const act = useCallback(async (o: AskOpts & { path: string; payload: (reason: string) => unknown; done: string }) => {
    const r = await ask(o)
    if (!r.ok) return null
    try {
      const res = await api.post(o.path, o.payload(r.reason))
      toast(o.done, true)
      refreshAll()
      return res ?? {}
    } catch (e: any) {
      toast(`Not done: ${e?.message || e}`, false)
      return null
    }
  }, [ask, toast])

  const close = (ok: boolean) => {
    if (!pending) return
    pending.resolve({ ok, reason: reason.trim() })
    setPending(null)
  }

  useEffect(() => {
    if (!pending) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close(false) }
    window.addEventListener('keydown', onKey)
    if (!pending.reason) confirmRef.current?.focus()
    return () => window.removeEventListener('keydown', onKey)
  })

  return (
    <DialogCtx.Provider value={{ ask, toast, act }}>
      {children}
      {pending && (
        <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/50 p-3 sm:items-center" onClick={() => close(false)}>
          <div role="dialog" aria-modal="true" aria-labelledby="dlg-title" data-testid="confirm-dialog"
            className="w-full max-w-md rounded-2xl bg-white p-5 shadow-xl dark:bg-slate-900" onClick={(e) => e.stopPropagation()}>
            <h3 id="dlg-title" className="text-lg font-semibold">{pending.title}</h3>
            <div className="mt-2 text-sm text-slate-600 dark:text-slate-300">{pending.body}</div>
            {pending.reason && (
              <label className="mt-3 block text-sm">
                <span className="text-slate-500 dark:text-slate-400">Reason (optional, saved in the log)</span>
                <input autoFocus value={reason} maxLength={200} onChange={(e) => setReason(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') close(true) }}
                  placeholder={pending.reasonPlaceholder || 'e.g. results week'}
                  className="mt-1 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-base dark:border-slate-700 dark:bg-slate-800" />
              </label>
            )}
            <div className="mt-5 flex gap-2">
              <Button className="flex-1" onClick={() => close(false)} testid="confirm-cancel">Cancel</Button>
              <button ref={confirmRef} data-testid="confirm-ok" onClick={() => close(true)}
                className={clsx('flex-1 rounded-xl px-3 py-2 text-sm font-semibold text-white',
                  pending.danger ? 'bg-red-600 hover:bg-red-700' : 'bg-blue-600 hover:bg-blue-700')}>
                {pending.confirmLabel}
              </button>
            </div>
          </div>
        </div>
      )}
      <div className="pointer-events-none fixed inset-x-0 bottom-20 z-50 flex flex-col items-center gap-2 px-3 md:bottom-4">
        {toasts.map((t) => (
          <div key={t.id} role="status" data-testid="toast"
            className={clsx('pointer-events-auto flex max-w-md items-start gap-2 rounded-xl px-4 py-2.5 text-sm text-white shadow-lg',
              t.ok ? 'bg-slate-800 dark:bg-slate-700' : 'bg-red-700')}>
            {t.ok ? <CheckCircle2 size={18} className="shrink-0 text-green-400" /> : <XCircle size={18} className="shrink-0" />}
            <span>{t.msg}</span>
          </div>
        ))}
      </div>
    </DialogCtx.Provider>
  )
}
