import React, { useEffect, useState } from 'react'
import { Shield, AlertTriangle } from 'lucide-react'
import { useStore } from '../../store'
import { api } from '../../api/client'
import { Btn, Badge, Modal, Input } from '../ui'

export default function SebiTab() {
  const { addToast } = useStore()
  const [status, setStatus] = useState<any>(null)
  const [killConfirm, setKillConfirm] = useState(false)
  const [resetModal, setResetModal] = useState(false)
  const [resetSecret, setResetSecret] = useState('')
  const [auditDate, setAuditDate] = useState(new Date().toISOString().split('T')[0])
  const [auditLog, setAuditLog] = useState<any[]>([])
  const [loading, setLoading] = useState(false)

  const fetchStatus = () => api.sebiStatus().then(r => setStatus(r.data)).catch(() => {})
  useEffect(() => {
    fetchStatus()
    const t = setInterval(fetchStatus, 5000)
    return () => clearInterval(t)
  }, [])

  const handleKillSwitch = async () => {
    setLoading(true)
    try {
      await api.sebiKillSwitch('Manual kill switch from UI')
      addToast('⛔ Kill switch triggered — all trading HALTED', 'error')
      fetchStatus()
    } catch (e: any) {
      addToast(e.response?.data?.detail || 'Kill switch failed', 'error')
    } finally {
      setLoading(false)
      setKillConfirm(false)
    }
  }

  const handleResume = async () => {
    setLoading(true)
    try {
      await api.sebiResume()
      addToast('Trading resumed', 'buy')
      fetchStatus()
    } catch (e: any) {
      addToast(e.response?.data?.detail || 'Resume failed', 'error')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = async () => {
    try {
      await api.sebiResetKillSwitch(resetSecret)
      addToast('Kill switch reset — trading re-enabled', 'buy')
      setResetModal(false)
      setResetSecret('')
      fetchStatus()
    } catch (e: any) {
      addToast(e.response?.data?.detail || 'Reset failed', 'error')
    }
  }

  const fetchAuditLog = () => {
    api.sebiAuditLog(auditDate).then(r => setAuditLog(r.data.records || [])).catch(() => {})
  }

  const isKilled = status?.kill_switch_state === 'KILLED'

  return (
    <div className="h-full overflow-auto p-4 space-y-4">
      {/* Status banner */}
      <div className={`flex items-center gap-3 rounded-xl p-4 border ${
        isKilled ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'
      }`}>
        <Shield className={`w-6 h-6 ${isKilled ? 'text-red-600' : 'text-green-600'}`} />
        <div>
          <div className={`font-bold ${isKilled ? 'text-red-800' : 'text-green-800'}`}>
            SEBI Compliance: {status?.kill_switch_state ?? '…'}
          </div>
          <div className="text-xs text-slate-600 mt-0.5">
            Orders logged: {status?.audit_log_count ?? '—'} · IP whitelist: {status?.whitelisted_ips?.length ?? '—'} IPs
          </div>
        </div>
      </div>

      {/* Kill switch buttons */}
      <div className="bg-white border border-slate-200 rounded-xl p-4">
        <div className="text-sm font-semibold text-slate-800 mb-3">Emergency Controls</div>
        <div className="flex gap-3 flex-wrap">
          {!killConfirm ? (
            <Btn variant="danger" onClick={() => setKillConfirm(true)} disabled={isKilled}>
              🛑 Emergency Kill Switch
            </Btn>
          ) : (
            <>
              <div className="w-full flex items-center gap-2 text-red-700 text-sm bg-red-50 rounded-lg p-2 mb-2">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                This will HALT ALL trading immediately. Are you sure?
              </div>
              <Btn variant="danger" onClick={handleKillSwitch} disabled={loading}>
                {loading ? 'Halting...' : 'Confirm Kill Switch'}
              </Btn>
              <Btn variant="outline" onClick={() => setKillConfirm(false)}>Cancel</Btn>
            </>
          )}
          {isKilled && (
            <>
              <Btn variant="buy" onClick={handleResume} disabled={loading}>
                ▶ Resume Trading
              </Btn>
              <Btn variant="outline" onClick={() => setResetModal(true)}>
                Reset Kill Switch
              </Btn>
            </>
          )}
        </div>
      </div>

      {/* Audit log */}
      <div className="bg-white border border-slate-200 rounded-xl p-4">
        <div className="flex items-center gap-3 mb-3">
          <span className="text-sm font-semibold text-slate-800">Audit Log</span>
          <input
            type="date"
            value={auditDate}
            onChange={e => setAuditDate(e.target.value)}
            className="text-xs border border-slate-200 rounded px-2 py-1 text-slate-700"
          />
          <Btn variant="outline" size="sm" onClick={fetchAuditLog}>Fetch</Btn>
        </div>
        {auditLog.length === 0 ? (
          <div className="text-sm text-slate-400">No records (click Fetch)</div>
        ) : (
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {auditLog.slice(0, 20).map((r, i) => (
              <div key={i} className="text-xs font-mono text-slate-600 border-b border-slate-50 py-1">
                <span className="text-slate-400">{r.ts?.split('T')[1]?.split('.')[0] ?? ''}</span>
                {' · '}
                <span className={r.decision === 'ALLOWED' ? 'text-green-600' : 'text-red-600'}>{r.decision}</span>
                {' · '}{r.strategy} · {r.symbol}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Reset modal */}
      <Modal open={resetModal} onClose={() => setResetModal(false)} title="Reset Kill Switch">
        <div className="space-y-4">
          <p className="text-sm text-slate-600">Enter the kill switch reset secret to re-enable trading.</p>
          <Input type="password" value={resetSecret} onChange={e => setResetSecret(e.target.value)} placeholder="Secret" />
          <div className="flex gap-2">
            <Btn onClick={handleReset} disabled={!resetSecret}>Confirm Reset</Btn>
            <Btn variant="outline" onClick={() => setResetModal(false)}>Cancel</Btn>
          </div>
        </div>
      </Modal>
    </div>
  )
}
