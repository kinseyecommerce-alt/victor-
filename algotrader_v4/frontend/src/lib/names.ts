// Plain-English labels for the engine's ids and codes.

export const STRATEGY_ORDER = ['auc_fade', 'attn_on', 'vrp_winged', 'pead_h60']

export const STRATEGY_INFO: Record<string, { title: string; what: string; when: string }> = {
  auc_fade: {
    title: 'Opening-auction fade',
    what: 'When a stock opens far away from yesterday’s close at the 09:15 auction, it takes the opposite side and gets out at 09:20.',
    when: 'Enters 09:15 · exits 09:20 · intraday',
  },
  attn_on: {
    title: 'Attention overnight buy',
    what: 'At 15:21 it buys the stocks trading more than 3× their usual volume and rising, and sells them at the next morning’s opening auction.',
    when: 'Buys 15:21 · sells next day 09:15 · one night',
  },
  vrp_winged: {
    title: 'NIFTY weekly strangle (with protection)',
    what: 'At 10:20 it sells a NIFTY weekly strangle and buys protective “wings” further out, so the worst case is capped. Held to expiry.',
    when: 'Enters 10:20 · settles on expiry day',
  },
  pead_h60: {
    title: 'Post-earnings drift',
    what: 'Buys stocks that react strongly upward to their quarterly results and holds them for about 60 trading sessions.',
    when: 'Enters 09:15 after results · holds ~60 sessions',
  },
}

export function strategyTitle(name: string, label?: string): string {
  return STRATEGY_INFO[name]?.title || label || name
}

export type TState = 'ACTIVE' | 'REDUCE_ONLY' | 'HALTED'

export const STATE_LABEL: Record<string, string> = {
  ACTIVE: 'Active',
  REDUCE_ONLY: 'Exit-only',
  HALTED: 'Stopped',
}

export const STATE_HELP: Record<string, string> = {
  ACTIVE: 'Opens and closes trades by its rule.',
  REDUCE_ONLY: 'Opens no new trades. Open trades still exit by the rule.',
  HALTED: 'Opens no new trades and cancels pending approvals. Open trades still leave at their tested exit (risk exits only).',
}

export function stateLabel(s: unknown): string {
  return typeof s === 'string' ? STATE_LABEL[s] || s : '—'
}

export function stateTone(s: unknown): 'green' | 'amber' | 'red' | 'slate' {
  if (s === 'ACTIVE') return 'green'
  if (s === 'REDUCE_ONLY') return 'amber'
  if (s === 'HALTED') return 'red'
  return 'slate'
}

const BLOCK: Record<string, string> = {
  live_mode_refused: 'Refuses LIVE mode (paper only)',
  no_kite_data: 'No Kite market data (Kite not connected)',
  validated_only_off: 'Validated trading switched off on the server',
  kill_switch: 'Switched off on the server (setting)',
  drawdown_kill: 'Stopped by its own drawdown limit — needs a manual re-enable',
  disabled: 'Switched off on the dashboard',
  stopped: 'Engine stopped (or strategy paused today)',
  daily_loss_stop: 'Hit today’s loss limit',
  awaiting_approval: 'Waiting for your approval',
  approval_rejected: 'You rejected today’s entries',
  approval_expired: 'Today’s approval window expired',
  approval_cancelled: 'Today’s approval was cancelled',
}

export function blockText(code: unknown): string | null {
  if (code === null || code === undefined || code === '') return null
  const c = String(code)
  if (c.startsWith('trading_state:')) return `State is ${stateLabel(c.split(':')[1])} — no new trades`
  if (c.startsWith('no_data')) return 'Price history not up to date — no new trades'
  return BLOCK[c] || c.replace(/_/g, ' ')
}

export const DRIFT_LABEL: Record<string, string> = {
  ok: 'In line with the lab',
  warn: 'Warning',
  alert: 'Alert',
  error: 'Monitor error',
}

export const FILL_KIND: Record<string, string> = {
  MARKETABLE: 'Limit at the best price',
  AUCTION: 'Opening auction',
  SETTLEMENT: 'Expiry settlement',
  RECONCILE: 'Reconciliation',
  CORPORATE_ACTION: 'Corporate action',
  LIVE_PROTECTED: 'Live protected order',
}

export const PERIOD_LABEL: Record<string, string> = {
  DISC: 'Discovery period (data the rule was found on)',
  FINAL: 'Final test (past unseen data)',
}
