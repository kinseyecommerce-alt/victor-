import { useState } from 'react'
import { V2 } from '../lib/api'
import { fixed, inr, num, pnlClass, plain, when } from '../lib/format'
import { FILL_KIND, STRATEGY_ORDER, strategyTitle } from '../lib/names'
import { usePoll } from '../lib/poll'
import { Badge, Card, Collapse, LoadState, Notice, PageHeader, Segmented, Table } from '../components/ui'
import { TradeList } from '../components/trades'

const sel = 'w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-base sm:text-sm dark:border-slate-700 dark:bg-slate-800'

export function Trades() {
  const [tab, setTab] = useState<'positions' | 'history' | 'fills'>('positions')
  const [strategy, setStrategy] = useState('')
  const [since, setSince] = useState('')
  const [n, setN] = useState('200')
  const positions = usePoll(`${V2}/positions`)
  const qs = new URLSearchParams({ n })
  if (strategy) qs.set('strategy', strategy)
  if (since) qs.set('since', since)
  const trades = usePoll(tab === 'history' ? `${V2}/trades?${qs}` : null, 'slow')
  const drift = usePoll(tab === 'fills' ? `${V2}/drift` : null, 'slow')
  const fills = usePoll(tab === 'fills' ? `${V2}/fills?n=200${strategy ? `&strategy=${strategy}` : ''}` : null, 'slow')

  const pos: any[] = positions.data?.positions || []
  const orphans: any[] = positions.data?.orphan_rows || []
  const broker: any[] = positions.data?.broker_rows || []
  const stats = trades.data?.stats || {}
  const statRows = Object.values<any>(stats).sort((a, b) => STRATEGY_ORDER.indexOf(a?.strategy) - STRATEGY_ORDER.indexOf(b?.strategy))
  const tradeRows: any[] = trades.data?.trades || []
  const shownNet = tradeRows.reduce((a, t) => a + (num(t?.net) ?? 0), 0)

  return (
    <div className="space-y-4">
      <PageHeader title="Positions & trades" subtitle="Paper positions, closed trades and how good the paper fills were." polls={[positions, ...(tab === 'history' ? [trades] : tab === 'fills' ? [fills, drift] : [])]} />
      <Segmented options={[{ value: 'positions', label: `Open (${pos.length})` }, { value: 'history', label: 'History' }, { value: 'fills', label: 'Fill quality' }]} value={tab} onChange={setTab} testid="trades-tabs" />

      {tab === 'positions' && (
        <LoadState poll={positions}>
          {orphans.length > 0 && (
            <Notice tone="amber" title={`${orphans.length} leftover position(s) from the old engine`}>
              No current strategy owns these rows. Intraday ones are closed by the 15:15 square-off; delivery (CNC) ones stay until handled.
              <div className="mt-1 font-mono text-xs">{orphans.map((o: any) => `${o?.tradingsymbol || o?.symbol || '?'} ${plain(o?.quantity ?? o?.qty)} ${o?.product || ''}`).join(' · ')}</div>
            </Notice>
          )}
          <Card title="Open positions">
            {pos.length === 0 ? <div className="py-2 text-sm text-slate-500">No open positions.</div> : (
              <ul className="divide-y divide-slate-100 dark:divide-slate-800" data-testid="positions-list">
                {pos.map((p, i) => (
                  <li key={i} className="py-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="min-w-0 truncate text-sm"><Badge tone={p?.side === 'BUY' ? 'green' : 'red'}>{p?.side === 'BUY' ? 'Long' : 'Short'}</Badge> <b>{p?.symbol}</b> × {plain(p?.qty)}</span>
                      <span className={`shrink-0 text-sm font-semibold tabular-nums ${pnlClass(p?.unrealised)}`}>{inr(p?.unrealised, { signed: true })}</span>
                    </div>
                    <div className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                      {strategyTitle(String(p?.strategy || ''))} · {p?.product} · entry {fixed(p?.entry_px)} · {when(p?.opened)}{p?.overnight ? ' · held overnight' : ''}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </Card>
          <Collapse title={`Paper broker rows (${broker.length})`}>
            <Table head={['Symbol', 'Product', 'Qty', 'Avg price', 'Last', 'P&L']} rows={broker.map((r: any) => [r?.tradingsymbol, r?.product, plain(r?.quantity), fixed(r?.average_price), fixed(r?.last_price), <span className={pnlClass(r?.pnl)}>{inr(r?.pnl, { signed: true })}</span>])} empty="No broker rows." />
          </Collapse>
        </LoadState>
      )}

      {tab === 'history' && (
        <>
          <Card>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
              <label className="text-xs text-slate-500">Strategy
                <select className={sel} value={strategy} onChange={(e) => setStrategy(e.target.value)} data-testid="filter-strategy">
                  <option value="">All strategies</option>
                  {STRATEGY_ORDER.map((s) => <option key={s} value={s}>{strategyTitle(s)}</option>)}
                </select>
              </label>
              <label className="text-xs text-slate-500">From date
                <input type="date" className={sel} value={since} onChange={(e) => setSince(e.target.value)} data-testid="filter-since" />
              </label>
              <label className="text-xs text-slate-500">Show
                <select className={sel} value={n} onChange={(e) => setN(e.target.value)}>
                  {['50', '200', '500', '2000'].map((x) => <option key={x} value={x}>last {x}</option>)}
                </select>
              </label>
            </div>
          </Card>
          <Card title="Results per strategy (all time, after costs)">
            <LoadState poll={trades}>
              <Table testid="stats-table" head={['Strategy', 'Trades', 'Win rate', 'Net P&L', 'Per trade', 'Profit factor', 'Max drawdown']}
                rows={statRows.map((s) => [
                  <span>{strategyTitle(String(s?.strategy || ''))}{s?.kept === false && <Badge tone="slate" className="ml-1">removed</Badge>}</span>,
                  plain(s?.trades), s?.win_rate === null || s?.win_rate === undefined ? '—' : `${fixed(s.win_rate, 1)} %`,
                  <span className={pnlClass(s?.net_pnl)}>{inr(s?.net_pnl, { signed: true })}</span>, inr(s?.avg_net_per_trade, { signed: true }), fixed(s?.profit_factor), inr(s?.max_drawdown),
                ])} />
            </LoadState>
          </Card>
          <Card title={`Closed trades${trades.data ? ` (${plain(tradeRows.length)} of ${plain(trades.data.count)})` : ''}`} right={tradeRows.length ? <span className={`text-sm font-semibold ${pnlClass(shownNet)}`}>{inr(shownNet, { signed: true })}</span> : null}>
            <LoadState poll={trades}><TradeList trades={tradeRows} /></LoadState>
          </Card>
        </>
      )}

      {tab === 'fills' && (
        <>
          <Notice tone="blue">Slippage = how much worse the paper fill was than the price the rule saw, in basis points (1 bp = 0.01 %). The lab assumed the slippage shown as “lab”.</Notice>
          <Card title="Slippage vs the lab fill model">
            <LoadState poll={drift}>
              <div className="space-y-3">
                {STRATEGY_ORDER.filter((s) => drift.data?.strategies?.[s]).map((s) => {
                  const fq = drift.data.strategies[s]?.fill_quality || {}
                  const live = num(fq.entry_slip_bps_mean)
                  const lab = num(fq.lab_slippage_bps)
                  const worse = live !== null && lab !== null && live > lab
                  return (
                    <div key={s} className="rounded-xl border border-slate-100 p-3 dark:border-slate-800">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-sm font-medium">{strategyTitle(s)}</span>
                        {live === null ? <Badge>no fills yet</Badge> : <Badge tone={worse ? 'amber' : 'green'}>{worse ? 'Worse than lab' : 'Within lab'}</Badge>}
                      </div>
                      <div className="mt-1 grid grid-cols-3 gap-2 text-xs">
                        <div><div className="text-slate-500">Paper (mean)</div><b>{live === null ? '—' : `${fixed(live, 1)} bps`}</b></div>
                        <div><div className="text-slate-500">Lab</div><b>{lab === null ? '—' : `${fixed(lab, 1)} bps`}</b></div>
                        <div><div className="text-slate-500">Fills</div><b>{plain(fq.entry_slip_n)}</b></div>
                      </div>
                      {fq.lab_fill_model && <div className="mt-1 text-xs text-slate-500">Lab fill model: {String(fq.lab_fill_model)}</div>}
                    </div>
                  )
                })}
              </div>
            </LoadState>
          </Card>
          <Card title="Recent fills (quote vs fill)">
            <LoadState poll={fills}>
              <Table testid="fills-table" head={['Time', 'Strategy', 'Symbol', 'Side', 'Qty', 'Quote', 'Fill', 'Slip (bps)', 'Kind']}
                rows={(fills.data?.fills || []).map((f: any) => [when(f?.t), strategyTitle(String(f?.agent || '')), f?.symbol, f?.side, plain(f?.qty), fixed(f?.quote), fixed(f?.fill), fixed(f?.slip_bps, 1), FILL_KIND[f?.kind] || f?.kind])}
                empty="No fills logged yet." />
            </LoadState>
          </Card>
        </>
      )}
    </div>
  )
}
