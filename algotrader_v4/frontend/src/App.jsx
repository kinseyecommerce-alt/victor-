import React, { useState, useEffect, useRef } from 'react'

const API = import.meta.env.VITE_API_BASE_URL || ''

function App() {
    const [health, setHealth] = useState('checking...')
    const [botStatus, setBotStatus] = useState({})
    const [positions, setPositions] = useState([])
    const [signals, setSignals] = useState([])
    const [pnl, setPnl] = useState(0)
    const [wsStatus, setWsStatus] = useState('disconnected')
    const [ticks, setTicks] = useState([])
    const wsRef = useRef(null)

  useEffect(() => {
        fetch(API + '/health').then(r => r.json()).then(d => setHealth(d.status || 'ok')).catch(() => setHealth('offline'))
        fetch(API + '/api/bot/status').then(r => r.json()).then(setBotStatus).catch(() => {})
        fetch(API + '/api/positions').then(r => r.json()).then(d => setPositions(d.positions || [])).catch(() => {})
        fetch(API + '/api/signals').then(r => r.json()).then(d => setSignals(d.signals || [])).catch(() => {})
        fetch(API + '/api/pnl').then(r => r.json()).then(d => setPnl(d.total_pnl || 0)).catch(() => {})
        const interval = setInterval(() => {
                fetch(API + '/api/bot/status').then(r => r.json()).then(setBotStatus).catch(() => {})
                fetch(API + '/api/positions').then(r => r.json()).then(d => setPositions(d.positions || [])).catch(() => {})
                fetch(API + '/api/pnl').then(r => r.json()).then(d => setPnl(d.total_pnl || 0)).catch(() => {})
        }, 5000)
        return () => clearInterval(interval)
  }, [])

  useEffect(() => {
        const wsUrl = (API || window.location.origin).replace('https', 'wss').replace('http', 'ws') + '/ws'
        try {
                const ws = new WebSocket(wsUrl)
                wsRef.current = ws
                ws.onopen = () => setWsStatus('connected')
                ws.onclose = () => setWsStatus('disconnected')
                ws.onerror = () => setWsStatus('error')
                ws.onmessage = (e) => {
                          try { setTicks(prev => [JSON.parse(e.data), ...prev].slice(0, 10)) } catch {}
                }
        } catch {}
        return () => { if (wsRef.current) wsRef.current.close() }
  }, [])

  const botCmd = (cmd) => fetch(API + '/api/bot/' + cmd, { method: 'POST', headers: { 'X-API-Key': 'jag_api_key_32chars_changeme_now' } }).then(() => fetch(API + '/api/bot/status').then(r => r.json()).then(setBotStatus)).catch(() => {})

  return React.createElement('div', { style: { minHeight: '100vh', background: '#0f172a', color: '#f8fafc', fontFamily: 'system-ui', padding: '1rem' } },
                                 React.createElement('div', { style: { maxWidth: '1400px', margin: '0 auto' } },
                                                           React.createElement('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', paddingBottom: '1rem', borderBottom: '1px solid #334155' } },
                                                                                       React.createElement('h1', { style: { fontSize: '1.5rem', fontWeight: 'bold', color: '#60a5fa' } }, 'Nirma Trade - AlgoTrader Pro'),
                                                                                       React.createElement('div', { style: { display: 'flex', gap: '1rem', alignItems: 'center' } },
                                                                                                                     React.createElement('span', { style: { padding: '4px 12px', borderRadius: '9999px', fontSize: '0.75rem', background: wsStatus === 'connected' ? '#16a34a' : '#dc2626', color: 'white' } }, 'WS: ' + wsStatus),
                                                                                                                     React.createElement('span', { style: { padding: '4px 12px', borderRadius: '9999px', fontSize: '0.75rem', background: health === 'ok' ? '#16a34a' : '#dc2626', color: 'white' } }, 'API: ' + health),
                                                                                                                     React.createElement('span', { style: { padding: '4px 12px', borderRadius: '9999px', fontSize: '0.75rem', background: '#2563eb', color: 'white' } }, (botStatus.trading_mode || 'PAPER') + ' MODE')
                                                                                                                   )
                                                                                     ),
                                                           React.createElement('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.5rem' } },
                                                                                       React.createElement('div', { style: { background: '#1e293b', borderRadius: '0.75rem', padding: '1rem', border: '1px solid #334155' } },
                                                                                                                     React.createElement('div', { style: { color: '#94a3b8', fontSize: '0.875rem' } }, 'Total PnL'),
                                                                                                                     React.createElement('div', { style: { fontSize: '1.5rem', fontWeight: 'bold', color: pnl >= 0 ? '#4ade80' : '#f87171' } }, '\u20B9' + pnl.toFixed(2))
                                                                                                                   ),
                                                                                       React.createElement('div', { style: { background: '#1e293b', borderRadius: '0.75rem', padding: '1rem', border: '1px solid #334155' } },
                                                                                                                     React.createElement('div', { style: { color: '#94a3b8', fontSize: '0.875rem' } }, 'Open Positions'),
                                                                                                                     React.createElement('div', { style: { fontSize: '1.5rem', fontWeight: 'bold' } }, positions.length)
                                                                                                                   ),
                                                                                       React.createElement('div', { style: { background: '#1e293b', borderRadius: '0.75rem', padding: '1rem', border: '1px solid #334155' } },
                                                                                                                     React.createElement('div', { style: { color: '#94a3b8', fontSize: '0.875rem' } }, 'Bot Status'),
                                                                                                                     React.createElement('div', { style: { fontSize: '1.25rem', fontWeight: 'bold', color: botStatus.is_running ? '#4ade80' : '#f87171' } }, botStatus.is_running ? 'RUNNING' : 'STOPPED')
                                                                                                                   ),
                                                                                       React.createElement('div', { style: { background: '#1e293b', borderRadius: '0.75rem', padding: '1rem', border: '1px solid #334155' } },
                                                                                                                     React.createElement('div', { style: { color: '#94a3b8', fontSize: '0.875rem' } }, 'Active Signals'),
                                                                                                                     React.createElement('div', { style: { fontSize: '1.5rem', fontWeight: 'bold' } }, signals.length)
                                                                                                                   )
                                                                                     ),
                                                           React.createElement('div', { style: { display: 'flex', gap: '0.75rem', marginBottom: '1.5rem' } },
                                                                                       React.createElement('button', { onClick: () => botCmd('start'), style: { padding: '0.5rem 1rem', background: '#16a34a', color: 'white', border: 'none', borderRadius: '0.5rem', cursor: 'pointer' } }, 'Start Bot'),
                                                                                       React.createElement('button', { onClick: () => botCmd('stop'), style: { padding: '0.5rem 1rem', background: '#dc2626', color: 'white', border: 'none', borderRadius: '0.5rem', cursor: 'pointer' } }, 'Stop Bot'),
                                                                                       React.createElement('button', { onClick: () => botCmd('pause'), style: { padding: '0.5rem 1rem', background: '#d97706', color: 'white', border: 'none', borderRadius: '0.5rem', cursor: 'pointer' } }, 'Pause'),
                                                                                       React.createElement('button', { onClick: () => botCmd('resume'), style: { padding: '0.5rem 1rem', background: '#2563eb', color: 'white', border: 'none', borderRadius: '0.5rem', cursor: 'pointer' } }, 'Resume')
                                                                                     ),
                                                           positions.length > 0 && React.createElement('div', { style: { background: '#1e293b', borderRadius: '0.75rem', padding: '1rem', border: '1px solid #334155', marginBottom: '1rem' } },
                                                                                                               React.createElement('h2', { style: { marginBottom: '0.75rem', fontWeight: 'bold' } }, 'Open Positions'),
                                                                                                               React.createElement('table', { style: { width: '100%', borderCollapse: 'collapse' } },
                                                                                                                                             React.createElement('thead', null, React.createElement('tr', { style: { color: '#94a3b8', textAlign: 'left' } },
                                                                                                                                                                                                                React.createElement('th', { style: { padding: '0.5rem' } }, 'Symbol'),
                                                                                                                                                                                                                React.createElement('th', { style: { padding: '0.5rem' } }, 'Qty'),
                                                                                                                                                                                                                React.createElement('th', { style: { padding: '0.5rem' } }, 'Avg'),
                                                                                                                                                                                                                React.createElement('th', { style: { padding: '0.5rem' } }, 'LTP'),
                                                                                                                                                                                                                React.createElement('th', { style: { padding: '0.5rem' } }, 'PnL')
                                                                                                                                                                                                              )),
                                                                                                                                             React.createElement('tbody', null, positions.map((p, i) => React.createElement('tr', { key: i, style: { borderTop: '1px solid #334155' } },
                                                                                                                                                                                                                                        React.createElement('td', { style: { padding: '0.5rem' } }, p.symbol),
                                                                                                                                                                                                                                        React.createElement('td', { style: { padding: '0.5rem' } }, p.quantity),
                                                                                                                                                                                                                                        React.createElement('td', { style: { padding: '0.5rem' } }, p.average_price),
                                                                                                                                                                                                                                        React.createElement('td', { style: { padding: '0.5rem' } }, p.last_price),
                                                                                                                                                                                                                                        React.createElement('td', { style: { padding: '0.5rem', color: p.pnl >= 0 ? '#4ade80' : '#f87171' } }, p.pnl)
                                                                                                                                                                                                                                      )))
                                                                                                                                           )
                                                                                                             ),
                                                           ticks.length > 0 && React.createElement('div', { style: { background: '#1e293b', borderRadius: '0.75rem', padding: '1rem', border: '1px solid #334155' } },
                                                                                                           React.createElement('h2', { style: { marginBottom: '0.75rem', fontWeight: 'bold' } }, 'Live Ticks'),
                                                                                                           ticks.slice(0, 5).map((t, i) => React.createElement('div', { key: i, style: { fontSize: '0.75rem', color: '#94a3b8', padding: '0.25rem 0' } }, JSON.stringify(t)))
                                                                                                         )
                                                         )
                               )
}

export default App
