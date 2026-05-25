import{useState,useEffect,useRef,useCallback}from'react'
import{AreaChart,Area,XAxis,YAxis,CartesianGrid,Tooltip,ResponsiveContainer}from'recharts'
import{Activity,TrendingUp,Zap,Shield,BarChart2,Clock,DollarSign,Wifi,WifiOff}from'lucide-react'
const API=import.meta.env.VITE_API_BASE_URL||''
const H={'X-API-Key':import.meta.env.VITE_API_KEY||'','Content-Type':'application/json'}
function usePoll(url,ms=6000){
  const[data,setData]=useState(null)
  const go=useCallback(()=>{fetch(API+url,{headers:H}).then(r=>r.ok?r.json():null).then(d=>d&&setData(d)).catch(()=>{})},[url])
  useEffect(()=>{go();const t=setInterval(go,ms);return()=>clearInterval(t)},[go,ms])
  return{data,refetch:go}
}
function Card({children,className}){return<div className={'bg-gray-900 border border-gray-800 rounded-xl p-4 '+(className||'')}>{children}</div>div>}
function Stat({icon:I,label,value,color}){
  const c={green:'text-green-400',red:'text-red-400',blue:'text-blue-400',yellow:'text-yellow-400'}[color||'green']
  return<Card className="flex items-start gap-3"><div className={'p-2 rounded-lg bg-gray-800 '+c}><I size={18}/></div>div><div><p className="text-xs text-gray-500">{label}</p>p><p className={'text-xl font-bold '+c}>{value||'---'}</p>p></div>div></Card>Card>
    }
    export default function App(){
      const[log,setLog]=useState([])
        const[ws,setWs]=useState('off')
          const[ticks,setTicks]=useState({})
            const[curve,setCurve]=useState([])
              const wsr=useRef(null)
                const lg=m=>setLog(l=>[{t:new Date().toLocaleTimeString(),m},...l.slice(0,49)])
                  const{data:health}=usePoll('/health',10000)
                    const{data:pos,refetch:rPos}=usePoll('/portfolio/positions',8000)
                      const{data:regime}=usePoll('/market/regime',15000)
                        const{data:adapt}=usePoll('/adaptive/status',10000)
                          const{data:bot}=usePoll('/bot/status',5000)
                            const{data:sigs}=usePoll('/signals/latest',6000)
                              useEffect(()=>{
                                const url=(API.replace('https','wss').replace('http','ws')||'ws://localhost:8000')+'/ws'
                                  const conn=()=>{
                                    const w=new WebSocket(url);wsr.current=w
                                      w.onopen=()=>{setWs('on');lg('WS connected')}
                                        w.onclose=()=>{setWs('off');setTimeout(conn,3000)}
                                          w.onerror=()=>setWs('err')
                                            w.onmessage=e=>{try{const d=JSON.parse(e.data);if(d.type==='tick'){setTicks(p=>({...p,[d.symbol]:d}));setCurve(p=>[...p.slice(-60),{t:new Date().toLocaleTimeString(),v:d.pnl||(p.at(-1)?.v||0)}])}}catch{}}
                                  }
                                    conn();return()=>wsr.current?.close()
                              },[])
                                const act=a=>{fetch(API+'/bot/'+a,{method:'POST',headers:H}).then(r=>r.json()).then(d=>lg('bot '+a+': '+(d.status||d.detail||'ok'))).catch(e=>lg('err: '+e));setTimeout(rPos,1200)}
                                  const pnl=pos?.reduce((s,p)=>s+(p.last_price-p.average_price)*p.quantity,0)||0
                                    const sl=Array.isArray(sigs)?sigs:(sigs?.signals||[])
                                      return(
                                        <div className="min-h-screen bg-gray-950 text-gray-100">
                                        <header className="border-b border-gray-800 px-6 py-3 flex items-center justify-between sticky top-0 bg-gray-950/95 backdrop-blur z-10">
                                        <div className="flex items-center gap-3">
                                        <div className="p-1.5 bg-green-500/20 rounded-lg"><Activity className="text-green-400" size={20}/></div>div>
                                        <div><h1 className="font-bold text-lg">Nirma Trade</h1>h1><p className="text-xs text-gray-500">AlgoTrader Pro v4</p>p></div>div>
                                        </div>div>
                                        <div className="flex items-center gap-3 text-xs">
                                        <span className={'flex items-center gap-1 '+(ws==='on'?'text-green-400':'text-gray-500')}>{ws==='on'?<Wifi size={12}/>:<WifiOff size={12}/>} {ws}</span>span>
                                        <span className={'px-2 py-1 rounded-full '+(health?.status==='ok'?'bg-green-900/40 text-green-400':'bg-red-900/40 text-red-400')}>{health?.status==='ok'?'Online':'Offline'}</span>span>
                                        <span className="bg-blue-900/40 text-blue-400 px-2 py-1 rounded-full">{bot?.mode||'PAPER'}</span>span>
                                        </div>div>
                                        </header>header>
                                        <main className="max-w-7xl mx-auto px-4 py-6 space-y-5">
                                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                        <Stat icon={DollarSign} label="Total PnL" value={'Rs '+pnl.toFixed(0)} color={pnl>=0?'green':'red'}/>
                                        <Stat icon={BarChart2} label="Positions" value={pos?.length||0} color="blue"/>
                                        <Stat icon={Zap} label="Regime" value={regime?.regime} color="yellow"/>
                                        <Stat icon={Shield} label="Strategy" value={adapt?.active_strategy} color="blue"/>
                                        </div>div>
                                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                                        <Card className="lg:col-span-2">
                                        <h2 className="text-sm font-semibold text-gray-400 mb-3 flex items-center gap-2"><TrendingUp size={13}/> Live Equity</h2>h2>
                                        <ResponsiveContainer width="100%" height={190}>
                                        <AreaChart data={curve}>
                                        <defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#22c55e" stopOpacity={0.3}/><stop offset="95%" stopColor="#22c55e" stopOpacity={0}/></linearGradient>linearGradient></defs>defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937"/>
                                        <XAxis dataKey="t" tick={{fill:'#6b7280',fontSize:9}} tickLine={false}/>
                                        <YAxis tick={{fill:'#6b7280',fontSize:9}} tickLine={false} axisLine={false}/>
                                        <Tooltip contentStyle={{background:'#111827',border:'1px solid #374151',borderRadius:8,fontSize:11}}/>
                                        <Area type="monotone" dataKey="v" stroke="#22c55e" fill="url(#g)" strokeWidth={2} dot={false}/>
                                        </AreaChart>AreaChart>
                                        </ResponsiveContainer>ResponsiveContainer>
                                        </Card>Card>
                                        <div className="space-y-3">
                                        <Card>
                                        <h2 className="text-sm font-semibold text-gray-400 mb-3">Bot Controls</h2>h2>
                                        <div className="grid grid-cols-2 gap-2">
                                          {[['start','Start','bg-green-700'],['stop','Stop','bg-red-700'],['pause','Pause','bg-yellow-700'],['resume','Resume','bg-blue-700']].map(([a,l,c])=>(<button key={a} onClick={()=>act(a)} className={'py-2 rounded-lg text-xs font-medium text-white hover:opacity-80 '+c}>{l}</button>button>))}
                                        </div>div>
                                        </Card>Card>
                                        <Card>
                                        <h2 className="text-sm font-semibold text-gray-400 mb-2">Live Ticks</h2>h2>
                                          {!Object.keys(ticks).length?<p className="text-xs text-gray-500">Waiting...</p>p>:<div className="space-y-1 max-h-28 overflow-y-auto">{Object.entries(ticks).map(([s,d])=>(<div key={s} className="flex justify-between text-xs"><span className="font-medium text-white">{s}</span>span><span>{d.price?.toFixed(2)}</span>span><span className={d.change>=0?'text-green-400':'text-red-400'}>{d.change>=0?'+':''}{d.change?.toFixed(2)}%</span>span></div>div>))}</div>div>}
                                        </Card>Card>
                                        </div>div>
                                        </div>div>
                                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                                        <Card>
                                        <h2 className="text-sm font-semibold text-gray-400 mb-2">Positions</h2>h2>
                                          {!pos?.length?<p className="text-xs text-gray-500 mt-2">No open positions</p>p>:<table className="w-full text-xs mt-2"><thead><tr className="text-gray-500 border-b border-gray-800">{['Symbol','Qty','Avg','LTP','PnL'].map(h=><th key={h} className="text-left py-1 pr-2">{h}</th>th>)}</tr>tr></thead>thead><tbody>{pos.map((p,i)=>{const pnl=(p.last_price-p.average_price)*p.quantity;return(<tr key={i} className="border-b border-gray-800/40"><td className="py-1.5 pr-2 font-medium text-white">{p.tradingsymbol}</td>td><td className="pr-2">{p.quantity}</td>td><td className="pr-2">{p.average_price?.toFixed(1)}</td>td><td className="pr-2">{p.last_price?.toFixed(1)}</td>td><td className={pnl>=0?'text-green-400':'text-red-400'}>{pnl.toFixed(0)}</td>td></tr>tr>)})}</tbody>tbody></table>table>}
                                        </Card>Card>
                                        <Card>
                                        <h2 className="text-sm font-semibold text-gray-400 mb-2">Signals</h2>h2>
                                          {!sl.length?<p className="text-xs text-gray-500 mt-2">No signals</p>p>:<div className="space-y-1.5 mt-2">{sl.slice(0,8).map((s,i)=>(<div key={i} className="flex items-center gap-2 text-xs"><span className="font-medium text-white w-16 truncate">{s.symbol}</span>span><span className={'px-1.5 py-0.5 rounded-full '+(s.action==='BUY'?'bg-green-900/40 text-green-400':'bg-red-900/40 text-red-400')}>{s.action}</span>span><span className="text-gray-500 flex-1 truncate">{s.strategy}</span>span><span className="text-yellow-400">{((s.confidence||0)*100).toFixed(0)}%</span>span></div>div>))}</div>div>}
                                        </Card>Card>
                                        </div>div>
                                        <Card>
                                        <h2 className="text-sm font-semibold text-gray-400 mb-2 flex items-center gap-2"><Clock size={13}/> Log</h2>h2>
                                        <div className="max-h-36 overflow-y-auto space-y-1 font-mono">
                                          {!log.length&&<p className="text-xs text-gray-500">No activity yet...</p>p>}
                                          {log.map((l,i)=><div key={i} className="flex gap-2 text-xs"><span className="text-gray-600 shrink-0">{l.t}</span>span><span className="text-gray-300">{l.m}</span>span></div>div>)}
                                        </div>div>
                                        </Card>Card>
                                        </main>main>
                                        </div>div>
                                        )
    }</Card>
