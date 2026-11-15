import { Component, ReactNode, useEffect, useState } from 'react'
import { V2 } from './lib/api'
import { MarketContext, refreshAll, usePoll } from './lib/poll'
import { useRoute } from './lib/router'
import { DialogProvider } from './components/Dialogs'
import { Layout } from './components/Layout'
import { Login } from './components/Login'
import { Button, Notice } from './components/ui'
import { Overview } from './pages/Overview'
import { Strategies } from './pages/Strategies'
import { StrategyDetail } from './pages/StrategyDetail'
import { Trades } from './pages/Trades'
import { Risk } from './pages/Risk'
import { ForwardDetail, ForwardLab } from './pages/ForwardLab'
import { Shadows } from './pages/Shadows'
import { More, System } from './pages/System'

class PageBoundary extends Component<{ children: ReactNode; resetKey: string }, { error: Error | null }> {
  state = { error: null as Error | null }
  static getDerivedStateFromError(error: Error) { return { error } }
  componentDidUpdate(prev: { resetKey: string }) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null })
  }
  render() {
    if (this.state.error) {
      return (
        <Notice tone="red" title="This page hit a problem">
          <div className="text-xs">{String(this.state.error.message || this.state.error)}</div>
          <Button className="mt-2" onClick={() => this.setState({ error: null })}>Try again</Button>
        </Notice>
      )
    }
    return this.props.children
  }
}

function Shell({ onAuth }: { onAuth: () => void }) {
  const route = useRoute()
  const overview = usePoll(`${V2}/overview`)
  const marketOpen = Boolean(overview.data?.market?.open)
  useEffect(() => { if (overview.error?.status === 401) onAuth() }, [overview.error, onAuth])

  let page: ReactNode
  switch (route[0] || '') {
    case '': page = <Overview overview={overview} />; break
    case 'strategies': page = route[1] ? <StrategyDetail name={route[1]} /> : <Strategies />; break
    case 'trades': page = <Trades />; break
    case 'risk': page = <Risk />; break
    case 'forward': page = route[1] ? <ForwardDetail id={route[1]} /> : <ForwardLab />; break
    case 'shadows': page = <Shadows />; break
    case 'system': page = <System overview={overview} />; break
    case 'more': page = <More />; break
    default: page = <Notice>Page not found. <a className="underline" href="#/">Go to the overview</a></Notice>
  }
  return (
    <MarketContext.Provider value={{ open: marketOpen }}>
      <Layout route={route} overview={overview.data}>
        <PageBoundary resetKey={route.join('/')}>{page}</PageBoundary>
      </Layout>
    </MarketContext.Provider>
  )
}

export default function App() {
  const [needLogin, setNeedLogin] = useState(false)
  const [session, setSession] = useState(0)
  useEffect(() => {
    const on = () => setNeedLogin(true)
    window.addEventListener('auth-required', on)
    return () => window.removeEventListener('auth-required', on)
  }, [])
  return (
    <DialogProvider>
      {needLogin
        ? <Login onDone={() => { setNeedLogin(false); setSession((s) => s + 1); setTimeout(refreshAll, 50) }} />
        : <Shell key={session} onAuth={() => setNeedLogin(true)} />}
    </DialogProvider>
  )
}
