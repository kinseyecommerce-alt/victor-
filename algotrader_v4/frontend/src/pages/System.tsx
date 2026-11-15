import { ChevronRight, LogOut, Moon, Sun, XCircle, CheckCircle2, Circle } from 'lucide-react'
import { api, auth, V2 } from '../lib/api'
import { fixed, hhmm, plain, relative, when } from '../lib/format'
import { strategyTitle } from '../lib/names'
import { Poll, usePoll } from '../lib/poll'
import { href } from '../lib/router'
import { useDialogs } from '../components/Dialogs'
import { NAV, useTheme } from '../components/Layout'
import { Badge, Button, Card, Collapse, Dot, KV, LoadState, Notice, PageHeader, Progress } from '../components/ui'
import { jobLabel } from './Overview'

const CHECK_LABEL: Record<string, string> = {
  static_ip: 'Fixed (static) internet address registered with Kite',
  api_key: 'API key protects the controls',
  jwt_secret: 'Dashboard login secret set',
  kill_switch_reset_secret: 'Secret for resetting the SEBI kill switch',
  kite_keys: 'Kite API key and secret',
  kite_auto_login: 'Daily automatic Kite login',
  kite_session: 'A working Kite session today',
  admin_ip_whitelist: 'Admin IP whitelist (kill-switch actions)',
  telegram: 'Telegram alerts',
  approval_mode: 'Approvals ON for the first LIVE weeks',
  strategies_live_capable: 'At least one strategy allowed to trade real money',
}

export function System({ overview }: { overview: Poll }) {
  const sys = usePoll(`${V2}/system`, 'slow')
  const rd = usePoll(`${V2}/readiness`, 'slow')
  const arch = usePoll(`${V2}/archiver`, 'slow')
  const { act, toast, ask } = useDialogs()
  const [dark, toggleTheme] = useTheme()
  const s = sys.data
  const r = rd.data
  const a = arch.data
  const running = overview.data?.bot_running

  const connectKite = async () => {
    try {
      const res = await api.get('/auth/login-url')
      if (res?.login_url) window.open(res.login_url, '_blank', 'noopener')
      else toast('No Kite login link (Kite API key not set?)', false)
    } catch (e: any) { toast(`Kite login link unavailable: ${e?.message || e}`, false) }
  }
  const autoLogin = () => act({
    title: 'Run the automatic Kite login now?', body: 'Logs in to Zerodha with the saved user id, password and TOTP seed (normally done at 08:45 every trading day).',
    confirmLabel: 'Log in now', path: '/api/auth/kite/auto-login', payload: () => ({}), done: 'Kite connected',
  })
  const bot = (start: boolean) => act({
    title: start ? 'Start the engine?' : 'Stop the engine?',
    body: start ? 'Every enabled strategy is set running again (this also clears an earlier Stop).'
      : 'All strategies stop taking new trades. The Stop is remembered — it lasts until you press Start (also across restarts and days).',
    confirmLabel: start ? 'Start' : 'Stop engine', danger: !start,
    path: start ? '/api/bot/start' : '/api/bot/stop', payload: () => ({}), done: start ? 'Engine started' : 'Engine stopped',
  })
  const signOut = async () => {
    const res = await ask({ title: 'Sign out?', body: 'You will need the password or API key to come back.', confirmLabel: 'Sign out' })
    if (res.ok) auth.signOut()
  }

  const usedPct = a && a.max_mb ? Math.min(1, (Number(a.archive_mb) || 0) / Number(a.max_mb)) : null

  return (
    <div className="space-y-4">
      <PageHeader title="System" subtitle="Readiness for real money, data, jobs and versions." polls={[sys, rd, arch]} />

      <Card title="LIVE (real-money) readiness">
        <LoadState poll={rd}>
          <div className="space-y-3">
            <Notice tone="red" title="LIVE trading is not possible yet">
              Everything runs on paper. None of the four strategies is allowed to trade real money — each one refuses LIVE mode by design. This checklist only shows what would be needed.
            </Notice>
            <ul className="space-y-2" data-testid="readiness-checks">
              {(r?.checks || []).map((c: any) => (
                <li key={c?.check} className="flex items-start gap-2 text-sm">
                  {c?.ok ? <CheckCircle2 size={18} className="mt-0.5 shrink-0 text-green-600" /> : c?.blocker ? <XCircle size={18} className="mt-0.5 shrink-0 text-red-600" /> : <Circle size={18} className="mt-0.5 shrink-0 text-slate-400" />}
                  <div className="min-w-0">
                    <div className="font-medium">{CHECK_LABEL[c?.check] || c?.check} {c?.blocker && !c?.ok && <Badge tone="red">blocks LIVE</Badge>}</div>
                    <div className="break-words text-xs text-slate-500 dark:text-slate-400">{c?.detail}</div>
                  </div>
                </li>
              ))}
            </ul>
            {r?.strategies && (
              <Collapse title="Per strategy">
                <ul className="space-y-1 text-sm">
                  {Object.entries<any>(r.strategies).map(([n, x]) => <li key={n}><b>{strategyTitle(n)}:</b> {x?.live_capable ? 'live-capable' : 'paper only'} <span className="text-xs text-slate-500">— {x?.why}</span></li>)}
                </ul>
              </Collapse>
            )}
          </div>
        </LoadState>
      </Card>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Card title="Kite (Zerodha)">
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-2"><Dot tone={s?.kite?.connected ? 'green' : 'red'} /> {s?.kite?.connected ? `Connected${s?.kite?.account ? ` (${s.kite.account})` : ''}` : 'Not connected'}</div>
            {!s?.kite?.connected && (s?.kite?.detail || s?.kite?.error) && <div className="text-xs text-slate-500">{s.kite.detail || s.kite.error}</div>}
            <div className="text-xs text-slate-500">API keys {s?.kite?.keys ? 'set' : 'not set'} · automatic login {s?.kite?.auto_login ? 'configured' : 'not configured'}</div>
            <div className="flex flex-wrap gap-2 pt-1">
              <Button onClick={connectKite} testid="kite-connect">Connect Kite (manual login)</Button>
              <Button onClick={autoLogin} disabled={!s?.kite?.auto_login} testid="kite-auto">Automatic login now</Button>
            </div>
          </div>
        </Card>
        <Card title="Engine">
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-2"><Dot tone={running ? 'green' : 'red'} /> {running === undefined ? '…' : running ? 'Running' : 'Stopped'}</div>
            <div className="text-xs text-slate-500">Price feed: {s?.feed?.source || '—'} · {plain(s?.feed?.quoted)}/{plain(s?.feed?.symbols)} symbols quoted{s?.feed?.last_tick_age_s !== null && s?.feed?.last_tick_age_s !== undefined ? ` · last tick ${fixed(s.feed.last_tick_age_s, 1)} s ago` : ''}</div>
            <div className="text-xs text-slate-500">SEBI kill switch: {s?.sebi?.state || s?.sebi?.kill_switch_state || '—'}{s?.sebi?.kill_reason ? ` (${s.sebi.kill_reason})` : ''}</div>
            <div className="flex gap-2 pt-1">
              <Button tone="success" onClick={() => bot(true)} disabled={running === true} testid="bot-start">Start</Button>
              <Button tone="danger" onClick={() => bot(false)} disabled={running === false} testid="bot-stop">Stop</Button>
            </div>
          </div>
        </Card>
      </div>

      <Card title="Option-bar archiver">
        <LoadState poll={arch}>
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-2"><Badge tone={a?.enabled ? 'green' : 'slate'}>{a?.enabled ? 'On' : 'Off'}</Badge> Saves NIFTY option minute bars every day for future research.</div>
            <div className="flex justify-between text-xs"><span>{fixed(a?.archive_mb, 1)} MB used of {fixed(a?.max_mb, 0)} MB</span><span>~{fixed(a?.estimate_mb_per_week, 0)} MB per week</span></div>
            <Progress value={usedPct} tone={usedPct !== null && usedPct > 0.85 ? 'amber' : 'blue'} />
            <div className="text-xs text-slate-500">Files: {a?.files !== undefined ? plain(a.files) : '—'} · last run: {a?.last_run ? when(typeof a.last_run === 'string' ? a.last_run : a.last_run?.date || a.last_run?.at) : '—'}</div>
          </div>
        </LoadState>
      </Card>

      <Card title={`Scheduled jobs (${(s?.jobs || []).length})`}>
        <LoadState poll={sys}>
          <ul className="divide-y divide-slate-100 text-sm dark:divide-slate-800" data-testid="all-jobs">
            {(s?.jobs || []).map((j: any) => (
              <li key={j.id + j.next_run} className="flex justify-between gap-2 py-1.5">
                <span className="min-w-0 truncate">{jobLabel(String(j.id))} <span className="font-mono text-xs text-slate-400">{j.id}</span></span>
                <span className="shrink-0 tabular-nums text-slate-500">{hhmm(j.next_run)} · {relative(j.next_run)}</span>
              </li>
            ))}
          </ul>
        </LoadState>
      </Card>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Card title="Paper book">
          <KV data={{ 'Saved to disk': s?.paper_book?.enabled, 'Book day': s?.paper_book?.book_day, 'Last restore': s?.paper_book?.last_restore?.restored ? 'restored' : s?.paper_book?.last_restore?.reason, 'Leftover rows': (s?.paper_book?.orphan_rows || []).length }} />
        </Card>
        <Card title="Version">
          <KV data={{ 'App version': s?.version, 'API version': s?.api_version, Mode: s?.mode, 'Server time': when(s?.time), 'Signed in as': auth.username() || (auth.apiKey() ? 'API key' : 'no login needed') }} />
        </Card>
      </div>

      <Card title="This device">
        <div className="flex flex-wrap gap-2">
          <Button onClick={toggleTheme}>{dark ? <Sun size={16} /> : <Moon size={16} />} {dark ? 'Light mode' : 'Dark mode'}</Button>
          {auth.hasCredentials() && <Button onClick={signOut} testid="sign-out"><LogOut size={16} /> Sign out</Button>}
        </div>
      </Card>
    </div>
  )
}

/** Phone "More" tab. */
export function More() {
  return (
    <div className="space-y-4">
      <PageHeader title="More" />
      <Card>
        <ul className="divide-y divide-slate-100 dark:divide-slate-800">
          {NAV.filter((n) => ['forward', 'builder', 'shadows', 'system'].includes(n.id)).map((n) => (
            <li key={n.id}><a href={href(n.id)} className="flex items-center gap-3 py-3 text-sm font-medium" data-testid={`more-${n.id}`}><n.icon size={18} /> <span className="flex-1">{n.label}</span><ChevronRight size={16} className="text-slate-400" /></a></li>
          ))}
        </ul>
      </Card>
    </div>
  )
}
