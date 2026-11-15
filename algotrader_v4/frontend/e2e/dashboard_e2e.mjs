// End-to-end check of the dashboard against a running backend (PAPER).
//
//   BASE_URL=http://127.0.0.1:8000 API_KEY=... ADMIN_USER=admin ADMIN_PASS=... OUT=/tmp/shots \
//     node algotrader_v4/frontend/e2e/dashboard_e2e.mjs
//
// Needs Playwright with Chromium (npm i -g playwright && npx playwright install chromium).
// Loads every page at phone size (iPhone 13) and desktop size, fails on any uncaught page error or unexpected
// console error, signs in through the login screen, and clicks the write controls (strategy state switch,
// dashboard on/off switch, whole-app state, approvals mode + approve / reject, brake reset), checking the result
// through the API. It CHANGES state on the target server: point it only at a local / throw-away PAPER instance.
// The forward-lab page is checked twice: as served (404 → "Not enabled yet") and with a mocked response.
// The builder page: as served (if the server has builder batch BUILDER_BATCH, default 0001, it is opened and a
// candidate expanded) and with a mocked response that has a proven cell.
import { createRequire } from 'module'
import fs from 'fs'
import path from 'path'

const require = createRequire(import.meta.url)
let playwright
try { playwright = require('playwright') } catch {
  const { execSync } = require('child_process')
  playwright = require(path.join(execSync('npm root -g').toString().trim(), 'playwright'))
}
const { chromium, devices } = playwright

const BASE = (process.env.BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
const KEY = process.env.API_KEY || ''
const USER = process.env.ADMIN_USER || ''
const PASS = process.env.ADMIN_PASS || ''
const OUT = process.env.OUT || ''
if (OUT) fs.mkdirSync(OUT, { recursive: true })

const results = []
const errors = []
const ok = (name, pass, detail = '') => { results.push({ name, pass, detail }); console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`) }

async function apiGet(p) {
  const r = await fetch(BASE + p, { headers: KEY ? { 'X-API-Key': KEY } : {} })
  if (!r.ok) throw new Error(`${p} → ${r.status}`)
  return r.json()
}
async function apiPost(p, body) {
  const r = await fetch(BASE + p, { method: 'POST', headers: { 'Content-Type': 'application/json', ...(KEY ? { 'X-API-Key': KEY } : {}) }, body: JSON.stringify(body) })
  return { status: r.status, body: await r.json().catch(() => null) }
}

// Expected network noise: the forward lab is 404 until it is deployed; the login test starts without credentials.
const EXPECTED = [/forward-lab.*404|404.*forward-lab/i, /status of 404/i, /status of 401/i]

function watch(page, label) {
  page.on('pageerror', (e) => { errors.push(`[${label}] uncaught: ${e.message}`) })
  page.on('console', (m) => {
    if (m.type() !== 'error') return
    const t = m.text()
    if (EXPECTED.some((re) => re.test(t))) return
    errors.push(`[${label}] console: ${t}`)
  })
}

async function shot(page, name) {
  if (!OUT) return
  await page.waitForTimeout(250)
  // full-page capture: let the fixed phone tab bar sit at the end of the page instead of across the middle
  const h = await page.addStyleTag({ content: '[data-testid=phone-nav]{position:static!important} [data-testid=toast]{display:none!important}' })
  await page.screenshot({ path: path.join(OUT, name + '.png'), fullPage: true })
  await h.evaluate((el) => el.remove())
}

async function waitIdle(page) {
  await page.waitForLoadState('networkidle').catch(() => {})
  await page.waitForFunction(() => !document.body.innerText.includes('Loading…'), null, { timeout: 15000 }).catch(() => {})
}

async function visit(page, hash, label) {
  await page.goto(`${BASE}/dashboard#/${hash}`)
  await waitIdle(page)
  const crashed = await page.locator('text=This page hit a problem').count()
  const [sw, cw] = await page.evaluate(() => [document.documentElement.scrollWidth, document.documentElement.clientWidth])
  ok(`${label}: #/${hash} renders`, crashed === 0 && sw <= cw + 1, sw > cw + 1 ? `page wider than the screen (${sw} > ${cw})` : '')
}

async function confirm(page, reason) {
  const dlg = page.getByTestId('confirm-dialog')
  await dlg.waitFor({ timeout: 5000 })
  if (reason !== undefined) {
    const input = dlg.locator('input')
    if (await input.count()) await input.fill(reason)
  }
  await page.getByTestId('confirm-ok').click()
  await page.getByTestId('toast').first().waitFor({ timeout: 10000 })
  await page.waitForTimeout(400)
}

const PAGES = ['', 'strategies', 'strategies/auc_fade', 'strategies/vrp_winged', 'trades', 'risk', 'forward', 'builder', 'shadows', 'system']

const MOCK_FL = {
  enabled: true, as_of: '2026-11-13',
  rule: { version: 'forward-lab/v1', family_size: 48, alpha: 0.05, z_required: 4.36, min_trades: 50, min_days: 20 },
  last_run: { date: '2026-11-13', status: 'ok', strategies_run: 48, errors: 0 },
  summary: { n_strategies: 48, qualified: 1, demoted: 3, consistency_fail: 2 },
  strategies: [
    { id: 'AUC-FADE-3PCT', family: 'auction', group: 'proven', trades: 212, days: 96, net: 48210.5, pf: 1.31, z: 4.61, status: 'qualified' },
    { id: 'SWING-ATTN-ON-3X', family: 'attention', group: 'proven', trades: 140, days: 70, net: 12890, pf: 1.12, z: 1.9, status: 'consistent' },
    { id: 'VRP-STRANGLE-016-WINGED', family: 'options', group: 'candidate', trades: 14, days: 70, net: -8120, pf: 0.8, z: null, status: 'watching' },
    { id: 'EV-PEAD-LONG-H60', family: 'events', group: 'candidate', trades: 33, days: 41, net: 61200, pf: 1.6, z: 2.4, status: 'watching' },
    { id: 'GAP-REV-1PCT', family: 'auction', group: 'forward', trades: 180, days: 96, net: -22100, pf: 0.86, z: -1.7, status: 'demoted' },
    { id: 'ORB-15M-NIFTY', family: 'breakout', group: 'forward', trades: 95, days: 96, net: 3100, pf: 1.02, z: 0.3, status: 'inconsistent' },
  ],
}
const MOCK_FL_DETAIL = {
  id: 'AUC-FADE-3PCT',
  daily: Array.from({ length: 30 }, (_, i) => ({ date: `2026-10-${String(i + 1).padStart(2, '0')}`, trades: 2, net: Math.round(Math.sin(i / 3) * 900 + 450) })),
  rule_status: { trades: 212, days: 96, z: 4.61, z_required: 4.36, cleared: true },
}


const MOCK_BUILDER = {
  enabled: true, rule_version: 'builder/v1', rule_sha256: 'a455009021b9e2cf', lab_trials_count: 2552, n_used: 3372,
  evaluated: 800, final_looks: 4, promoted: 1, chain_ok: true,
  agents: ['intraday', 'scalping', 'swing', 'options', 'futures'], market_types: ['RISING', 'FALLING', 'SIDEWAYS', 'HIGH_VOL'],
  grid: ['intraday', 'scalping', 'swing', 'options', 'futures'].flatMap((a) => ['RISING', 'FALLING', 'SIDEWAYS', 'HIGH_VOL'].map((m) => ({
    agent: a, mtype: m, proven: a === 'swing' && m === 'RISING' ? 1 : 0, in_forward_lab: a === 'swing' && m === 'RISING' ? 1 : 0, qualified: 0, ready: 0 }))),
  batches: [
    { id: '0002', created: '2026-11-21T06:00:05', hash: 'ab12cd34ef56ab12cd34', status: 'evaluated', candidates: 400, discovery_passers: 25, shuffle_expected: 7, shuffle_passers: 7, validation_passers: 2, final_confirmed: 1, proven: 1, errata: 0 },
    { id: '0001', created: '2026-11-15T08:17:11', hash: 'f506bc1888f72821', status: 'evaluated', candidates: 400, discovery_passers: 27, shuffle_expected: 6, shuffle_passers: 6, validation_passers: 2, final_confirmed: 0, proven: 0, errata: 1 },
  ],
}
const MOCK_BUILDER_BATCH = {
  id: '0002', created: '2026-11-21T06:00:05', hash: 'ab12cd34ef56ab12cd34', snapshot_end: 20261120, status: 'evaluated',
  discovery_passers: 25, shuffle_passers: 7, validation_tested: 20, validation_passers: 2, final_confirmed: 1, proven: ['bld0002_swi_ris_03_abcdef'], n_proof: 3372,
  windows: { discovery: [0, 20241120], validation: [20241121, 20251120], final: [20251121, 20261120] },
  planted: { gate_intraday: true, gate_scalping: true, gate_swing: true, gate_options: false, gate_futures: true }, errata: [],
  candidates: [
    { id: 'bld0002_swi_ris_03_abcdef', agent: 'swing', mtype: 'RISING', block: 'deliv_up', stage: 'proven',
      text: "Swing in a RISING market (NIFTY above a rising 50-day average): when delivery % >= its 20-day mean + 15 points on >= 2x volume and a >= 2 % up day, buy the stock at the next session's open; hold 20 sessions, exit at the close.",
      discovery: { trades: 812, pf: 1.42, net: 912000, h1: 401000, h2: 511000, dsr: 0.97, passed: true, shuffle_passed: false, why: [] },
      validation: { passed: true, trades: 120, pf: 1.3, net: 88000, reasons: [] }, final: { confirmed: true, trades: 70, pf: 1.2, net: 31000 },
      proof: { dsr: 0.961, gates: { validation: true, final: true, dsr: true, placebo: true, shuffle: true, planted: true }, proven: true } },
    { id: 'bld0002_int_fal_00_123456', agent: 'intraday', mtype: 'FALLING', block: 'gap_fade', stage: 'discovery',
      text: "Intraday in a FALLING market: when the stock opens >= 2% away from yesterday's close (call auction), trade the stock against the move at 09:15 (MIS); exit at 15:15.",
      discovery: { trades: 300, pf: 0.9, net: -12000, h1: -5000, h2: -7000, dsr: 0.0, passed: false, shuffle_passed: false, why: ['H1 net <= 0', 'H2 net <= 0'] } },
  ],
}

async function main() {
  const health = await apiGet('/api/v2/health')
  ok('backend health', health.status === 'ok' && health.mode === 'PAPER', `mode ${health.mode}`)
  if (health.mode !== 'PAPER') throw new Error('refusing to run against a non-PAPER server')

  // Two approval tickets with a deadline later today, so approve / reject can be clicked whatever the time.
  const ist = new Date(Date.now() + 5.5 * 3600e3).toISOString().slice(0, 10)
  for (const s of ['auc_fade', 'attn_on']) await apiPost('/api/v2/approvals', { strategy: s, deadline: `${ist}T23:59:00+05:30`, note: 'e2e' })

  const browser = await chromium.launch()
  try {
    // ── phone ─────────────────────────────────────────────────────────────
    const phoneCtx = await browser.newContext({ ...devices['iPhone 13'], deviceScaleFactor: 2, colorScheme: 'light' })
    const phone = await phoneCtx.newPage()
    watch(phone, 'phone')

    await phone.goto(`${BASE}/dashboard`)
    if (KEY && USER && PASS) {
      await phone.getByTestId('login-form').waitFor({ timeout: 10000 })
      await shot(phone, 'phone-login')
      await phone.fill('input[name=username]', USER)
      await phone.fill('input[name=password]', PASS)
      await phone.getByTestId('login-submit').click()
      await phone.getByRole('heading', { name: 'Overview' }).waitFor({ timeout: 10000 })
      ok('phone: password sign-in shows the overview', true)
    } else if (KEY) {
      await phone.evaluate((k) => localStorage.setItem('apiKey', k), KEY)
    }

    for (const p of PAGES) {
      await visit(phone, p, 'phone')
      await shot(phone, `phone-${p.replace('/', '-') || 'overview'}`)
      if (p === 'builder') {
        ok('phone: builder page shows the 5 x 4 grid', (await phone.getByTestId('builder-grid').count()) === 1)
      }
      if (p === 'forward') {
        ok('phone: forward lab shows "Not enabled yet" (404 / disabled) or the live page',
          (await phone.getByTestId('forward-not-enabled').count()) === 1 || (await phone.getByTestId('forward-maths').count()) === 1)
      }
    }
    await visit(phone, 'more', 'phone')
    await shot(phone, 'phone-more')
    for (const tab of ['History', 'Fill quality']) {
      await visit(phone, 'trades', 'phone')
      await phone.getByTestId('trades-tabs').getByRole('radio', { name: tab }).click()
      await waitIdle(phone)
      await shot(phone, `phone-trades-${tab.toLowerCase().replace(' ', '-')}`)
    }
    // filter
    await phone.getByTestId('trades-tabs').getByRole('radio', { name: 'History' }).click()
    await phone.getByTestId('filter-strategy').selectOption('attn_on')
    await waitIdle(phone)
    const filtered = await phone.locator('[data-testid=trade-cards] li').allInnerTexts()
    ok('phone: strategy filter narrows trade history', filtered.every((t) => !t.includes('Opening-auction')), `${filtered.length} rows`)

    // strategy state switch: cancel leaves it, confirm changes it, then back
    await visit(phone, 'strategies', 'phone')
    const sw = phone.getByTestId('state-switch-vrp_winged')
    await sw.getByRole('radio', { name: 'Exit-only' }).click()
    await phone.getByTestId('confirm-cancel').click()
    let st = await apiGet('/api/v2/trading-state')
    ok('state switch: Cancel changes nothing', st.strategies.vrp_winged.state === 'ACTIVE')
    await sw.getByRole('radio', { name: 'Exit-only' }).click()
    await confirm(phone, 'e2e test')
    st = await apiGet('/api/v2/trading-state')
    ok('state switch: vrp_winged → REDUCE_ONLY', st.strategies.vrp_winged.state === 'REDUCE_ONLY' && st.strategies.vrp_winged.reason === 'e2e test')
    await shot(phone, 'phone-strategies-exit-only')
    await sw.getByRole('radio', { name: 'Stopped' }).click()
    await confirm(phone)
    st = await apiGet('/api/v2/trading-state')
    ok('state switch: vrp_winged → HALTED', st.strategies.vrp_winged.state === 'HALTED')
    await sw.getByRole('radio', { name: 'Active' }).click()
    await confirm(phone)
    st = await apiGet('/api/v2/trading-state')
    ok('state switch: vrp_winged → ACTIVE', st.strategies.vrp_winged.state === 'ACTIVE')

    // dashboard on/off toggle
    await visit(phone, 'strategies/attn_on', 'phone')
    await phone.getByTestId('enable-attn_on').click()
    await confirm(phone)
    let card = await apiGet('/api/v2/strategies/attn_on')
    ok('enable toggle: attn_on off', card.enabled.toggle === false)
    await phone.getByTestId('enable-attn_on').click()
    await confirm(phone)
    card = await apiGet('/api/v2/strategies/attn_on')
    ok('enable toggle: attn_on back on', card.enabled.toggle === true)

    // approvals
    await visit(phone, 'risk', 'phone')
    await phone.getByTestId('approval-mode').click()
    await confirm(phone)
    let ap = await apiGet('/api/v2/approvals')
    ok('approvals mode → ON', ap.mode === true)
    await phone.getByTestId('pending-approvals').waitFor({ timeout: 10000 })
    await shot(phone, 'phone-risk-approvals')
    const pend = ap.pending.reduce((m, t) => ({ ...m, [t.strategy]: t.id }), {})
    if (pend.auc_fade && pend.attn_on) {
      await phone.getByTestId(`approve-${pend.auc_fade}`).click()
      await confirm(phone, 'looks fine')
      await phone.getByTestId(`reject-${pend.attn_on}`).click()
      await confirm(phone, 'not today')
      ap = await apiGet('/api/v2/approvals')
      const byId = Object.fromEntries(ap.recent.map((t) => [t.id, t.status]))
      ok('approve / reject buttons', byId[pend.auc_fade] === 'approved' && byId[pend.attn_on] === 'rejected', JSON.stringify(byId))
    } else {
      ok('approve / reject buttons', false, 'no pending tickets to decide (already decided today? use a fresh ledger)')
    }
    await phone.getByTestId('approval-mode').click()
    await confirm(phone)
    ap = await apiGet('/api/v2/approvals')
    ok('approvals mode → off', ap.mode === false)

    // whole-app state
    const g = phone.getByTestId('state-switch-global')
    await g.getByRole('radio', { name: 'Exit-only' }).click()
    await confirm(phone, 'e2e global')
    st = await apiGet('/api/v2/trading-state')
    ok('whole-app state → REDUCE_ONLY', st.global.state === 'REDUCE_ONLY')
    await waitIdle(phone)
    await shot(phone, 'phone-risk-global-exit-only')
    await g.getByRole('radio', { name: 'Active' }).click()
    await confirm(phone)
    st = await apiGet('/api/v2/trading-state')
    ok('whole-app state → ACTIVE', st.global.state === 'ACTIVE')

    // brake reset (only offered once the brake has tripped; seed a large paper loss to test it)
    let rb = await apiGet('/api/v2/risk-budget')
    if (rb.brake.state !== 'ACTIVE') {
      await waitIdle(phone)
      await shot(phone, 'phone-risk-brake-tripped')
      await phone.getByTestId('brake-reset').click()
      await confirm(phone, 'e2e reset')
      rb = await apiGet('/api/v2/risk-budget')
      ok('brake reset (tripped → ACTIVE, measured afresh)', rb.brake.state === 'ACTIVE' && Boolean(rb.brake.since), `since ${rb.brake.since}`)
    } else {
      ok('brake reset button disabled while the brake is off', await phone.getByTestId('brake-reset').isDisabled())
    }

    // dark mode
    await visit(phone, '', 'phone')
    await phone.getByTestId('theme-toggle').click()
    await waitIdle(phone)
    await shot(phone, 'phone-dark-overview')
    await visit(phone, 'strategies', 'phone')
    await shot(phone, 'phone-dark-strategies')
    await phone.getByTestId('theme-toggle').click()

    // builder: the served batch (if any) with a candidate expanded
    const bb = process.env.BUILDER_BATCH || '0001'
    const served = await fetch(`${BASE}/api/v2/builder/batches/${bb}`, { headers: KEY ? { 'X-API-Key': KEY } : {} })
    if (served.ok) {
      await visit(phone, `builder/${bb}`, 'phone')
      const n = await phone.getByTestId('builder-candidate').count()
      ok(`phone: builder batch ${bb} lists its candidates`, n > 0 && n <= 40, `${n} shown, the rest behind 'Show more'`)
      await phone.getByTestId('builder-candidate').first().locator('button').click()
      ok('phone: candidate detail opens', (await phone.getByTestId('builder-candidate-detail').count()) === 1)
      await shot(phone, `phone-builder-${bb}`)
    }
    // mocked builder (a proven cell)
    await phone.route('**/api/v2/builder', (r) => r.fulfill({ json: MOCK_BUILDER }))
    await phone.route('**/api/v2/builder/batches/*', (r) => r.fulfill({ json: MOCK_BUILDER_BATCH }))
    await visit(phone, 'builder', 'phone (mocked builder)')
    const cellTxt = await phone.getByTestId('cell-swing-RISING').innerText()
    const emptyTxt = await phone.getByTestId('cell-options-FALLING').innerText()
    ok('builder (mock): proven cell and empty cell', cellTxt.includes('1 proven') && emptyTxt.includes('none proven yet'), `${cellTxt} | ${emptyTxt}`)
    await shot(phone, 'phone-builder-mock')
    await visit(phone, 'builder/0002', 'phone (mocked builder)')
    await phone.getByTestId('builder-candidate').first().locator('button').click()
    const det = await phone.getByTestId('builder-candidate-detail').innerText()
    ok('builder (mock): proven candidate detail', det.includes('0.961') || det.includes('0.970'), det.slice(0, 80))
    await shot(phone, 'phone-builder-batch-mock')

    // mocked forward lab
    await phone.route('**/api/v2/forward-lab', (r) => r.fulfill({ json: MOCK_FL }))
    await phone.route('**/api/v2/forward-lab/*', (r) => r.fulfill({ json: MOCK_FL_DETAIL }))
    await visit(phone, 'forward', 'phone (mocked forward lab)')
    const maths = await phone.getByTestId('forward-maths').innerText()
    ok('forward lab (mock): maths bar', maths.includes('4.36') && maths.includes('48'), maths)
    await shot(phone, 'phone-forward-mock')
    await visit(phone, 'forward/AUC-FADE-3PCT', 'phone (mocked forward lab)')
    await shot(phone, 'phone-forward-detail-mock')
    await phoneCtx.close()

    // ── desktop ───────────────────────────────────────────────────────────
    const deskCtx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2, colorScheme: 'light' })
    const desk = await deskCtx.newPage()
    watch(desk, 'desktop')
    await desk.goto(`${BASE}/dashboard`)
    if (KEY) {
      await desk.getByTestId('login-form').waitFor({ timeout: 10000 })
      await desk.getByRole('radio', { name: 'API key' }).click()
      await desk.fill('input[name=apikey]', KEY)
      await desk.getByTestId('login-submit').click()
      await desk.getByRole('heading', { name: 'Overview' }).waitFor({ timeout: 10000 })
      ok('desktop: API-key sign-in shows the overview', true)
    }
    for (const p of PAGES) {
      await visit(desk, p, 'desktop')
      await shot(desk, `desktop-${p.replace('/', '-') || 'overview'}`)
    }
    await visit(desk, 'trades', 'desktop')
    await desk.getByTestId('trades-tabs').getByRole('radio', { name: 'History' }).click()
    await waitIdle(desk)
    await shot(desk, 'desktop-trades-history')
    await desk.route('**/api/v2/forward-lab', (r) => r.fulfill({ json: MOCK_FL }))
    await visit(desk, 'forward', 'desktop (mocked forward lab)')
    await desk.locator('[data-testid=forward-table] th', { hasText: 'Net P&L' }).click()
    const firstRow = await desk.locator('[data-testid=forward-table] tbody tr').first().innerText()
    ok('forward lab (mock): sortable table', firstRow.includes('EV-PEAD-LONG-H60'), firstRow.split('\n')[0])
    await shot(desk, 'desktop-forward-mock')
    await desk.route('**/api/v2/builder', (r) => r.fulfill({ json: MOCK_BUILDER }))
    await desk.route('**/api/v2/builder/batches/*', (r) => r.fulfill({ json: MOCK_BUILDER_BATCH }))
    await visit(desk, 'builder', 'desktop (mocked builder)')
    await shot(desk, 'desktop-builder-mock')
    await visit(desk, 'builder/0002', 'desktop (mocked builder)')
    await desk.getByTestId('builder-filter').selectOption('passed')
    ok('builder (mock): filter "passed discovery"', (await desk.getByTestId('builder-candidate').count()) === 1)
    await shot(desk, 'desktop-builder-batch-mock')
    await desk.getByTestId('theme-toggle-desktop').click()
    await visit(desk, '', 'desktop')
    await shot(desk, 'desktop-dark-overview')
    await deskCtx.close()
  } finally {
    await browser.close()
  }

  ok('no uncaught page errors / unexpected console errors', errors.length === 0, errors.slice(0, 10).join(' | '))
  const failed = results.filter((r) => !r.pass)
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`)
  process.exit(failed.length ? 1 : 0)
}

main().catch((e) => { console.error(e); process.exit(2) })
