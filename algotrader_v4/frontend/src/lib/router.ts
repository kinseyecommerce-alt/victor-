import { useEffect, useState } from 'react'

// Hash routing (#/strategies/auc_fade): works under /dashboard without any server-side route changes.

export function parseHash(): string[] {
  const h = window.location.hash.replace(/^#\/?/, '')
  return h.split('?')[0].split('/').filter(Boolean).map(decodeURIComponent)
}

export function useRoute(): string[] {
  const [route, setRoute] = useState<string[]>(parseHash())
  useEffect(() => {
    const on = () => { setRoute(parseHash()); window.scrollTo(0, 0) }
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return route
}

export function href(...parts: string[]): string {
  return '#/' + parts.map(encodeURIComponent).join('/')
}

export function go(...parts: string[]) {
  window.location.hash = href(...parts).slice(1)
}
