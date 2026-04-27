import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import { RetrievalMode, ResultScope, SearchResponse, SearchResult } from './types'
import RagPanel from './RagPanel'
import HowItWorks from './HowItWorks'
import { TRANSCRIPT_DIMENSION_NAMES, CHUNK_DIMENSION_NAMES } from './svdDimensionNames'

interface LlmSuggestions {
  comedian?: string
  yearMin?: number
  yearMax?: number
  retrievalHint?: 'tfidf' | 'svd'
  note?: string
}

const RESULTS_PER_PAGE = 5
const MAX_TOTAL_RESULTS = 25
const YEAR_MIN = 1965
const YEAR_MAX = 2026
const SEARCH_DEBOUNCE_MS = 200

const SEARCH_LOADING_PHRASES = [
  'Digging through the archives…',
  'Checking the bit book…',
  'Sifting through 500+ specials…',
  'Asking the crowd…',
  'Consulting the green room…',
  'Mining the punchlines…',
  'Flipping through the notebooks…',
  'Running the tape…',
  'Searching the catalogue…',
  'Hunting through the set lists…',
  'Trawling the transcripts…',
  'Reading the room…',
  'Scanning the specials…',
  'Sorting through the callbacks…',
  'Parsing the premises…',
  'Weighing the word counts…',
  'Cross-referencing the bits…',
  'Checking with the booker…',
  'Asking Norm…',
  'Listening to the tapes…',
  'Reviewing the club footage…',
  'Matching topics to bits…',
  'Projecting into latent space…',
  'Pulling relevant chunks…',
  'Ranking by relevance…',
  'Vectorizing your query…',
  'Finding the setup and punchline…',
  'Cueing up the highlights…',
  'Calling in a favor from the back catalogue…',
  'Reformulating with AI…',
]

const SPECIAL_TYPE_OPTIONS = [
  '', 'Special', 'TV Appearance', 'Roast', 'Interview',
  'Monologue', 'Award Show', 'Speech', 'Crowd Work Special',
]

type Page = 'search' | 'howto'

// ── Tag color — consistent hash per tag text ──────────────────────────────────
const TAG_PALETTES = ['amber', 'sage', 'steel', 'rose', 'stone', 'teal'] as const

function tagColorClass(tag: string): string {
  let h = 0
  for (let i = 0; i < tag.length; i++) {
    h = Math.imul(31, h) + tag.charCodeAt(i) | 0
  }
  return `rcard-tag-${TAG_PALETTES[Math.abs(h) % TAG_PALETTES.length]}`
}

// ── Noise particles ───────────────────────────────────────────────────────────
function NoiseParticles(): JSX.Element {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const words = ['ha', 'ha', 'heh', 'lol', 'ha', 'haha', 'heh', 'ha']
    // classic comedy icons as inline SVGs
    const mic = `<svg width="12" height="20" viewBox="0 0 16 24" fill="none"><rect x="3" y="0" width="10" height="18" rx="5" fill="currentColor"/><line x1="3" y1="6" x2="13" y2="6" stroke="white" stroke-width="1.2" opacity=".5"/><line x1="3" y1="10" x2="13" y2="10" stroke="white" stroke-width="1.2" opacity=".5"/><rect x="7" y="18" width="2" height="6" rx="1" fill="currentColor" opacity=".7"/></svg>`
    const star = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"/></svg>`
    const bubble = `<svg width="15" height="13" viewBox="0 0 24 20" fill="currentColor"><path d="M21 2H3a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h4l3 4 3-4h8a1 1 0 0 0 1-1V3a1 1 0 0 0-1-1z"/></svg>`
    const hat = `<svg width="16" height="10" viewBox="0 0 28 16" fill="currentColor"><ellipse cx="14" cy="14" rx="13" ry="3"/><rect x="8" y="2" width="12" height="12" rx="2"/><rect x="6" y="10" width="16" height="3" rx="1"/></svg>`
    const spotlight = `<svg width="12" height="16" viewBox="0 0 16 22" fill="currentColor"><polygon points="5,0 11,0 16,22 0,22" opacity=".7"/><circle cx="8" cy="2" r="3"/></svg>`
    const icons = [mic, star, bubble, hat, spotlight]
    for (let i = 0; i < 44; i++) {
      const p = document.createElement('span')
      p.className = 'noise-particle'
      const iconRoll = i % 7
      if (iconRoll === 0 || iconRoll === 3) {
        p.innerHTML = icons[Math.floor(Math.random() * icons.length)]
        p.style.color = 'var(--cord)'
        p.style.fontSize = (0.9 + Math.random() * 0.6) + 'rem'
      } else {
        p.textContent = words[i % words.length]
        p.style.fontFamily = "'Kanit', sans-serif"
        p.style.fontWeight = '700'
        p.style.color = 'var(--cord)'
        p.style.fontSize = (0.7 + Math.random() * 1.3) + 'rem'
      }
      const x = Math.random() * 100, y = Math.random() * 130
      const d = 15 + Math.random() * 20, dl = -Math.random() * 30
      const r = (Math.random() - 0.5) * 40, op = 0.055 + Math.random() * 0.055
      p.style.left = `${x}vw`; p.style.top = `${y}vh`
      p.style.animationDuration = `${d}s`; p.style.animationDelay = `${dl}s`
      p.style.setProperty('--r', `${r}deg`); p.style.setProperty('--op', `${op}`)
      el.appendChild(p)
    }
    return () => { el.innerHTML = '' }
  }, [])
  return <div ref={ref} className="noise-layer" aria-hidden="true" />
}

// ── Platform-colored watch links ──────────────────────────────────────────────
interface WatchLink { label: string; url: string; cls: string }

function platformLinkClass(platform: string): string {
  const p = (platform || '').toLowerCase()
  if (p.includes('netflix')) return 'rcard-link rcard-link-netflix'
  if (p.includes('max') || p.includes('hbo')) return 'rcard-link rcard-link-hbo'
  if (p.includes('hulu')) return 'rcard-link rcard-link-hulu'
  if (p.includes('amazon') || p.includes('prime')) return 'rcard-link rcard-link-amazon'
  if (p.includes('apple')) return 'rcard-link rcard-link-apple'
  if (p.includes('comedy central') || p.includes('cc.com')) return 'rcard-link rcard-link-cc'
  if (p.includes('peacock')) return 'rcard-link rcard-link-peacock'
  if (p.includes('paramount')) return 'rcard-link rcard-link-paramount'
  if (p.includes('disney')) return 'rcard-link rcard-link-disney'
  return 'rcard-link rcard-link-w'
}

function getWatchLinks(result: SearchResult): WatchLink[] {
  const links: WatchLink[] = []
  const q = encodeURIComponent(`${result.comedian} ${result.special_title || result.title}`)
  if (result.url)
    links.push({ label: '📄 Transcript →', url: result.url, cls: 'rcard-link rcard-link-t' })
  if (result.watch_url)
    links.push({ label: `Watch on ${result.watch_platform || 'Watch'} →`, url: result.watch_url, cls: platformLinkClass(result.watch_platform) })
  else
    links.push({ label: 'Search on YouTube →', url: `https://www.youtube.com/results?search_query=${q}`, cls: 'rcard-link rcard-link-yt' })
  return links
}

// ── BitInsight — auto-fetch tags, lazy-fetch summary on click ─────────────────

const BIT_PHRASES = [
  'Watching the tape again…', 'Breaking down the bit…', 'Asking Mitch Hedberg…',
  'Timing the punchline…', 'Running it in the shower…', 'Consulting the writers room…',
  'Checking the joke notebook…', 'Reading between the ha-ha\'s…', 'Clearing it with the booker…',
  'Replaying it at half speed…', 'Counting the beats…', 'Checking if it\'s been done before…',
  'Dissecting the setup…', 'Getting notes from the ghost of Norm…', 'Finding the turn…',
]

interface BitInsightProps {
  result: SearchResult
  query: string
  useLlm: boolean
  onTagsReady: (tags: string[]) => void
  onTagsLoadingChange: (loading: boolean) => void
}

function BitInsight({ result, query, useLlm, onTagsReady, onTagsLoadingChange }: BitInsightProps): JSX.Element | null {
  const [open, setOpen] = useState(false)
  const [summaryLoading, setSummaryLoading] = useState(false)
  const [bitPhrase, setBitPhrase] = useState(() => BIT_PHRASES[Math.floor(Math.random() * BIT_PHRASES.length)])
  const [summary, setSummary] = useState('')
  const [summaryFetched, setSummaryFetched] = useState(false)
  const bitTimerRef = useRef<number | null>(null)

  // Auto-fetch tags on mount (tags_only=true — cheaper)
  useEffect(() => {
    if (!useLlm) return
    let cancelled = false
    onTagsLoadingChange(true)

    const snippet = (result.display_snippet || result.content || '').slice(0, 600)
    fetch('/api/bit-summary', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ snippet, comedian: result.comedian, special_title: result.special_title || result.title, query, tags_only: true }),
    })
      .then(r => r.json())
      .then(data => { if (!cancelled && data.tags?.length) onTagsReady(data.tags) })
      .catch(() => {})
      .finally(() => { if (!cancelled) onTagsLoadingChange(false) })

    return () => { cancelled = true; onTagsLoadingChange(false) }
  }, [result.chunk_id])

  const handleToggle = async () => {
    const willOpen = !open
    setOpen(willOpen)
    if (willOpen && !summaryFetched) {
      setSummaryLoading(true)
      bitTimerRef.current = window.setInterval(() => {
        setBitPhrase(BIT_PHRASES[Math.floor(Math.random() * BIT_PHRASES.length)])
      }, 1800)
      try {
        const snippet = (result.display_snippet || result.content || '').slice(0, 600)
        const res = await fetch('/api/bit-summary', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ snippet, comedian: result.comedian, special_title: result.special_title || result.title, query }),
        })
        const data = await res.json()
        setSummary(data.summary || 'No summary available.')
        if (data.tags?.length) onTagsReady(data.tags)
      } catch {
        setSummary('Could not load summary.')
      } finally {
        setSummaryLoading(false)
        setSummaryFetched(true)
        if (bitTimerRef.current) { clearInterval(bitTimerRef.current); bitTimerRef.current = null }
      }
    }
  }

  if (!useLlm) return null

  return (
    <div className="bit-wrap">
      <button className={`bit-toggle${open ? ' open' : ''}`} onClick={() => void handleToggle()}>
        <span>✦ On this bit</span>
        <span className="bit-chev">▼</span>
      </button>
      <div className={`bit-body${open ? ' open' : ''}`}>
        {summaryLoading
          ? <div className="bit-loading-row">
              <div className="bit-dots"><span /><span /><span /></div>
              <span className="bit-phrase">{bitPhrase}</span>
            </div>
          : summary}
      </div>
    </div>
  )
}

// ── Pagination bar ────────────────────────────────────────────────────────────
interface PaginationBarProps {
  page: number; totalPages: number; totalResults: number; query: string
  onPrev: () => void; onNext: () => void; resultsPerPage: number
}

function PaginationBar({ page, totalPages, totalResults, query, onPrev, onNext, resultsPerPage }: PaginationBarProps): JSX.Element | null {
  if (totalPages <= 1) return null
  const from = page * resultsPerPage + 1
  const to = Math.min((page + 1) * resultsPerPage, totalResults)
  return (
    <div className="pgbar">
      <button className="pgbtn" disabled={page === 0} onClick={onPrev}>← Previous</button>
      <span className="pginfo">
        Showing {from}–{to} of {totalResults}{query ? <> for <em>"{query}"</em></> : ''}
      </span>
      <button className="pgbtn" disabled={page >= totalPages - 1} onClick={onNext}>Next →</button>
    </div>
  )
}

// ── Comedian autocomplete input ───────────────────────────────────────────────
interface ComedianInputProps { value: string; onChange: (v: string) => void; suggestions: string[] }

function ComedianInput({ value, onChange, suggestions }: ComedianInputProps): JSX.Element {
  const [open, setOpen] = useState(false)
  const [cursor, setCursor] = useState(-1)
  const wrapRef = useRef<HTMLDivElement>(null)

  const filtered = value.trim().length > 0
    ? suggestions.filter(s => s.toLowerCase().includes(value.toLowerCase())).slice(0, 8)
    : []

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const pick = (name: string) => { onChange(name); setOpen(false); setCursor(-1) }

  const highlightMatch = (name: string) => {
    const idx = name.toLowerCase().indexOf(value.toLowerCase())
    if (idx === -1) return name
    return <>{name.slice(0, idx)}<mark>{name.slice(idx, idx + value.length)}</mark>{name.slice(idx + value.length)}</>
  }

  return (
    <div className="autocomplete-wrap" ref={wrapRef}>
      <input
        className="sb-input"
        type="text"
        value={value}
        placeholder="Any comedian"
        onChange={e => { onChange(e.target.value); setOpen(true); setCursor(-1) }}
        onFocus={() => { if (filtered.length) setOpen(true) }}
        onKeyDown={e => {
          if (!open || !filtered.length) return
          if (e.key === 'ArrowDown') { e.preventDefault(); setCursor(c => Math.min(c + 1, filtered.length - 1)) }
          else if (e.key === 'ArrowUp') { e.preventDefault(); setCursor(c => Math.max(c - 1, 0)) }
          else if (e.key === 'Enter' && cursor >= 0) { e.preventDefault(); pick(filtered[cursor]) }
          else if (e.key === 'Escape') setOpen(false)
        }}
      />
      {open && filtered.length > 0 && (
        <div className="autocomplete-dropdown">
          {filtered.map((name, i) => (
            <div key={name} className={`autocomplete-item${i === cursor ? ' active' : ''}`}
              onMouseDown={() => pick(name)}>
              {highlightMatch(name)}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Empty state ───────────────────────────────────────────────────────────────
type DemoPhase = 'typing' | 'loading' | 'results' | 'filter' | 'filter-reload' | 'results2' | 'ai' | 'ai-show' | 'erasing'

const DEMO_DATA = [
  {
    q: 'men not going to therapy', activeFilter: 'chunks' as const,
    r1: '<strong>Trevor Noah</strong> · Son of Patricia — "Men will google symptoms for three hours before admitting they might need help."',
    r2: '<strong>Ali Wong</strong> · Baby Cobra — "He\'s not broken, he\'s just never been asked how he feels. Ever."',
    ai: 'Comedians dissect male emotional avoidance with equal parts frustration and sympathy — therapy as the punchline men keep refusing to get to.',
  },
  {
    q: 'petition to ban dating apps', activeFilter: 'profanity' as const,
    r1: '<strong>Aziz Ansari</strong> · Modern Romance — "Every bad decision I\'ve made in the last five years started with a swipe."',
    r2: '<strong>John Mulaney</strong> · The Comeback Kid — "We replaced meeting people with scrolling past them at speed."',
    ai: 'Comedians agree: dating apps promised infinite choice and delivered infinite anxiety — a catalogue of disappointment with better UI.',
  },
  {
    q: 'pretending to have your life together', activeFilter: 'year' as const,
    r1: '<strong>Mike Birbiglia</strong> · My Girlfriend\'s Boyfriend — "I bought a planner. That\'s as far as I got."',
    r2: '<strong>Iliza Shlesinger</strong> · War Paint — "We\'re all performing adulthood for an audience that\'s also performing adulthood."',
    ai: 'Comedians expose the shared performance of competence — everyone faking confidence while quietly hoping no one checks the group chat.',
  },
]

function EmptyState({ onSearch }: { onSearch: (q: string) => void }): JSX.Element {
  const [demoText, setDemoText] = useState('')
  const [qIdx, setQIdx] = useState(0)
  const [charIdx, setCharIdx] = useState(0)
  const [phase, setPhase] = useState<DemoPhase>('typing')
  const [showR1, setShowR1] = useState(false)
  const [showR2, setShowR2] = useState(false)
  const [showAi, setShowAi] = useState(false)
  const [activeFilter, setActiveFilter] = useState<'chunks' | 'year' | 'profanity' | null>(null)
  const [activeAi, setActiveAi] = useState(false)
  const [loaderWidth, setLoaderWidth] = useState(0)
  const timerRef = useRef<number | null>(null)

  const go = (nextPhase: DemoPhase, delay: number) => {
    timerRef.current = window.setTimeout(() => setPhase(nextPhase), delay)
  }

  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current)
    const d = DEMO_DATA[qIdx]

    if (phase === 'typing') {
      if (charIdx < d.q.length) {
        timerRef.current = window.setTimeout(() => {
          setDemoText(d.q.slice(0, charIdx + 1)); setCharIdx(c => c + 1)
        }, 85)
      } else { go('loading', 300) }
    } else if (phase === 'loading') {
      setLoaderWidth(0)
      timerRef.current = window.setTimeout(() => setLoaderWidth(100), 50)
      go('results', 1000)
    } else if (phase === 'results') {
      setShowR1(true)
      timerRef.current = window.setTimeout(() => { setShowR2(true); go('filter', 1400) }, 320)
    } else if (phase === 'filter') {
      setActiveFilter(d.activeFilter)
      go('filter-reload', 800)
    } else if (phase === 'filter-reload') {
      setShowR1(false); setShowR2(false)
      setLoaderWidth(0)
      timerRef.current = window.setTimeout(() => setLoaderWidth(100), 50)
      go('results2', 1000)
    } else if (phase === 'results2') {
      setShowR1(true)
      timerRef.current = window.setTimeout(() => { setShowR2(true); go('ai', 1200) }, 300)
    } else if (phase === 'ai') {
      setActiveAi(true); go('ai-show', 500)
    } else if (phase === 'ai-show') {
      setShowAi(true); go('erasing', 2800)
    } else if (phase === 'erasing') {
      if (charIdx > 0) {
        timerRef.current = window.setTimeout(() => { setCharIdx(c => c - 1); setDemoText(d.q.slice(0, charIdx - 1)) }, 35)
      } else {
        setShowR1(false); setShowR2(false); setShowAi(false)
        setActiveFilter(null); setActiveAi(false); setLoaderWidth(0)
        timerRef.current = window.setTimeout(() => {
          setQIdx(i => (i + 1) % DEMO_DATA.length); setPhase('typing')
        }, 500)
      }
    }
    return () => { if (timerRef.current) clearTimeout(timerRef.current) }
  }, [phase, charIdx, qIdx])

  const d = DEMO_DATA[qIdx]

  return (
    <div className="e-grid">
      <div className="ecard">
        <div className="eicon">🎤</div>
        <h3 className="ecard-title">Try a search</h3>
        <p className="ep">Live demo — typing, filtering, and AI summary:</p>
        <div className="demo-bar">
          <span>{demoText}</span><span className="demo-cursor" />
        </div>
        <div className="demo-filter-row">
          <span>Filters:</span>
          <span className={`demo-filter-chip${activeFilter === 'chunks' ? ' active' : ''}`}>Chunks</span>
          <span className={`demo-filter-chip${activeFilter === 'year' ? ' active' : ''}`}>1990–2010</span>
          <span className={`demo-filter-chip${activeFilter === 'profanity' ? ' active' : ''}`}>Exclude profanity</span>
          <span className={`demo-ai-chip${activeAi ? ' active' : ''}`}>✦ AI</span>
        </div>
        <div className="demo-loader">
          <div className="demo-loader-fill" style={{ width: `${loaderWidth}%` }} />
        </div>
        <div className="demo-results">
          <div className={`demo-result${showR1 ? ' show' : ''}`} dangerouslySetInnerHTML={{ __html: d.r1 }} />
          <div className={`demo-result${showR2 ? ' show' : ''}`} dangerouslySetInnerHTML={{ __html: d.r2 }} />
        </div>
        <div className={`demo-ai-block${showAi ? ' show' : ''}`}>
          <div className="demo-ai-lbl">✦ AI Summary</div>
          {d.ai}
        </div>
      </div>
      <div className="ecard">
        <div className="eicon">⚙️</div>
        <h3 className="ecard-title">How it works</h3>
        <div className="how-step"><div className="sn">1</div><div>500+ transcripts indexed as full specials and short semantic chunks.</div></div>
        <div className="how-step"><div className="sn">2</div><div>Matched via <strong>TF-IDF</strong> or <strong>SVD</strong>. Use filters on the left to narrow by comedian, year, or type.</div></div>
        <div className="how-step"><div className="sn">3</div><div>Best sentence highlighted. Tap <strong>✦ On this bit</strong> for an AI read on the specific moment.</div></div>
        <div className="how-step"><div className="sn">4</div><div><strong>✦ AI Summary</strong> in the sidebar synthesises the full results page.</div></div>
        <div className="e-tag-row">
          <span className="etag">TF-IDF</span><span className="etag">SVD / LSA</span>
          <span className="etag">Chunks</span><span className="etag">✦ AI summary</span>
          <span className="etag">✦ On this bit</span><span className="etag">500+ specials</span>
        </div>
        <div className="demo-chip-row" style={{ marginTop: '4px' }}>
          {['men not going to therapy', 'petition to ban dating apps', 'pretending to have your life together'].map(q => (
            <button key={q} className="demo-chip" onClick={() => onSearch(q)}>{q}</button>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Main app ──────────────────────────────────────────────────────────────────
function App(): JSX.Element {
  const [activePage, setActivePage] = useState<Page>('search')
  const [darkMode, setDarkMode] = useState(false)
  const [useLlm, setUseLlm] = useState<boolean | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [allResults, setAllResults] = useState<SearchResult[]>([])
  const [resolvedComedian, setResolvedComedian] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)
  const [page, setPage] = useState(0)
  const [comedian, setComedian] = useState('')
  const [specialType, setSpecialType] = useState('')
  const [yearMin, setYearMin] = useState(YEAR_MIN)
  const [yearMax, setYearMax] = useState(YEAR_MAX)
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>('tfidf')
  const [resultScope, setResultScope] = useState<ResultScope>('full')
  const [excludeProfanity, setExcludeProfanity] = useState(false)
  const [cardTags, setCardTags] = useState<Record<string, string[]>>({})
  const [loadingTagCards, setLoadingTagCards] = useState<Set<string>>(new Set())
  const [comedianList, setComedianList] = useState<string[]>([])
  const [modifiedQuery, setModifiedQuery] = useState<string | null>(null)
  const [queryWasModified, setQueryWasModified] = useState(false)
  const [llmSuggestions, setLlmSuggestions] = useState<LlmSuggestions | null>(null)
  const [ragTrigger, setRagTrigger] = useState(0)
  const [searchPhrase, setSearchPhrase] = useState(SEARCH_LOADING_PHRASES[0])
  const [expandedSvd, setExpandedSvd] = useState<Set<string>>(new Set())

  const debounceRef = useRef<number | null>(null)
  const latestRef = useRef(0)
  const searchPhraseTimerRef = useRef<number | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)

  useEffect(() => { fetch('/api/config').then(r => r.json()).then(d => setUseLlm(d.use_llm)) }, [])
  useEffect(() => { fetch('/api/comedians').then(r => r.json()).then(d => setComedianList(d)).catch(() => {}) }, [])
  useEffect(() => { document.body.classList.toggle('dark', darkMode) }, [darkMode])

  const clearDebounce = () => {
    if (debounceRef.current !== null) { window.clearTimeout(debounceRef.current); debounceRef.current = null }
  }

  const buildSearchContext = () => {
    const q = searchTerm.trim()
    if (q) return q
    const parts: string[] = []
    if (comedian.trim()) parts.push(comedian.trim())
    if (specialType.trim()) parts.push(specialType.trim())
    if (yearMin !== YEAR_MIN || yearMax !== YEAR_MAX) parts.push(`${yearMin}–${yearMax}`)
    return parts.join(', ')
  }

  const stopSearchPhrases = () => {
    if (searchPhraseTimerRef.current !== null) {
      clearInterval(searchPhraseTimerRef.current)
      searchPhraseTimerRef.current = null
    }
  }

  const startSearchPhrases = () => {
    stopSearchPhrases()
    setSearchPhrase(SEARCH_LOADING_PHRASES[Math.floor(Math.random() * SEARCH_LOADING_PHRASES.length)])
    searchPhraseTimerRef.current = window.setInterval(() => {
      setSearchPhrase(SEARCH_LOADING_PHRASES[Math.floor(Math.random() * SEARCH_LOADING_PHRASES.length)])
    }, 3000)
  }

  const runSearch = async (query: string) => {
    const trimmed = query.trim()
    if (!trimmed) {
      latestRef.current += 1
      stopSearchPhrases()
      setAllResults([]); setResolvedComedian(null); setPage(0)
      setHasSearched(false); setLoading(false); setCardTags({})
      setModifiedQuery(null); setQueryWasModified(false); setLlmSuggestions(null); return
    }
    const id = latestRef.current + 1; latestRef.current = id
    if (abortControllerRef.current) abortControllerRef.current.abort()
    const abort = new AbortController()
    abortControllerRef.current = abort
    setLoading(true); setHasSearched(true); setCardTags({})
    setModifiedQuery(null); setQueryWasModified(false); setLlmSuggestions(null)
    startSearchPhrases()

    const buildParams = (q: string, com: string) => new URLSearchParams({
      query: q, top_k: String(MAX_TOTAL_RESULTS),
      retrieval_mode: retrievalMode, result_scope: resultScope,
      comedian: com, special_type: specialType,
      year_min: String(yearMin), year_max: String(yearMax),
      exclude_profanity: String(excludeProfanity), max_chunks_per_doc: '2',
    })

    let searchQuery = trimmed
    let searchComedian = comedian

    if (useLlm) {
      // Wait for LLM reformulation first, then run a single search
      try {
        const mqData = await fetch('/api/reformulate-query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: trimmed }),
          signal: abort.signal,
        }).then(r => r.ok ? r.json() : null).catch(() => null)

        if (id !== latestRef.current) { stopSearchPhrases(); return }

        if (mqData) {
          const mq: string = (mqData.modified_query || '').trim()
          const llmComedian: string = mqData.comedian || ''
          const queryChanged = mq && mq.toLowerCase() !== trimmed.toLowerCase()

          const sugg: LlmSuggestions = {}
          if (mqData.year_min) sugg.yearMin = mqData.year_min
          if (mqData.year_max) sugg.yearMax = mqData.year_max
          if (mqData.retrieval_hint === 'tfidf' || mqData.retrieval_hint === 'svd') sugg.retrievalHint = mqData.retrieval_hint
          if (mqData.note) sugg.note = mqData.note
          setLlmSuggestions(Object.keys(sugg).length > 0 ? sugg : null)

          if (mq) { setModifiedQuery(mq); setQueryWasModified(!!queryChanged); searchQuery = mq }
          if (llmComedian) { searchComedian = llmComedian }
        }
      } catch { /* LLM failed — search with original query */ }

      if (id !== latestRef.current) { stopSearchPhrases(); return }
    }

    try {
      const data: SearchResponse = await fetch(`/api/search?${buildParams(searchQuery, searchComedian)}`, { signal: abort.signal }).then(r => r.json())
      if (id !== latestRef.current) { stopSearchPhrases(); return }
      setAllResults(data.results ?? []); setResolvedComedian(data.resolved_comedian ?? null); setPage(0)
      if (searchComedian && searchComedian !== comedian) setComedian(searchComedian)
      if ((data.results ?? []).length > 0) { setRagTrigger(t => t + 1) }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') return
      if (id === latestRef.current) setAllResults([])
    } finally {
      if (id === latestRef.current) { setLoading(false); stopSearchPhrases() }
    }
  }

  const handleDemoSearch = (q: string) => { setSearchTerm(q); void runSearch(q) }

  useEffect(() => {
    clearDebounce()
    if (!searchTerm.trim()) {
      latestRef.current += 1
      stopSearchPhrases()
      setAllResults([]); setResolvedComedian(null); setPage(0)
      setHasSearched(false); setLoading(false); setCardTags({})
      setModifiedQuery(null); setQueryWasModified(false); setLlmSuggestions(null); return
    }
    debounceRef.current = window.setTimeout(() => { void runSearch(searchTerm) }, SEARCH_DEBOUNCE_MS)
    return clearDebounce
  }, [searchTerm, comedian, specialType, yearMin, yearMax, retrievalMode, resultScope, excludeProfanity])

  const visibleResults = useMemo(() => {
    const s = page * RESULTS_PER_PAGE; return allResults.slice(s, s + RESULTS_PER_PAGE)
  }, [allResults, page])

  const totalPages = Math.ceil(allResults.length / RESULTS_PER_PAGE)

  const renderSnippet = (result: SearchResult): JSX.Element | string => {
    if (result.snippet_sentences && result.snippet_sentences.length > 0) {
      const base = result.global_snippet_start ?? result.snippet_sentence_start ?? 0
      return (
        <>
          {result.snippet_sentences.map((s, i) => (
            <span key={`${result.chunk_id}-s${i}`} className={(base + i) === result.best_sentence_index ? 'highlight-sentence' : ''}>
              {s + ' '}
            </span>
          ))}
        </>
      )
    }
    return result.display_snippet || result.content
  }

  if (useLlm === null) return <></>

  const pgProps = {
    page, totalPages, totalResults: allResults.length,
    query: searchTerm.trim(), onPrev: () => setPage(p => p - 1),
    onNext: () => setPage(p => p + 1), resultsPerPage: RESULTS_PER_PAGE,
  }

  return (
    <div className="app-root">
      <NoiseParticles />

      {activePage === 'search' && (
        <aside className="sidebar">
          <div className="sb-heading">Filters</div>

          <div className="sb-sect">
            <div className="sb-lbl">Comedian{comedian && <button className="sb-clear-x" onClick={() => setComedian('')}>×</button>}</div>
            <ComedianInput value={comedian} onChange={setComedian} suggestions={comedianList} />
          </div>
          <div className="sb-sect">
            <div className="sb-lbl">Special type{specialType && <button className="sb-clear-x" onClick={() => setSpecialType('')}>×</button>}</div>
            <select className="sb-sel" value={specialType} onChange={e => setSpecialType(e.target.value)}>
              {SPECIAL_TYPE_OPTIONS.map(o => <option key={o} value={o}>{o || 'Any type'}</option>)}
            </select>
          </div>
          <div className="sb-sect">
            <div className="sb-lbl">Year range{(yearMin !== YEAR_MIN || yearMax !== YEAR_MAX) && <button className="sb-clear-x" onClick={() => { setYearMin(YEAR_MIN); setYearMax(YEAR_MAX) }}>×</button>}</div>
            <div className="range-wrap">
              <div className="range-vals"><span>{yearMin}</span><span>{yearMax}</span></div>
              <input type="range" className="range-from" min={YEAR_MIN} max={YEAR_MAX} value={yearMin}
                onChange={e => setYearMin(Math.min(Number(e.target.value), yearMax))} />
              <input type="range" className="range-to" min={YEAR_MIN} max={YEAR_MAX} value={yearMax}
                onChange={e => setYearMax(Math.max(Number(e.target.value), yearMin))} />
            </div>
          </div>
          <div className="sb-sect">
            <div className="sb-lbl">Profanity</div>
            <div className="prof-pills">
              <button className={`ppill${!excludeProfanity ? ' ppill-inc' : ''}`} onClick={() => setExcludeProfanity(false)}>Include</button>
              <button className={`ppill${excludeProfanity ? ' ppill-exc' : ''}`} onClick={() => setExcludeProfanity(true)}>Exclude</button>
            </div>
          </div>
          <button className="clear-all-btn" onClick={() => { setComedian(''); setSpecialType(''); setYearMin(YEAR_MIN); setYearMax(YEAR_MAX); setExcludeProfanity(false) }}>↺ Clear all filters</button>

          {useLlm && (
            <>
              <hr className="ai-divider" />
              <div className="ai-section-lbl"><span>✦</span> AI Summary</div>
              <RagPanel
                query={modifiedQuery ?? buildSearchContext()}
                visibleResults={visibleResults}
                hasResults={allResults.length > 0 && hasSearched}
                externalTrigger={ragTrigger}
                inline
              />
              {(!allResults.length || !hasSearched) && (
                <p className="ai-empty-note">Run a search to generate an AI summary.</p>
              )}
            </>
          )}
        </aside>
      )}

      <div className="main">
        <nav className="app-nav">
          <button className={`nav-btn${activePage === 'search' ? ' nav-btn-active' : ''}`} onClick={() => setActivePage('search')}>Search</button>
          <button className={`nav-btn${activePage === 'howto' ? ' nav-btn-active' : ''}`} onClick={() => setActivePage('howto')}>How it works</button>
          <div className="nav-sep" />
          <button className="dkbtn" onClick={() => setDarkMode(d => !d)}>{darkMode ? '☀ Light' : '◑ Dark'}</button>
        </nav>

        {activePage === 'howto' ? <HowItWorks /> : (
          <>
            <div className="logo-wrap">
              <svg id="lsvg" width="420" height="150" viewBox="0 0 360 136" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ maxWidth: '100%', height: 'auto' }}>
                <defs>
                  <linearGradient id="logo-cord" x1="312" y1="68" x2="14" y2="116" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stopColor="#7c3aed"/><stop offset="55%" stopColor="#7c3aed" stopOpacity="0.5"/><stop offset="100%" stopColor="#7c3aed" stopOpacity="0.04"/>
                  </linearGradient>
                </defs>
                <text className="logo-text" x="14" y="82" fontFamily="'Kanit', sans-serif" fontSize="64" fontWeight="700" fill="#2b1e3f" letterSpacing="-2">laughDB</text>
                <path className="cord" d="M 294 64 C 308 66, 320 74, 318 88 C 316 100, 304 104, 294 98" stroke="#7c3aed" strokeWidth="2.2" strokeLinecap="round" opacity="0.58" fill="none"/>
                <ellipse cx="297" cy="68" rx="4.5" ry="3" fill="var(--bg)"/>
                <path className="cord" d="M 292 62 C 300 64, 314 70, 318 84 C 322 98, 310 108, 296 104 C 282 100, 278 88, 286 80 C 292 74, 302 78, 300 86" stroke="#7c3aed" strokeWidth="2.2" strokeLinecap="round" fill="none"/>
                <path className="cord" d="M 282 102 C 266 106, 248 108, 232 104" stroke="#7c3aed" strokeWidth="2.2" strokeLinecap="round" opacity="0.8" fill="none"/>
                <path className="cord" d="M 232 104 C 222 100, 218 92, 224 88 C 230 84, 240 88, 238 96 C 236 104, 226 106, 222 102" stroke="#7c3aed" strokeWidth="2.2" strokeLinecap="round" fill="none"/>
                <path className="cord" d="M 220 100 C 192 104, 112 108, 14 104" stroke="url(#logo-cord)" strokeWidth="2.2" strokeLinecap="round" fill="none"/>
                <g className="mic-group">
                  <rect x="280" y="8" width="16" height="30" rx="8" fill="#7c3aed"/>
                  <line x1="280" y1="18" x2="296" y2="18" stroke="#f7f4ef" strokeWidth="1.3" opacity="0.45"/>
                  <line x1="280" y1="25" x2="296" y2="25" stroke="#f7f4ef" strokeWidth="1.3" opacity="0.45"/>
                  <line x1="280" y1="32" x2="296" y2="32" stroke="#f7f4ef" strokeWidth="1.3" opacity="0.45"/>
                  <rect x="286" y="38" width="4" height="10" rx="2" fill="#7c3aed" opacity="0.65"/>
                  <circle cx="285" cy="16" r="2.2" fill="white" opacity="0.9"/><circle cx="291" cy="16" r="2.2" fill="white" opacity="0.9"/>
                  <circle className="pup-l" cx="285" cy="16" r="1.1" fill="#2b1e3f"/><circle className="pup-r" cx="291" cy="16" r="1.1" fill="#2b1e3f"/>
                </g>
                <g className="sw">
                  <path d="M274 18 C270 21 270 27 274 30" stroke="#7c3aed" strokeWidth="1.8" strokeLinecap="round" fill="none"/>
                  <path d="M270 14 C264 19 264 29 270 34" stroke="#7c3aed" strokeWidth="1.8" strokeLinecap="round" fill="none" opacity="0.6"/>
                  <path d="M266 10 C258 17 258 31 266 38" stroke="#7c3aed" strokeWidth="1.5" strokeLinecap="round" fill="none" opacity="0.3"/>
                </g>
              </svg>
            </div>
            <p className="brand-sub">Search stand-up comedy transcripts by topic</p>

            <div className="sw-wrap">
              <div className="sbox-inner" onClick={() => document.getElementById('search-input')?.focus()}>
                <img src={SearchIcon} alt="" className="sbox-icon" />
                <input id="search-input" className="sbox" placeholder="Search comedy transcript topics"
                  value={searchTerm} onChange={e => setSearchTerm(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); void runSearch(searchTerm) } }} />
                <button className="sbox-btn" onClick={() => void runSearch(searchTerm)}>Search</button>
              </div>
            </div>

            <div className="pills-row">
              <div className="mpill">
                {(['tfidf', 'svd'] as RetrievalMode[]).map(m => (
                  <button key={m} className={`mopt${retrievalMode === m ? ' mopt-on' : ''}`} onClick={() => setRetrievalMode(m)}>
                    {m === 'tfidf' ? 'TF-IDF' : 'SVD'}
                  </button>
                ))}
              </div>
              <div className="mpill">
                {(['full', 'chunks'] as ResultScope[]).map(s => (
                  <button key={s} className={`mopt${resultScope === s ? ' mopt-on' : ''}`} onClick={() => setResultScope(s)}>
                    {s === 'full' ? 'Full special' : 'Chunks'}
                  </button>
                ))}
              </div>
            </div>

            <div className="results-column">
              {!hasSearched && !loading && <EmptyState onSearch={handleDemoSearch} />}
              {resolvedComedian && <div className="info-banner">Best matched comedian: <strong>{resolvedComedian}</strong></div>}
              {excludeProfanity && allResults.length > 0 && (
                <div className="warning-banner">
                  {resultScope === 'chunks' ? 'Profanity filter on. These chunks are clean — the full special may still contain profanity.' : 'Profanity filter on. Results containing profanity are excluded.'}
                </div>
              )}
              {loading && (
                <div className="search-loading-banner">
                  <div className="slb-dots"><span /><span /><span /></div>
                  <span className="slb-phrase">{searchPhrase}</span>
                </div>
              )}
              {!loading && hasSearched && visibleResults.length === 0 && <div className="info-banner">No results found.</div>}

              {!loading && allResults.length > 0 && <PaginationBar {...pgProps} />}

              {!loading && modifiedQuery && allResults.length > 0 && (
                <div className="modified-query-banner">
                  <div className="mqb-main">
                    {queryWasModified ? 'LLM modified query:' : 'LLM kept your query:'} <em>{modifiedQuery}</em>
                  </div>
                  {(llmSuggestions?.yearMin || llmSuggestions?.yearMax) && (
                    <div className="mqb-suggestion">💡 Era detected: <strong>{llmSuggestions.yearMin ?? '?'}–{llmSuggestions.yearMax ?? 'present'}</strong> — adjust the year range filter</div>
                  )}
                  {llmSuggestions?.retrievalHint && llmSuggestions.retrievalHint !== retrievalMode && (
                    <div className="mqb-suggestion">💡 Try <strong>{llmSuggestions.retrievalHint === 'svd' ? 'SVD' : 'TF-IDF'}</strong> mode for this type of query</div>
                  )}
                  {llmSuggestions?.note && (
                    <div className="mqb-note">⚠️ {llmSuggestions.note}</div>
                  )}
                </div>
              )}

              {visibleResults.length > 0 && <div className="rcard-list">
                {visibleResults.map((result, idx) => {
                  const tags = cardTags[result.chunk_id] ?? []
                  return (
                    <div key={`${result.chunk_id}-${idx}`} className="rcard-d2">
                      <div className="rcard-left">
                        <div className="rcard-comedian">{result.comedian || 'Unknown'}</div>
                        <div className="rcard-show">{result.special_title || result.title}</div>
                        <div className="rcard-year-info">{result.release_date}{result.watch_platform ? ` · ${result.watch_platform}` : ''}</div>
                        <div className="rcard-score-wrap">
                          <div className="rcard-score-num">{result.similarity_percent?.toFixed(0) ?? '—'}<span className="rcard-pct">%</span></div>
                          <div className="rcard-score-lbl">relevance</div>
                        </div>
                      </div>
                      <div className="rcard-right">
                        <p className="rcard-snippet">{renderSnippet(result)}</p>

                        {result.retrieval_mode === 'svd' && (
                          <div className="svd-explanation-box">
                            <button
                              className={`svd-toggle-btn${expandedSvd.has(result.chunk_id) ? ' open' : ''}`}
                              onClick={() => setExpandedSvd(prev => {
                                const next = new Set(prev)
                                next.has(result.chunk_id) ? next.delete(result.chunk_id) : next.add(result.chunk_id)
                                return next
                              })}
                            >
                              <span>{expandedSvd.has(result.chunk_id) ? 'Hide SVD explanation' : 'Show SVD explanation'}</span>
                              <span className="svd-chev">▼</span>
                            </button>
                            {expandedSvd.has(result.chunk_id) && (() => {
                              const dimNames = result.result_scope === 'chunks' ? CHUNK_DIMENSION_NAMES : TRANSCRIPT_DIMENSION_NAMES
                              return (<>
                                {result.svd_positive_dimensions && result.svd_positive_dimensions.length > 0 && (
                                  <div className="svd-positive">
                                    <p className="svd-heading"><strong>Positive latent dimensions</strong></p>
                                    {result.svd_positive_dimensions.map(dim => (
                                      <div key={`pos-${result.chunk_id}-${dim.dimension}`} className="svd-dimension">
                                        <span className="meta-pill subtle-pill">Dim {dim.dimension}{dimNames[dim.dimension] ? ` · ${dimNames[dim.dimension]}` : ''} <span className="svd-contribution">· {dim.contribution.toFixed(3)}</span></span>
                                        <p className="svd-terms">+ {dim.top_positive_terms?.join(', ')}</p>
                                        <p className="svd-terms">− {dim.top_negative_terms?.join(', ')}</p>
                                      </div>
                                    ))}
                                  </div>
                                )}
                                {result.svd_negative_dimensions && result.svd_negative_dimensions.length > 0 && (
                                  <div className="svd-negative">
                                    <p className="svd-heading"><strong>Negative latent dimensions</strong></p>
                                    {result.svd_negative_dimensions.map(dim => (
                                      <div key={`neg-${result.chunk_id}-${dim.dimension}`} className="svd-dimension">
                                        <span className="meta-pill subtle-pill">Dim {dim.dimension}{dimNames[dim.dimension] ? ` · ${dimNames[dim.dimension]}` : ''} <span className="svd-contribution">· {dim.contribution.toFixed(3)}</span></span>
                                        <p className="svd-terms">+ {dim.top_positive_terms?.join(', ')}</p>
                                        <p className="svd-terms">− {dim.top_negative_terms?.join(', ')}</p>
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </>)
                            })()}
                          </div>
                        )}

                        <BitInsight result={result} query={buildSearchContext()} useLlm={useLlm}
                          onTagsReady={t => setCardTags(prev => ({ ...prev, [result.chunk_id]: t }))}
                          onTagsLoadingChange={loading => setLoadingTagCards(prev => {
                            const next = new Set(prev)
                            loading ? next.add(result.chunk_id) : next.delete(result.chunk_id)
                            return next
                          })} />

                        <div className="rcard-bottom">
                          <div className="rcard-tags">
                            {useLlm && loadingTagCards.has(result.chunk_id) && tags.length === 0 && (
                              <span className="rcard-tag rcard-tag-loading">
                                <span className="tag-loading-dots"><span /><span /><span /></span>
                              </span>
                            )}
                            {tags.map(tag => <span key={tag} className={`rcard-tag ${tagColorClass(tag)}`}>{tag}</span>)}
                            {!useLlm && <span className="rcard-tag rcard-tag-mode">{retrievalMode === 'svd' ? 'SVD' : 'TF-IDF'}</span>}
                            {!useLlm && <span className="rcard-tag rcard-tag-mode">{resultScope === 'chunks' ? 'Chunks' : 'Full special'}</span>}
                            {result.has_profanity && <span className="rcard-tag rcard-tag-warn">🐫 Contains profanity</span>}
                          </div>
                          <div className="rcard-links">
                            {getWatchLinks(result).map(link => (
                              <a key={link.label} href={link.url} target="_blank" rel="noreferrer" className={link.cls}>{link.label}</a>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>}

              {!loading && allResults.length > 0 && <PaginationBar {...pgProps} />}
            </div>
          </>
        )}
      </div>
    </div>
  )
}

export default App
