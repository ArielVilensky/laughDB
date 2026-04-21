import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import { RetrievalMode, ResultScope, SearchResponse, SearchResult } from './types'
import RagPanel from './RagPanel'

const RESULTS_PER_PAGE = 5
const MAX_TOTAL_RESULTS = 25
const YEAR_MIN = 1965
const YEAR_MAX = 2026
const SEARCH_DEBOUNCE_MS = 200

const SPECIAL_TYPE_OPTIONS = [
  '',
  'Special',
  'TV Appearance',
  'Roast',
  'Interview',
  'Monologue',
  'Award Show',
  'Speech',
  'Crowd Work Special',
]

interface WatchLink {
  label: string
  url: string
  cls: string
}

function getWatchLinks(result: SearchResult): WatchLink[] {
  const links: WatchLink[] = []
  const query = encodeURIComponent(`${result.comedian} ${result.special_title || result.title}`)

  if (result.url) {
    links.push({
      label: 'Transcript',
      url: result.url,
      cls: 'resource-link transcript-link',
    })
  }

  if (result.watch_url) {
    const platform = result.watch_platform || 'Watch'
    links.push({
      label: `Watch on ${platform}`,
      url: result.watch_url,
      cls: 'resource-link watch-link',
    })
  } else {
    links.push({
      label: 'Search on YouTube',
      url: `https://www.youtube.com/results?search_query=${query}`,
      cls: 'resource-link youtube-link',
    })
  }

  return links
}

function formatRetrievalMode(mode?: RetrievalMode): string {
  if (mode === 'tfidf') return 'TF-IDF'
  if (mode === 'svd') return 'SVD'
  return '—'
}

function App(): JSX.Element {
  const [useLlm, setUseLlm] = useState<boolean | null>(null)
  const [searchTerm, setSearchTerm] = useState<string>('')

  const [allResults, setAllResults] = useState<SearchResult[]>([])
  const [resolvedComedian, setResolvedComedian] = useState<string | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [hasSearched, setHasSearched] = useState<boolean>(false)
  const [page, setPage] = useState<number>(0)

  const [comedian, setComedian] = useState<string>('')
  const [specialType, setSpecialType] = useState<string>('')
  const [yearMin, setYearMin] = useState<number>(YEAR_MIN)
  const [yearMax, setYearMax] = useState<number>(YEAR_MAX)
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>('tfidf')
  const [resultScope, setResultScope] = useState<ResultScope>('full')
  const [excludeProfanity, setExcludeProfanity] = useState<boolean>(false)

  const debounceRef = useRef<number | null>(null)
  const latestRequestRef = useRef<number>(0)

  useEffect(() => {
    fetch('/api/config')
      .then((r) => r.json())
      .then((data) => setUseLlm(data.use_llm))
  }, [])

  const clearPendingDebounce = (): void => {
    if (debounceRef.current !== null) {
      window.clearTimeout(debounceRef.current)
      debounceRef.current = null
    }
  }

  const hasActiveFilters = (): boolean => {
    return (
      comedian.trim() !== '' ||
      specialType.trim() !== '' ||
      yearMin !== YEAR_MIN ||
      yearMax !== YEAR_MAX ||
      excludeProfanity ||
      resultScope !== 'full' ||
      retrievalMode !== 'tfidf'
    )
  }

  const buildSearchContext = (): string => {
    const q = searchTerm.trim()
    if (q) return q
    const parts: string[] = []
    if (comedian.trim()) parts.push(comedian.trim())
    if (specialType.trim()) parts.push(specialType.trim())
    if (yearMin !== YEAR_MIN || yearMax !== YEAR_MAX) parts.push(`${yearMin}–${yearMax}`)
    return parts.join(', ')
  }

  const runSearch = async (query: string): Promise<void> => {
    const trimmed = query.trim()
    const filtersActive = hasActiveFilters()

    if (!trimmed && !filtersActive) {
      setAllResults([])
      setResolvedComedian(null)
      setPage(0)
      setHasSearched(false)
      setLoading(false)
      return
    }

    const requestId = latestRequestRef.current + 1
    latestRequestRef.current = requestId

    setLoading(true)
    setHasSearched(true)

    try {
      const params = new URLSearchParams({
        query: trimmed,
        top_k: String(MAX_TOTAL_RESULTS),
        retrieval_mode: retrievalMode,
        result_scope: resultScope,
        comedian,
        special_type: specialType,
        year_min: String(yearMin),
        year_max: String(yearMax),
        exclude_profanity: String(excludeProfanity),
        max_chunks_per_doc: '2',
      })

      const response = await fetch(`/api/search?${params.toString()}`)
      const data: SearchResponse = await response.json()

      if (requestId !== latestRequestRef.current) {
        return
      }

      setAllResults(data.results ?? [])
      setResolvedComedian(data.resolved_comedian ?? null)
      setPage(0)
    } finally {
      if (requestId === latestRequestRef.current) {
        setLoading(false)
      }
    }
  }

  const handleImmediateSearch = async (value?: string): Promise<void> => {
    const query = value ?? searchTerm
    clearPendingDebounce()
    await runSearch(query)
  }

  useEffect(() => {
    clearPendingDebounce()

    const trimmed = searchTerm.trim()
    const filtersActive = hasActiveFilters()

    if (!trimmed && !filtersActive) {
      setAllResults([])
      setResolvedComedian(null)
      setPage(0)
      setHasSearched(false)
      setLoading(false)
      return
    }

    debounceRef.current = window.setTimeout(() => {
      void runSearch(searchTerm)
    }, SEARCH_DEBOUNCE_MS)

    return () => {
      clearPendingDebounce()
    }
  }, [
    searchTerm,
    comedian,
    specialType,
    yearMin,
    yearMax,
    retrievalMode,
    resultScope,
    excludeProfanity,
  ])

  const visibleResults = useMemo(() => {
    const start = page * RESULTS_PER_PAGE
    return allResults.slice(start, start + RESULTS_PER_PAGE)
  }, [allResults, page])

  const totalPages = Math.ceil(allResults.length / RESULTS_PER_PAGE)
  const canGoPrev = page > 0
  const canGoNext = page < totalPages - 1

  const handleClearFilters = (): void => {
    setComedian('')
    setSpecialType('')
    setYearMin(YEAR_MIN)
    setYearMax(YEAR_MAX)
    setRetrievalMode('tfidf')
    setResultScope('full')
    setExcludeProfanity(false)
  }

  const renderSnippet = (result: SearchResult): JSX.Element | string => {
    if (result.snippet_sentences && result.snippet_sentences.length > 0) {
      const snippetGlobalStart =
        result.global_snippet_start ??
        result.snippet_sentence_start ??
        0

      return (
        <>
          {result.snippet_sentences.map((sentence, i) => {
            const absoluteIndex = snippetGlobalStart + i
            const isBest = absoluteIndex === result.best_sentence_index

            return (
              <span
                key={`${result.chunk_id}-sentence-${i}`}
                className={isBest ? 'highlight-sentence' : ''}
              >
                {sentence + ' '}
              </span>
            )
          })}
        </>
      )
    }

    return result.display_snippet || result.content
  }

  if (useLlm === null) return <></>

  return (
    <div className={`full-body-container ${useLlm ? 'llm-mode' : ''}`}>
      <div className="top-text">
        <div className="brand-block">
          <h1 className="brand-title">laughDB</h1>
          <p className="brand-subtitle">Search stand-up comedy transcripts by topic</p>
        </div>

        <div
          className="input-box"
          onClick={() => document.getElementById('search-input')?.focus()}
        >
          <img src={SearchIcon} alt="search" />
          <input
            id="search-input"
            placeholder="Search comedy transcript topics"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                void handleImmediateSearch((e.target as HTMLInputElement).value)
              }
            }}
          />
          <button
            className="search-button"
            onClick={() => void handleImmediateSearch(searchTerm)}
          >
            Search
          </button>
        </div>
      </div>

      <div className="results-layout">
        <div className="results-column">
          {resolvedComedian && (
            <div className="info-banner">
              Best matched comedian: <strong>{resolvedComedian}</strong>
            </div>
          )}

          {excludeProfanity && allResults.length > 0 && (
            <div className="warning-banner">
              {resultScope === 'chunks'
                ? 'Profanity filter is on. The chunks shown do not contain profanity, but the broader transcript may still contain it.'
                : 'Profanity filter is on. Results containing profanity are excluded.'}
            </div>
          )}

          {loading && <div className="info-banner">Loading results...</div>}

          {!loading && hasSearched && visibleResults.length === 0 && (
            <div className="info-banner">No results found.</div>
          )}

          <div id="answer-box">
            {visibleResults.map((result, index) => (
              <div key={`${result.chunk_id}-${index}`} className="episode-item">
                <h3 className="episode-title">
                  {result.comedian || 'Unknown'}
                </h3>

                <p className="episode-rating">
                  <strong>{result.special_title || result.title}</strong>
                  {result.release_date ? ` (${result.release_date})` : ''}
                  {result.special_type ? ` • ${result.special_type}` : ''}
                </p>

                <p className="episode-desc">{renderSnippet(result)}</p>

                <div className="result-meta-row">
                  <span className="meta-pill score-pill">
                    Match: {result.similarity_percent?.toFixed(1) ?? '—'}%
                  </span>

                  <span className="meta-pill subtle-pill">
                    Mode: {formatRetrievalMode(result.retrieval_mode)}
                  </span>

                  <span className="meta-pill subtle-pill">
                    Scope: {result.result_scope === 'full' ? 'Full transcript' : 'Chunks'}
                  </span>

                  {result.has_profanity && (
                    <span className="meta-pill subtle-pill">Contains profanity</span>
                  )}
                </div>

                {result.retrieval_mode === 'svd' && (
                  <div className="svd-explanation-box">
                    {result.svd_positive_dimensions && result.svd_positive_dimensions.length > 0 && (
                      <div className="svd-positive">
                        <p className="svd-heading"><strong>Positive latent dimensions</strong></p>
                        {result.svd_positive_dimensions.map((dim) => (
                          <div key={`pos-${result.chunk_id}-${dim.dimension}`} className="svd-dimension">
                            <span className="meta-pill subtle-pill">
                              Dimension {dim.dimension}{' '}
                              <span className="svd-contribution">
                                • contribution {dim.contribution.toFixed(3)}
                              </span>
                            </span>
                            <p className="svd-terms">
                              Query weight: {dim.query_weight.toFixed(3)} • Result weight:{' '}
                              {dim.chunk_weight.toFixed(3)}
                            </p>
                            <p className="svd-terms">+ {dim.top_positive_terms?.join(', ')}</p>
                            <p className="svd-terms">− {dim.top_negative_terms?.join(', ')}</p>
                          </div>
                        ))}
                      </div>
                    )}

                    {result.svd_negative_dimensions && result.svd_negative_dimensions.length > 0 && (
                      <div className="svd-negative">
                        <p className="svd-heading"><strong>Negative latent dimensions</strong></p>
                        {result.svd_negative_dimensions.map((dim) => (
                          <div key={`neg-${result.chunk_id}-${dim.dimension}`} className="svd-dimension">
                            <span className="meta-pill subtle-pill">
                              Dimension {dim.dimension}{' '}
                              <span className="svd-contribution">
                                • contribution {dim.contribution.toFixed(3)}
                              </span>
                            </span>
                            <p className="svd-terms">
                              Query weight: {dim.query_weight.toFixed(3)} • Result weight:{' '}
                              {dim.chunk_weight.toFixed(3)}
                            </p>
                            <p className="svd-terms">+ {dim.top_positive_terms?.join(', ')}</p>
                            <p className="svd-terms">− {dim.top_negative_terms?.join(', ')}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                <div className="resource-links-row">
                  {getWatchLinks(result).map((link) => (
                    <a
                      key={`${result.chunk_id}-${link.label}`}
                      href={link.url}
                      target="_blank"
                      rel="noreferrer"
                      className={link.cls}
                    >
                      {link.label}
                    </a>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {!loading && allResults.length > 0 && (
            <div className="results-summary">
              Showing {page * RESULTS_PER_PAGE + 1}–
              {Math.min((page + 1) * RESULTS_PER_PAGE, allResults.length)} of {allResults.length}
            </div>
          )}

          {totalPages > 1 && (
            <div className="pagination-block">
              <button
                className="page-arrow"
                disabled={!canGoPrev}
                onClick={() => setPage((p) => p - 1)}
              >
                ←
              </button>

              <div className="page-dots">
                {Array.from({ length: totalPages }).map((_, i) => (
                  <button
                    key={i}
                    className={`page-dot ${page === i ? 'active' : ''}`}
                    onClick={() => setPage(i)}
                  />
                ))}
              </div>

              <button
                className="page-arrow"
                disabled={!canGoNext}
                onClick={() => setPage((p) => p + 1)}
              >
                →
              </button>
            </div>
          )}
        </div>

        <aside className="filters-panel">
          <h2 className="filters-title">Search controls</h2>

          <div className="filter-group">
            <label>Result scope</label>
            <div className="toggle-row compact-toggle-row">
              <button
                className={`toggle-pill compact-pill ${resultScope === 'full' ? 'active' : ''}`}
                onClick={() => setResultScope('full')}
              >
                Full transcripts
              </button>
              <button
                className={`toggle-pill compact-pill ${resultScope === 'chunks' ? 'active' : ''}`}
                onClick={() => setResultScope('chunks')}
              >
                Chunks
              </button>
            </div>
          </div>

          <div className="filter-group">
            <label>Retrieval mode</label>
            <div className="toggle-row compact-toggle-row">
              {(['tfidf', 'svd'] as RetrievalMode[]).map((mode) => (
                <button
                  key={mode}
                  className={`toggle-pill compact-pill ${retrievalMode === mode ? 'active' : ''}`}
                  onClick={() => setRetrievalMode(mode)}
                >
                  {mode === 'tfidf' ? 'TF-IDF' : 'SVD'}
                </button>
              ))}
            </div>
          </div>

          <div className="filters-divider" />

          <h3 className="filters-subtitle">Search filters</h3>

          <div className="filter-group">
            <label htmlFor="comedian-filter">Comedian</label>
            <input
              id="comedian-filter"
              type="text"
              value={comedian}
              onChange={(e) => setComedian(e.target.value)}
              placeholder="Filter by comedian"
            />
          </div>

          <div className="filter-group">
            <label htmlFor="special-type-filter">Special type</label>
            <select
              id="special-type-filter"
              value={specialType}
              onChange={(e) => setSpecialType(e.target.value)}
            >
              {SPECIAL_TYPE_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option || 'All special types'}
                </option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Release year range</label>
            <div className="range-inputs">
              <div>
                <span>From: {yearMin}</span>
                <input
                  type="range"
                  min={YEAR_MIN}
                  max={YEAR_MAX}
                  value={yearMin}
                  onChange={(e) => setYearMin(Math.min(Number(e.target.value), yearMax))}
                />
              </div>
              <div>
                <span>To: {yearMax}</span>
                <input
                  type="range"
                  min={YEAR_MIN}
                  max={YEAR_MAX}
                  value={yearMax}
                  onChange={(e) => setYearMax(Math.max(Number(e.target.value), yearMin))}
                />
              </div>
            </div>
          </div>

          <div className="filter-group">
            <label>Profanity filter</label>
            <div className="toggle-row compact-toggle-row">
              <button
                className={`toggle-pill compact-pill ${!excludeProfanity ? 'active' : ''}`}
                onClick={() => setExcludeProfanity(false)}
              >
                Include
              </button>
              <button
                className={`toggle-pill compact-pill ${excludeProfanity ? 'active' : ''}`}
                onClick={() => setExcludeProfanity(true)}
              >
                Exclude
              </button>
            </div>
          </div>

          <button className="clear-filters-btn" onClick={handleClearFilters}>
            Clear filters
          </button>
        </aside>
      </div>

      {useLlm && (
        <RagPanel
          query={buildSearchContext()}
          visibleResults={visibleResults}
          hasResults={allResults.length > 0 && hasSearched}
        />
      )}
    </div>
  )
}

export default App