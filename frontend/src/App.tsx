import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import { RetrievalMode, ResultScope, SearchResponse, SearchResult } from './types'
import Chat from './Chat'

const RESULTS_PER_PAGE = 5
const MAX_TOTAL_RESULTS = 50
const YEAR_MIN = 1965
const YEAR_MAX = 2025

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

const MAX_CHUNKS_PER_TRANSCRIPT_OPTIONS = [1, 2, 3, 4, 5]

interface WatchLink {
  label: string
  url: string
  cls: string
}

const PLATFORM_CLS: Record<string, string> = {
  'Netflix': 'watch-netflix',
  'Max': 'watch-hbo',
  'Amazon Prime': 'watch-amazon',
  'Amazon Prime Video': 'watch-amazon',
  'HBO': 'watch-hbo',
  'Comedy Central': 'watch-cc',
  'Peacock': 'watch-peacock',
}

function getWatchLinks(result: SearchResult): WatchLink[] {
  const links: WatchLink[] = []
  const query = encodeURIComponent(`${result.comedian} ${result.special_title || result.title}`)

  if (result.watch_url && result.watch_platform) {
    links.push({
      label: result.watch_platform,
      url: result.watch_url,
      cls: PLATFORM_CLS[result.watch_platform] ?? 'watch-generic',
    })
  }

  links.push({ label: 'YouTube', url: `https://www.youtube.com/results?search_query=${query}`, cls: 'watch-youtube' })

  return links
}

function App(): JSX.Element {
  const [useLlm, setUseLlm] = useState<boolean | null>(null)
  const [searchTerm, setSearchTerm] = useState<string>('')

  const [allResults, setAllResults] = useState<SearchResult[]>([])
  const [resolvedComedian, setResolvedComedian] = useState<string | null>(null)
  const [loading, setLoading] = useState<boolean>(false)

  const [page, setPage] = useState<number>(0)

  const [comedian, setComedian] = useState<string>('')
  const [specialType, setSpecialType] = useState<string>('')
  const [yearMin, setYearMin] = useState<number>(YEAR_MIN)
  const [yearMax, setYearMax] = useState<number>(YEAR_MAX)
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>('tfidf')
  const [resultScope, setResultScope] = useState<ResultScope>('full')
  const [excludeProfanity, setExcludeProfanity] = useState<boolean>(false)
  const [maxChunksPerTranscript, setMaxChunksPerTranscript] = useState<number>(2)

  const debounceRef = useRef<number | null>(null)

  useEffect(() => {
    fetch('/api/config')
      .then((r) => r.json())
      .then((data) => setUseLlm(data.use_llm))
  }, [])

  const handleSearch = async (value?: string): Promise<void> => {
    const query = value ?? searchTerm
    setSearchTerm(query)

    if (query.trim() === '') {
      setAllResults([])
      setResolvedComedian(null)
      setPage(0)
      return
    }

    setLoading(true)

    try {
      const params = new URLSearchParams({
        query,
        top_k: String(MAX_TOTAL_RESULTS),
        retrieval_mode: retrievalMode,
        result_scope: resultScope,
        comedian,
        special_type: specialType,
        year_min: String(yearMin),
        year_max: String(yearMax),
        exclude_profanity: String(excludeProfanity),
        max_chunks_per_doc: String(maxChunksPerTranscript),
      })

      const response = await fetch(`/api/search?${params.toString()}`)
      const data: SearchResponse = await response.json()

      setAllResults(data.results ?? [])
      setResolvedComedian(data.resolved_comedian ?? null)
      setPage(0)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (debounceRef.current !== null) {
      window.clearTimeout(debounceRef.current)
    }

    const trimmed = searchTerm.trim()

    if (!trimmed) {
      setAllResults([])
      setResolvedComedian(null)
      setPage(0)
      return
    }

    debounceRef.current = window.setTimeout(() => {
      void handleSearch(searchTerm)
    }, 250)

    return () => {
      if (debounceRef.current !== null) {
        window.clearTimeout(debounceRef.current)
      }
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
    maxChunksPerTranscript,
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
    setResultScope('chunks')
    setExcludeProfanity(false)
    setMaxChunksPerTranscript(2)
  }

  const renderSnippet = (result: SearchResult): JSX.Element | string => {
    if (result.snippet_sentences && result.snippet_sentences.length > 0) {
      return (
        <>
          {result.snippet_sentences.map((sentence, i) => {
            const absoluteIndex = (result.snippet_sentence_start ?? 0) + i
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
                void handleSearch((e.target as HTMLInputElement).value)
              }
            }}
          />
          <button
            className="search-button"
            onClick={() => void handleSearch(searchTerm)}
          >
            Search
          </button>
        </div>
      </div>

      <div className="results-layout">
        <div className="results-column">
          {resolvedComedian && (
            <div className="info-banner">
              Best matched for comedian: <strong>{resolvedComedian}</strong>
            </div>
          )}

          {excludeProfanity && allResults.length > 0 && (
            <div className="warning-banner">
              Profanity filter is on. Results containing profanity are excluded.
            </div>
          )}

          {loading && <div className="info-banner">Loading results...</div>}

          {!loading && searchTerm.trim() !== '' && visibleResults.length === 0 && (
            <div className="info-banner">No results found.</div>
          )}

          <div id="answer-box">
            {visibleResults.map((result, index) => (
              <div key={`${result.chunk_id}-${index}`} className="episode-item">
                <h3 className="episode-title">
                  {result.comedian || 'Unknown Comedian'}
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
                    Mode: {result.retrieval_mode}
                  </span>

                  <span className="meta-pill subtle-pill">
                    Scope: {result.result_scope === 'full' ? 'Full transcript' : 'Chunk'}
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
                            <p className="svd-terms">
                              + {dim.top_positive_terms?.join(', ')}
                            </p>
                            <p className="svd-terms">
                              − {dim.top_negative_terms?.join(', ')}
                            </p>
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
                            <p className="svd-terms">
                              + {dim.top_positive_terms?.join(', ')}
                            </p>
                            <p className="svd-terms">
                              − {dim.top_negative_terms?.join(', ')}
                            </p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                <div className="episode-actions">
                  <a href={result.url} target="_blank" rel="noreferrer" className="action-btn transcript-btn">
                    View transcript
                  </a>
                  {getWatchLinks(result).map((link, i) => (
                    <a key={i} href={link.url} target="_blank" rel="noreferrer" className={`action-btn watch-btn ${link.cls}`}>
                      {link.label}
                    </a>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {allResults.length > RESULTS_PER_PAGE && (
            <div className="pagination-block">
              <button
                className="page-arrow"
                onClick={() => setPage((prev) => Math.max(prev - 1, 0))}
                disabled={!canGoPrev}
              >
                ←
              </button>

              <div className="page-dots">
                {Array.from({ length: totalPages }).map((_, i) => (
                  <button
                    key={i}
                    className={`page-dot ${page === i ? 'active' : ''}`}
                    onClick={() => setPage(i)}
                    aria-label={`Go to page ${i + 1}`}
                  />
                ))}
              </div>

              <button
                className="page-arrow"
                onClick={() => setPage((prev) => Math.min(prev + 1, totalPages - 1))}
                disabled={!canGoNext}
              >
                →
              </button>
            </div>
          )}

          {allResults.length > RESULTS_PER_PAGE && (
            <p className="more-results-note">
              Showing {page * RESULTS_PER_PAGE + 1}–
              {Math.min((page + 1) * RESULTS_PER_PAGE, allResults.length)} of {allResults.length} results
            </p>
          )}

          {useLlm && <Chat />}
        </div>

        <aside className="filters-panel">
          <h2 className="filters-title">Filters</h2>

          <div className="filter-group">
            <label>Retrieval mode</label>
            <select
              value={retrievalMode}
              onChange={(e) => setRetrievalMode(e.target.value as RetrievalMode)}
            >
              <option value="tfidf">Basic TF-IDF</option>
              <option value="svd">SVD</option>
              <option value="embedding">Sentence Embeddings</option>
            </select>
          </div>

          <div className="filter-group">
            <label>Result scope</label>
            <div className="toggle-row">
              <button
                type="button"
                className={`toggle-pill ${resultScope === 'chunks' ? 'active' : ''}`}
                onClick={() => setResultScope('chunks')}
              >
                Chunks
              </button>
              <button
                type="button"
                className={`toggle-pill ${resultScope === 'full' ? 'active' : ''}`}
                onClick={() => setResultScope('full')}
              >
                Full transcripts
              </button>
            </div>
          </div>

          {resultScope === 'chunks' && (
            <div className="filter-group">
              <label>Max chunks per transcript</label>
              <select
                value={maxChunksPerTranscript}
                onChange={(e) => setMaxChunksPerTranscript(Number(e.target.value))}
              >
                {MAX_CHUNKS_PER_TRANSCRIPT_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="filter-group">
            <label>Comedian</label>
            <input
              type="text"
              value={comedian}
              onChange={(e) => setComedian(e.target.value)}
              placeholder="Optional comedian name"
            />
          </div>

          <div className="filter-group">
            <label>Special type</label>
            <select
              value={specialType}
              onChange={(e) => setSpecialType(e.target.value)}
            >
              {SPECIAL_TYPE_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option || 'Any'}
                </option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Year range</label>
            <div className="range-inputs">
              <div>
                <span>From</span>
                <input
                  type="range"
                  min={YEAR_MIN}
                  max={YEAR_MAX}
                  value={yearMin}
                  onChange={(e) => {
                    const value = Number(e.target.value)
                    setYearMin(Math.min(value, yearMax))
                  }}
                />
                <div className="range-value">{yearMin}</div>
              </div>

              <div>
                <span>To</span>
                <input
                  type="range"
                  min={YEAR_MIN}
                  max={YEAR_MAX}
                  value={yearMax}
                  onChange={(e) => {
                    const value = Number(e.target.value)
                    setYearMax(Math.max(value, yearMin))
                  }}
                />
                <div className="range-value">{yearMax}</div>
              </div>
            </div>
          </div>

          <div className="filter-group">
            <label>Profanity</label>
            <div className="profanity-toggle">
              <button
                type="button"
                className={`profanity-btn ${excludeProfanity ? 'active' : ''}`}
                onClick={() => setExcludeProfanity(true)}
              >
                Hide profanity
              </button>
              <button
                type="button"
                className={`profanity-btn ${!excludeProfanity ? 'active' : ''}`}
                onClick={() => setExcludeProfanity(false)}
              >
                Show profanity
              </button>
            </div>
          </div>

          <div className="filter-actions">
            <button
              type="button"
              className="apply-filters-btn"
              onClick={() => void handleSearch(searchTerm)}
            >
              Apply
            </button>
            <button
              type="button"
              className="clear-filters-btn"
              onClick={() => {
                handleClearFilters()
                if (searchTerm.trim()) {
                  window.setTimeout(() => {
                    void handleSearch(searchTerm)
                  }, 0)
                }
              }}
            >
              Clear
            </button>
          </div>
        </aside>
      </div>
    </div>
  )
}

export default App