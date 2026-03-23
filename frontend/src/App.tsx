import { useState, useEffect, useRef } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import { SearchResult } from './types'
import Chat from './Chat'

function App(): JSX.Element {
  const [useLlm, setUseLlm] = useState<boolean | null>(null)
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const debounceRef = useRef<number | null>(null)

  useEffect(() => {
    fetch('/api/config')
      .then(r => r.json())
      .then(data => setUseLlm(data.use_llm))
  }, [])

  const handleSearch = (value: string): void => {
    setSearchTerm(value)

    if (debounceRef.current) {
      window.clearTimeout(debounceRef.current)
    }

    if (value.trim() === '') {
      setResults([])
      return
    }

    setIsLoading(true)
    setSearchError(null)
    debounceRef.current = window.setTimeout(async () => {
      try {
        const response = await fetch(`/api/search?query=${encodeURIComponent(value)}`)
        if (!response.ok) throw new Error(`Server error: ${response.status}`)
        const data: SearchResult[] = await response.json()
        setResults(data)
      } catch (e) {
        console.error("Search failed:", e)
        setSearchError("Search failed. Please try again.")
        setResults([])
      } finally {
        setIsLoading(false)
      }
    }, 300)
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
            onChange={(e) => handleSearch(e.target.value)}
          />
        </div>
      </div>

      <div id="answer-box">
        {isLoading && <p className="search-status">Searching…</p>}
        {searchError && <p className="search-status search-error">{searchError}</p>}
        {!isLoading && !searchError && results.map((result, index) => (
          <div key={index} className="episode-item">
            <h3 className="episode-title">
              {result.comedian || 'Unknown Comedian'}
            </h3>

            <p className="episode-rating">
              <strong>{result.special_title || result.title}</strong>
              {result.release_date ? ` (${result.release_date})` : ''}
            </p>

            <p className="episode-desc">
              {result.context_sentences.map((sentence, i) => {
                const isBest = sentence.trim() === result.best_sentence.trim()
                return (
                  <span
                    key={i}
                    className={isBest ? 'highlight-sentence' : ''}
                  >
                    {sentence + ' '}
                  </span>
                )
              })}
            </p>

            <p className="episode-rating">
              <a
                href={result.url}
                target="_blank"
                rel="noreferrer"
              >
                View transcript
              </a>
            </p>
          </div>
        ))}
      </div>

      {useLlm && <Chat onSearchTerm={handleSearch} />}
    </div>
  )
}

export default App