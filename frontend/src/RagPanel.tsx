import { useEffect, useRef, useState } from 'react'
import { SearchResult } from './types'

const LOADING_PHRASES = [
  'Watching the homework tapes...',
  'Consulting the ghost of Mitch Hedberg...',
  'Running it by the hecklers...',
  'Finding the punchline...',
  'Checking the bit notebook...',
  'Clearing it with the club booker...',
  'Stealing material (ethically)...',
  'Timing the rimshots...',
  'Asking the open mic committee...',
  'Reading between the one-liners...',
  'Warming up the crowd...',
  'Adjusting the mic stand...',
  'Flipping through the joke graveyard...',
  'Getting notes from the writers room...',
  'Googling "how to be funny"...',
  'Testing it on the spouse first...',
  'Waiting for the laugh track...',
  'Counting to three before the punchline...',
  'Reviewing the tape from last Thursday...',
  'Arguing with the network notes...',
  'Checking if this bit was already done...',
  'Pulling the B-material just in case...',
  'Running it in the shower...',
  'Getting the light from the back of the room...',
  'Asking the opening act for their opinion...',
]

interface FollowUpQuestion {
  id: string
  label: string
  prompt: string
}

const FOLLOW_UP_QUESTIONS: FollowUpQuestion[] = [
  {
    id: 'evolution',
    label: 'How has this topic evolved over the decades?',
    prompt: 'How has this comedy topic evolved across different decades? Reference the years of the results shown.',
  },
  {
    id: 'comparison',
    label: 'How do these comedians approach it differently?',
    prompt: 'How do the comedians in these results approach this topic differently from each other? Be specific about each comedian.',
  },
  {
    id: 'audience',
    label: 'What crowds are these specials best suited for?',
    prompt: 'What kind of audience or crowd would enjoy these specials the most? Consider comedy taste, sensibilities, and what to expect.',
  },
]

interface RagPanelProps {
  query: string
  visibleResults: SearchResult[]
  hasResults: boolean
  externalTrigger?: number
  inline?: boolean
}

function RagPanel({ query, visibleResults, hasResults, externalTrigger, inline }: RagPanelProps): JSX.Element | null {
  const [isOpen, setIsOpen] = useState(false)
  const [vibes, setVibes] = useState<string[]>([])
  const [summary, setSummary] = useState('')
  const [loading, setLoading] = useState(false)
  const [phraseIndex, setPhraseIndex] = useState(0)
  const [stale, setStale] = useState(false)
  const [usedQuestionIds, setUsedQuestionIds] = useState<string[]>([])
  const [activeQuestionLabel, setActiveQuestionLabel] = useState<string | null>(null)

  const lastSummarizedQuery = useRef<string | null>(null)
  const phraseTimer = useRef<number | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    abortRef.current?.abort()
    setSummary('')
    setVibes([])
    setStale(false)
    setLoading(false)
    setUsedQuestionIds([])
    setActiveQuestionLabel(null)
    lastSummarizedQuery.current = null
    if (!inline) setIsOpen(false)
  }, [query])

  useEffect(() => {
    if (lastSummarizedQuery.current !== null && lastSummarizedQuery.current === query) {
      setStale(true)
    }
  }, [visibleResults])

  const startPhrases = () => {
    setPhraseIndex(Math.floor(Math.random() * LOADING_PHRASES.length))
    phraseTimer.current = window.setInterval(() => {
      setPhraseIndex(i => (i + 1) % LOADING_PHRASES.length)
    }, 2200)
  }

  const stopPhrases = () => {
    if (phraseTimer.current !== null) {
      clearInterval(phraseTimer.current)
      phraseTimer.current = null
    }
  }

  const resultPayload = () =>
    visibleResults.map((r, i) => ({
      rank: i + 1,
      comedian: r.comedian,
      special_title: r.special_title || r.title,
      release_date: r.release_date,
      display_snippet: r.display_snippet || r.content,
      similarity_percent: r.similarity_percent,
    }))

  const streamResponse = async (
    followup: FollowUpQuestion | null,
    onVibes: (tags: string[]) => void,
    onText: (text: string) => void,
    signal: AbortSignal,
  ): Promise<boolean> => {
    const body: Record<string, unknown> = { query, results: resultPayload() }
    if (followup) body.followup = followup.prompt

    const response = await fetch('/api/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    })

    if (!response.ok) {
      const data = await response.json()
      onText('Error: ' + (data.error ?? response.status))
      return false
    }

    let fullText = ''
    let vibesProcessed = followup !== null
    let summaryStart = 0

    const reader = response.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          const data = JSON.parse(line.slice(6))
          if (data.error) { onText('Error: ' + data.error); return false }
          if (!data.content) continue

          fullText += data.content

          if (!vibesProcessed) {
            const nlIdx = fullText.indexOf('\n')
            if (nlIdx !== -1) {
              vibesProcessed = true
              const firstLine = fullText.slice(0, nlIdx).trim()
              if (!followup && firstLine.toUpperCase().startsWith('VIBES:')) {
                const tags = firstLine.slice(firstLine.indexOf(':') + 1)
                  .split(',')
                  .map(t => t.trim())
                  .filter(Boolean)
                onVibes(tags)
                summaryStart = nlIdx
              }
              onText(fullText.slice(summaryStart).trimStart())
            }
          } else {
            onText(fullText.slice(summaryStart).trimStart())
          }
        } catch { /* ignore malformed SSE lines */ }
      }
    }
    return true
  }

  const callApi = async (followup: FollowUpQuestion | null = null) => {
    if (loading) return
    abortRef.current?.abort()
    abortRef.current = new AbortController()

    setSummary('')
    setStale(false)
    setLoading(true)
    setActiveQuestionLabel(followup?.label ?? null)
    startPhrases()

    try {
      const ok = await streamResponse(
        followup,
        (tags) => setVibes(tags),
        (text) => setSummary(text),
        abortRef.current.signal,
      )

      if (ok) {
        lastSummarizedQuery.current = query
        if (followup) {
          setUsedQuestionIds(prev => [...prev, followup.id])
        } else {
          setUsedQuestionIds([])
        }
      }
    } catch (err: unknown) {
      if ((err as Error).name !== 'AbortError') {
        setSummary('Something went wrong. Please try again.')
      }
    } finally {
      setLoading(false)
      stopPhrases()
    }
  }

  const handleOpen = () => {
    if (!inline) setIsOpen(true)
    if (!summary && !loading) void callApi()
  }

  useEffect(() => {
    if (externalTrigger && externalTrigger > 0) handleOpen()
  }, [externalTrigger])

  const remainingQuestions = FOLLOW_UP_QUESTIONS.filter(q => !usedQuestionIds.includes(q.id))
  const allQuestionsUsed = summary && !loading && usedQuestionIds.length === FOLLOW_UP_QUESTIONS.length

  if (!query.trim() || !hasResults) return null

  // ── Inline mode (rendered inside sidebar) ─────────────────────────────────
  if (inline) {
    return (
      <div className="rag-inline">
        {vibes.length > 0 && (
          <div className="rag-vibes">
            {vibes.map(v => <span key={v} className="rag-vibe-pill">{v}</span>)}
          </div>
        )}

        {loading && (
          <div className="rag-loading">
            {activeQuestionLabel && <p className="rag-active-question">"{activeQuestionLabel}"</p>}
            <div className="rag-dots"><span /><span /><span /></div>
            <p className="rag-phrase">{LOADING_PHRASES[phraseIndex]}</p>
          </div>
        )}

        {!loading && stale && summary && (
          <button className="rag-stale-btn" onClick={() => void callApi()}>
            ↺ Results changed — refresh
          </button>
        )}

        {!loading && summary && (
          <>
            {activeQuestionLabel && (
              <p className="rag-followup-label">"{activeQuestionLabel}"</p>
            )}
            <p className="rag-summary">{summary}</p>
          </>
        )}

        {!loading && !summary && (
          <button className="ai-btn" onClick={() => void callApi()}>
            ✨ Generate summary
          </button>
        )}

        {!loading && summary && !stale && !allQuestionsUsed && remainingQuestions.length > 0 && (
          <div className="rag-questions">
            <p className="rag-questions-label">Explore further</p>
            {remainingQuestions.map(q => (
              <button key={q.id} className="rag-question-btn" onClick={() => void callApi(q)}>
                {q.label}
              </button>
            ))}
          </div>
        )}

        {allQuestionsUsed && (
          <p className="rag-done-msg">Try a different search to explore more!</p>
        )}

        {!loading && summary && !stale && (
          <button className="rag-refresh-btn" style={{ marginTop: '10px' }} onClick={() => void callApi()}>
            ↺ Refresh summary
          </button>
        )}
      </div>
    )
  }

  // ── Float mode (legacy, not used when inline) ─────────────────────────────
  return (
    <>
      {!isOpen && (
        <button className="rag-fab" onClick={handleOpen} title="Summarize results with AI">
          ✨ Summarize with AI
        </button>
      )}

      {isOpen && (
        <div className="rag-panel">
          <div className="rag-panel-header">
            <span className="rag-panel-title">✨ AI Summary</span>
            <button className="rag-panel-close" onClick={() => setIsOpen(false)} aria-label="Close">✕</button>
          </div>

          <div className="rag-panel-body">
            {vibes.length > 0 && (
              <div className="rag-vibes">
                {vibes.map(v => <span key={v} className="rag-vibe-pill">{v}</span>)}
              </div>
            )}

            {loading && (
              <div className="rag-loading">
                {activeQuestionLabel && <p className="rag-active-question">"{activeQuestionLabel}"</p>}
                <div className="rag-dots"><span /><span /><span /></div>
                <p className="rag-phrase">{LOADING_PHRASES[phraseIndex]}</p>
              </div>
            )}

            {!loading && stale && summary && (
              <button className="rag-stale-btn" onClick={() => void callApi()}>
                ↺ Results changed — refresh summary
              </button>
            )}

            {!loading && summary && (
              <>
                {activeQuestionLabel && (
                  <p className="rag-followup-label">"{activeQuestionLabel}"</p>
                )}
                <p className="rag-summary">{summary}</p>
              </>
            )}

            {!loading && !summary && (
              <button className="rag-summarize-btn" onClick={() => void callApi()}>
                Summarize these results
              </button>
            )}

            {!loading && summary && !stale && !allQuestionsUsed && remainingQuestions.length > 0 && (
              <div className="rag-questions">
                <p className="rag-questions-label">Explore further</p>
                {remainingQuestions.map(q => (
                  <button key={q.id} className="rag-question-btn" onClick={() => void callApi(q)}>
                    {q.label}
                  </button>
                ))}
              </div>
            )}

            {allQuestionsUsed && <p className="rag-done-msg">Try a different search query to explore more!</p>}
          </div>

          {!loading && summary && !stale && (
            <div className="rag-panel-footer">
              <button className="rag-refresh-btn" onClick={() => void callApi()}>↺ Refresh</button>
            </div>
          )}
        </div>
      )}
    </>
  )
}

export default RagPanel
