import { useEffect, useRef, useState } from 'react'
import './HowItWorks.css'

const TOC_SECTIONS = [
  { id: 'retrieval', label: 'Retrieval modes' },
  { id: 'svd-dims', label: 'SVD dimensions' },
  { id: 'scope', label: 'Search scope' },
  { id: 'matrix', label: 'Quick guide' },
  { id: 'filters', label: 'Other filters' },
  { id: 'ai', label: '✦ AI summary' },
]

function HowItWorks(): JSX.Element {
  const [activeId, setActiveId] = useState('retrieval')
  const observerRef = useRef<IntersectionObserver | null>(null)

  useEffect(() => {
    observerRef.current = new IntersectionObserver(
      entries => {
        const visible = entries.filter(e => e.isIntersecting)
        if (visible.length) setActiveId(visible[0].target.id)
      },
      { rootMargin: '-20% 0px -60% 0px', threshold: 0 }
    )
    TOC_SECTIONS.forEach(({ id }) => {
      const el = document.getElementById(id)
      if (el) observerRef.current!.observe(el)
    })
    return () => observerRef.current?.disconnect()
  }, [])

  return (
    <div className="howto-layout">
      <nav className="howto-toc">
        <div className="howto-toc-label">On this page</div>
        {TOC_SECTIONS.map(({ id, label }) => (
          <a
            key={id}
            href={`#${id}`}
            className={`howto-toc-link${activeId === id ? ' active' : ''}`}
            onClick={e => { e.preventDefault(); document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' }) }}
          >
            {label}
          </a>
        ))}
      </nav>

    <div className="howto-page">
      <div className="howto-hero">
        <h1 className="howto-title">How laughDB works</h1>
        <p className="howto-sub">
          A quick guide to the search modes, scopes, and filters — so you always find exactly the bit you're looking for.
        </p>
      </div>

      {/* Retrieval modes */}
      <section id="retrieval" className="howto-section">
        <h2 className="howto-section-title">Retrieval modes</h2>
        <p className="howto-section-desc">Two different algorithms for matching your query to transcripts.</p>
        <div className="howto-grid-2">

          <div className="howto-card">
            <div className="howto-card-header">
              <div className="howto-icon howto-icon-purple">TF</div>
              <h3 className="howto-card-title">TF-IDF</h3>
              <span className="howto-badge howto-badge-default">Default</span>
            </div>
            <p className="howto-card-full">Term Frequency–Inverse Document Frequency</p>
            <p className="howto-card-body">
              Scores each transcript by how often your search words appear in it, weighted against how common those words
              are across all transcripts. Rare, specific words get a bigger boost.
            </p>
            <div className="howto-card-when">
              <div className="howto-when-label">Best for</div>
              <ul className="howto-when-list">
                <li>Specific keywords — comedian names, catchphrases, named topics</li>
                <li>Quotes or lines you half-remember</li>
                <li>Filtering by exact subject matter (e.g. "airplane food", "cancel culture")</li>
              </ul>
            </div>
            <div className="howto-example">"Bill Burr marriage"</div>
          </div>

          <div className="howto-card">
            <div className="howto-card-header">
              <div className="howto-icon howto-icon-blue">SVD</div>
              <h3 className="howto-card-title">SVD</h3>
              <span className="howto-badge howto-badge-alt">Semantic</span>
            </div>
            <p className="howto-card-full">Singular Value Decomposition (Latent Semantic Analysis)</p>
            <p className="howto-card-body">
              Learns hidden themes and relationships across all transcripts by decomposing the word-document matrix.
              Matches your query to results that share the same <em>concept</em>, even if they use different words.
            </p>
            <div className="howto-card-when">
              <div className="howto-when-label">Best for</div>
              <ul className="howto-when-list">
                <li>Themes and abstract concepts — "loneliness", "class anxiety", "growing up"</li>
                <li>Exploratory searches where you don't know the exact wording</li>
                <li>Finding unexpected comedians covering a theme you care about</li>
              </ul>
            </div>
            <div className="howto-example">"feeling like an outsider"</div>
          </div>

        </div>
      </section>

      {/* SVD dimensions */}
      <section id="svd-dims" className="howto-section">
        <h2 className="howto-section-title">SVD dimensions — how to read them</h2>
        <p className="howto-section-desc">
          When SVD mode is active, each result shows the latent dimensions that drove the match. Here's what those numbers and word lists actually mean.
        </p>

        <div className="howto-card" style={{ marginBottom: 16 }}>
          <div className="howto-card-header">
            <div className="howto-icon howto-icon-blue">∑</div>
            <h3 className="howto-card-title">What is a latent dimension?</h3>
          </div>
          <p className="howto-card-body">
            SVD breaks the entire transcript corpus down into hidden axes of meaning — called <strong>latent dimensions</strong> — by finding groups of words that tend to appear together across specials. Each dimension captures a recurring theme, style, or context in the data: one might correspond to <em>racial identity and social commentary</em>, another to <em>family and parenting</em>, another to <em>political absurdity</em>.
          </p>
          <p className="howto-card-body" style={{ marginTop: 8 }}>
            Neither your query nor any transcript is described in terms of the original words. Instead, both are projected into this 130-dimensional space, and similarity is measured there. That's why SVD can match on <em>concept</em> rather than exact wording.
          </p>
        </div>

        <div className="howto-grid-2" style={{ marginBottom: 16 }}>
          <div className="howto-card">
            <div className="howto-card-header">
              <div className="howto-icon howto-icon-green">+</div>
              <h3 className="howto-card-title">Positive words</h3>
              <span className="howto-badge howto-badge-default">High loading</span>
            </div>
            <p className="howto-card-body">
              Words that <strong>define</strong> this dimension — they appear heavily in transcripts that score high on it. When your query and a result both have a high positive score on a dimension, these words hint at what shared theme pulled them together.
            </p>
            <div className="howto-card-when">
              <div className="howto-when-label">Example — Dim 7</div>
              <ul className="howto-when-list">
                <li>Positive: <em>love, kids, dad, mom, baby, dog, kid, old</em> → dimension captures <strong>family & parenting</strong> comedy — specials that revolve around raising children, domestic life, and pets score high here</li>
              </ul>
            </div>
          </div>

          <div className="howto-card">
            <div className="howto-card-header">
              <div className="howto-icon howto-icon-warm">−</div>
              <h3 className="howto-card-title">Negative words</h3>
              <span className="howto-badge howto-badge-alt">Low loading</span>
            </div>
            <p className="howto-card-body">
              Words that <strong>oppose</strong> this dimension — they appear heavily in transcripts that score low (negative) on it. A result with a strong negative dimension score is matching on the absence of the positive theme, or through an opposing concept.
            </p>
            <div className="howto-card-when">
              <div className="howto-when-label">Example — Dim 7</div>
              <ul className="howto-when-list">
                <li>Negative: <em>guy, women, men, guys, look</em> → adult-perspective, relationship-focused stand-up sits at the opposite end of the same axis — a completely different style of comedy captured by the same dimension</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="howto-card">
          <div className="howto-card-header">
            <div className="howto-icon howto-icon-purple">±</div>
            <h3 className="howto-card-title">Reading a dimension match in results</h3>
          </div>
          <p className="howto-card-body">
            Each result card (in SVD mode) shows the top dimensions contributing to the match, tagged as <strong>+dim N</strong> or <strong>−dim N</strong>:
          </p>
          <div className="howto-card-when" style={{ marginTop: 8 }}>
            <div className="howto-when-label">What the signs mean</div>
            <ul className="howto-when-list">
              <li><strong>+dim N</strong> — both your query and this transcript score in the <em>same direction</em> on dimension N (both positive, or both negative). The positive-loading words describe what the shared theme is.</li>
              <li><strong>−dim N</strong> — your query and this transcript score in <em>opposite directions</em>. The connection is contrastive — the result is relevant because it sits at the other end of that axis from your query, which can surface unexpected but thematically related results.</li>
            </ul>
          </div>
          <p className="howto-card-body" style={{ marginTop: 10 }}>
            The word lists shown under each dimension tag are a best-effort label — they describe the concept that dimension captured from the data. They won't always be clean (proper nouns, character names, and OCR artifacts can appear), but the pattern across 5–6 words usually points to a recognizable theme.
          </p>
        </div>
      </section>

      {/* Scope */}
      <section id="scope" className="howto-section">
        <h2 className="howto-section-title">Search scope</h2>
        <p className="howto-section-desc">Controls what unit of content gets scored and returned.</p>
        <div className="howto-grid-2">

          <div className="howto-card">
            <div className="howto-card-header">
              <div className="howto-icon howto-icon-warm">▬</div>
              <h3 className="howto-card-title">Full special</h3>
              <span className="howto-badge howto-badge-default">Default</span>
            </div>
            <p className="howto-card-body">
              The entire transcript of each show is treated as a single document. The most relevant 4–6 sentences are
              extracted and shown as your snippet, with the best-matching sentence highlighted.
            </p>
            <div className="howto-card-when">
              <div className="howto-when-label">Best for</div>
              <ul className="howto-when-list">
                <li>Discovering which shows or comedians cover a topic at all</li>
                <li>Getting a broad sense of how a comedian approaches a theme</li>
                <li>When you want results across many different specials</li>
              </ul>
            </div>
            <div className="howto-example">Find all shows about "immigration"</div>
          </div>

          <div className="howto-card">
            <div className="howto-card-header">
              <div className="howto-icon howto-icon-green">≡</div>
              <h3 className="howto-card-title">Chunks</h3>
              <span className="howto-badge howto-badge-alt">Precise</span>
            </div>
            <p className="howto-card-body">
              Each special is split into short overlapping segments of roughly 8–12 sentences. Chunks are scored
              individually, so you can surface the exact moment in a show where the topic comes up.
            </p>
            <div className="howto-card-when">
              <div className="howto-when-label">Best for</div>
              <ul className="howto-when-list">
                <li>Finding the specific bit or punchline, not just the show</li>
                <li>Comparing how different segments of long specials handle the same topic</li>
                <li>When full-special results feel too diffuse</li>
              </ul>
            </div>
            <div className="howto-example">Find the exact bit about "airport security"</div>
          </div>

        </div>
      </section>

      {/* Quick guide matrix */}
      <section id="matrix" className="howto-section">
        <h2 className="howto-section-title">Quick guide — which combination to use</h2>
        <p className="howto-section-desc">Pick based on what you're trying to find.</p>
        <div className="howto-matrix">
          <div className="howto-matrix-header howto-matrix-corner"></div>
          <div className="howto-matrix-header">Full special</div>
          <div className="howto-matrix-header">Chunks</div>

          <div className="howto-matrix-row-label">TF-IDF</div>
          <div className="howto-matrix-cell">
            <div className="howto-cell-combo">TF-IDF + Full</div>
            <p>Find which <strong>shows</strong> mention a specific keyword or topic. Best starting point for most searches.</p>
            <div className="howto-cell-try">Try: "airplane food" or "dating apps"</div>
          </div>
          <div className="howto-matrix-cell">
            <div className="howto-cell-combo">TF-IDF + Chunks</div>
            <p>Find the <strong>exact moment</strong> in a show where a specific word or phrase is used. Great for locating a half-remembered bit.</p>
            <div className="howto-cell-try">Try: "my wife" or "cancel culture"</div>
          </div>

          <div className="howto-matrix-row-label">SVD</div>
          <div className="howto-matrix-cell">
            <div className="howto-cell-combo">SVD + Full</div>
            <p>Find which <strong>shows</strong> thematically explore a concept, even without using the exact words. Good for browsing.</p>
            <div className="howto-cell-try">Try: "loneliness" or "growing up poor"</div>
          </div>
          <div className="howto-matrix-cell howto-matrix-cell-highlight">
            <div className="howto-cell-combo">SVD + Chunks</div>
            <p>Find the <strong>specific segment</strong> of a show that most deeply explores a theme. Most targeted option for concept searches.</p>
            <div className="howto-cell-try">Try: "feeling like an outsider"</div>
          </div>
        </div>
      </section>

      {/* Other filters */}
      <section id="filters" className="howto-section">
        <h2 className="howto-section-title">Other filters</h2>
        <div className="howto-grid-3">

          <div className="howto-card howto-card-sm">
            <div className="howto-card-header">
              <div className="howto-icon howto-icon-purple">🎤</div>
              <h3 className="howto-card-title">Comedian filter</h3>
            </div>
            <p className="howto-card-body">Type a comedian's name to restrict results to their work. Fuzzy matching means you don't need exact spelling.</p>
          </div>

          <div className="howto-card howto-card-sm">
            <div className="howto-card-header">
              <div className="howto-icon howto-icon-warm">📅</div>
              <h3 className="howto-card-title">Year range</h3>
            </div>
            <p className="howto-card-body">Slide to narrow results to a specific era. Useful for comparing how comedy about a topic has changed over time.</p>
          </div>

          <div className="howto-card howto-card-sm">
            <div className="howto-card-header">
              <div className="howto-icon howto-icon-green">🔇</div>
              <h3 className="howto-card-title">Profanity filter</h3>
            </div>
            <p className="howto-card-body">Exclude results that contain profanity. In Chunks mode, the chunk itself is clean — but the broader show may still contain it.</p>
          </div>

        </div>
      </section>

      {/* AI summary */}
      <section id="ai" className="howto-section">
        <h2 className="howto-section-title">✦ AI summary</h2>
        <div className="howto-card howto-card-ai">
          <p className="howto-card-body">
            Every search automatically generates an <strong>✦ AI Summary</strong> in the sidebar. The AI reads
            the visible snippets and synthesises how different comedians approach your topic — themes, patterns,
            and differences in perspective — without you having to do anything.
          </p>
          <p className="howto-card-body" style={{ marginTop: '10px' }}>
            You can ask follow-up questions like <em>"How has this topic evolved over the decades?"</em> or{' '}
            <em>"How do these comedians approach it differently?"</em> to go deeper. The summary updates
            automatically whenever you run a new search.
          </p>
          <div className="howto-ai-note">AI summaries are generated fresh each time and are based only on the current page of results.</div>
        </div>
      </section>

    </div>
    </div>
  )
}

export default HowItWorks
