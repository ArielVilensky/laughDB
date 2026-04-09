export type RetrievalMode = 'tfidf' | 'svd' | 'embedding'
export type ResultScope = 'chunks' | 'full'

export interface SvdDimensionMatch {
  dimension: number
  query_weight: number
  chunk_weight: number
  contribution: number
  direction: 'positive' | 'negative'
  top_positive_terms?: string[]
  top_negative_terms?: string[]
}

export interface SearchResult {
  chunk_id: string
  doc_id: number
  comedian: string
  special_title: string
  release_date: string
  title: string
  url: string
  platform: string
  special_type: string

  content: string
  display_snippet: string

  chunk_sentences?: string[]
  best_sentence?: string
  best_sentence_index?: number | null
  sentence_score?: number

  snippet_sentences?: string[]
  snippet_sentence_start?: number
  snippet_sentence_end?: number

  global_snippet_start?: number
  global_snippet_end?: number

  has_profanity: boolean
  profanity_terms?: string[]

  base_score?: number
  proximity_feature?: number
  comedian_feature?: number

  similarity_score?: number
  similarity_percent?: number

  retrieval_mode?: RetrievalMode
  result_scope?: ResultScope

  svd_positive_dimensions?: SvdDimensionMatch[]
  svd_negative_dimensions?: SvdDimensionMatch[]
}

export interface SearchResponse {
  query?: string
  results: SearchResult[]
  resolved_comedian?: string | null
  known_comedians?: string[]
  known_special_types?: string[]
}