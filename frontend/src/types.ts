export interface SearchResult {
  doc_id: number
  comedian: string
  special_title: string
  release_date: string
  title: string
  url: string
  transcript_score: number
  sentence_score: number
  best_sentence_index: number | null
  best_sentence: string
  context_sentences: string[]
  context: string
  platform: string
}