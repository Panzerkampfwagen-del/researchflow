// TypeScript interfaces mirroring the backend Pydantic contracts.

export interface ResearchPlan {
  goal: string
  subtopics: string[]
  search_queries: string[]
  year_start: number
  year_end: number
}

export interface ResearchSession {
  session_id: string
  query: string
  status: string
  plan: ResearchPlan | null
  paper_count: number
  report_ready: boolean
  created_at: string | null
  completed_at: string | null
}

export interface PaperMetadata {
  id: string
  arxiv_id: string | null
  semantic_scholar_id: string | null
  title: string
  authors: string[]
  abstract: string | null
  year: number | null
  venue: string | null
  citation_count: number
  url: string | null
  relevance_score?: number | null
}

export interface PaperAnalysis {
  paper_id: string
  problem: string
  methodology: string
  datasets: string[]
  metrics: string[]
  key_results: string
  limitations: string
  confidence: number
}

export interface ResearchGap {
  description: string
  supporting_evidence: string[]
  opportunity: string
}

export interface ResearchReport {
  session_id: string
  executive_summary: string
  methodology_comparison: Record<string, unknown>[]
  research_gaps: ResearchGap[]
  trends: string[]
  future_directions: string[]
  citations: Record<string, unknown>[]
  markdown_content: string
}

export interface AgentEvent {
  event: string
  data: Record<string, any>
}

export interface EvalMetrics {
  retrieval: {
    precision_at_5: number
    precision_at_10: number
    recall_at_10: number
    ndcg_at_10: number
  }
  extraction: Record<string, number>
  cost: {
    total_tokens: number
    total_cost_usd: number
    latency_ms: Record<string, number>
  }
}

export interface GraphNode {
  id: string
  type: 'paper' | 'method' | 'dataset' | 'metric'
  label: string
  year?: number
  url?: string
}

export interface GraphLink {
  source: string
  target: string
  relation: string
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphLink[]
}
