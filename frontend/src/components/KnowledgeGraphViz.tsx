import { useEffect, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { apiUrl } from '../config'
import type { GraphData } from '../types'

interface KnowledgeGraphVizProps {
  sessionId: string | null
  done: boolean
}

const NODE_COLORS: Record<string, string> = {
  paper: '#3b82f6',
  method: '#10b981',
  dataset: '#f59e0b',
  metric: '#8b5cf6',
}

const LEGEND = [
  { type: 'paper', label: 'Paper' },
  { type: 'method', label: 'Method' },
  { type: 'dataset', label: 'Dataset' },
  { type: 'metric', label: 'Metric' },
]

export default function KnowledgeGraphViz({ sessionId, done }: KnowledgeGraphVizProps) {
  const [data, setData] = useState<{ nodes: any[]; links: any[] } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(600)

  useEffect(() => {
    const update = () => {
      if (containerRef.current) setWidth(containerRef.current.clientWidth)
    }
    update()
    window.addEventListener('resize', update)
    return () => window.removeEventListener('resize', update)
  }, [data])

  useEffect(() => {
    if (!done || !sessionId) return
    setError(null)
    fetch(apiUrl(`/api/knowledge-graph/${sessionId}`))
      .then(async (resp) => {
        if (!resp.ok) throw new Error(`Graph unavailable (${resp.status})`)
        return resp.json()
      })
      .then((graph: GraphData) =>
        setData({
          nodes: graph.nodes.map((n) => ({ ...n })),
          links: graph.edges.map((e) => ({ ...e })),
        }),
      )
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load graph'))
  }, [sessionId, done])

  if (!done) {
    return <p className="text-sm text-slate-500">The knowledge graph will appear here when the run completes.</p>
  }
  if (error) return <p className="text-sm text-red-400">{error}</p>
  if (!data) return <p className="text-sm text-slate-400">Loading knowledge graph…</p>
  if (data.nodes.length === 0) return <p className="text-sm text-slate-500">No graph data for this session.</p>

  return (
    <div ref={containerRef} className="relative rounded-lg border border-navy-border bg-[#060a14]">
      <div className="absolute right-3 top-3 z-10 rounded-md bg-navy/80 p-2 text-xs">
        {LEGEND.map((item) => (
          <div key={item.type} className="flex items-center gap-2">
            <span
              className="inline-block h-3 w-3 rounded-sm"
              style={{ backgroundColor: NODE_COLORS[item.type] }}
            />
            <span className="text-slate-300">{item.label}</span>
          </div>
        ))}
      </div>
      <ForceGraph2D
        graphData={data}
        width={width}
        height={480}
        backgroundColor="#060a14"
        nodeColor={(node: any) => NODE_COLORS[node.type] ?? '#94a3b8'}
        nodeVal={(node: any) => (node.type === 'paper' ? 8 : 3)}
        nodeLabel={(node: any) => `${node.type}: ${node.label}`}
        linkColor={() => '#334155'}
        linkLabel={(link: any) => link.relation}
        linkDirectionalArrowLength={3}
        linkDirectionalArrowRelPos={1}
      />
    </div>
  )
}
