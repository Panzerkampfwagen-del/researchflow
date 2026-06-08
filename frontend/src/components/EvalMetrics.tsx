import { useEffect, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { apiUrl } from '../config'
import type { EvalMetrics as EvalMetricsType } from '../types'

interface EvalMetricsProps {
  sessionId: string | null
  done: boolean
}

const BAR_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-navy-border bg-navy-light p-4">
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-300">{title}</h3>
      {children}
    </div>
  )
}

function ScoreChart({ data }: { data: { name: string; value: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} />
        <YAxis domain={[0, 1]} stroke="#94a3b8" fontSize={11} />
        <Tooltip
          contentStyle={{ background: '#0a0f1e', border: '1px solid #1e293b', borderRadius: 8 }}
        />
        <Bar dataKey="value" radius={[4, 4, 0, 0]}>
          {data.map((_, i) => (
            <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

export default function EvalMetrics({ sessionId, done }: EvalMetricsProps) {
  const [metrics, setMetrics] = useState<EvalMetricsType | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!done || !sessionId) return
    setError(null)
    fetch(apiUrl(`/api/evals/${sessionId}`))
      .then(async (resp) => {
        if (!resp.ok) throw new Error(`Metrics unavailable (${resp.status})`)
        return resp.json()
      })
      .then((data: EvalMetricsType) => setMetrics(data))
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load metrics'))
  }, [sessionId, done])

  if (!done) {
    return <p className="text-sm text-slate-500">Evaluation metrics will appear here when the run completes.</p>
  }
  if (error) return <p className="text-sm text-red-400">{error}</p>
  if (!metrics) return <p className="text-sm text-slate-400">Loading metrics…</p>

  const retrievalData = [
    { name: 'P@5', value: metrics.retrieval.precision_at_5 },
    { name: 'P@10', value: metrics.retrieval.precision_at_10 },
    { name: 'R@10', value: metrics.retrieval.recall_at_10 },
    { name: 'NDCG@10', value: metrics.retrieval.ndcg_at_10 },
  ]
  const extractionData = Object.entries(metrics.extraction).map(([key, value]) => ({
    name: key.replace(/_f1$/, ''),
    value,
  }))
  const latencyData = Object.entries(metrics.cost.latency_ms).map(([name, ms]) => ({ name, ms }))

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <Card title="Retrieval Quality">
        <ScoreChart data={retrievalData} />
      </Card>
      <Card title="Extraction Accuracy (F1)">
        <ScoreChart data={extractionData} />
      </Card>
      <Card title="Cost">
        <div className="space-y-2 font-mono text-sm">
          <div className="flex justify-between">
            <span className="text-slate-400">Total tokens</span>
            <span className="text-slate-100">{metrics.cost.total_tokens.toLocaleString()}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Total cost</span>
            <span className="text-success">${metrics.cost.total_cost_usd.toFixed(5)}</span>
          </div>
        </div>
      </Card>
      <Card title="Per-Agent Latency (ms)">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart
            layout="vertical"
            data={latencyData}
            margin={{ top: 5, right: 10, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis type="number" stroke="#94a3b8" fontSize={11} />
            <YAxis type="category" dataKey="name" stroke="#94a3b8" fontSize={11} width={70} />
            <Tooltip
              contentStyle={{ background: '#0a0f1e', border: '1px solid #1e293b', borderRadius: 8 }}
            />
            <Bar dataKey="ms" fill="#3b82f6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}
