import { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { apiUrl } from '../config'
import type { ResearchReport } from '../types'

interface ReportViewerProps {
  sessionId: string | null
  done: boolean
}

export default function ReportViewer({ sessionId, done }: ReportViewerProps) {
  const [report, setReport] = useState<ResearchReport | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!done || !sessionId) return
    setLoading(true)
    setError(null)
    fetch(apiUrl(`/api/reports/${sessionId}`))
      .then(async (resp) => {
        if (!resp.ok) throw new Error(`Report unavailable (${resp.status})`)
        return resp.json()
      })
      .then((data: ResearchReport) => setReport(data))
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load report'))
      .finally(() => setLoading(false))
  }, [sessionId, done])

  const exportMarkdown = () => {
    if (!report) return
    const blob = new Blob([report.markdown_content], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `researchflow-${sessionId}.md`
    link.click()
    URL.revokeObjectURL(url)
  }

  if (!done) {
    return <p className="text-sm text-slate-500">The report will appear here when synthesis completes.</p>
  }
  if (loading) return <p className="text-sm text-slate-400">Loading report…</p>
  if (error) return <p className="text-sm text-red-400">{error}</p>
  if (!report) return null

  return (
    <div>
      <div className="mb-4 flex justify-end">
        <button
          onClick={exportMarkdown}
          className="rounded-lg border border-navy-border bg-navy-light px-4 py-1.5 text-sm text-slate-200 transition hover:border-accent"
        >
          Export Markdown
        </button>
      </div>
      <article className="markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{report.markdown_content}</ReactMarkdown>
      </article>
    </div>
  )
}
