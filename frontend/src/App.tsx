import { useState } from 'react'
import AgentTimeline from './components/AgentTimeline'
import EvalMetrics from './components/EvalMetrics'
import KnowledgeGraphViz from './components/KnowledgeGraphViz'
import QueryInput from './components/QueryInput'
import ReportViewer from './components/ReportViewer'
import { useResearchStream } from './hooks/useResearchStream'

type Tab = 'report' | 'graph' | 'evals'

const TABS: { key: Tab; label: string }[] = [
  { key: 'report', label: 'Report' },
  { key: 'graph', label: 'Knowledge Graph' },
  { key: 'evals', label: 'Evaluation Metrics' },
]

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [tab, setTab] = useState<Tab>('report')
  const { events, status, done } = useResearchStream(sessionId)

  const inProgress = sessionId !== null && !done

  return (
    <div className="min-h-screen bg-navy text-slate-100">
      <header className="border-b border-navy-border px-6 py-4">
        <div className="mx-auto flex max-w-7xl items-center gap-3">
          <span className="font-mono text-xl font-semibold text-accent">ResearchFlow</span>
          <span className="text-sm text-slate-400">
            Multi-stage retrieval &amp; synthesis pipeline — citation-backed reports
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-6">
        <section className="mb-6">
          <QueryInput onSessionCreated={setSessionId} disabled={inProgress} />
          {sessionId && (
            <p className="mt-3 text-center font-mono text-xs text-slate-500">
              session {sessionId} · {status}
            </p>
          )}
        </section>

        {sessionId && (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
            <aside className="lg:col-span-2">
              <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
                Agent Timeline
              </h2>
              <div className="rounded-lg border border-navy-border bg-navy-light p-4">
                <AgentTimeline events={events} />
              </div>
            </aside>

            <div className="lg:col-span-3">
              <div className="mb-3 flex gap-1 border-b border-navy-border">
                {TABS.map((t) => (
                  <button
                    key={t.key}
                    onClick={() => setTab(t.key)}
                    className={`px-4 py-2 text-sm transition ${
                      tab === t.key
                        ? 'border-b-2 border-accent text-accent'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
              <div className="rounded-lg border border-navy-border bg-navy-light p-5">
                {tab === 'report' && <ReportViewer sessionId={sessionId} done={done} />}
                {tab === 'graph' && <KnowledgeGraphViz sessionId={sessionId} done={done} />}
                {tab === 'evals' && <EvalMetrics sessionId={sessionId} done={done} />}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
