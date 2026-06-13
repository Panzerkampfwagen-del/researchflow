import { useState } from 'react'
import AgentTimeline from './components/AgentTimeline'
import KnowledgeGraphViz from './components/KnowledgeGraphViz'
import QueryInput from './components/QueryInput'
import ReportViewer from './components/ReportViewer'
import { useResearchStream } from './hooks/useResearchStream'

type Tab = 'report' | 'graph'

const TABS: { key: Tab; label: string }[] = [
  { key: 'report', label: 'Report' },
  { key: 'graph', label: 'Knowledge Graph' },
]

const STAGES = [
  { name: 'Planner', desc: 'Decomposes the question into subtopics and targeted search queries' },
  { name: 'Discovery', desc: 'Searches arXiv + Semantic Scholar, ranks with hybrid retrieval' },
  { name: 'Analysis', desc: 'Extracts methodology, datasets, metrics and limitations per paper' },
  { name: 'Synthesis', desc: 'Detects research gaps and writes a citation-backed report' },
]

const GITHUB_URL = 'https://github.com/Panzerkampfwagen-del/researchflow'

function Logo() {
  return (
    <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-violet-500 shadow-lg shadow-blue-500/20">
      <svg
        width="20"
        height="20"
        viewBox="0 0 24 24"
        fill="none"
        stroke="white"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        <circle cx="5" cy="6" r="2" />
        <circle cx="19" cy="6" r="2" />
        <circle cx="12" cy="18" r="2" />
        <path d="M7 7l4 9M17 7l-4 9" />
      </svg>
    </span>
  )
}

function GitHubIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M12 .5C5.7.5.5 5.7.5 12c0 5.1 3.3 9.4 7.9 10.9.6.1.8-.2.8-.6v-2c-3.2.7-3.9-1.4-3.9-1.4-.5-1.3-1.3-1.7-1.3-1.7-1.1-.7 0-.7 0-.7 1.2.1 1.8 1.2 1.8 1.2 1 1.8 2.7 1.3 3.4 1 .1-.7.4-1.3.7-1.6-2.6-.3-5.3-1.3-5.3-5.7 0-1.3.5-2.3 1.2-3.1-.1-.3-.5-1.5.1-3.1 0 0 1-.3 3.3 1.2a11.5 11.5 0 016 0c2.3-1.5 3.3-1.2 3.3-1.2.6 1.6.2 2.8.1 3.1.8.8 1.2 1.8 1.2 3.1 0 4.4-2.7 5.4-5.3 5.7.4.4.8 1.1.8 2.2v3.3c0 .3.2.7.8.6 4.6-1.5 7.9-5.8 7.9-10.9C23.5 5.7 18.3.5 12 .5z" />
    </svg>
  )
}

function StatusPill({ status, done, inProgress }: { status: string; done: boolean; inProgress: boolean }) {
  const failed = status === 'failed' || status === 'error'
  const dot = failed ? 'bg-red-500' : done ? 'bg-success' : 'bg-accent animate-pulse'
  const text = failed ? 'text-red-400' : done ? 'text-success' : 'text-accent'
  const label = failed ? 'failed' : done ? 'completed' : inProgress ? 'running' : status
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-navy-border bg-navy-light px-3 py-1 text-xs">
      <span className={`h-2 w-2 rounded-full ${dot}`} />
      <span className={`font-mono ${text}`}>{label}</span>
    </span>
  )
}

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [tab, setTab] = useState<Tab>('report')
  const { events, status, done } = useResearchStream(sessionId)

  const inProgress = sessionId !== null && !done

  return (
    <div className="flex min-h-screen flex-col bg-navy text-slate-100">
      <header className="sticky top-0 z-20 border-b border-navy-border bg-navy/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center gap-3 px-6 py-3">
          <Logo />
          <div className="flex flex-col leading-tight">
            <span className="font-mono text-lg font-semibold text-slate-100">ResearchFlow</span>
            <span className="text-[11px] text-slate-400">Multi-stage retrieval &amp; synthesis</span>
          </div>
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
            className="ml-auto flex items-center gap-1.5 rounded-lg border border-navy-border px-3 py-1.5 text-sm text-slate-300 transition hover:border-accent hover:text-accent"
          >
            <GitHubIcon /> GitHub
          </a>
        </div>
      </header>

      <main className="mx-auto w-full max-w-7xl flex-1 px-6 py-8">
        {!sessionId ? (
          <section className="mx-auto max-w-3xl">
            <div className="mb-8 text-center">
              <h1 className="bg-gradient-to-r from-blue-400 via-sky-300 to-violet-400 bg-clip-text text-4xl font-bold tracking-tight text-transparent sm:text-5xl">
                Literature review, automated end to end
              </h1>
              <p className="mx-auto mt-4 max-w-2xl text-base leading-relaxed text-slate-400">
                Give ResearchFlow a research question. It plans the search, pulls papers from arXiv
                and Semantic Scholar, extracts structured findings, and writes a citation-backed
                report with the open research gaps — streamed live as each stage runs.
              </p>
            </div>

            <QueryInput onSessionCreated={setSessionId} disabled={inProgress} />

            <div className="mt-12 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {STAGES.map((s, i) => (
                <div key={s.name} className="rounded-xl border border-navy-border bg-navy-light/60 p-4">
                  <div className="mb-2 flex items-center gap-2">
                    <span className="flex h-6 w-6 items-center justify-center rounded-full bg-accent/15 font-mono text-xs text-accent">
                      {i + 1}
                    </span>
                    <span className="text-sm font-semibold text-slate-200">{s.name}</span>
                  </div>
                  <p className="text-xs leading-relaxed text-slate-400">{s.desc}</p>
                </div>
              ))}
            </div>
          </section>
        ) : (
          <>
            <section className="mb-6">
              <QueryInput onSessionCreated={setSessionId} disabled={inProgress} />
              <div className="mt-3 flex items-center justify-center gap-3">
                <StatusPill status={status} done={done} inProgress={inProgress} />
                <span className="font-mono text-xs text-slate-500">session {sessionId.slice(0, 8)}</span>
              </div>
            </section>

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
              <aside className="lg:col-span-2">
                <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
                  Agent Timeline
                </h2>
                <div className="rounded-xl border border-navy-border bg-navy-light p-4">
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
                <div className="rounded-xl border border-navy-border bg-navy-light p-5">
                  {tab === 'report' && <ReportViewer sessionId={sessionId} done={done} />}
                  {tab === 'graph' && <KnowledgeGraphViz sessionId={sessionId} done={done} />}
                </div>
              </div>
            </div>
          </>
        )}
      </main>

      <footer className="border-t border-navy-border px-6 py-5 text-center text-xs text-slate-500">
        ResearchFlow · multi-stage retrieval-and-synthesis pipeline ·{' '}
        <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="text-slate-400 transition hover:text-accent">
          source on GitHub
        </a>
      </footer>
    </div>
  )
}
