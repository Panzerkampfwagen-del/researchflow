import type { AgentEvent } from '../types'

interface AgentTimelineProps {
  events: AgentEvent[]
}

type AgentStatus = 'pending' | 'running' | 'complete' | 'error'

const AGENTS = ['planner', 'discovery', 'analysis', 'synthesis'] as const
const LABELS: Record<string, string> = {
  planner: 'Planner',
  discovery: 'Discovery',
  analysis: 'Paper Analysis',
  synthesis: 'Synthesis',
}

const DOT_CLASS: Record<AgentStatus, string> = {
  pending: 'bg-slate-600',
  running: 'bg-accent animate-pulse',
  complete: 'bg-success',
  error: 'bg-red-500',
}

const LIFECYCLE = new Set(['agent_start', 'agent_complete', 'agent_error'])
const LIFECYCLE_STATUS: Record<string, AgentStatus> = {
  agent_start: 'running',
  agent_complete: 'complete',
  agent_error: 'error',
}

function statusFor(agent: string, events: AgentEvent[]): AgentStatus {
  // Latest lifecycle event wins, so a retried-then-completed agent shows
  // 'complete' rather than a stale 'error' from an earlier failed attempt.
  const forAgent = events.filter((e) => e.data?.agent === agent && LIFECYCLE.has(e.event))
  const last = forAgent[forAgent.length - 1]
  return last ? LIFECYCLE_STATUS[last.event] : 'pending'
}

function completeEvent(agent: string, events: AgentEvent[]): AgentEvent | undefined {
  return events.find((e) => e.event === 'agent_complete' && e.data?.agent === agent)
}

function errorEvent(agent: string, events: AgentEvent[]): AgentEvent | undefined {
  return events.find((e) => e.event === 'agent_error' && e.data?.agent === agent)
}

export default function AgentTimeline({ events }: AgentTimelineProps) {
  const papersFound = events.filter((e) => e.event === 'papers_found')
  const analyzed = events.filter((e) => e.event === 'paper_analyzed')
  const lastAnalyzed = analyzed[analyzed.length - 1]
  const verification = events.find((e) => e.event === 'citation_verification')

  return (
    <div className="font-mono text-sm">
      <ol className="relative border-l border-navy-border pl-6">
        {AGENTS.map((agent) => {
          const status = statusFor(agent, events)
          const done = completeEvent(agent, events)
          const err = errorEvent(agent, events)
          return (
            <li key={agent} className="mb-6 animate-fade-in-up">
              <span
                className={`absolute -left-[7px] mt-1.5 h-3.5 w-3.5 rounded-full border-2 border-navy ${DOT_CLASS[status]}`}
              />
              <div className="flex items-center gap-2">
                <span className="rounded bg-navy-light px-2 py-0.5 text-xs uppercase tracking-wide text-slate-300">
                  {LABELS[agent]}
                </span>
                <span
                  className={
                    status === 'error'
                      ? 'text-red-400'
                      : status === 'complete'
                        ? 'text-success'
                        : status === 'running'
                          ? 'text-accent'
                          : 'text-slate-500'
                  }
                >
                  {status}
                </span>
              </div>

              {done && (
                <div className="mt-1 text-xs text-slate-400">
                  {done.data.tokens ?? 0} tok · {done.data.latency_ms ?? 0}ms · $
                  {Number(done.data.cost_usd ?? 0).toFixed(5)}
                </div>
              )}
              {err && <div className="mt-1 text-xs text-red-400">{err.data.message}</div>}

              {agent === 'discovery' && papersFound.length > 0 && (
                <div className="mt-2 space-y-1 border-l border-navy-border pl-4">
                  {papersFound.map((p, i) => (
                    <div key={i} className="text-xs text-slate-400">
                      ↳ {p.data.source}: {p.data.count} papers (total {p.data.total_so_far})
                    </div>
                  ))}
                </div>
              )}

              {agent === 'analysis' && lastAnalyzed && (
                <div className="mt-2 border-l border-navy-border pl-4">
                  <div className="text-xs text-slate-400">
                    Analyzing papers: {lastAnalyzed.data.index} / {lastAnalyzed.data.total}
                  </div>
                  <div className="mt-1 h-1.5 w-full overflow-hidden rounded bg-navy-light">
                    <div
                      className="h-full rounded bg-accent transition-all"
                      style={{
                        width: `${
                          (Number(lastAnalyzed.data.index) /
                            Math.max(Number(lastAnalyzed.data.total), 1)) *
                          100
                        }%`,
                      }}
                    />
                  </div>
                </div>
              )}

              {agent === 'synthesis' && verification && (
                <div className="mt-2 border-l border-navy-border pl-4 text-xs">
                  <div className="text-slate-400">
                    Citations verified: {verification.data.verified} / {verification.data.total} ·
                    hallucination {(Number(verification.data.hallucination_rate) * 100).toFixed(1)}%
                  </div>
                  {Array.isArray(verification.data.unsupported) &&
                    verification.data.unsupported.length > 0 && (
                      <div className="mt-1 text-warning">
                        ⚠️ {verification.data.unsupported.length} unsupported reference(s)
                      </div>
                    )}
                </div>
              )}
            </li>
          )
        })}
      </ol>
    </div>
  )
}
