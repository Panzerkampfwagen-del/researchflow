import { useState } from 'react'
import { apiUrl } from '../config'

interface QueryInputProps {
  onSessionCreated: (sessionId: string) => void
  disabled: boolean
}

const CURRENT_YEAR = new Date().getFullYear()

export default function QueryInput({ onSessionCreated, disabled }: QueryInputProps) {
  const [query, setQuery] = useState('')
  const [yearStart, setYearStart] = useState(2020)
  const [yearEnd, setYearEnd] = useState(CURRENT_YEAR)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    if (query.trim().length < 10) {
      setError('Query must be at least 10 characters.')
      return
    }
    setLoading(true)
    try {
      const resp = await fetch(apiUrl('/api/research'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, year_start: yearStart, year_end: yearEnd }),
      })
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}))
        throw new Error(body.error || `Request failed (${resp.status})`)
      }
      const data = await resp.json()
      onSessionCreated(data.session_id)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start research')
    } finally {
      setLoading(false)
    }
  }

  const isDisabled = disabled || loading

  return (
    <form onSubmit={submit} className="w-full max-w-3xl mx-auto">
      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        disabled={isDisabled}
        placeholder="e.g. Recent advances in post-training quantization for large language models"
        rows={3}
        className="w-full resize-none rounded-lg bg-navy-light border border-navy-border px-4 py-3 text-base text-slate-100 placeholder-slate-500 focus:border-accent focus:outline-none disabled:opacity-50"
      />
      <div className="mt-3 flex flex-wrap items-center gap-4">
        <label className="flex items-center gap-2 text-sm text-slate-400">
          From
          <input
            type="number"
            value={yearStart}
            min={1990}
            max={CURRENT_YEAR}
            disabled={isDisabled}
            onChange={(e) => setYearStart(Number(e.target.value))}
            className="w-24 rounded bg-navy-light border border-navy-border px-2 py-1 font-mono text-slate-100 focus:border-accent focus:outline-none"
          />
        </label>
        <label className="flex items-center gap-2 text-sm text-slate-400">
          To
          <input
            type="number"
            value={yearEnd}
            min={1990}
            max={CURRENT_YEAR}
            disabled={isDisabled}
            onChange={(e) => setYearEnd(Number(e.target.value))}
            className="w-24 rounded bg-navy-light border border-navy-border px-2 py-1 font-mono text-slate-100 focus:border-accent focus:outline-none"
          />
        </label>
        <button
          type="submit"
          disabled={isDisabled}
          className="ml-auto flex items-center gap-2 rounded-lg bg-accent px-5 py-2 font-medium text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading && (
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
          )}
          {loading ? 'Starting…' : 'Run Research'}
        </button>
      </div>
      {error && <p className="mt-2 text-sm text-red-400">{error}</p>}
    </form>
  )
}
