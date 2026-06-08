import { useEffect, useState } from 'react'
import { apiUrl } from '../config'
import type { AgentEvent } from '../types'

interface StreamState {
  events: AgentEvent[]
  status: string
  done: boolean
}

// Subscribes to the SSE progress stream for a session and accumulates events.
export function useResearchStream(sessionId: string | null): StreamState {
  const [events, setEvents] = useState<AgentEvent[]>([])
  const [status, setStatus] = useState<string>('idle')
  const [done, setDone] = useState<boolean>(false)

  useEffect(() => {
    if (!sessionId) return

    setEvents([])
    setStatus('connecting')
    setDone(false)

    const source = new EventSource(apiUrl(`/api/research/${sessionId}/stream`))

    source.onmessage = (message) => {
      let parsed: AgentEvent
      try {
        parsed = JSON.parse(message.data)
      } catch {
        return
      }
      if (parsed.event === 'ping') return

      setEvents((prev) => [...prev, parsed])
      setStatus(parsed.event)

      if (parsed.event === 'done' || parsed.event === 'failed') {
        setDone(true)
        source.close()
      }
    }

    source.onerror = () => {
      setStatus('error')
      source.close()
    }

    return () => {
      source.close()
    }
  }, [sessionId])

  return { events, status, done }
}
