// Base URL for the backend API. Empty in dev (Vite proxies /api to :8000).
// VITE_API_URL overrides the default production URL (useful for staging).
const PROD_URL = 'https://researchflow-j62g.onrender.com'
export const API_BASE: string =
  import.meta.env.VITE_API_URL ?? (import.meta.env.PROD ? PROD_URL : '')

export const apiUrl = (path: string): string => `${API_BASE}${path}`
