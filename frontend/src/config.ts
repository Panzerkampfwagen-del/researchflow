// Base URL for the backend API. Empty in dev (Vite proxies /api to :8000);
// set VITE_API_URL to the deployed backend origin in production.
export const API_BASE: string = import.meta.env.VITE_API_URL ?? ''

export const apiUrl = (path: string): string => `${API_BASE}${path}`
