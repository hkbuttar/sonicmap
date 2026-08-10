export const API_BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')

export async function api<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`)
  const contentType = response.headers.get('content-type') ?? ''
  if (!contentType.includes('application/json')) {
    const deploymentHint = API_BASE
      ? `The configured API at ${API_BASE} returned ${contentType || 'an unknown content type'}.`
      : 'VITE_API_URL is not configured, so the request reached the frontend instead of FastAPI.'
    throw new Error(`${deploymentHint} Set VITE_API_URL to the Render service origin and redeploy.`)
  }
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(detail.detail ?? `Request failed: ${response.status}`)
  }
  return response.json() as Promise<T>
}

export type MetricRow = {
  model?: string
  method?: string
  metric: string
  mean?: number
  ci_low?: number
  ci_high?: number
  augmented?: boolean
}

export type EmbeddingPoint = {
  track_id: string
  label: string
  label_index: number
  x: number
  y: number
}
