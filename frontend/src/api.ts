export const API_BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')

export async function api<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`)
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
