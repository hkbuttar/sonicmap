import { afterEach, describe, expect, it, vi } from 'vitest'
import { api } from './api'

describe('API deployment errors', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('explains when the frontend returns HTML instead of API JSON', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<!doctype html>', {
      status: 200,
      headers: { 'content-type': 'text/html' },
    })))
    await expect(api('/api/health')).rejects.toThrow('VITE_API_URL is not configured')
  })
})
