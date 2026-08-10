import { describe, expect, it } from 'vitest'
import { fmt } from './components'

describe('dashboard formatting', () => {
  it('formats experiment metrics consistently', () => {
    expect(fmt(0.758718, 1)).toBe('0.8')
    expect(fmt(0.758718)).toBe('0.759')
  })
})
