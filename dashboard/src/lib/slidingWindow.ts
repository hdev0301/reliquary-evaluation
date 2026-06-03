import { classifyWindow } from './classify'
import type { WindowDetail, WindowStatus } from './types'

export const WINDOW_CAPACITY = 72

// Merge a fresh poll into the existing Map and prune anything outside the
// trailing 72-window window. The right edge is anchored to the highest window
// number we've ever seen (either from incoming or from prev), so a transient
// empty response doesn't make the strip jump backwards.
export function mergeWindows(
  prev: Map<number, WindowStatus>,
  incoming: WindowDetail[],
): { merged: Map<number, WindowStatus>; latestWindow: number } {
  const next = new Map(prev)

  for (const rec of incoming) {
    if (typeof rec?.window !== 'number') continue
    next.set(rec.window, classifyWindow(rec))
  }

  let incomingMax = -Infinity
  for (const r of incoming) {
    if (typeof r?.window === 'number' && r.window > incomingMax) incomingMax = r.window
  }
  let prevMax = -Infinity
  for (const w of prev.keys()) {
    if (w > prevMax) prevMax = w
  }
  const latestWindow = Math.max(incomingMax, prevMax)
  if (!Number.isFinite(latestWindow)) return { merged: next, latestWindow: 0 }

  const cutoff = latestWindow - (WINDOW_CAPACITY - 1)
  for (const w of [...next.keys()]) {
    if (w < cutoff) next.delete(w)
  }

  return { merged: next, latestWindow }
}

// Build a dense length-72 array for render. Missing entries become 'blank'.
// Order: index 0 = oldest, index 71 = newest (reading left -> right).
export function materializeStrip(
  merged: Map<number, WindowStatus>,
  latestWindow: number,
): WindowStatus[] {
  if (latestWindow <= 0) {
    return Array.from({ length: WINDOW_CAPACITY }, (_, i) => emptySlot(-WINDOW_CAPACITY + i + 1))
  }
  const start = latestWindow - (WINDOW_CAPACITY - 1)
  const out: WindowStatus[] = []
  for (let w = start; w <= latestWindow; w++) {
    out.push(merged.get(w) ?? emptySlot(w))
  }
  return out
}

function emptySlot(window: number): WindowStatus {
  return {
    window,
    bucket: 'blank',
    submitted: 0,
    accepted: 0,
    soft: 0,
    hard: 0,
    score: 0,
    topReason: null,
    createdAt: null,
    slots: [],
  }
}
