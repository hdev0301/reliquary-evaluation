import { reasonSeverity } from './reasons'
import type { Bucket, WindowDetail, WindowStatus } from './types'

// Protocol cap from reliquary docs: MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW = 8.
export const MAX_SLOTS_PER_WINDOW = 8

// Reconcile validator counters with the miner_reject_reasons fallback. Observed
// anomaly (window 11433 in the probe): submitted=4 but accepted/soft/hard=0,
// while miner_reject_reasons={batch_filled:4}. In that case we re-derive the
// soft/hard split from the reason map so the dot column isn't empty.
function bucketCounts(r: WindowDetail): {
  accepted: number
  soft: number
  hard: number
  submitted: number
} {
  let accepted = r.accepted ?? 0
  let soft = r.soft_failed ?? 0
  let hard = r.hard_failed ?? 0
  const submitted = r.submitted ?? 0

  if (submitted > 0 && accepted + soft + hard === 0 && r.miner_reject_reasons) {
    for (const [reason, count] of Object.entries(r.miner_reject_reasons)) {
      const sev = reasonSeverity(reason)
      if (sev === 'soft') soft += count
      else hard += count
    }
  }
  return { accepted, soft, hard, submitted }
}

// One dot per actual submission, ordered bottom-up by severity:
//   index 0 (bottom) ... accepted ... soft ... hard ... (top) index N-1
// Capped at MAX_SLOTS_PER_WINDOW = 8 per protocol.
function buildSlots(accepted: number, soft: number, hard: number): Bucket[] {
  const slots: Bucket[] = []
  for (let i = 0; i < accepted; i++) slots.push('accepted')
  for (let i = 0; i < soft; i++) slots.push('soft')
  for (let i = 0; i < hard; i++) slots.push('hard')
  return slots.slice(0, MAX_SLOTS_PER_WINDOW)
}

// Summary bucket for tooltip + accent purposes. Accept-dominant priority:
// any acceptance wins the window; otherwise red beats brown beats blank.
function summaryBucket(
  accepted: number,
  soft: number,
  hard: number,
  submitted: number,
): Bucket {
  if (submitted === 0 && accepted + soft + hard === 0) return 'blank'
  if (accepted > 0) return 'accepted'
  if (hard > 0) return 'hard'
  if (soft > 0) return 'soft'
  return 'blank'
}

export function classifyWindow(r: WindowDetail): WindowStatus {
  const { accepted, soft, hard, submitted } = bucketCounts(r)
  const slots = buildSlots(accepted, soft, hard)
  const bucket = summaryBucket(accepted, soft, hard, submitted)

  let topReason: string | null = null
  if (r.miner_reject_reasons) {
    const entries = Object.entries(r.miner_reject_reasons)
    if (entries.length > 0) {
      entries.sort((a, b) => b[1] - a[1])
      topReason = entries[0][0]
    }
  }

  return {
    window: r.window,
    bucket,
    submitted,
    accepted,
    soft,
    hard,
    score: r.score ?? 0,
    topReason,
    createdAt: r.created_at ?? null,
    slots,
  }
}
