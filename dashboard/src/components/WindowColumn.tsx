import { MAX_SLOTS_PER_WINDOW } from '@/lib/classify'
import type { WindowStatus } from '@/lib/types'
import WindowDot from './WindowDot'

const LABEL_BY_BUCKET: Record<WindowStatus['bucket'], string> = {
  accepted: 'accepted',
  soft: 'soft-failed',
  hard: 'hard-failed',
  blank: 'no submission',
}

function relativeTime(iso: string | null): string {
  if (!iso) return ''
  const t = Date.parse(iso)
  if (!Number.isFinite(t)) return ''
  const ms = Date.now() - t
  if (ms < 60_000) return `${Math.round(ms / 1000)}s ago`
  if (ms < 3_600_000) return `${Math.round(ms / 60_000)}m ago`
  if (ms < 86_400_000) return `${Math.round(ms / 3_600_000)}h ago`
  return `${Math.round(ms / 86_400_000)}d ago`
}

// One window = one vertical column of up to 8 dots, anchored to the bottom.
// Each dot is one actual submission (accepted/soft-fail/hard-fail). A window
// with no submissions renders an empty column to preserve the time axis.
export default function WindowColumn({ status }: { status: WindowStatus }) {
  const { bucket, window, submitted, accepted, soft, hard, score, topReason, createdAt, slots } =
    status
  const lines = [
    `window ${window} - ${LABEL_BY_BUCKET[bucket]}`,
    submitted === 0
      ? null
      : `submitted ${submitted} / acc ${accepted} / soft ${soft} / hard ${hard}`,
    submitted === 0 ? null : `score ${score.toFixed(3)}`,
    topReason ? `top reason: ${topReason}` : null,
    createdAt ? relativeTime(createdAt) : null,
  ].filter(Boolean) as string[]
  const tooltip = lines.join('\n')
  const aria =
    submitted === 0
      ? `Window ${window}: no submission`
      : `Window ${window}: ${accepted} accepted, ${soft} soft-failed, ${hard} hard-failed of ${submitted} submitted`
  const visibleSlots = slots.slice(0, MAX_SLOTS_PER_WINDOW)
  return (
    <div
      className="window-col"
      role="img"
      aria-label={aria}
      title={tooltip}
      data-bucket={bucket}
    >
      {visibleSlots.map((b, i) => (
        <WindowDot key={i} bucket={b} />
      ))}
    </div>
  )
}
