import type { WindowStatus } from '@/lib/types'
import WindowColumn from './WindowColumn'

interface Props {
  windows: WindowStatus[]
  latestWindow: number
}

export default function WindowStrip({ windows, latestWindow }: Props) {
  const first = windows[0]?.window
  const last = windows[windows.length - 1]?.window
  const slotCounts = windows.reduce(
    (acc, w) => {
      for (const s of w.slots) acc[s]++
      return acc
    },
    { accepted: 0, soft: 0, hard: 0, blank: 0 },
  )
  const windowsWithSubmissions = windows.filter((w) => w.submitted > 0).length
  return (
    <section className="strip-section">
      <div className="strip-header">
        <h2>Last 72 windows &middot; per-submission dots</h2>
        <span className="strip-range mono">
          w{first ?? '-'} &rarr; w{last ?? '-'}{' '}
          {latestWindow ? <span className="muted">(live: w{latestWindow})</span> : null}
        </span>
      </div>
      <div className="window-strip" aria-label="Last 72 windows, one column per window, one dot per submission">
        {windows.map((w) => (
          <WindowColumn key={w.window} status={w} />
        ))}
      </div>
      <div className="legend" aria-hidden="true">
        <span className="legend-item">
          <span className="swatch" style={{ background: 'var(--accepted)' }} />
          accepted ({slotCounts.accepted})
        </span>
        <span className="legend-item">
          <span className="swatch" style={{ background: 'var(--soft)' }} />
          soft-failed ({slotCounts.soft})
        </span>
        <span className="legend-item">
          <span className="swatch" style={{ background: 'var(--hard)' }} />
          hard-failed ({slotCounts.hard})
        </span>
        <span className="legend-item muted">
          {windowsWithSubmissions} / {windows.length} windows with submissions
        </span>
      </div>
    </section>
  )
}
