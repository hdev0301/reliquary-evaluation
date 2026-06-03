import type { Bucket } from '@/lib/types'

// One submission slot inside a window column. Pure presentational.
export default function WindowDot({ bucket }: { bucket: Bucket }) {
  return <span className="dot" data-bucket={bucket} aria-hidden="true" />
}
