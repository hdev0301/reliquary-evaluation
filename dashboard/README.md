# Reliquary miner dashboard

Live submission-status heatbar for a single Bittensor subnet 81 miner.

Polls `https://www.reliqua.ai/api/miners/<hotkey>` every ~7s and renders the last
72 windows as a strip of colored dots:

| Color  | Bucket       | Trigger                                              |
| ------ | ------------ | ---------------------------------------------------- |
| green  | accepted     | `accepted > 0` for that window                       |
| brown  | soft-failed  | only `soft_failed > 0` (transient / retry-friendly)  |
| red    | hard-failed  | `hard_failed > 0` and no acceptance                  |
| blank  | no submission | window absent or `submitted == 0`                   |

When the validator publishes a new window, the oldest entry on the left drops
off automatically — the strip always shows exactly the trailing 72 windows.

## Run

```powershell
cd d:\Work\Bittensor\Reliquadotai\reliquary-evaluation\dashboard
npm install
npm run dev
# open http://localhost:3000
```

Override the hotkey via URL:

```
http://localhost:3000?hotkey=5SomeOtherSS58Address...
```

Or set the default in `.env.local` (copy `.env.local.example`):

```
NEXT_PUBLIC_DEFAULT_HOTKEY=5HGr6joke42gGZxMHsDTuJEepnmbaihM7KdUwVtq2kA6TNAN
RELIQUA_BASE_URL=https://www.reliqua.ai
```

## Design notes

- **Proxy route** at `/api/miner/[hotkey]` — sidesteps CORS, lets us set
  `Cache-Control: no-store`, and gives us server-side log visibility when
  the upstream schema drifts. Validates the hotkey against the SS58 regex
  before issuing the upstream fetch (defense in depth against SSRF).
- **Sliding window** lives in a `Map<number, WindowStatus>` ref so that
  per-poll merges don't trigger render churn. A `version` counter forces
  the re-render after each merge. The right edge is anchored to
  `max(seen_window)` across all polls; a transient empty response never
  shifts the strip backwards.
- **Soft / hard reason split** mirrors `reliquary/protocol/submission.py`.
  Unknown reasons default to `hard` (over-flag, never silently miss).
- **One dot per window**, not one per submission slot. A window with up to
  `MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW = 8` rollouts collapses to a single
  best-of color (accepted > hard > soft > blank). Tooltip surfaces the
  underlying counts and the top reject reason.

## Files

```
src/
├── app/
│   ├── layout.tsx            root layout
│   ├── page.tsx              server shell, reads ?hotkey=
│   ├── globals.css           dark theme + dot grid CSS
│   └── api/miner/[hotkey]/route.ts   proxy to reliqua.ai
├── components/
│   ├── MinerDashboard.tsx    client orchestrator
│   ├── MinerHeader.tsx
│   ├── StatsRow.tsx
│   ├── WindowStrip.tsx
│   ├── WindowDot.tsx
│   └── RejectBreakdown.tsx
└── lib/
    ├── types.ts              MinerResponse / WindowDetail / WindowStatus
    ├── reasons.ts            SOFT_FAIL_REASONS / HARD_FAIL_REASONS sets
    ├── classify.ts           WindowDetail -> Bucket
    ├── slidingWindow.ts      mergeWindows / materializeStrip
    └── poll.ts               useMinerPoll hook
```

## Known limits

- API has no pagination; the upstream returns only what it returns
  (typically ~30 to ~72 recent windows). Older windows render as blank
  until they age off entirely.
- Reject-reason enum drift: when `reliquary/protocol/submission.py` adds a
  new variant, classify it in `src/lib/reasons.ts`.
