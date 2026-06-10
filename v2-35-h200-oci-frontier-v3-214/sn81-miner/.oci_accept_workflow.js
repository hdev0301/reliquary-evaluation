export const meta = {
  name: 'oci-accept-strategy',
  description: 'Map SN81 OCI accept/reject mechanics, diagnose low accept rate, design + score strategies to raise accepted submits',
  phases: [
    { title: 'Understand', detail: '4 parallel readers: validator accept path, miner gen/submit path, opencode reward+cases, log/data forensics' },
    { title: 'Diagnose', detail: 'synthesize ranked root causes of low accept rate' },
    { title: 'Design', detail: '6 diverse strategists propose concrete accept-raising changes' },
    { title: 'Score', detail: 'adversarial judges score each strategy for feasibility/impact/risk' },
    { title: 'Synthesize', detail: 'final ranked recommendation' },
  ],
}

const REPO = '/root/sn81-miner'
const REL = '/root/reliquary/reliquary'

const READER_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['area', 'mechanism_facts', 'accept_levers', 'failure_modes', 'surprises'],
  properties: {
    area: { type: 'string' },
    mechanism_facts: {
      type: 'array', description: 'Precise, code-cited facts about how the mechanism works. Cite file:line.',
      items: { type: 'string' }, minItems: 3,
    },
    accept_levers: {
      type: 'array', description: 'Knobs/code paths that change how many submits get ACCEPTED.',
      items: {
        type: 'object', additionalProperties: false,
        required: ['lever', 'where', 'current_value', 'effect_on_accepts'],
        properties: {
          lever: { type: 'string' }, where: { type: 'string' },
          current_value: { type: 'string' }, effect_on_accepts: { type: 'string' },
        },
      },
    },
    failure_modes: {
      type: 'array', description: 'Each reject reason / supply loss this area causes, with exact trigger condition and the lever that fixes it.',
      items: {
        type: 'object', additionalProperties: false,
        required: ['reason', 'exact_trigger', 'fix'],
        properties: { reason: { type: 'string' }, exact_trigger: { type: 'string' }, fix: { type: 'string' } },
      },
    },
    surprises: { type: 'array', description: 'Non-obvious findings, bugs, dead code, or config that contradicts the docs.', items: { type: 'string' } },
  },
}

const DIAGNOSIS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['primary_bottleneck', 'bottleneck_chain', 'root_causes_ranked', 'baseline_vs_now', 'key_numbers'],
  properties: {
    primary_bottleneck: { type: 'string', description: 'supply (too few in-zone groups) OR race (batch_filled) OR divergence (out_of_zone/reward_mismatch) OR cooldown/dup — pick the dominant one with evidence.' },
    bottleneck_chain: { type: 'string', description: 'The full funnel from prompt -> rollouts -> in-zone group -> seal -> ACCEPTED, marking where the biggest drop happens.' },
    root_causes_ranked: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['cause', 'evidence', 'severity'],
        properties: { cause: { type: 'string' }, evidence: { type: 'string' }, severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] } },
      }, minItems: 3,
    },
    baseline_vs_now: { type: 'string', description: 'Why did logs/miner.log.baseline get 104 accepts while current gets ~2? What config/state changed?' },
    key_numbers: { type: 'array', items: { type: 'string' } },
  },
}

const STRATEGY_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['name', 'angle', 'hypothesis', 'concrete_changes', 'expected_effect', 'risks', 'effort', 'verifiable_metric'],
  properties: {
    name: { type: 'string' },
    angle: { type: 'string' },
    hypothesis: { type: 'string', description: 'The causal claim: changing X raises accepts because Y.' },
    concrete_changes: {
      type: 'array', minItems: 1,
      items: {
        type: 'object', additionalProperties: false,
        required: ['target', 'change', 'rationale'],
        properties: {
          target: { type: 'string', description: 'exact env var / file:func / script' },
          change: { type: 'string', description: 'precise new value or code change' },
          rationale: { type: 'string' },
        },
      },
    },
    expected_effect: { type: 'string', description: 'Quantified if possible: e.g. "lifts in-zone yield from ~0.1/window to ~3/window".' },
    risks: { type: 'array', items: { type: 'string' } },
    effort: { type: 'string', enum: ['trivial', 'low', 'medium', 'high'] },
    verifiable_metric: { type: 'string', description: 'What log line/number confirms it worked, measurable within ~1 hour of mining.' },
  },
}

const SCORE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['strategy_name', 'feasibility', 'impact', 'risk', 'grounded_in_code', 'critique', 'corrections', 'overall'],
  properties: {
    strategy_name: { type: 'string' },
    feasibility: { type: 'integer', minimum: 1, maximum: 5 },
    impact: { type: 'integer', minimum: 1, maximum: 5 },
    risk: { type: 'integer', minimum: 1, maximum: 5, description: '1=safe, 5=could tank accepts further' },
    grounded_in_code: { type: 'boolean', description: 'Does the proposed change reference levers that ACTUALLY exist in the code as read?' },
    critique: { type: 'string', description: 'Adversarial: where does this fail or rest on a wrong assumption?' },
    corrections: { type: 'string', description: 'Fixes/adjustments that make it correct or stronger.' },
    overall: { type: 'number', description: 'weighted score 0-10' },
  },
}

// ---- empirical facts I (the orchestrator) already established; agents must corroborate or refute ----
const GROUNDING = `
EMPIRICAL FACTS established by the orchestrator (corroborate against code/logs, refute if wrong):
- This is SN81/Reliquary OCI mining (nvidia/OpenCodeInstruct env). Reward is validator-authoritative
  passed/total over HIDDEN structured tests. Miner mines a local-reward subset (reconstructed cases + local grader).
- In-zone = k correct + (8-k) wrong with sigma >= 0.43 across the 8-rollout group.
- Current live config (opencode/run_miner.sh): CURATE=1, TARGET_K=4 (sigma 0.5 target), SCREEN_P_LOW=0.10,
  SCREEN_P_HIGH=0.80, OVERSAMPLE=48, HOT_FRAC=0.0, ALLCORR_BURN=1, PREDICT_BLIND=1, PREDICT_LEAD_MS=1800,
  DECOOL_SNIPE=1, MAX_NEW_TOKENS=1024, SCREEN_MAX_TOKENS=768, MAX_NUM_SEQS=512, GPU_MEM=0.65.
- CURRENT miner.log (ckpt 6477312861): screens show 2-10/24 "promising"; the dropped prompts are
  OVERWHELMINGLY allcorr (e.g. "extreme=12 [allcorr=10 allwrong=2]"); pregen batches show
  avg_n_correct ~= 20-23 out of ~25 completions => the live checkpoint SOLVES nearly all pooled prompts
  => few/no scatter groups => 11x "+0/24 in-zone", 2x "+1/24". Only 2 accepts this run.
- Historical accept counts: baseline=104 (best; 31 stale_round, 13 batch_filled), breadth-trial=22
  (43 batch_filled), chunk-only=2 (32 batch_filled), revert-confirm=17 (30 batch_filled, 12 stale_round),
  miner_v2=24 (54 batch_filled). => batch_filled is the dominant reject historically; supply (allcorr) is
  the dominant problem NOW.
- divergent_idx.json is tiny (554 bytes); cooled_idx.json is large (26KB) and growing (per-prompt cooldown
  is effectively permanent: BATCH_PROMPT_COOLDOWN_WINDOWS=1e6).
Relevant paths: ${REPO}/opencode/{run_miner.sh,build_pool.sh,build_frontier_pool.sh,grow_data.sh,build_local_subset.py,grader.sh,RUNBOOK.md},
${REPO}/dataprep/build_opencode_pool.py, ${REPO}/bin/run_miner.sh, ${REPO}/logs/*, ${REPO}/opencode/{logs,data,diagnostics}/*,
${REL}/miner/{engine.py,pregen.py,frontier.py,submitter.py}, ${REL}/validator/{server.py,batcher.py,batch_selection.py,cooldown.py,dedup.py,reward_shape.py,verifier.py},
${REL}/environment/{opencodeinstruct.py,grader_client.py,base.py}.
`

phase('Understand')
const readers = await parallel([
  () => agent(`You are mapping the VALIDATOR acceptance path of SN81/Reliquary so we can raise ACCEPTED OCI submits.
${GROUNDING}
Read closely: ${REL}/validator/server.py, batcher.py, batch_selection.py, cooldown.py, dedup.py, reward_shape.py, verifier.py.
Answer with PRECISION (cite file:line):
1. The exact seal/window mechanism: how the 8 shared seal slots fill, what "batch_filled" means, what ordering decides WHICH groups win the seal (is there a canonical sha256(prompt_idx) top-8 ranking? where?), window length/timing.
2. Exact in-zone test: the sigma>=0.43 band, what k values count as in-zone (2..6/8?), where reward_shape enforces it.
3. Every REJECT reason the validator can emit (batch_filled, out_of_zone, reward_mismatch, hash_duplicate, prompt_in_cooldown, stale_round, worker_dropped, future_round) and the EXACT trigger condition for each.
4. The cooldown rule (BATCH_PROMPT_COOLDOWN_WINDOWS) and dedup/hash_duplicate rule — how long is a won prompt dead, what exactly is hashed.
5. reward_mismatch / out_of_zone vs the miner's reconstructed cases: how the validator re-grades on its OWN pinned hidden cases and what divergence triggers a reject.
Return the structured object. accept_levers = anything a miner can do to win more seals / avoid rejects.`,
    { label: 'read:validator', phase: 'Understand', schema: READER_SCHEMA }),

  () => agent(`You are mapping the MINER generation+submission path of SN81/Reliquary so we can raise ACCEPTED OCI submits.
${GROUNDING}
Read closely: ${REL}/miner/engine.py, pregen.py, frontier.py, submitter.py, and ${REPO}/bin/run_miner.sh + ${REPO}/opencode/run_miner.sh (env wiring).
Answer with PRECISION (cite file:line):
1. The full pregen pipeline: cheap-screen (SCREEN_P_LOW/HIGH band, "promising" count), deep-mine (OVERSAMPLE), the curate step (CURATE, TARGET_K, RELIQUARY_CORRECT_BAND/WRONG_BAND), how an in-zone 8-group is assembled and the sigma computed. Where does "+N/24 in-zone groups" come from?
2. Why does the CURRENT pool produce mostly allcorr (too-easy) screens? What in pregen decides a prompt is "not_curatable"? How does ALLCORR_BURN work and is it actually firing?
3. The predictive seal-fire path: PREDICT_BLIND, PREDICT_MIN_WINDOWS, PREDICT_LEAD_MS, PREDICT_POST_MS — how the engine predicts the window boundary and fires early to beat batch_filled. Is it engaging (does L converge)? What logs prove it.
4. HOT_POOL / HOT_FRAC and DECOOL_SNIPE behavior and why HOT_FRAC=0.0 now.
5. The frontier predictor (frontier.py): features used, online update, persistence (/root/frontier_model.npz), epsilon-explore — how it selects prompts and whether it actually improves in-zone density.
6. Every supply-side loss (group never forms) and submit-side timing loss the miner controls.
Return the structured object. accept_levers = miner knobs that change accepts.`,
    { label: 'read:miner', phase: 'Understand', schema: READER_SCHEMA }),

  () => agent(`You are mapping the OPENCODE REWARD + test-case reconstruction path of SN81/Reliquary so we can raise ACCEPTED OCI submits.
${GROUNDING}
Read closely: ${REL}/environment/opencodeinstruct.py, grader_client.py, base.py; ${REPO}/dataprep/build_opencode_pool.py; ${REPO}/opencode/build_local_subset.py, build_pool.sh, build_frontier_pool.sh, grow_data.sh, grader.sh.
Answer with PRECISION (cite file:line):
1. How compute_reward works for opencode (passed/total). Why OCI_PROMPT_ONLY=1 yields reward 0. How the local subset + local grader restore a real reward signal (exact parity with validator worker.py?).
2. Case reconstruction: how build_opencode_pool.py / grow_data.sh rebuild hidden test cases from the PUBLIC nvidia/OpenCodeInstruct, and exactly HOW these can diverge from the validator's pinned HIDDEN cases (=> out_of_zone/reward_mismatch). How big is that divergence risk per prompt?
3. The scatter/canon pool builders: build_pool.sh (GPU scatter-screen, pass-fraction band) vs build_frontier_pool.sh (canon-filter = keep lowest sha256(prompt_idx) KEEP_FRAC, divergence-burn). What does canon-filter buy at seal time?
4. As the validator checkpoint advances (~hourly), why does a screened pool go stale (more allcorr)? What's the right way to keep prompt difficulty intermediate for the LIVE checkpoint?
5. Is the local grader running / configured right? What makes "screen: 0/24 promising allwrong=24" vs "allcorr" — distinguish "cases wrong/missing" from "model too good".
Return the structured object. accept_levers = data/reward changes that change accepts.`,
    { label: 'read:reward-cases', phase: 'Understand', schema: READER_SCHEMA }),

  () => agent(`You are doing EMPIRICAL forensics on SN81 OCI mining logs+data to find where ACCEPTED submits are lost. Be quantitative.
${GROUNDING}
Use Bash (grep/awk/python) over: ${REPO}/logs/miner.log and miner.log.* and miner_v2.log; ${REPO}/opencode/logs/*; and the data/diagnostics:
${REPO}/diagnostics/{reject_dump.jsonl,opencode_pool_meta.json}, ${REPO}/opencode/diagnostics/opencode_pool_meta.json,
${REPO}/opencode/data/{divergent_idx.json,cooled_idx.json,submitted_idx.json,inzone_pool_opencode.json,oci_cases_cache.json,gradeable_universe.json,screen_candidates.json}.
Quantify and report (cite the numbers):
1. Per log file: accepts, and the full reject-reason distribution. Confirm/correct: baseline=104, current~2, batch_filled dominant historically.
2. In-zone yield over time: distribution of "+N/24 in-zone groups". How many windows produce 0 groups? Mean groups/window per run.
3. Scatter forensics from the screen lines: allcorr vs allwrong vs promising ratios. Is the live checkpoint too good (allcorr) or are cases wrong (allwrong)? Pull avg_n_correct / avg_completions trends.
4. Pool stats: size of inzone_pool_opencode.json, gradeable_universe, how many idxs are in cooled_idx.json (dead forever) vs pool size — is the pool being eaten by cooldown? size of divergent_idx (case divergence rate).
5. The CRUCIAL diff: what about the baseline run made 104 accepts work that the current run lacks? (config in run_miner.sh history/backups, pool freshness, checkpoint number, window timing, predictive fire). Inspect ${REPO}/backups/run_miner.sh.* and ${REPO}/opencode/run_miner.sh.bak-* if present.
6. Seal-race timing: from FIRE/verdict lines, distribution of 'over=' values and t_since_window_seen; are we arriving late (over>0 => batch_filled)?
Return the structured object. mechanism_facts = quantified findings; accept_levers = what the data says to change.`,
    { label: 'forensics:logs-data', phase: 'Understand', schema: READER_SCHEMA }),
])

const [valR, minR, rewR, logR] = readers
const readersBlob = JSON.stringify({ validator: valR, miner: minR, reward_cases: rewR, forensics: logR }, null, 1)

phase('Diagnose')
const diagnosis = await agent(`You are the diagnostician. Below are 4 structured reports (validator path, miner path, opencode reward/cases, empirical forensics) on why SN81 OCI mining gets FEW accepted submits.
${GROUNDING}

REPORTS:
${readersBlob}

Produce a single coherent root-cause diagnosis. Trace the funnel prompt -> 8 rollouts -> in-zone group -> seal slot -> ACCEPTED and mark where the biggest drops are. Reconcile the historical batch_filled dominance with the current allcorr/supply starvation. Decide the PRIMARY bottleneck NOW and rank root causes. Explain the baseline(104) vs now(2) gap concretely. Be quantitative.`,
  { label: 'diagnose', phase: 'Diagnose', schema: DIAGNOSIS_SCHEMA })

const diagBlob = JSON.stringify(diagnosis, null, 1)

phase('Design')
const ANGLES = [
  { key: 'supply', brief: 'SUPPLY-MAXIMIZER: raise in-zone groups/window. The live checkpoint solves pooled prompts allcorr. Target prompts at intermediate difficulty FOR THE LIVE CHECKPOINT (screen bands, k tuning toward the scarce wrong-side, oversample, allcorr-burn aggressiveness, online difficulty targeting). Make scatter groups actually form.' },
  { key: 'race', brief: 'RACE-WINNER: win more seal slots / cut batch_filled. Predictive fire timing (LEAD_MS/POST_MS/MIN_WINDOWS), staging more ready groups before window-open, exploiting the canonical seal ordering (sha256 rank) so our groups are the ones that win the 8 slots.' },
  { key: 'divergence', brief: 'DIVERGENCE-ELIMINATOR: cut out_of_zone/reward_mismatch from reconstructed-case vs hidden-case mismatch. Better/verified case reconstruction, sigma margin (k=4 center), divergence-burn, choosing prompts whose reconstructed cases provably match.' },
  { key: 'pool-frontier', brief: 'POOL+FRONTIER: keep the pool intermediate as the checkpoint advances ~hourly without a GPU treadmill. Frontier online-learning tuning, canon-filter KEEP_FRAC, growing the gradeable universe, decool-snipe, refresh cadence.' },
  { key: 'baseline-restore', brief: 'BASELINE-RESTORE: the baseline run got 104 accepts. Identify the exact config/state that made it work (from backups + forensics) and restore/adapt it to the current checkpoint. Prefer a proven config over novel theory.' },
  { key: 'contrarian', brief: 'CONTRARIAN/ARBITRAGE: rethink the game. e.g. exploit cooldown/canonical-ordering mechanics, target a different prompt sub-distribution where the model is weaker, run multiple staged groups, or change WHAT we submit. Find an edge others miss. Must stay within the validator rules (no cheating the reward).' },
]
const strategies = await parallel(ANGLES.map(a => () =>
  agent(`You are a strategist for SN81 OCI mining. GOAL: maximize ACCEPTED submits per hour. Propose ONE concrete strategy from this angle:
ANGLE: ${a.brief}

DIAGNOSIS (root causes):
${diagBlob}

${GROUNDING}

Requirements: every concrete_change must reference a lever that ACTUALLY EXISTS (env var in run_miner.sh, a function in the reliquary code, or a build script) OR be a clearly-specified small code/config change. Be specific with values. Give a hypothesis (changing X raises accepts because Y), expected_effect (quantified vs the current ~2/run), risks, and a verifiable_metric observable in miner.log within ~1h. Do NOT propose anything that games/falsifies the validator reward. Return the structured object.`,
    { label: `design:${a.key}`, phase: 'Design', schema: STRATEGY_SCHEMA })))

const goodStrategies = strategies.filter(Boolean)

phase('Score')
const scored = await pipeline(goodStrategies,
  s => parallel([
    () => agent(`Adversarially score this SN81 OCI accept-raising strategy on FEASIBILITY given the real code.
STRATEGY: ${JSON.stringify(s)}
DIAGNOSIS: ${diagBlob}
${GROUNDING}
Verify each concrete_change actually maps to a real lever/code path — read the relevant file to confirm. If a change references something that doesn't exist or works differently, say so. Score feasibility/impact/risk, set grounded_in_code, give a sharp critique and corrections. Return the structured object.`,
      { label: `score-feas:${s.name}`, phase: 'Score', schema: SCORE_SCHEMA }),
    () => agent(`Adversarially score this SN81 OCI accept-raising strategy on EXPECTED IMPACT + RISK.
STRATEGY: ${JSON.stringify(s)}
DIAGNOSIS: ${diagBlob}
${GROUNDING}
Will this actually move accepts up given the diagnosed primary bottleneck? Could it backfire (e.g. fewer in-zone groups, more out_of_zone, lost races)? Is it attacking the real bottleneck or a secondary one? Score feasibility/impact/risk, set grounded_in_code, critique + corrections. Return the structured object.`,
      { label: `score-impact:${s.name}`, phase: 'Score', schema: SCORE_SCHEMA }),
  ]).then(votes => ({ strategy: s, scores: votes.filter(Boolean) }))
)

const scoredClean = scored.filter(Boolean).map(x => {
  const sc = x.scores
  const avg = (k) => sc.length ? sc.reduce((a, v) => a + (v[k] || 0), 0) / sc.length : 0
  return {
    name: x.strategy.name,
    strategy: x.strategy,
    avg_feasibility: avg('feasibility'),
    avg_impact: avg('impact'),
    avg_risk: avg('risk'),
    avg_overall: avg('overall'),
    all_grounded: sc.every(v => v.grounded_in_code),
    critiques: sc.map(v => v.critique),
    corrections: sc.map(v => v.corrections),
  }
}).sort((a, b) => b.avg_overall - a.avg_overall)

phase('Synthesize')
const finalRec = await agent(`You are the lead. Synthesize the single best ACTION PLAN to raise ACCEPTED submits for SN81 OCI mining, drawing from the scored strategies. Integrate complementary ideas (supply + race + divergence can compose). Be concrete and ordered by impact-per-effort.
DIAGNOSIS: ${diagBlob}
SCORED STRATEGIES (sorted by score, with critiques+corrections):
${JSON.stringify(scoredClean, null, 1)}
${GROUNDING}
Output a prose plan: (1) the ONE root cause to fix first and the exact change (env var values / file edits), (2) the next 2-3 changes in priority order, (3) what to leave alone, (4) the metric to watch in miner.log to confirm each works, (5) any quick experiment to run. Reference exact knobs and files. Apply the judges' corrections — don't repeat refuted claims.`,
  { label: 'synthesize', phase: 'Synthesize' })

return {
  diagnosis,
  ranked_strategies: scoredClean,
  recommendation: finalRec,
}
