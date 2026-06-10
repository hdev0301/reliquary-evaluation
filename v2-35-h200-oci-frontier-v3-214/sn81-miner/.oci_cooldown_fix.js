export const meta = {
  name: 'oci-cooldown-fix',
  description: 'Decide the proper fix for OCI prompt_in_cooldown rejects: (a) avoid low-sha256/contested prompts vs (b) fold validator cooldown set in. Read-only.',
  phases: [
    { title: 'Understand', detail: '3 parallel readers: validator cooldown+/state coverage, miner selection/canon/frontier, live empirical forensics' },
    { title: 'Decide', detail: 'synthesize root cause + pick a/b/both/other with exact change' },
    { title: 'Verify', detail: 'adversarial panel checks the recommendation' },
  ],
}

const REPO = '/root/reliquary/reliquary'
const SN = '/root/sn81-miner'
const VURL = 'http://86.38.238.30:8080'

const GROUNDING = `
SITUATION (SN81/Reliquary OCI miner, live). Since a 19:33 restart, EVERY fired group is rejected
prompt_in_cooldown (4/4, 0 accepts in ~5 min). Cooldown is PERMANENT (BATCH_PROMPT_COOLDOWN_WINDOWS=1e6
in constants.py): any prompt ever WON (entered a training batch) is dead forever, subnet-wide.
Current setup:
- Pool = BROAD gradeable universe, 24768 idxs (built by opencode/build_frontier_pool.sh: canon-filter
  keeps the LOWEST sha256(prompt_idx) KEEP_FRAC=0.7 of ~35k gradeable, "to win the validator's
  canonical seal ties"), UNION a 542-idx dense GPU-screened overlay (front-loaded).
- Miner loads cooled_idx.json (3985 idxs) = local blocklist, excludes it from selection+firing, and
  APPENDS every newly-observed prompt_in_cooldown idx. Path: ${SN}/opencode/data/cooled_idx.json.
- Frontier predictor was "PRE-TRAINED on 219 positives (live cooldown WINNERS) vs 800 negatives".
- Fire order = SUBMIT-ORDER mode=short_then_canonical (picks lowest vcost / lowest canon-rank first).
- decool-snipe is now OFF.
TWO CANDIDATE FIXES the user is choosing between:
  (a) Stop selecting low-sha256 / "winner-like" prompts that tend to already be cooled (i.e. the
      canon-filter + frontier-on-winners + canonical fire-order are STEERING us toward contested,
      already-won prompts).
  (b) Fold the validator's published cooldown set in more aggressively (pull /state cooldown_prompts
      and/or /verdicts often, exclude them from the pool/selection up front instead of discovering
      each cooled prompt by firing and eating a rejected window).
Decide which is the PROPER fix (a, b, both, or a better option), grounded in code + live data.
Read-only: do NOT edit any files. You MAY curl ${VURL}/state and ${VURL}/verdicts/<ss58>.
`

const READER_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['area', 'facts', 'why_firing_cooled', 'fix_a_assessment', 'fix_b_assessment', 'key_numbers'],
  properties: {
    area: { type: 'string' },
    facts: { type: 'array', items: { type: 'string' }, minItems: 3, description: 'precise code/data-cited facts (file:line or measured numbers)' },
    why_firing_cooled: { type: 'string', description: 'the mechanism by which the miner ends up firing already-cooled prompts' },
    fix_a_assessment: { type: 'string', description: 'would avoiding low-sha256/winner-like prompts fix it? side effects (e.g. loses seal-tie advantage)?' },
    fix_b_assessment: { type: 'string', description: 'would folding the validator cooldown set in fix it? KEY: is /state cooldown_prompts COMPLETE or PARTIAL — what % of the pool/cooled set does it cover?' },
    key_numbers: { type: 'array', items: { type: 'string' } },
  },
}

const DECISION_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['root_cause', 'recommended_fix', 'rationale', 'exact_change', 'expected_effect', 'why_not_the_other', 'risks'],
  properties: {
    root_cause: { type: 'string' },
    recommended_fix: { type: 'string', enum: ['a', 'b', 'both', 'other'] },
    rationale: { type: 'string' },
    exact_change: { type: 'string', description: 'concrete: which file/knob, what change' },
    expected_effect: { type: 'string' },
    why_not_the_other: { type: 'string' },
    risks: { type: 'array', items: { type: 'string' } },
  },
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['agrees', 'strongest_objection', 'correction', 'confidence'],
  properties: {
    agrees: { type: 'boolean' },
    strongest_objection: { type: 'string' },
    correction: { type: 'string' },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
  },
}

phase('Understand')
const readers = await parallel([
  () => agent(`Map the VALIDATOR cooldown + what /state and /verdicts actually publish, to decide an OCI prompt_in_cooldown fix.
${GROUNDING}
Read: ${REPO}/validator/cooldown.py, ${REPO}/validator/server.py (find the /state and /verdicts handlers and EXACTLY which cooldown fields they serialize), ${REPO}/validator/dedup.py, ${REPO}/validator/batch_selection.py, ${REPO}/constants.py.
Also LIVE-QUERY: curl -s ${VURL}/state and inspect the JSON keys, especially 'cooldown_prompts' (its LENGTH) and any checkpoint/window fields. (You can use bash + python/jq.)
ANSWER PRECISELY:
1. Exactly when does a prompt enter cooldown (won/selected-for-batch?), and for how long (permanent?). cite cooldown.py / batch_selection.py.
2. Does /state publish the FULL global cooldown set or only a PARTIAL/recent slice? How many idxs does it ACTUALLY return right now (measure)? This is the crux for fix (b): if /state is near-complete, (b) works; if it's a tiny partial slice, (b) can't pre-exclude most cooled prompts.
3. Does /verdicts/<hotkey> reveal additional cooled prompts (our own prompt_in_cooldown verdicts)?
4. Is there ANY endpoint/way to get the complete cooled set? If not, what's the best obtainable approximation?
Return the structured object.`,
    { label: 'read:validator-cooldown', phase: 'Understand', schema: READER_SCHEMA }),

  () => agent(`Map the MINER selection/canon-filter/frontier/fire-order to explain why it fires already-cooled prompts and which fix helps.
${GROUNDING}
Read: ${REPO}/miner/engine.py (cooled blocklist load + append + how selection/fire excludes it; the FIRE path), ${REPO}/miner/pregen.py (SUBMIT-ORDER short_then_canonical, _submit_sort_key, _canonical_key, how candidates are chosen), ${REPO}/miner/frontier.py (the pre-train on "cooldown winners" — does it bias selection TOWARD prompts that are now cooled? the decool-snipe path), ${SN}/opencode/build_frontier_pool.sh (canon-filter = lowest sha256 KEEP_FRAC).
ANSWER PRECISELY (cite file:line):
1. The chain that selects which prompt gets fired: pool -> frontier ranking -> SUBMIT-ORDER short_then_canonical. Does canonical/sha256 ordering make it prefer the MOST-CONTESTED (lowest sha256) prompts = the ones most likely already won/cooled?
2. The frontier was pre-trained on "219 positives = live cooldown winners". Does that bias it toward selecting prompts SIMILAR TO (or identical to) already-won/cooled prompts? Is that a feature-generalization (fine) or memorized-idx (bad) effect?
3. When a fire is rejected prompt_in_cooldown, does the engine reliably ADD that idx to cooled_idx.json so it's not re-fired? Does it persist? So is the problem "re-firing the same cooled prompt" or "discovering many distinct cooled prompts one-by-one"?
4. For fix (a): is there a knob to avoid low-sha256/contested prompts (e.g. invert/disable canon-filter, change SUBMIT-ORDER, not pre-train frontier on winners)? For fix (b): where would folding the /state cooldown set into selection go (engine cooled-provider? build_frontier_pool exclusion?)?
Return the structured object.`,
    { label: 'read:miner-selection', phase: 'Understand', schema: READER_SCHEMA }),

  () => agent(`Do LIVE empirical forensics to quantify the OCI prompt_in_cooldown problem and which fix wins. Be numeric.
${GROUNDING}
Use bash/python over ${SN}/logs/miner.log, ${SN}/opencode/data/{cooled_idx.json,inzone_pool_opencode.json,divergent_idx.json,submitted_idx.json}, and LIVE curl ${VURL}/state.
MEASURE and report:
1. Since ~19:33: how many distinct prompts were fired, how many distinct got prompt_in_cooldown? Are they being RE-fired (same idx repeatedly) or all-distinct (discovering new cooled idxs each time)?
2. Of the prompts being fired/rejected, what are their sha256-rank positions — are they concentrated at the LOW-sha256 end (canon-filter favored)? (compute sha256(idx) for the rejected idxs vs the pool distribution).
3. Pool overlap: |inzone_pool_opencode.json| (24768), |cooled_idx.json| (3985), and CRUCIALLY pull /state cooldown_prompts: how many idxs, and |pool ∩ /state_cooldown|? What fraction of the broad pool is ALREADY known-cooled via /state but NOT yet in cooled_idx.json? (This sizes how much fix (b) would pre-exclude.)
4. Compare to BEFORE the broad pool: did the dense 551-idx pool have fewer cooldown rejects? (grep older log segments). Was the broad pool the regression?
5. Estimate: if we excluded /state cooldown_prompts from the pool (fix b), how many of the recent rejects would have been prevented? If we dropped the canon-filter low-sha256 bias (fix a), would the fired prompts have been less-cooled?
Return the structured object with key_numbers populated heavily.`,
    { label: 'forensics:cooldown', phase: 'Understand', schema: READER_SCHEMA }),
])

const rb = JSON.stringify({ validator: readers[0], miner: readers[1], forensics: readers[2] }, null, 1)

phase('Decide')
const decision = await agent(`You are the decision-maker. Given these 3 read-only reports on the OCI prompt_in_cooldown problem, decide the PROPER fix the user is choosing between: (a) avoid low-sha256/winner-like prompts, (b) fold the validator cooldown set in more aggressively — or BOTH, or a better OTHER option.
${GROUNDING}
REPORTS:
${rb}
Decide based on the EVIDENCE — especially: (i) is /state's cooldown set complete enough for (b) to actually pre-exclude the cooled prompts, or is it too partial? (ii) is the canon-filter/winner-bias genuinely steering toward cooled prompts (making (a) necessary), or incidental? (iii) is the real problem the BROAD pool itself (full of historically-won prompts) such that the proper fix differs from both? Give the exact change and why the other option is insufficient.`,
  { label: 'decide', phase: 'Decide', schema: DECISION_SCHEMA })

phase('Verify')
const verdicts = await parallel([1, 2, 3].map(i => () =>
  agent(`Adversarially verify this recommended fix for OCI prompt_in_cooldown rejects. Try to REFUTE it. Lens ${i}: ${['does /state actually cover enough for fix b? is the coverage claim verified against live data?', 'does the recommended change have a side effect — e.g. losing seal-tie wins, shrinking the pool back toward starvation, or fighting the frontier?', 'is there a simpler/more-correct fix being missed (e.g. the broad pool itself, or the fire-order, or just letting cooled_idx.json accrue)?'][i-1]}
RECOMMENDATION: ${JSON.stringify(decision)}
${GROUNDING}
Verify against the code/live data if needed (read-only, may curl ${VURL}/state). Return whether you agree, the strongest objection, a correction, and your confidence.`,
    { label: `verify:${i}`, phase: 'Verify', schema: VERDICT_SCHEMA })))

return { decision, verdicts: verdicts.filter(Boolean), reports: { validator: readers[0], miner: readers[1], forensics: readers[2] } }
