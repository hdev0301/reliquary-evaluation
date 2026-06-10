"""Deep OCI (opencode) forensics for SN81. Reads one /api/miner/<hk> JSON. Unlike the
math deep dive, the load-bearing signals for code are: REWARD CONTINUITY (binary {0,1}
curation vs honest partial-pass) and SIGMA SCATTER (canonical k-zone vs continuous),
plus long-completion length, OCI prompt-idx subset, timing, rejects. Usage:
  python _deep_oci.py <dir> <basename.json>
"""
import json, re, os, sys, statistics as st
from collections import Counter

DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get('TEMP', '.'), 'sn81deep')
FN  = sys.argv[2] if len(sys.argv) > 2 else '5ECEJH9M.json'
LBL = os.path.splitext(FN)[0]
d = json.load(open(os.path.join(DIR, FN), encoding='utf-8-sig'))
m = d.get('miner') or {}
wd = d.get('window_detail') or []
HEX16 = re.compile(r'^[0-9a-f]{16}$')
def f2(x):
    try: return float(x)
    except (TypeError, ValueError): return None
def i2(x):
    try: return int(float(x))
    except (TypeError, ValueError): return None
def sigma_label(v):
    for k in range(0, 9):
        s = (k*(8-k))**0.5/8
        if abs(v - s) < 1e-6: return 'k=%d|%d' % (k, 8-k)
    return 'NONCANON'

print('='*96)
print('DEEP OCI FORENSICS — %s  (generated_at=%s)' % (LBL, d.get('generated_at')))
print('='*96)
print('\n[MINER-LEVEL]')
for k in ['uid','window','score','rank','rollout_count','valid_rollouts','unique_rollouts','unique_ratio',
          'avg_reward','success_rate','hard_failed','soft_failed','stability','streak','participation','trend',
          'trend_slope','share_of_emission','cumulative_tao','estimated_daily_tao','upload_lag_ms',
          'upload_lag_p50_ms','upload_lag_p95_ms','response_time_ms','response_time_p95_ms','last_seen','status']:
    if k in m: print('  %-22s %s' % (k, m[k]))

agg_rej = Counter(); sig_exact = Counter(); rew_hist = Counter(); alllen = []; eos = Counter()
idxs = []; nsamp = 0; ncode = nmath = 0; canon = noncanon = 0
samples = []; prompts_seen = {}
print('\n[PER-WINDOW TRAJECTORY] (%d windows)' % len(wd))
print('  %-9s %-8s %-5s %-5s %-5s %s' % ('window','score','sub','acc','hard','top_reject'))
for w in sorted(wd, key=lambda x: i2(x.get('window')) or 0):
    rr = w.get('miner_reject_reasons') or w.get('window_reject_summary') or {}
    if isinstance(rr, dict):
        for k, v in rr.items(): agg_rej[k] += i2(v) or 0
    topr = ' '.join('%s=%s' % (k, v) for k, v in sorted(rr.items(), key=lambda x: -(i2(x[1]) or 0))[:3]) if isinstance(rr, dict) and rr else ''
    print('  %-9s %-8s %-5s %-5s %-5s %s' % (w.get('window'), w.get('score'), w.get('submitted'),
        w.get('accepted'), w.get('hard_failed'), topr))
    for s in (w.get('samples') or []):
        gt = str(s.get('ground_truth') or '').strip(); nsamp += 1
        if HEX16.match(gt): ncode += 1
        else: nmath += 1
        sg = f2(s.get('sigma'))
        if sg is not None:
            sig_exact[round(sg, 6)] += 1
            (canon and 0)  # noop
            if sigma_label(sg) == 'NONCANON': noncanon += 1
            else: canon += 1
        rw = f2(s.get('reward'))
        if rw is not None: rew_hist[round(rw, 4)] += 1
        ln = i2(s.get('completion_length'))
        if ln: alllen.append(ln)
        eos[str(s.get('eos_terminated'))] += 1
        ix = i2(s.get('prompt_idx'))
        if ix is not None: idxs.append(ix)
        samples.append({'window': w.get('window'), 'prompt_idx': s.get('prompt_idx'), 'gt': gt,
            'reward': s.get('reward'), 'sigma': s.get('sigma'), 'len': ln, 'eos': s.get('eos_terminated'),
            'prompt': s.get('prompt'), 'completion': s.get('completion_text')})
        if s.get('prompt'): prompts_seen[ix] = s.get('prompt')

print('\n[ENV] code(hex16)=%d  math=%d  (of %d)' % (ncode, nmath, nsamp))
print('\n[REWARD CONTINUITY]  (binary {0,1} => curation; spread => honest partial-pass)')
print('  distinct reward values=%d  mean=%.4f' % (len(rew_hist), (sum(r*c for r,c in rew_hist.items())/sum(rew_hist.values())) if rew_hist else 0))
for r, c in sorted(rew_hist.items(), key=lambda x: -x[1])[:14]:
    print('    reward=%-7s n=%-3d %s' % (r, c, '#'*min(40, c)))
print('\n[SIGMA SCATTER]  canonical(k-zone)=%d  NONCANON(continuous)=%d  => %d%% non-canonical' % (
    canon, noncanon, round(100*noncanon/max(1, canon+noncanon))))
for sv, c in sorted(sig_exact.items(), key=lambda x: -x[1])[:14]:
    print('    sigma=%-9s n=%-3d -> %s' % (sv, c, sigma_label(sv)))
print('  ... %d distinct sigma values total' % len(sig_exact))

print('\n[REWARD x EOS] eos: ' + '  '.join('%s=%d' % (k, v) for k, v in eos.most_common()))
if alllen:
    s = sorted(alllen)
    def q(p): return s[min(len(s)-1, int(p*len(s)))]
    print('[LENGTH]  min=%d p10=%d p25=%d med=%d p75=%d p90=%d p95=%d max=%d mean=%d' % (
        s[0], q(.1), q(.25), int(st.median(s)), q(.75), q(.9), q(.95), s[-1], int(st.mean(s))))
if idxs:
    si = sorted(idxs)
    print('[PROMPT_IDX] n=%d distinct=%d  min=%d max=%d  (OCI subset; max=>%dk rows)' % (
        len(idxs), len(set(idxs)), si[0], si[-1], si[-1]//1000))
    buckets = Counter(ix//10000 for ix in idxs)
    print('  idx histogram (per 10k): ' + '  '.join('%d0k:%d' % (b, n) for b, n in sorted(buckets.items())))
print('\n[REJECT REASONS aggregated] ' + '  '.join('%s=%d' % (k, v) for k, v in agg_rej.most_common()))

samples.sort(key=lambda x: -(x['len'] or 0))
pick = samples[:6] + samples[len(samples)//2:len(samples)//2+2] + samples[-3:]
with open(os.path.join(DIR, '%s_samples.txt' % LBL), 'w', encoding='utf-8') as f:
    for i, s in enumerate(pick):
        f.write('\n' + '='*90 + '\nSAMPLE %d | window=%s idx=%s reward=%s sigma=%s len=%s eos=%s\n' % (
            i, s['window'], s['prompt_idx'], s['reward'], s['sigma'], s['len'], s['eos']))
        f.write('--- PROMPT ---\n%s\n' % ((s['prompt'] or '')[:1400]))
        f.write('--- COMPLETION (first 2200 chars) ---\n%s\n' % ((s['completion'] or '')[:2200]))
with open(os.path.join(DIR, '%s_prompts.txt' % LBL), 'w', encoding='utf-8') as f:
    for ix, p in sorted(prompts_seen.items(), key=lambda x: x[0] or 0):
        f.write('[idx=%s] %s\n' % (ix, (p or '').replace('\n', ' ')[:280]))
print('\n[ARTIFACTS] wrote %s_samples.txt (%d), %s_prompts.txt (%d)  DIR=%s' % (
    LBL, len(pick), LBL, len(prompts_seen), DIR))
