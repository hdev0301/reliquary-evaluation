"""Deep single-miner forensics for SN81. Reads one /api/miner/<hk> JSON and emits:
  - console: miner-level fields, per-window trajectory, EXACT sigma histogram (curation
    proof), format/topic/length detail, reject forensics, timing, uniqueness, eos rate.
  - artifacts (for qualitative LLM reading): full samples dump, distinct-prompt list,
    per-window JSON table.
"""
import json, re, os, sys, statistics as st
from collections import Counter, defaultdict

DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get('TEMP', '.'), 'sn81deep')
FN = sys.argv[2] if len(sys.argv) > 2 else '5Hp6EPJd.json'
LBL = re.sub(r'^[A-Z]_', '', os.path.splitext(os.path.basename(FN))[0])
SRC = os.path.join(DIR, FN)
d = json.load(open(SRC, encoding='utf-8-sig'))
m = d.get('miner') or {}
wd = d.get('window_detail') or []
hist = d.get('history') or []
timing = d.get('timing') or {}

HEX16 = re.compile(r'^[0-9a-f]{16}$')
PLAINNUM = re.compile(r'^[\-\+]?\d+(\.\d+)?$')

def f2(x):
    try: return float(x)
    except (TypeError, ValueError): return None
def i2(x):
    try: return int(float(x))
    except (TypeError, ValueError): return None
def numeric(a): return bool(PLAINNUM.fullmatch(str(a).strip()))

def feats(a):
    a = str(a); core = re.sub(r'\\(text|frac|sqrt|pi|circ|begin|end|pmatrix|bmatrix|cdot|times|left|right|sin|cos|tan|log|ln|theta|alpha|beta)', '', a)
    return {'plain_int': bool(re.fullmatch(r'-?\d+', a.strip())), 'decimal': bool(re.fullmatch(r'-?\d+\.\d+', a.strip())),
            'fraction': ('\\frac' in a) or bool(re.search(r'\b\d+/\d+\b', a)), 'radical': '\\sqrt' in a, 'pi': '\\pi' in a,
            'tuple': (',' in a) or bool(re.search(r'[()\[\]]', a)), 'matrix': 'matrix' in a,
            'has_var': bool(re.search(r'[a-zA-Z]', core))}
FKEYS = ['plain_int','decimal','fraction','radical','pi','tuple','matrix','has_var']
def topic(p):
    p=(p or '').lower()
    if any(w in p for w in ['sin','cos','tan','angle','triangle']): return 'trig'
    if any(w in p for w in ['circle','area','perimeter','coordinate','radius','volume']): return 'geom'
    if any(w in p for w in ['probability','percent','%']): return 'prob/pct'
    if '\\sqrt' in p or 'square root' in p: return 'radical'
    if any(w in p for w in ['matrix','vector','determinant']): return 'matrix'
    if any(w in p for w in ['polynomial','factor','roots','quadratic']): return 'poly'
    if any(w in p for w in ['sequence','series','geometric','arithmetic seq']): return 'seq'
    if any(w in p for w in ['log','exponent']): return 'logexp'
    if any(w in p for w in ['$','cost','buy','bought','spent','how many','total','each']): return 'wordprob'
    return 'other'

print('='*96)
print('DEEP FORENSICS — %s   (src=%s, generated_at=%s)' % (LBL, SRC, d.get('generated_at')))
print('='*96)
print('\n[MINER-LEVEL]')
for k in ['hotkey','coldkey','uid','window','score','rank','rollout_count','valid_rollouts','unique_rollouts',
          'unique_ratio','avg_reward','success_rate','hard_failed','soft_failed','stability','streak',
          'participation','trend','trend_slope','share_of_emission','estimated_daily_tao','cumulative_tao',
          'upload_lag_ms','upload_lag_p50_ms','upload_lag_p95_ms','response_time_ms','response_time_p50_ms',
          'response_time_p95_ms','last_seen','status']:
    if k in m: print('  %-22s %s' % (k, m[k]))
if timing: print('\n[TIMING]'); [print('  %-22s %s' % (k, v)) for k, v in timing.items()]

# ---- per-window trajectory ----
print('\n[PER-WINDOW TRAJECTORY] (%d windows)' % len(wd))
print('  %-9s %-8s %-5s %-5s %-5s %-5s %s' % ('window','score','sub','acc','soft','hard','top_reject'))
agg_rej = Counter(); sig_exact = Counter(); allrew = []; alllen = []; eos = Counter(); idxs = []
fmt = Counter(); top = Counter(); nmath = nsym = 0; nsamp = 0
samples_for_dump = []; prompts_seen = {}
for w in sorted(wd, key=lambda x: i2(x.get('window')) or 0):
    rr = w.get('miner_reject_reasons') or w.get('window_reject_summary') or {}
    if isinstance(rr, dict):
        for k, v in rr.items(): agg_rej[k] += i2(v) or 0
    topr = ''
    if isinstance(rr, dict) and rr:
        topr = ' '.join('%s=%s' % (k, v) for k, v in sorted(rr.items(), key=lambda x: -(i2(x[1]) or 0))[:3])
    print('  %-9s %-8s %-5s %-5s %-5s %-5s %s' % (
        w.get('window'), w.get('score'), w.get('submitted'), w.get('accepted'),
        w.get('soft_failed'), w.get('hard_failed'), topr))
    for s in (w.get('samples') or []):
        gt = s.get('ground_truth'); nsamp += 1
        sg = f2(s.get('sigma'))
        if sg is not None: sig_exact[repr(round(sg, 10))] += 1
        rw = f2(s.get('reward'));
        if rw is not None: allrew.append(rw)
        ln = i2(s.get('completion_length'))
        if ln: alllen.append(ln)
        eos[str(s.get('eos_terminated'))] += 1
        ix = i2(s.get('prompt_idx'))
        if ix is not None: idxs.append(ix)
        if numeric(gt): nmath += 1
        else: nsym += 1
        for k, v in feats(gt).items():
            if v: fmt[k] += 1
        top[topic(s.get('prompt'))] += 1
        samples_for_dump.append({'window': w.get('window'), 'prompt_idx': s.get('prompt_idx'),
            'ground_truth': gt, 'reward': s.get('reward'), 'sigma': s.get('sigma'),
            'len': ln, 'eos': s.get('eos_terminated'),
            'prompt': s.get('prompt'), 'completion': s.get('completion_text')})
        if s.get('prompt'): prompts_seen[ix] = s.get('prompt')

print('\n[CURATION SIGNATURE] exact sigma values across %d samples:' % nsamp)
KMAP = {round((2*6)**0.5/8*1,10): 'k2or6', }
def sigma_label(v):
    # sigma of a population of k ones and (8-k) zeros, mean=k/8: sqrt(k*(8-k))/8
    for k in range(0, 9):
        s = (k*(8-k))**0.5/8
        if abs(v - s) < 1e-6: return 'k=%d|%d (%.4f)' % (k, 8-k, s)
    return 'NONCANON(%.5f)' % v
for sv, c in sorted(sig_exact.items(), key=lambda x: -x[1]):
    fv = float(sv)
    print('  %-22s n=%-3d  -> %s' % (sv, c, sigma_label(fv)))

print('\n[REWARD]  mean=%.4f  min=%.3f  max=%.3f  distinct=%s' % (
    (sum(allrew)/len(allrew)) if allrew else 0, min(allrew) if allrew else 0,
    max(allrew) if allrew else 0, sorted(set(round(r,3) for r in allrew))))
print('[EOS]     ' + '  '.join('%s=%d' % (k, v) for k, v in eos.most_common()))
if alllen:
    s = sorted(alllen)
    def q(p): return s[min(len(s)-1, int(p*len(s)))]
    print('[LENGTH]  min=%d p10=%d p25=%d med=%d p75=%d p90=%d max=%d mean=%d' % (
        s[0], q(.1), q(.25), int(st.median(s)), q(.75), q(.9), s[-1], int(st.mean(s))))
if idxs:
    si = sorted(idxs)
    print('[PROMPT_IDX] n=%d distinct=%d  min=%d max=%d  (OMI ~880k/2-shard => max/880k=%.2f)' % (
        len(idxs), len(set(idxs)), si[0], si[-1], si[-1]/880000.0))
    buckets = Counter(ix//100000 for ix in idxs)
    print('  idx histogram (per 100k): ' + '  '.join('%d00k:%d' % (b, n) for b, n in sorted(buckets.items())))
print('\n[FORMAT %%] ' + '  '.join('%s=%d' % (k, round(100*fmt[k]/nsamp)) for k in FKEYS if fmt[k]))
print('[NUM/SYM] %d/%d' % (round(100*nmath/nsamp), round(100*nsym/nsamp)))
print('[TOPIC %%] ' + '  '.join('%s=%d' % (k, round(100*v/nsamp)) for k, v in top.most_common()))
print('\n[REJECT REASONS aggregated] ' + '  '.join('%s=%d' % (k, v) for k, v in agg_rej.most_common()))

# ---- artifacts ----
# full samples dump (sorted longest-completion first to inspect the curatable long tail + a few short)
samples_for_dump.sort(key=lambda x: -(x['len'] or 0))
pick = samples_for_dump[:8] + samples_for_dump[-4:]
with open(os.path.join(DIR, '%s_samples.txt' % LBL), 'w', encoding='utf-8') as f:
    for i, s in enumerate(pick):
        f.write('\n' + '='*90 + '\n')
        f.write('SAMPLE %d | window=%s prompt_idx=%s gt=%r reward=%s sigma=%s len=%s eos=%s\n' % (
            i, s['window'], s['prompt_idx'], s['ground_truth'], s['reward'], s['sigma'], s['len'], s['eos']))
        f.write('--- PROMPT ---\n%s\n' % (s['prompt'] or ''))
        f.write('--- COMPLETION (first 2500 chars) ---\n%s\n' % ((s['completion'] or '')[:2500]))
with open(os.path.join(DIR, '%s_prompts.txt' % LBL), 'w', encoding='utf-8') as f:
    for ix, p in sorted(prompts_seen.items(), key=lambda x: x[0] or 0):
        f.write('[idx=%s] %s\n' % (ix, (p or '').replace('\n', ' ')[:300]))
with open(os.path.join(DIR, '%s_windows.json' % LBL), 'w', encoding='utf-8') as f:
    json.dump([{k: w.get(k) for k in ['window','created_at','score','submitted','accepted','soft_failed',
        'hard_failed','response_time_ms','miner_reject_reasons','window_reject_summary']} for w in wd], f, indent=1)
print('\n[ARTIFACTS] wrote %s_samples.txt (%d samples), %s_prompts.txt (%d prompts), %s_windows.json' % (
    LBL, len(pick), LBL, len(prompts_seen), LBL))
print('  DIR=%s' % DIR)
