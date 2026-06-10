"""Profile data-source distribution for a set of SN81 miners.
Reads single-miner JSON dumps (/api/miner/<hotkey>) and reports the env split
(math vs opencode), math answer-format / k / length distributions, code stats,
and a pool inference per miner. Classification matches _dist3.py / _mathdist.py.
"""
import json, re, os, sys, statistics as st
from collections import Counter

import glob
DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get('TEMP', '.'), 'sn81prof')
# auto-discover all *.json in DIR; label = filename stem with any leading "X_" prefix stripped
FILES = []
for p in sorted(glob.glob(os.path.join(DIR, '*.json'))):
    stem = os.path.splitext(os.path.basename(p))[0]
    lbl = stem.split('_', 1)[1] if re.match(r'^[A-Z]_', stem) else stem
    FILES.append((lbl, os.path.basename(p)))

HEX16    = re.compile(r'^[0-9a-f]{16}$')
PLAINNUM = re.compile(r'^[\-\+]?\d+(\.\d+)?$')
BS = chr(92)

def f2(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def i2(x):
    try: return int(float(x))
    except (TypeError, ValueError): return None

def env_of(gt):
    return 'code' if HEX16.match((gt or '').strip()) else 'math'

def numeric(a):
    return bool(PLAINNUM.fullmatch(str(a).strip()))

def feats(a):
    a = str(a)
    core = re.sub(r'\\(text|frac|sqrt|pi|circ|begin|end|pmatrix|bmatrix|cdot|times|left|right|sin|cos|tan|log|ln|theta|alpha|beta)', '', a)
    return {
        'plain_int': bool(re.fullmatch(r'-?\d+', a.strip())),
        'decimal':   bool(re.fullmatch(r'-?\d+\.\d+', a.strip())),
        'fraction':  ('\\frac' in a) or bool(re.search(r'\b\d+/\d+\b', a)),
        'radical':   '\\sqrt' in a,
        'pi':        '\\pi' in a,
        'tuple':     (',' in a) or bool(re.search(r'[()\[\]]', a)),
        'matrix':    'matrix' in a,
        'has_var':   bool(re.search(r'[a-zA-Z]', core)),
    }
FKEYS = ['plain_int', 'decimal', 'fraction', 'radical', 'pi', 'tuple', 'matrix', 'has_var']

def topic(p):
    p = (p or '').lower()
    if any(w in p for w in ['sin', 'cos', 'tan', 'angle', 'triangle']): return 'trig'
    if any(w in p for w in ['circle', 'area', 'perimeter', 'coordinate', 'radius', 'volume']): return 'geom'
    if any(w in p for w in ['probability', 'percent', '%']): return 'prob/pct'
    if '\\sqrt' in p or 'square root' in p: return 'radical'
    if any(w in p for w in ['matrix', 'vector', 'determinant']): return 'matrix'
    if any(w in p for w in ['polynomial', 'factor', 'roots', 'quadratic']): return 'poly'
    if any(w in p for w in ['sequence', 'series', 'geometric', 'arithmetic seq']): return 'seq'
    if any(w in p for w in ['log', 'exponent']): return 'logexp'
    if any(w in p for w in ['$', 'cost', 'buy', 'bought', 'spent', 'how many', 'total', 'each']): return 'wordprob'
    return 'other'

def kof(s):
    s = f2(s)
    if s is None: return '?'
    return {0.5: 'k4', 0.4841: 'k3/5', 0.433: 'k2/6'}.get(round(s, 4), '%.3f' % round(s, 4))

def pct(c, n):
    return '  '.join('%s=%d%%' % (k, round(100 * v / n)) for k, v in c.most_common())

def quant(vals):
    if not vals: return (0, 0, 0, 0)
    s = sorted(vals)
    def q(p): return s[min(len(s) - 1, int(p * len(s)))]
    return (q(0.10), int(st.median(s)), q(0.90), int(st.mean(s)))

def infer_pool(math_n, code_n, num_pct, ks, med_len, fmt):
    parts = []
    tot = math_n + code_n
    if tot == 0: return 'no accepted samples'
    if code_n > math_n:
        parts.append('CODING / OpenCodeInstruct dominant')
        kc = ks.most_common(1)
        if kc and kc[0][0] == 'k4': parts.append('k=4 binary curation (zone-center, ~OCI #1 strategy)')
        elif kc and kc[0][0] == 'k2/6': parts.append('k=6 binary curation')
        else: parts.append('honest/continuous (partial-pass)')
    else:
        if num_pct >= 80:
            parts.append('openmath NUMERIC (gsm8k plain int/decimal)')
        elif num_pct <= 45:
            parts.append('openmath SYMBOLIC (math/augmented frac|radical|var)')
        else:
            parts.append('openmath NUMERIC+SYMBOLIC blend (~%d/%d)' % (num_pct, 100 - num_pct))
        kc = ks.most_common(1)
        if kc:
            parts.append('%s curation' % kc[0][0])
    return ' | '.join(parts)

print('=' * 92)
print('SN81 DATA-SOURCE PROFILE  (dir=%s)' % DIR)
print('=' * 92)

summary = []
for label, fn in FILES:
    path = os.path.join(DIR, fn)
    try:
        d = json.load(open(path, encoding='utf-8-sig'))
    except Exception as e:
        print('\n### %s: load error %s' % (label, e)); continue
    m = d.get('miner') or {}
    wd = d.get('window_detail') or []

    senv = Counter()
    msrc = {'n': 0, 'num': 0, 'f': Counter(), 't': Counter(), 'k': Counter(), 'len': [], 'rew': [], 'idx': set()}
    csrc = {'n': 0, 'k': Counter(), 'len': [], 'rew': [], 'idx': set(), 'sample_prompt': None}
    sub_tot = acc_tot = 0
    rejects = Counter()

    for w in wd:
        sub_tot += i2(w.get('submitted')) or 0
        acc_tot += i2(w.get('accepted')) or 0
        rr = w.get('miner_reject_reasons') or w.get('window_reject_summary') or {}
        if isinstance(rr, dict):
            for k, v in rr.items():
                rejects[k] += i2(v) or 0
        for s in (w.get('samples') or []):
            gt = s.get('ground_truth')
            e = env_of(gt); senv[e] += 1
            ln = i2(s.get('completion_length'))
            rw = f2(s.get('reward'))
            idx = i2(s.get('prompt_idx'))
            if e == 'math':
                msrc['n'] += 1
                if numeric(gt): msrc['num'] += 1
                for k, v in feats(gt).items():
                    if v: msrc['f'][k] += 1
                msrc['t'][topic(s.get('prompt'))] += 1
                msrc['k'][kof(s.get('sigma'))] += 1
                if ln: msrc['len'].append(ln)
                if rw is not None: msrc['rew'].append(rw)
                if idx is not None: msrc['idx'].add(idx)
            else:
                csrc['n'] += 1
                csrc['k'][kof(s.get('sigma'))] += 1
                if ln: csrc['len'].append(ln)
                if rw is not None: csrc['rew'].append(rw)
                if idx is not None: csrc['idx'].add(idx)
                if csrc['sample_prompt'] is None and s.get('prompt'):
                    csrc['sample_prompt'] = s['prompt'][:160]

    n = senv['math'] + senv['code']
    print('\n' + '-' * 92)
    print('### %s   uid=%s  rank=%s  score=%s' % (label, m.get('uid'), m.get('rank'), m.get('score')))
    print('    cumTAO=%s  trend=%s  daily_tao=%s  emission_share=%s' % (
        m.get('cumulative_tao'), m.get('trend'), m.get('estimated_daily_tao'), m.get('share_of_emission')))
    print('    success_rate=%s  participation=%s  valid/unique_rollouts=%s/%s  status=%s' % (
        m.get('success_rate'), m.get('participation'), m.get('valid_rollouts'), m.get('unique_rollouts'), m.get('status')))
    print('    windows=%d  submitted=%d  accepted=%d  accepted_samples_seen=%d' % (len(wd), sub_tot, acc_tot, n))
    if rejects:
        print('    reject_reasons: ' + '  '.join('%s=%d' % (k, v) for k, v in rejects.most_common(6)))
    if not n:
        print('    NO accepted samples in window_detail');
        summary.append((label, m.get('uid'), 0, 0, 0, '?', 0, 'no samples')); continue

    me, ce = senv['math'], senv['code']
    print('    ENV SPLIT:  math=%d (%d%%)   code/opencode=%d (%d%%)' % (me, round(100*me/n), ce, round(100*ce/n)))

    num_pct = 0
    if msrc['n']:
        num_pct = round(100 * msrc['num'] / msrc['n'])
        p10, med, p90, mean = quant(msrc['len'])
        ix = sorted(msrc['idx'])
        rmean = (sum(msrc['rew']) / len(msrc['rew'])) if msrc['rew'] else 0
        print('  [MATH]  n=%d  num/sym=%d/%d  reward_mean=%.3f' % (msrc['n'], num_pct, 100 - num_pct, rmean))
        print('          len: p10=%d med=%d p90=%d mean=%d' % (p10, med, p90, mean))
        print('          distinct_prompt_idx=%d  idx_range=[%s..%s]' % (
            len(msrc['idx']), ix[0] if ix else '-', ix[-1] if ix else '-'))
        print('          fmt%%:   ' + '  '.join('%s=%d' % (k, round(100*msrc['f'][k]/msrc['n'])) for k in FKEYS if msrc['f'][k]))
        print('          k(sigma): ' + pct(msrc['k'], msrc['n']))
        print('          topic%%: ' + pct(msrc['t'], msrc['n']))
    if csrc['n']:
        p10, med, p90, mean = quant(csrc['len'])
        ix = sorted(csrc['idx'])
        rmean = (sum(csrc['rew']) / len(csrc['rew'])) if csrc['rew'] else 0
        print('  [CODE]  n=%d  reward_mean=%.3f' % (csrc['n'], rmean))
        print('          len: p10=%d med=%d p90=%d mean=%d' % (p10, med, p90, mean))
        print('          distinct_prompt_idx=%d  idx_range=[%s..%s]' % (
            len(csrc['idx']), ix[0] if ix else '-', ix[-1] if ix else '-'))
        print('          k(sigma): ' + pct(csrc['k'], csrc['n']))
        print('          sample_prompt: %s' % csrc['sample_prompt'])

    dom_k = (msrc['k'] if me >= ce else csrc['k'])
    dom_len = quant(msrc['len'] if me >= ce else csrc['len'])[1]
    pool = infer_pool(me, ce, num_pct, dom_k, dom_len, msrc['f'])
    print('  >> POOL INFERENCE: ' + pool)
    summary.append((label, m.get('uid'), me, ce, num_pct, dom_k.most_common(1)[0][0] if dom_k else '?', dom_len, pool))

print('\n' + '=' * 92)
print('CROSS-MINER SUMMARY')
print('=' * 92)
print('%-9s %-7s %-6s %-6s %-7s %-6s %-7s %s' % ('hotkey', 'uid', 'math', 'code', 'num%', 'k', 'medlen', 'pool'))
for label, uid, me, ce, num_pct, k, med, pool in summary:
    print('%-9s %-7s %-6d %-6d %-7s %-6s %-7d %s' % (
        label, uid, me, ce, ('%d' % num_pct) if (me >= ce) else '-', k, med, pool))
