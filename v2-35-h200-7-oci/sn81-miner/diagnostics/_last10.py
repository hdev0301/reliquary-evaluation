"""Data-source distribution over ONLY THE LAST N WINDOWS per miner (recency view).
Auto-discovers *.json in a dir (single-miner /api/miner dumps). For each, takes the
N most-recent windows (by window number) and aggregates their accepted samples.
Usage: python _last10.py <dir> [N]
"""
import json, re, os, sys, glob, statistics as st
from collections import Counter

DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get('TEMP', '.'), 'sn81last10')
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10

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
            'tuple': (',' in a) or bool(re.search(r'[()\[\]]', a)), 'has_var': bool(re.search(r'[a-zA-Z]', core))}
FKEYS = ['plain_int','decimal','fraction','radical','pi','tuple','has_var']
def kof(s):
    s = f2(s)
    if s is None: return '?'
    for k in range(9):
        if abs(s - (k*(8-k))**0.5/8) < 1e-6: return {0:'k0/8',1:'k1/7',2:'k2|6',3:'k3|5',4:'k4'}.get(min(k,8-k),'k%d'%k)
    return 'cont(%.3f)' % s

print('=' * 100)
print('LAST %d WINDOWS — data-source distribution per miner   (dir=%s)' % (N, DIR))
print('=' * 100)
rows = []
for path in sorted(glob.glob(os.path.join(DIR, '*.json'))):
    stem = os.path.splitext(os.path.basename(path))[0]
    lbl = stem.split('_', 1)[1] if re.match(r'^[A-Z]_', stem) else stem
    try:
        d = json.load(open(path, encoding='utf-8-sig'))
    except Exception as e:
        print('\n%s: load err %s' % (lbl, e)); continue
    m = d.get('miner') or {}
    wd = d.get('window_detail') or []
    wd = sorted(wd, key=lambda w: i2(w.get('window')) or 0)
    last = wd[-N:]
    wins = [i2(w.get('window')) for w in last if i2(w.get('window')) is not None]
    senv = Counter(); num = 0; fc = Counter(); kk = Counter(); lens = []; n = 0
    sub = acc = 0; rej = Counter()
    for w in last:
        sub += i2(w.get('submitted')) or 0; acc += i2(w.get('accepted')) or 0
        rr = w.get('miner_reject_reasons') or {}
        if isinstance(rr, dict):
            for k, v in rr.items(): rej[k] += i2(v) or 0
        for s in (w.get('samples') or []):
            gt = s.get('ground_truth'); e = 'code' if HEX16.match((gt or '').strip()) else 'math'
            senv[e] += 1; n += 1
            if e == 'math':
                if numeric(gt): num += 1
                for k, v in feats(gt).items():
                    if v: fc[k] += 1
            kk[kof(s.get('sigma'))] += 1
            ln = i2(s.get('completion_length'))
            if ln: lens.append(ln)
    print('\n' + '-' * 100)
    print('### %-9s uid=%s rank=%s score=%s cumTAO=%s trend=%s  | windows %s..%s (%d shown)' % (
        lbl, m.get('uid'), m.get('rank'), m.get('score'), m.get('cumulative_tao'), m.get('trend'),
        wins[0] if wins else '-', wins[-1] if wins else '-', len(last)))
    if not n:
        print('    no accepted samples in last %d windows' % N); rows.append((lbl, '-', 0, 0, '-', '-', 0)); continue
    me, ce = senv['math'], senv['code']
    env = 'MATH' if me >= ce else 'CODE'
    numpct = round(100*num/me) if me else 0
    med = int(st.median(lens)) if lens else 0
    p90 = sorted(lens)[min(len(lens)-1, int(0.9*len(lens)))] if lens else 0
    print('    submitted=%d accepted=%d  samples=%d  ENV: math=%d code=%d -> %s' % (sub, acc, n, me, ce, env))
    if me:
        print('    num/sym=%d/%d   med_len=%d p90=%d' % (numpct, 100-numpct, med, p90))
        print('    fmt%%:  ' + '  '.join('%s=%d' % (k, round(100*fc[k]/me)) for k in FKEYS if fc[k]))
    print('    k(sigma): ' + '  '.join('%s=%d%%' % (k, round(100*v/n)) for k, v in kk.most_common(6)))
    if rej: print('    rejects: ' + '  '.join('%s=%d' % (k, v) for k, v in rej.most_common(4)))
    rows.append((lbl, env, me, ce, ('%d/%d'%(numpct,100-numpct)) if env=='MATH' else '-',
                 kk.most_common(1)[0][0] if kk else '-', med))

print('\n' + '=' * 100)
print('SUMMARY (last %d windows)' % N)
print('%-9s %-5s %-5s %-5s %-9s %-7s %s' % ('miner','env','math','code','num/sym','k','medlen'))
for lbl, env, me, ce, ns, k, med in rows:
    print('%-9s %-5s %-5d %-5d %-9s %-7s %d' % (lbl, env, me, ce, ns, k, med))
