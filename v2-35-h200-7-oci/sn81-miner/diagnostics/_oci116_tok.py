import json, re, os, statistics as st
from collections import Counter, defaultdict

TMP = os.environ.get('TEMP', os.environ.get('TMP', '.'))
d = json.load(open(os.path.join(TMP, 'm_5G3wUj.json'), encoding='utf-8-sig'))

samples = []
for w in d.get('window_detail', []) or []:
    samples.extend(w.get('samples') or [])

L = [s['completion_length'] for s in samples if s.get('completion_length')]
L.sort()
n = len(L)

def pct(p):
    return L[min(n-1, int(round((p/100.0)*(n-1))))]

print('uid 116 (5G3wUj) — completion_length (tokens), n=%d accepted samples\n' % n)
print('min=%d  p10=%d  p25=%d  p50=%d  p75=%d  p90=%d  p95=%d  p99=%d  max=%d' % (
    L[0], pct(10), pct(25), pct(50), pct(75), pct(90), pct(95), pct(99), L[-1]))
print('mean=%.0f  stdev=%.0f' % (st.mean(L), st.pstdev(L)))

bins = [(0,128),(128,192),(192,256),(256,384),(384,512),(512,768),(768,1024),(1024,1536),(1536,2048),(2048,99999)]
print('\nHISTOGRAM (token buckets):')
for lo,hi in bins:
    c = sum(1 for x in L if lo <= x < hi)
    lab = '%d-%d' % (lo, hi) if hi < 99999 else '%d+' % lo
    bar = '#' * c
    print('  %-10s %3d  %4.1f%%  %s' % (lab, c, 100*c/n, bar))

# cumulative coverage at candidate caps
print('\nCAP COVERAGE (%% of samples that fit under a given --max-new-tokens):')
for cap in [512, 768, 1024, 1280, 1536, 2048]:
    cov = sum(1 for x in L if x <= cap)
    print('  cap %4d -> %5.1f%% fit  (%d truncated)' % (cap, 100*cov/n, n-cov))

# by k
def kof(s):
    return {0.5:'k4',0.4841:'k3/5',0.433:'k2/6'}.get(round(float(s),4), str(round(float(s),4)))
byk = defaultdict(list)
for s in samples:
    if s.get('completion_length') and s.get('sigma') is not None:
        byk[kof(s['sigma'])].append(s['completion_length'])
print('\nBY SIGMA(k):')
for k,xs in sorted(byk.items(), key=lambda x:-len(x[1])):
    print('  %-6s n=%3d  median=%4d  p90=%4d  max=%4d' % (k, len(xs), int(st.median(xs)), sorted(xs)[min(len(xs)-1,int(0.9*(len(xs)-1)))], max(xs)))

# by problem type
def ptype(p):
    p=(p or '').lower()
    for kw,lab in [('linked list','list'),('binary tree','tree'),('tree','tree'),('graph','graph'),('sort','sort'),
                   ('array','array'),('list','list'),('string','string'),('matrix','matrix'),('dictionary','dict'),
                   ('recursion','recursion'),('palindrome','string'),('factorial','math'),('prime','math'),
                   ('fibonacci','math'),('stack','stack'),('class','oop')]:
        if kw in p: return lab
    return 'other'
byt = defaultdict(list)
for s in samples:
    if s.get('completion_length'): byt[ptype(s.get('prompt'))].append(s['completion_length'])
print('\nBY PROBLEM TYPE:')
for t,xs in sorted(byt.items(), key=lambda x:-len(x[1])):
    print('  %-10s n=%3d  median=%4d  max=%4d' % (t, len(xs), int(st.median(xs)), max(xs)))
