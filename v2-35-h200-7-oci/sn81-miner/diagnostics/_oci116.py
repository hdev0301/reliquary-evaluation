import json, re, os, statistics as st
from collections import Counter

TMP = os.environ.get('TEMP', os.environ.get('TMP', '.'))
d = json.load(open(os.path.join(TMP, 'm_5G3wUj.json'), encoding='utf-8-sig'))

def pct(xs, p):
    xs = sorted(xs);
    if not xs: return 0
    i = min(len(xs)-1, int(round((p/100.0)*(len(xs)-1))))
    return xs[i]

HEX16 = re.compile(r'^[0-9a-f]{16}$')

samples = []
per_window = []
for w in d.get('window_detail', []) or []:
    ss = w.get('samples') or []
    per_window.append((w.get('window'), w.get('submitted'), w.get('accepted'), w.get('soft_failed'), w.get('hard_failed'), len(ss)))
    samples.extend(ss)

n = len(samples)
print('TOTAL accepted samples in window_detail: %d  (windows=%d)' % (n, len(per_window)))

# reward / sigma binarity
rewards = [s.get('reward') for s in samples if s.get('reward') is not None]
sigmas = [round(float(s['sigma']), 4) for s in samples if s.get('sigma') is not None]
binary_reward = all(r in (0, 1, 0.0, 1.0) for r in rewards)
print('\n[REWARD] distinct values=%s  all-binary(0/1)=%s' % (sorted(set(rewards))[:8], binary_reward))
def kof(s):
    return {0.5:'k4', 0.4841:'k3/5', 0.433:'k2/6'}.get(s, str(s))
print('[SIGMA]  value counts=%s' % dict(Counter(kof(x) for x in sigmas).most_common()))
nonbinary_sigma = [x for x in sigmas if x not in (0.5, 0.4841, 0.433)]
print('         non-binary sigma values: %s' % (sorted(set(nonbinary_sigma)) or 'NONE -> fully all-or-nothing'))

# eos
eos = [s.get('eos_terminated') for s in samples if 'eos_terminated' in s]
print('[EOS]    terminated=%d/%d = %d%%' % (sum(1 for e in eos if e), len(eos), 100*sum(1 for e in eos if e)//max(1,len(eos))))

# completion length
L = [s['completion_length'] for s in samples if s.get('completion_length')]
print('\n[COMPLETION_LENGTH tokens]  n=%d' % len(L))
print('  min=%d  p10=%d  p25=%d  median=%d  p75=%d  p90=%d  p95=%d  max=%d  mean=%d'
      % (min(L), pct(L,10), pct(L,25), int(st.median(L)), pct(L,75), pct(L,90), pct(L,95), max(L), int(st.mean(L))))

# prompt_idx
idx = [s['prompt_idx'] for s in samples if s.get('prompt_idx') is not None]
print('\n[PROMPT_IDX]  distinct=%d / total=%d  reuse=%d  range[min=%d max=%d]'
      % (len(set(idx)), len(idx), len(idx)-len(set(idx)), min(idx), max(idx)))

# prompt features (coding problem typing)
def ptype(p):
    p = (p or '').lower()
    cats = []
    for kw, lab in [('linked list','linkedlist'),('binary tree','tree'),('tree','tree'),('graph','graph'),
                    ('sort','sort'),('array','array'),('list of','list'),('string','string'),('matrix','matrix'),
                    ('dictionary','dict'),('recursion','recursion'),('dynamic programming','dp'),('palindrome','string'),
                    ('factorial','math'),('prime','math'),('fibonacci','math'),('stack','stack'),('queue','queue'),
                    ('regex','regex'),('json','json'),('class','class-oop')]:
        if kw in p: cats.append(lab)
    return cats[0] if cats else 'other'
tp = Counter(ptype(s.get('prompt')) for s in samples)
plens = [len(str(s.get('prompt',''))) for s in samples]
has_io = sum(1 for s in samples if 'sample input' in (s.get('prompt') or '').lower() or 'sample output' in (s.get('prompt') or '').lower())
has_sig = sum(1 for s in samples if re.search(r'def\s+\w+\s*\(', s.get('prompt') or ''))
print('\n[PROMPT]  median_chars=%d  has_sample_io=%d%%  has_func_signature=%d%%' % (int(st.median(plens)), 100*has_io//n, 100*has_sig//n))
print('  problem types: %s' % dict(tp.most_common()))

# completion structure
def cfeats(c):
    c = c or ''
    return ('```' in c, bool(re.search(r'```python', c)), 'def ' in c, '<think' in c.lower(),
            c.lstrip().startswith('```') or c.lstrip().startswith('def') or c.lstrip().startswith('import') or c.lstrip().startswith('class'))
fb = [cfeats(s.get('completion_text')) for s in samples]
print('\n[COMPLETION STRUCTURE]')
print('  has_code_fence=%d%%  python_fence=%d%%  has_def=%d%%  has_think=%d%%  starts_with_code=%d%%'
      % (100*sum(f[0] for f in fb)//n, 100*sum(f[1] for f in fb)//n, 100*sum(f[2] for f in fb)//n,
         100*sum(f[3] for f in fb)//n, 100*sum(f[4] for f in fb)//n))

# per-window cadence
acc = [a for (_,_,a,_,_,_) in per_window if a is not None]
sub = [s for (_,s,_,_,_,_) in per_window if s is not None]
print('\n[PER-WINDOW]  accepted: median=%s max=%s  submitted: median=%s max=%s'
      % (int(st.median(acc)) if acc else '?', max(acc) if acc else '?', int(st.median(sub)) if sub else '?', max(sub) if sub else '?'))

print('\n=== sample prompts (first 2 lines each, varied) ===')
seen=set()
for s in samples:
    t=ptype(s.get('prompt'))
    if t in seen: continue
    seen.add(t)
    pr=(s.get('prompt') or '').replace(chr(10),' ')[:150]
    cl=s.get('completion_length'); k=kof(round(float(s['sigma']),4)) if s.get('sigma') else '?'
    print('[%s | len=%s | %s]  %s' % (t, cl, k, pr))
