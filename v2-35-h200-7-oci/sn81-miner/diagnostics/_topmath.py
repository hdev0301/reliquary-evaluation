import json, re, os, statistics as st
from collections import Counter, defaultdict

TMP = os.environ.get('TEMP', os.environ.get('TMP', '.'))
d = json.load(open(os.path.join(TMP, 'all_miners_live.json'), encoding='utf-8-sig'))
wins = d.get('windows', []) or []

HEX16=re.compile(r'^[0-9a-f]{16}$'); PLAIN=re.compile(r'^[\-\+]?\d+(\.\d+)?$'); SFRAC=re.compile(r'^[\-\+]?\d+/\d+$')
SYM=re.compile(r'\\(frac|sqrt|pi|cdot|times|circ|begin|matrix|pmatrix|text|sin|cos|tan|log|ln|sum|int|alpha|beta|theta|infty)'); BS=chr(92)
def src(g):
    g=(g or '').strip()
    if HEX16.match(g): return 'CODING'
    if PLAIN.match(g): return 'NUMERIC'
    if SFRAC.match(g): return 'simple-frac'
    if SYM.search(g) or re.search(r'[A-Za-z]',g) or ',' in g or BS in g: return 'SYMBOLIC'
    return 'other'
def kof(s):
    if s is None: return '?'
    return {0.5:'k4',0.4841:'k3/5',0.433:'k2/6'}.get(round(float(s),4),'%.3f'%round(float(s),4))
MATH={'NUMERIC','SYMBOLIC','simple-frac','other'}
KNOWN={'5F6VZ2ro':'uid39','5HEAK6g3':'uid181','5F7YBWD1':'uid243','5DARq6by':'?','5HQbAQ4U':'uid226',
       '5Hp6EPJd':'uid15','5G3wUjwf':'uid116(CODE)','5CkU7wLM':'?'}

tasksrc=Counter()
per_hk=defaultdict(lambda: {'tot':0,'math':0,'code':0,'NUMERIC':0,'SYMBOLIC':0,'simple-frac':0,'other':0,'k':Counter(),'len':[]})
for w in wins:
    tasksrc[w.get('task_source')]+=1
    for s in (w.get('samples') or []):
        hk=(s.get('hotkey') or '?')[:8]; b=src(s.get('ground_truth'))
        r=per_hk[hk]; r['tot']+=1
        if b=='CODING': r['code']+=1
        else:
            r['math']+=1; r[b]+=1; r['k'][kof(s.get('sigma'))]+=1
            if s.get('completion_length'): r['len'].append(s['completion_length'])

print('windows=%d  task_source dist=%s\n' % (len(wins), dict(tasksrc)))
ranked=sorted(per_hk.items(), key=lambda x:-x[1]['math'])
print('=== TOP OPENMATH MINERS (by accepted in-zone MATH groups over %d windows) ===' % len(wins))
print('%-9s %-12s %5s %5s %5s | %-22s | %-22s | %s' % ('hotkey','(uid)','math','code','tot','math-mix','sigma(k)','med_len'))
for hk,r in ranked[:15]:
    if r['math']==0: continue
    mix='N:%d%% S:%d%%'%(round(100*r['NUMERIC']/r['math']),round(100*r['SYMBOLIC']/r['math']))
    ks='  '.join('%s:%d%%'%(k,round(100*v/r['math'])) for k,v in r['k'].most_common(3))
    ml=int(st.median(r['len'])) if r['len'] else 0
    print('%-9s %-12s %5d %5d %5d | %-22s | %-22s | %d' % (hk, KNOWN.get(hk,''), r['math'], r['code'], r['tot'], mix, ks, ml))

# coding-dominant miners for contrast
print('\n=== (contrast) CODING-dominant hotkeys ===')
for hk,r in sorted(per_hk.items(), key=lambda x:-x[1]['code'])[:6]:
    if r['code']==0: continue
    print('  %-9s code=%d math=%d' % (hk, r['code'], r['math']))
