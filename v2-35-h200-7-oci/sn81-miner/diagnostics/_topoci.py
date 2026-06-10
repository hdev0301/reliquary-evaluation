import json, re, os, statistics as st
from collections import Counter, defaultdict

TMP = os.environ.get('TEMP', os.environ.get('TMP', '.'))
d = json.load(open(os.path.join(TMP, 'all_miners_live.json'), encoding='utf-8-sig'))
wins = d.get('windows', []) or []

HEX16 = re.compile(r'^[0-9a-f]{16}$')
BINARY_SIG = {0.0, 0.3307, 0.433, 0.4841, 0.5}   # sigma values reachable by binary k-of-8
def kof(s):
    return {0.5:'k4',0.4841:'k3/5',0.433:'k2/6',0.3307:'k1/7'}.get(round(float(s),4),'%.3f'%round(float(s),4))
def ptype(p):
    p=(p or '').lower()
    for kw,lab in [('linked list','list'),('binary tree','tree'),('tree','tree'),('graph','graph'),('sort','sort'),
                   ('array','array'),('list','list'),('string','string'),('matrix','matrix'),('dictionary','dict'),
                   ('recursion','recursion'),('palindrome','string'),('factorial','math'),('prime','math'),
                   ('fibonacci','math'),('stack','stack'),('queue','queue'),('class','oop'),('regex','regex')]:
        if kw in p: return lab
    return 'other'
KNOWN={'5G3wUjwf':'uid116','5HQbAQ4U':'uid226','5DARq6by':'(ex-math)','5ECEJH9M':'','5GxNhDLW':'','5GxSiKC5':''}

per=defaultdict(lambda:{'n':0,'sig':Counter(),'nonbin':0,'len':[],'type':Counter(),'idx':set()})
for w in wins:
    for s in (w.get('samples') or []):
        if not HEX16.match(str(s.get('ground_truth','')).strip()):
            continue   # coding only
        hk=(s.get('hotkey') or '?')[:8]; r=per[hk]; r['n']+=1
        sg=round(float(s['sigma']),4) if s.get('sigma') is not None else None
        if sg is not None:
            r['sig'][kof(sg)]+=1
            if sg not in BINARY_SIG: r['nonbin']+=1
        if s.get('completion_length'): r['len'].append(s['completion_length'])
        r['type'][ptype(s.get('prompt'))]+=1
        if s.get('prompt_idx') is not None: r['idx'].add(s['prompt_idx'])

ranked=sorted(per.items(), key=lambda x:-x[1]['n'])
print('=== TOP OPENCODE MINERS (accepted in-zone coding groups, last %d windows) ===\n' % len(wins))
print('%-9s %-9s %5s %6s | %-26s | %-9s | %s' % ('hotkey','(uid)','grps','nonbin','sigma(k)','med_len','top problem types'))
for hk,r in ranked:
    if r['n']<3: continue
    ml=int(st.median(r['len'])) if r['len'] else 0
    p90=sorted(r['len'])[min(len(r['len'])-1,int(0.9*(len(r['len'])-1)))] if r['len'] else 0
    ks='  '.join('%s:%d%%'%(k,round(100*v/r['n'])) for k,v in r['sig'].most_common(3))
    nb='%d%%'%round(100*r['nonbin']/r['n'])
    tt=' '.join('%s:%d'%(t,c) for t,c in r['type'].most_common(4))
    print('%-9s %-9s %5d %6s | %-26s | %4d/%-4d | %s' % (hk, KNOWN.get(hk,''), r['n'], nb, ks, ml, p90, tt))

print('\n(nonbin = %% of groups with NON-binary sigma => partial-pass/continuous reward, i.e. NOT clean k=6 curation)')
