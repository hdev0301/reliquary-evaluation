import json, re, os, statistics as st
from collections import Counter, defaultdict

TMP = os.environ.get('TEMP', os.environ.get('TMP', '.'))
d = json.load(open(os.path.join(TMP, 'all_miners_live.json'), encoding='utf-8-sig'))
wins = d.get('windows', []) or []
HEX16 = re.compile(r'^[0-9a-f]{16}$')

# top openmath miners (window-winners / high volume)
TOP = {'5F7YBWD1':'uid243','5CX7gQ4f':'','5HEAK6g3':'uid181','5Hp6EPJd':'uid15','5ED8ahWx':'','5F6VZ2ro':'uid39'}

def feats(a):
    a=str(a); core=re.sub(r'\\(text|frac|sqrt|pi|circ|begin|end|pmatrix|bmatrix|cdot|times|left|right|sin|cos|tan|log|ln|theta|alpha|beta)','',a)
    return {'plain_int':bool(re.fullmatch(r'-?\d+',a.strip())),'decimal':bool(re.fullmatch(r'-?\d+\.\d+',a.strip())),
            'fraction':('\\frac' in a) or bool(re.search(r'\b\d+/\d+\b',a)),'radical':'\\sqrt' in a,'pi':'\\pi' in a,
            'tuple':(',' in a) or bool(re.search(r'[()\[\]]',a)),'matrix':'matrix' in a,'has_var':bool(re.search(r'[a-zA-Z]',core))}
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
def kof(s):
    return {0.5:'k4',0.4841:'k3/5',0.433:'k2/6'}.get(round(float(s),4),'%.3f'%round(float(s),4))
FKEYS=['plain_int','decimal','fraction','radical','pi','tuple','matrix','has_var']

per=defaultdict(lambda:{'n':0,'num':0,'sym':0,'f':Counter(),'t':Counter(),'k':Counter(),'len':[]})
agg={'n':0,'num':0,'sym':0,'f':Counter(),'t':Counter(),'k':Counter(),'len':[]}
def numeric(a):
    a=str(a).strip(); return bool(re.fullmatch(r'[\-\+]?\d+(\.\d+)?',a))
for w in wins:
    for s in (w.get('samples') or []):
        gt=str(s.get('ground_truth','')).strip()
        if HEX16.match(gt): continue   # skip coding
        hk=(s.get('hotkey') or '?')[:8]
        if hk not in TOP: continue
        for tgt in (per[hk],agg):
            tgt['n']+=1
            if numeric(gt): tgt['num']+=1
            else: tgt['sym']+=1
            for k,v in feats(gt).items():
                if v: tgt['f'][k]+=1
            tgt['t'][topic(s.get('prompt'))]+=1
            if s.get('sigma') is not None: tgt['k'][kof(s['sigma'])]+=1
            if s.get('completion_length'): tgt['len'].append(s['completion_length'])

def line(label,r):
    n=r['n']
    if not n: return
    ml=int(st.median(r['len'])) if r['len'] else 0
    fs='  '.join('%s:%d'%(k,round(100*r['f'][k]/n)) for k in FKEYS if r['f'][k])
    ks='  '.join('%s:%d'%(k,round(100*v/n)) for k,v in r['k'].most_common(3))
    ts='  '.join('%s:%d'%(k,round(100*v/n)) for k,v in r['t'].most_common(5))
    print('%-9s n=%3d  num/sym=%d/%d  med_len=%d' % (label, n, round(100*r['num']/n), round(100*r['sym']/n), ml))
    print('    fmt%%: %s' % fs)
    print('    k%%:   %s' % ks)
    print('    top%%: %s' % ts)

print('=== TOP OPENMATH MINERS — answer-format / topic / k distribution ===\n')
for hk in sorted(TOP, key=lambda h:-per[h]['n']):
    line('%s %s'%(hk,TOP[hk]), per[hk])
    print()
print('=== AGGREGATE (what the winning openmath side collectively mines) ===')
line('ALL-TOP', agg)
