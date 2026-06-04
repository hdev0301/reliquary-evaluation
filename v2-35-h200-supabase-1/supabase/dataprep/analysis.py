import json, re, numpy as np
from collections import Counter

def load(f): return [json.loads(l) for l in open(f) if l.strip()]
W = load("/root/wf_data/winners.jsonl")
C = load("/root/wf_data/controls.jsonl")

# ---------- numeric extraction ----------
def numbers_in(text):
    out=[]
    for m in re.finditer(r'(?<![\w.])(\d[\d,]*\.?\d*|\.\d+)(?![\w])', text):
        s=m.group(1).rstrip('.').replace(',','')
        try: out.append(float(s))
        except: pass
    return out

KW = {
 'twice':r'\btwice\b','half':r'\bhalf\b|one[- ]half','each':r'\beach\b','total':r'\btotal\b',
 'remaining':r'remain','ratio':r'\bratio\b','per':r'\bper\b','discount':r'discount','tax':r'\btax\b',
}
STEM = {
 'how_many':r'how many','how_much':r'how much','what_is':r'what is|what was|what would',
}

def gt_class(g):
    g=str(g).strip(); gn=g.replace(',','').replace('$','').replace('%','').strip()
    hasdot='.' in gn
    try:
        v=float(gn)
        is_int = (abs(v-round(v))<1e-9 and not hasdot)
        is_dec = not is_int
        return int(is_int),int(is_dec),0,abs(v),len(re.sub(r'\D','',gn))
    except:
        return 0,0,1,0.0,len(re.sub(r'\D','',gn))

def feats(r):
    p=r["prompt"]; pl=p.lower(); words=p.split(); nums=numbers_in(p)
    f={}
    f["len_chars"]=len(p); f["len_words"]=len(words)
    f["n_numeric_tokens"]=len(nums)
    f["max_number"]=max(nums) if nums else 0.0
    f["decimal_in_prompt"]=int(bool(re.search(r'\d+\.\d+',p)))
    f["has_percent"]=int('%' in p or 'percent' in pl)
    f["has_fraction"]=int(bool(re.search(r'\b\d+/\d+\b|one[- ](third|fourth|fifth|half|quarter)|two[- ]thirds|three[- ]quarters',pl)))
    f["has_money"]=int('$' in p)
    gi,gd,gt_,mag,nd=gt_class(r["ground_truth"])
    f["gt_is_integer"]=gi; f["gt_is_decimal"]=gd; f["gt_is_text"]=gt_
    f["gt_magnitude"]=mag; f["gt_num_digits"]=nd
    f["has_decimal_answer"]=gd
    for k,pat in KW.items(): f["kw_"+k]=int(bool(re.search(pat,pl)))
    f["kw_count"]=sum(f["kw_"+k] for k in KW)
    for k,pat in STEM.items(): f["stem_"+k]=int(bool(re.search(pat,pl)))
    return f

def mat(rows):
    fs=[feats(r) for r in rows]; cols=list(fs[0].keys())
    return cols, np.array([[fr[c] for c in cols] for fr in fs],float)

def auc(pos,neg):
    pos=np.asarray(pos,float); neg=np.asarray(neg,float)
    allv=np.concatenate([pos,neg])
    order=allv.argsort(kind='mergesort'); ranks=np.empty(len(allv)); ranks[order]=np.arange(1,len(allv)+1)
    # tie correction: average ranks
    s=np.sort(allv); i=0
    while i<len(s):
        j=i
        while j+1<len(s) and s[j+1]==s[i]: j+=1
        if j>i:
            avg=(i+1+j+1)/2.0
            ranks[np.isin(allv,s[i])]=avg  # crude but ok for tied identical vals
        i=j+1
    ra=ranks[:len(pos)].sum(); u=ra-len(pos)*(len(pos)+1)/2.0
    return u/(len(pos)*len(neg))

cols, Xw = mat(W); _, Xc = mat(C)
# source-matched winners (augmented_gsm8k only)
Wg=[r for r in W if r["source"]=="augmented_gsm8k"]
_, Xwg = mat(Wg)

def table(Xpos, Xneg, label):
    print("="*100); print(f"{label}  (pos n={len(Xpos)}, neg n={len(Xneg)})"); print("="*100)
    rows=[]
    for i,c in enumerate(cols):
        wm=Xpos[:,i].mean(); cm=Xneg[:,i].mean(); a=auc(Xpos[:,i],Xneg[:,i])
        rows.append((c,wm,cm,a,abs(a-0.5)))
    rows.sort(key=lambda t:-t[4])
    print(f"{'feature':20s} {'win_mean':>11s} {'ctrl_mean':>11s} {'AUC':>7s} {'|AUC-.5|':>9s}")
    for c,wm,cm,a,s in rows:
        print(f"{c:20s} {wm:11.3f} {cm:11.3f} {a:7.3f} {s:9.3f}")
    return rows

print("\nWINNER k distribution:", dict(sorted(Counter(sum(int(x) for x in r['reward_vector']) for r in W).items())))
table(Xw, Xc, "ALL winners vs controls (CONFOUNDED: controls 100% gsm8k, winners 79.6% gsm8k)")
r2=table(Xwg, Xc, "SOURCE-MATCHED: augmented_gsm8k winners vs gsm8k controls (clean)")

# permutation null on best feature to check what |AUC-0.5| is 'just noise'
print("\n" + "#"*100)
print("PERMUTATION NULL: shuffle labels, max |AUC-0.5| over all features, 200 reps (source-matched sizes)")
allX=np.vstack([Xwg,Xc]); n1=len(Xwg)
rng=np.random.default_rng(0); maxnull=[]
for _ in range(200):
    perm=rng.permutation(len(allX)); P=allX[perm]
    m=max(abs(auc(P[:n1,i],P[n1:,i])-0.5) for i in range(len(cols)))
    maxnull.append(m)
maxnull=np.array(maxnull)
print(f"  null max|AUC-.5|: mean={maxnull.mean():.3f} p95={np.percentile(maxnull,95):.3f} max={maxnull.max():.3f}")
print(f"  observed top |AUC-.5| (source-matched) = {r2[0][4]:.3f} for '{r2[0][0]}'")
