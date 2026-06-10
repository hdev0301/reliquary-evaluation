import json, re, numpy as np, pandas as pd
from collections import Counter

def load(f):
    return [json.loads(l) for l in open(f) if l.strip()]

winners = load("/root/wf_data/winners.jsonl")
controls = load("/root/wf_data/controls.jsonl")

NUM_RE = re.compile(r'\d[\d,]*\.?\d*|\.\d+')  # numeric tokens incl commas/decimals
def numbers_in(text):
    out=[]
    for m in re.finditer(r'(?<![\w.])(\d[\d,]*\.?\d*|\.\d+)(?![\w])', text):
        s=m.group(1).replace(',','')
        try: out.append(float(s))
        except: pass
    return out

KW = {
 'twice':r'\btwice\b','half':r'\bhalf|one[- ]half\b','each':r'\beach\b',
 'total':r'\btotal\b','remaining':r'\bremain','percent':r'percent|%',
 'more_than':r'\bmore than\b','less_than':r'\bless than\b','fewer':r'\bfewer\b',
 'increase':r'\bincrease','decrease':r'\bdecrease','average':r'\baverage|\bmean\b',
 'ratio':r'\bratio\b','per':r'\bper\b','discount':r'\bdiscount','tax':r'\btax\b','tip':r'\btip\b',
}
STEM = {
 'how_many':r'how many','how_much':r'how much','what_is':r'what is|what was|what would',
 'find':r'\bfind\b','calculate':r'\bcalculate\b',
}

def gt_classify(g):
    g=str(g).strip()
    # try numeric
    gn=g.replace(',','').replace('$','').replace('%','').strip()
    is_int=is_dec=is_text=False; mag=np.nan; ndig=np.nan; hasdot=('.' in gn)
    try:
        v=float(gn)
        if abs(v-round(v))<1e-9 and not hasdot:
            is_int=True
        else:
            is_dec=True
        mag=abs(v); ndig=len(re.sub(r'\D','',gn))
    except:
        is_text=True
    return is_int,is_dec,is_text,mag,ndig,hasdot

def feats(r):
    p=r["prompt"]; pl=p.lower()
    words=p.split()
    nums=numbers_in(p)
    f={}
    f["len_chars"]=len(p)
    f["len_words"]=len(words)
    f["n_numeric_tokens"]=len(nums)
    f["max_number"]=max(nums) if nums else 0.0
    f["has_decimal"]=int(any('.' in m for m in re.findall(r'\d+\.\d+',p)))
    f["has_percent"]=int('%' in p or 'percent' in pl)
    f["has_fraction"]=int(bool(re.search(r'\b\d+/\d+\b|one[- ](third|fourth|fifth|half|quarter)|two[- ]thirds|three[- ]quarters',pl)))
    f["has_money"]=int('$' in p)
    g=str(r["ground_truth"])
    is_int,is_dec,is_text,mag,ndig,hasdot=gt_classify(g)
    f["gt_is_integer"]=int(is_int); f["gt_is_decimal"]=int(is_dec); f["gt_is_text"]=int(is_text)
    f["gt_magnitude"]=mag if not np.isnan(mag) else 0.0
    f["gt_num_digits"]=ndig if not np.isnan(ndig) else 0.0
    f["gt_has_dot"]=int(hasdot)
    for k,pat in KW.items(): f["kw_"+k]=int(bool(re.search(pat,pl)))
    f["kw_any_arith"]=int(any(f["kw_"+k] for k in KW))
    f["kw_count_arith"]=sum(f["kw_"+k] for k in KW)
    for k,pat in STEM.items(): f["stem_"+k]=int(bool(re.search(pat,pl)))
    return f

def build(rows): return pd.DataFrame([feats(r) for r in rows])

Wf=build(winners); Cf=build(controls)
# matched-source subset
Wf_g=build([r for r in winners if r["source"]=="augmented_gsm8k"])

def cohend(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    na,nb=len(a),len(b); va,vb=a.var(ddof=1),b.var(ddof=1)
    sp=np.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2)) if (na+nb-2)>0 else np.nan
    return (a.mean()-b.mean())/sp if sp and sp>0 else 0.0

def auc(a,b):
    # prob a-value > b-value (Mann-Whitney style separation, 0.5=no sep)
    a=np.asarray(a,float); b=np.asarray(b,float)
    allv=np.concatenate([a,b]); ranks=pd.Series(allv).rank().values
    ra=ranks[:len(a)].sum()
    u=ra-len(a)*(len(a)+1)/2
    return u/(len(a)*len(b))

def compare(W,C,label):
    rows=[]
    for col in W.columns:
        w=W[col].values; c=C[col].values
        wm=w.mean(); cm=c.mean()
        ratio=wm/cm if cm!=0 else (np.inf if wm!=0 else 1.0)
        pct=(wm-cm)/abs(cm)*100 if cm!=0 else np.nan
        d=cohend(w,c); a=auc(w,c)
        rows.append((col,wm,cm,ratio,pct,d,a))
    df=pd.DataFrame(rows,columns=["feature","winners","controls","ratio","pct_diff","cohen_d","auc"])
    df["abs_d"]=df["cohen_d"].abs()
    df["sep"]=(df["auc"]-0.5).abs()
    df=df.sort_values("abs_d",ascending=False).reset_index(drop=True)
    print("\n"+"="*100)
    print(f"COMPARISON: {label}   (winners n={len(W)}, controls n={len(C)})")
    print("="*100)
    pd.set_option("display.width",200,"display.max_rows",100,"display.float_format",lambda x:f"{x:8.3f}")
    print(df[["feature","winners","controls","ratio","pct_diff","cohen_d","auc"]].to_string(index=False))
    return df

print("Winners sources:",Counter(r["source"] for r in winners))
print("Controls sources:",Counter(r["source"] for r in controls))

dA=compare(Wf,Cf,"ALL winners vs ALL controls")
dB=compare(Wf_g,Cf,"augmented_gsm8k winners ONLY vs controls (source-matched)")

# Overall separability: train a tiny logistic-style check via simple linear separation using all numeric feats
def separability(W,C):
    X=np.vstack([W.values,C.values]).astype(float)
    y=np.array([1]*len(W)+[0]*len(C))
    # standardize
    mu=X.mean(0); sd=X.std(0); sd[sd==0]=1
    Xs=(X-mu)/sd
    # closed-form-ish: use numpy lstsq logistic surrogate via ridge on centered y
    # Simpler: 5-fold-ish single linear discriminant (no sklearn). Use pseudo-inverse.
    from numpy.linalg import pinv
    w=pinv(Xs.T@Xs+1e-2*np.eye(Xs.shape[1]))@Xs.T@(y-y.mean())
    scores=Xs@w
    a=auc(scores[y==1],scores[y==0])
    # best single-feature auc
    best=max(((c,abs(auc(W[c].values,C[c].values)-0.5)+0.5) for c in W.columns),key=lambda t:t[1])
    return a,best

aA,bestA=separability(Wf,Cf)
aB,bestB=separability(Wf_g,Cf)
print("\n"+"#"*100)
print("SEPARABILITY (in-sample linear discriminant AUC; 0.5=random, 1.0=perfect):")
print(f"  ALL winners vs controls:            multivar AUC={aA:.3f} | best single feat={bestA}")
print(f"  gsm8k-matched winners vs controls:  multivar AUC={aB:.3f} | best single feat={bestB}")
print("#"*100)

dA.to_pickle("/root/dA.pkl"); dB.to_pickle("/root/dB.pkl")

# ---- HONEST cross-validated separability ----
print("\n"+"#"*100)
print("CROSS-VALIDATED separability (5-fold, ridge linear discriminant). This is the honest signal.")
print("#"*100)

def cv_auc(W,C,seed=0,folds=5):
    X=np.vstack([W.values,C.values]).astype(float)
    y=np.array([1.0]*len(W)+[0.0]*len(C))
    rng=np.random.default_rng(seed)
    idx=rng.permutation(len(y))
    X,y=X[idx],y[idx]
    aucs=[]
    fold=np.array_split(np.arange(len(y)),folds)
    from numpy.linalg import pinv
    for k in range(folds):
        te=fold[k]; tr=np.setdiff1d(np.arange(len(y)),te)
        mu=X[tr].mean(0); sd=X[tr].std(0); sd[sd==0]=1
        Xtr=(X[tr]-mu)/sd; Xte=(X[te]-mu)/sd
        ytr=y[tr]
        w=pinv(Xtr.T@Xtr+1.0*np.eye(Xtr.shape[1]))@Xtr.T@(ytr-ytr.mean())
        s=Xte@w
        # auc on test
        order=pd.Series(s).rank().values; pos=y[te]==1
        if pos.sum()==0 or (~pos).sum()==0: continue
        u=order[pos].sum()-pos.sum()*(pos.sum()+1)/2
        aucs.append(u/(pos.sum()*(~pos).sum()))
    return np.mean(aucs)

import pandas as pd, numpy as np
res_all=np.mean([cv_auc(Wf,Cf,s) for s in range(10)])
res_g  =np.mean([cv_auc(Wf_g,Cf,s) for s in range(10)])
print(f"  ALL winners vs controls:           5-fold CV AUC = {res_all:.3f}")
print(f"  gsm8k-matched winners vs controls: 5-fold CV AUC = {res_g:.3f}")
print("  (0.50 = indistinguishable / semantic difficulty;  >0.65 = cheap text features carry real signal)")

# Single-feature CV-honest check for top movers: report median (robust) too
print("\nRobust check on top movers (median, since means are skewed):")
for col in ["gt_magnitude","max_number","has_fraction","has_decimal","kw_remaining","kw_total","gt_num_digits","len_words"]:
    print(f"  {col:16s} winners median={np.median(Wf[col]):10.3f}  controls median={np.median(Cf[col]):10.3f}")
