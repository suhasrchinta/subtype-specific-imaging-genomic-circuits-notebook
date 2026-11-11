"""
Version 2: Level 5 imaging vs genomic models on the verified overlap cohort,
with gene selection (top-k |Spearman| with pCR) nested inside each training fold, the same
folds for every model, and paired DeLong AUC differences. Missing expression values are
filled with the training-fold median.
"""
import os, numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
OUT = os.path.join(os.getcwd(), "results", "revision_r2")
d = pd.read_csv("datawith4visits.csv", skiprows=1).rename(columns={"CLINICAL-TRIAL-SUBJECT-ID": "Patient_ID"})
c = pd.read_csv("clinical_data.csv"); df = d.merge(c, on="Patient_ID"); df["Patient_ID"] = df.Patient_ID.astype(str)
C = [x for x in d.columns if x != "Patient_ID"]
expr = pd.read_csv("GSE194040_ISPY2ResID_AgilentGeneExp_990_FrshFrzn_meanCol_geneLevel_n988.txt", sep="\t", index_col=0)
expr.columns = expr.columns.astype(str); expr = expr[~expr.index.duplicated()]
m = df[df.Patient_ID.isin(expr.columns)].reset_index(drop=True)
G = expr[m.Patient_ID].T.reset_index(drop=True); y = m.pCR.astype(int).values; Xd = m[C].values
print("overlap n", len(m), "pCR", y.sum())
rf = lambda: RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
P = {k: np.zeros(len(y)) for k in ["DCE", "Gen50", "Gen200", "DCE+Gen50"]}
for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(Xd, y):
    Gtr = G.iloc[tr]; med = Gtr.median(); Gtr = Gtr.fillna(med); Gte = G.iloc[te].fillna(med)
    R = Gtr.rank().values; R = R - R.mean(0); yr = stats.rankdata(y[tr]); yr = yr - yr.mean()
    den = np.sqrt((R ** 2).sum(0)) * np.sqrt((yr ** 2).sum()); r = np.where(den > 0, (R * yr[:, None]).sum(0) / np.where(den > 0, den, 1), 0)
    order = np.argsort(-np.abs(r)); top50, top200 = G.columns[order[:50]], G.columns[order[:200]]
    P["DCE"][te] = rf().fit(Xd[tr], y[tr]).predict_proba(Xd[te])[:, 1]
    P["Gen50"][te] = rf().fit(Gtr[top50].values, y[tr]).predict_proba(Gte[top50].values)[:, 1]
    P["Gen200"][te] = rf().fit(Gtr[top200].values, y[tr]).predict_proba(Gte[top200].values)[:, 1]
    P["DCE+Gen50"][te] = rf().fit(np.hstack([Xd[tr], Gtr[top50].values]), y[tr]).predict_proba(np.hstack([Xd[te], Gte[top50].values]))[:, 1]
def delong(y, p1, p2):
    pos, neg = y == 1, y == 0
    def comp(p):
        psi = (p[pos][:, None] > p[neg][None, :]) + 0.5 * (p[pos][:, None] == p[neg][None, :]); return psi.mean(), psi.mean(1), psi.mean(0)
    a1, x1, y1 = comp(p1); a2, x2, y2 = comp(p2)
    v = (np.var(x1 - x2, ddof=1) / pos.sum()) + (np.var(y1 - y2, ddof=1) / neg.sum()); se = np.sqrt(v)
    return a1 - a2, se, 2 * stats.norm.sf(abs((a1 - a2) / se))
def boot(p):
    r = np.random.RandomState(42); v = []
    for _ in range(1000):
        i = r.randint(0, len(y), len(y))
        if 0 < y[i].sum() < len(i): v.append(roc_auc_score(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])
rows = [{"model": k, "n": len(y), "n_pcr": int(y.sum()), "auc": roc_auc_score(y, p), "lo": boot(p)[0], "hi": boot(p)[1]} for k, p in P.items()]
for a, b in [("DCE+Gen50", "DCE"), ("DCE", "Gen50"), ("DCE", "Gen200")]:
    dd, se, pv = delong(y, P[a], P[b]); rows.append({"model": f"{a} minus {b}", "auc": dd, "lo": dd - 1.96 * se, "hi": dd + 1.96 * se, "p": pv})
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "level5_nested_paired.csv"), index=False); print(R.round(4).to_string(index=False))
