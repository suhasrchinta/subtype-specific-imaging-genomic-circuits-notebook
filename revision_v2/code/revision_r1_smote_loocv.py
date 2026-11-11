"""
Version 2: SMOTE and LOOCV sensitivity checks, rerun with all resampling
inside the training folds. The original SMOTE/LOOCV code
could not be located, so these replace the earlier figures.

Model = locked primary model: RF(500 trees, class_weight='balanced',
random_state=42), Group C (28 features), StratifiedKFold(5, shuffle, 42).
SMOTE is applied only to each training fold via an imblearn Pipeline; the
held-out fold is never resampled.
"""
import os, json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

ROOT = os.getcwd(); OUT = os.path.join(ROOT, "results", "revision_r1")
os.makedirs(OUT, exist_ok=True)
d = pd.read_csv("datawith4visits.csv", skiprows=1).rename(columns={"CLINICAL-TRIAL-SUBJECT-ID": "Patient_ID"})
c = pd.read_csv("clinical_data.csv")
df = d.merge(c[["Patient_ID", "HR", "HER2", "pCR"]], on="Patient_ID")
df["sub"] = np.where(df.HER2 == 1, "HER2+", np.where(df.HR == 1, "HR+/HER2-", "TNBC"))
FEAT = [x for x in d.columns if x != "Patient_ID"]; assert len(FEAT) == 28
cv = StratifiedKFold(5, shuffle=True, random_state=42)

def rf(cw="balanced"): return RandomForestClassifier(n_estimators=500, class_weight=cw, random_state=42, n_jobs=-1)
def boot_ci(y, p, n=1000, seed=42):
    r = np.random.RandomState(seed); a = []
    for _ in range(n):
        i = r.randint(0, len(y), len(y))
        if len(set(y[i])) == 2: a.append(roc_auc_score(y[i], p[i]))
    return np.percentile(a, [2.5, 97.5])

import sys
WANT = sys.argv[1:] or ["Pooled", "HR+/HER2-", "HER2+", "TNBC"]
rows = []
for name, g in [("Pooled", df)] + [(s, df[df["sub"] == s]) for s in ("HR+/HER2-", "HER2+", "TNBC")]:
    if name not in WANT: continue
    X, y = g[FEAT].values, g.pCR.astype(int).values
    res = {"cohort": name, "n": len(y), "n_pcr": int(y.sum())}
    p = cross_val_predict(rf(), X, y, cv=cv, method="predict_proba")[:, 1]
    res["auc_5fold_balanced"] = roc_auc_score(y, p); res["ci_5fold"] = boot_ci(y, p).round(3).tolist()
    k = min(5, int(y.sum()) - 1)
    pipe = Pipeline([("smote", SMOTE(random_state=42, k_neighbors=k)), ("rf", rf(None))])
    ps = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    res["auc_5fold_smote_in_fold"] = roc_auc_score(y, ps); res["ci_smote"] = boot_ci(y, ps).round(3).tolist()
    res["delta_smote"] = res["auc_5fold_smote_in_fold"] - res["auc_5fold_balanced"]
    if True:
        pl = cross_val_predict(rf(), X, y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
        res["auc_loocv"] = roc_auc_score(y, pl); res["delta_loocv"] = res["auc_loocv"] - res["auc_5fold_balanced"]
    rows.append(res); print(res, flush=True)
R = pd.DataFrame(rows); f = os.path.join(OUT, "smote_loocv_in_fold.csv")
R.to_csv(f, index=False, mode="a", header=not os.path.exists(f))
print(R.round(3).to_string(index=False))
