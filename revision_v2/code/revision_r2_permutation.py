"""Permutation test (1000 label permutations) for the primary RF model.
Permutations are pre-generated from RandomState(42) so results do not depend on
how the work is split across machines/chunks. Usage: perm.py GROUP START END OUTDIR"""
import sys, os, time, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
grp, a, b, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
d = pd.read_csv("datawith4visits.csv", skiprows=1).rename(columns={"CLINICAL-TRIAL-SUBJECT-ID": "Patient_ID"})
c = pd.read_csv("clinical_data.csv"); df = d.merge(c[["Patient_ID", "HR", "HER2", "pCR"]], on="Patient_ID")
df["sub"] = np.where(df.HER2 == 1, "HER2+", np.where(df.HR == 1, "HR+/HER2-", "TNBC"))
g = df if grp == "Pooled" else df[df["sub"] == grp]
F = [x for x in d.columns if x != "Patient_ID"]; X, y = g[F].values, g.pCR.astype(int).values
cv = StratifiedKFold(5, shuffle=True, random_state=42)
rng = np.random.RandomState(42); perms = [rng.permutation(y) for _ in range(1000)]
def one(i):
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=1)
    return i, roc_auc_score(perms[i], cross_val_predict(rf, X, perms[i], cv=cv, method="predict_proba")[:, 1])
t = time.time()
res = Parallel(n_jobs=-1)(delayed(one)(i) for i in range(a, b))
os.makedirs(out, exist_ok=True)
pd.DataFrame(res, columns=["perm", "auc"]).to_csv(f"{out}/{grp.replace('/','-')}_{a}_{b}.csv", index=False)
print(grp, a, b, f"{time.time()-t:.1f}s", flush=True)
