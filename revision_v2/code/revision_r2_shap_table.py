"""
Version 2: out-of-fold SHAP ranks and stability for the
twelve imaging features carried into radiogenomics, recomputed on the verified
384-patient data with the primary model and folds. Stability = coefficient of
variation (SD/mean) of fold-level mean |SHAP| across the 5 folds.
"""
import os, numpy as np, pandas as pd, shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
OUT = os.path.join(os.getcwd(), "results", "revision_r2")
d = pd.read_csv("datawith4visits.csv", skiprows=1).rename(columns={"CLINICAL-TRIAL-SUBJECT-ID": "Patient_ID"})
c = pd.read_csv("clinical_data.csv"); df = d.merge(c, on="Patient_ID"); assert len(df) == 384
df["sub"] = np.where(df.HER2 == 1, "HER2+", np.where(df.HR == 1, "HR+/HER2-", "TNBC"))
F = [x for x in d.columns if x != "Patient_ID"]
TWELVE = ["FTV_pch_T0_T1", "LD_T3", "LD_pch_T0_T1", "LD_pch_T0_T2", "LD_T2", "FTV_pch_T0_T2", "FTV_pch_T0_T3",
          "BPE_5slice_mean_T0", "BPE_5slice_mean_T1", "BPE_5slice_mean_T2", "LD_T0", "VOLUME_TUM_BLU_V10"]
rows = []
for name in ["Pooled", "HR+/HER2-", "HER2+", "TNBC"]:
    g = (df if name == "Pooled" else df[df["sub"] == name]).reset_index(drop=True)
    X, y = g[F].values, g.pCR.astype(int).values; fold_means = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        m = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1).fit(X[tr], y[tr])
        sv = shap.TreeExplainer(m).shap_values(X[te]); sv = sv[:, :, 1] if np.ndim(sv) == 3 else sv[1]
        fold_means.append(np.abs(sv).mean(0))
    fm = np.array(fold_means); mean = fm.mean(0); cv = fm.std(0) / mean
    rank = pd.Series(mean, index=F).rank(ascending=False).astype(int)
    for i, f in enumerate(F):
        rows.append({"cohort": name, "feature": f, "mean_abs_shap": mean[i], "cv_across_folds": cv[i], "rank_of_28": int(rank[f]), "in_twelve": f in TWELVE})
    print(name, "top5:", list(pd.Series(mean, index=F).nlargest(5).index), flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "shap_stability_all.csv"), index=False)
T = R[R.in_twelve].pivot(index="feature", columns="cohort", values="rank_of_28").join(
    R[R.in_twelve].pivot(index="feature", columns="cohort", values="cv_across_folds").add_suffix("_cv"))
T.loc[TWELVE].to_csv(os.path.join(OUT, "shap_twelve_feature_table.csv")); print(T.loc[TWELVE].round(2).to_string())
