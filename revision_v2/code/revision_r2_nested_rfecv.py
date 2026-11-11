"""
Version 2: RFECV inside the training folds.
Outer: StratifiedKFold(5, shuffle, 42), the same folds as the primary model.
Inner: RFECV(RF 200 trees, step=1, StratifiedKFold(5, shuffle, 42), AUC, min 3 features)
fit on the outer training fold only; an RF(500) on the selected features is
scored on the untouched outer test fold. Also reports per-feature selection
frequency across the 5 outer folds, and (descriptively) the features RFECV
selects on the full cohort.
Usage: revision_r2_nested_rfecv.py COHORT FOLD|full|summarize
"""
import os, sys, json, pickle, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
OUT = os.path.join(os.getcwd(), "results", "revision_r2"); CACHE = os.path.join(OUT, "_rfecv_cache.pkl")
d = pd.read_csv("datawith4visits.csv", skiprows=1).rename(columns={"CLINICAL-TRIAL-SUBJECT-ID": "Patient_ID"})
c = pd.read_csv("clinical_data.csv"); df = d.merge(c, on="Patient_ID"); assert len(df) == 384
df["sub"] = np.where(df.HER2 == 1, "HER2+", np.where(df.HR == 1, "HR+/HER2-", "TNBC"))
F = [x for x in d.columns if x != "Patient_ID"]
name, what = sys.argv[1], sys.argv[2]
cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
if what != "summarize":
    g = (df if name == "Pooled" else df[df["sub"] == name]).reset_index(drop=True)
    X, y = g[F].values, g.pCR.astype(int).values
    inner = StratifiedKFold(5, shuffle=True, random_state=42)
    sel = lambda: RFECV(RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
                        step=1, cv=inner, scoring="roc_auc", min_features_to_select=3, n_jobs=1)
    if what == "full":
        r = sel().fit(X, y); cache.setdefault(name, {})["full"] = [f for f, s in zip(F, r.support_) if s]
        print(name, "full-cohort RFECV:", len(cache[name]["full"]), cache[name]["full"])
    else:
        k = int(what); tr, te = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))[k]
        r = sel().fit(X[tr], y[tr]); s = r.support_
        m = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1).fit(X[tr][:, s], y[tr])
        cache.setdefault(name, {})[k] = {"te": te, "p": m.predict_proba(X[te][:, s])[:, 1], "sel": [f for f, b in zip(F, s) if b], "y": y}
        print(name, "fold", k, "selected", int(s.sum()))
    pickle.dump(cache, open(CACHE, "wb"))
else:
    rows, freq = [], []
    for nm, e in cache.items():
        folds = [k for k in range(5) if k in e]
        if len(folds) < 5: continue
        y = e[0]["y"]; p = np.zeros(len(y))
        for k in folds: p[e[k]["te"]] = e[k]["p"]
        cnt = pd.Series([f for k in folds for f in e[k]["sel"]]).value_counts()
        rows.append({"cohort": nm, "n": len(y), "n_pcr": int(y.sum()), "nested_rfecv_auc": roc_auc_score(y, p),
                     "n_selected_per_fold": ";".join(str(len(e[k]["sel"])) for k in folds),
                     "full_cohort_selection": ", ".join(e.get("full", []))})
        for f in F: freq.append({"cohort": nm, "feature": f, "selected_in_folds": int(cnt.get(f, 0))})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "nested_rfecv.csv"), index=False)
    pd.DataFrame(freq).to_csv(os.path.join(OUT, "nested_rfecv_selection_frequency.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
