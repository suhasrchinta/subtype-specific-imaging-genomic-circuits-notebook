"""
Version 2: core prediction models on the verified 384-patient data
(datawith4visits.csv == TCIA Multi-feature-MRI-NACT-Data.xlsx; clinical_data.csv ==
TCIA ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx).

Stages (run separately to keep each call short):
  preds      out-of-fold predictions for every model x cohort (seed-42 StratifiedKFold(5))
  repeat     10 x repeated 5-fold CV for the primary model
  threshold  nested Youden threshold (chosen inside training folds only)
  summarize  AUC/PR-AUC with bootstrap CIs, DeLong comparisons, calibration, DCA
Feature sets:
  A  baseline (T0): FTV, sphericity, LD, BPE at T0 (4)
  B  early change T0->T1 percent change of the same four (4)
  T1 decision-timepoint set: A + the four T1 values + B (12)
  C  full longitudinal set (28)  <- primary
  D  clinical: age, menopausal status (+ HR/HER2 where they vary)
  E  C + D
  LR_C  L2 logistic regression on standardized C
  FTV   FTV-only RF (4 volumes + 3 percent changes)
  MP    MammaPrint binary (High1=0 / High2=1) logistic; A+MP; T1+MP
"""
import os, sys, json, pickle, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss

ROOT = os.getcwd(); OUT = os.path.join(ROOT, "results", "revision_r2"); os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "_preds_cache.pkl")
d = pd.read_csv("datawith4visits.csv", skiprows=1).rename(columns={"CLINICAL-TRIAL-SUBJECT-ID": "Patient_ID"})
c = pd.read_csv("clinical_data.csv")
df = d.merge(c, on="Patient_ID", how="inner"); assert len(df) == 384
df["sub"] = np.where(df.HER2 == 1, "HER2+", np.where(df.HR == 1, "HR+/HER2-", "TNBC"))
meno = df.menopausal_status.astype(str).str.lower()
df["meno_post"] = meno.str.startswith("post").astype(int)
df["age"] = df.Age_at_Screening
C = [x for x in d.columns if x != "Patient_ID"]; assert len(C) == 28
A = ["VOLUME_TUM_BLU_V10", "SPHERICITY_T0", "LD_T0", "BPE_5slice_mean_T0"]
B = ["FTV_pch_T0_T1", "Sphericity_pch_T0_T1", "LD_pch_T0_T1", "BPE_pch_T0_T1"]
T1 = A + ["VOLUME_TUM_BLU_V20", "SPHERICITY_T1", "LD_T1", "BPE_5slice_mean_T1"] + B
FTV = ["VOLUME_TUM_BLU_V10", "VOLUME_TUM_BLU_V20", "VOLUME_TUM_BLU_V30", "VOLUME_TUM_BLU_V40",
       "FTV_pch_T0_T1", "FTV_pch_T0_T2", "FTV_pch_T0_T3"]
COHORTS = ["Pooled", "HR+/HER2-", "HER2+", "TNBC", "HR+/HER2+", "HR-/HER2+"]

def cohort(name):
    if name == "Pooled": return df
    if name == "HR+/HER2+": return df[(df.HER2 == 1) & (df.HR == 1)]
    if name == "HR-/HER2+": return df[(df.HER2 == 1) & (df.HR == 0)]
    return df[df["sub"] == name]

def clin_cols(name):
    cols = ["age", "meno_post"]
    if name == "Pooled": cols += ["HR", "HER2"]
    if name == "HER2+": cols += ["HR"]
    return cols

def rf(): return RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
def lr(): return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))

def models(name):
    D = clin_cols(name)
    return {"A_baseline": (A, rf), "B_early_change": (B, rf), "T1_through_T1": (T1, rf),
            "C_full": (C, rf), "D_clinical": (D, lr), "E_C_plus_D": (C + D, rf),
            "LR_C": (C, lr), "FTV_only_RF": (FTV, rf),
            "MP_only": (["MP"], lr), "A_plus_MP": (A + ["MP"], rf), "T1_plus_MP": (T1 + ["MP"], rf)}

def oof(X, y, make, seed=42):
    return cross_val_predict(make(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]

stage = sys.argv[1]
cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}

if stage == "preds":
    want = sys.argv[2:] or COHORTS
    for name in want:
        g = cohort(name).reset_index(drop=True); y = g.pCR.astype(int).values
        folds = np.zeros(len(g), int)
        for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(g, y)): folds[te] = k
        cache[name] = {"ids": g.Patient_ID.values, "y": y, "fold": folds, "p": {}}
        ms = models(name) if name in ("Pooled", "HR+/HER2-", "HER2+", "TNBC") else {"C_full": (C, rf)}
        for m, (cols, make) in ms.items():
            cache[name]["p"][m] = oof(g[cols].values, y, make)
        if name == "HER2+": cache[name]["p"]["C_plus_HR"] = oof(g[C + ["HR"]].values, y, rf)
        print(name, {m: round(roc_auc_score(y, p), 3) for m, p in cache[name]["p"].items()}, flush=True)
        pickle.dump(cache, open(CACHE, "wb"))

elif stage == "repeat":
    rows = []
    for name in (sys.argv[2:] or ["Pooled", "HR+/HER2-", "HER2+", "TNBC"]):
        g = cohort(name); y = g.pCR.astype(int).values
        aucs = [roc_auc_score(y, oof(g[C].values, y, rf, seed=s)) for s in range(10)]
        rows.append({"cohort": name, "mean_auc_10x5": np.mean(aucs), "sd": np.std(aucs, ddof=1), "min": min(aucs), "max": max(aucs)})
        print(rows[-1], flush=True)
    f = os.path.join(OUT, "repeated_cv_10x5.csv")
    pd.DataFrame(rows).to_csv(f, index=False, mode="a", header=not os.path.exists(f))

elif stage == "threshold":
    rows = []
    for name in (sys.argv[2:] or ["Pooled", "HR+/HER2-", "HER2+", "TNBC"]):
        g = cohort(name).reset_index(drop=True); X = g[C].values; y = g.pCR.astype(int).values
        pred = np.zeros(len(y), int); thr_used = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
            inner = cross_val_predict(rf(), X[tr], y[tr], cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
            fpr, tpr, th = roc_curve(y[tr], inner); t = th[np.argmax(tpr - fpr)]; thr_used.append(float(t))
            m = rf().fit(X[tr], y[tr]); pred[te] = (m.predict_proba(X[te])[:, 1] >= t).astype(int)
        cache.setdefault("nested_threshold", {})[name] = {"pred": pred, "y": y, "thr": thr_used}
        print(name, "thresholds", np.round(thr_used, 3), flush=True)
    pickle.dump(cache, open(CACHE, "wb"))

elif stage == "summarize":
    rng = np.random.RandomState(42)
    def boot(y, p, f, n=1000):
        r = np.random.RandomState(42); v = []
        for _ in range(n):
            i = r.randint(0, len(y), len(y))
            if 0 < y[i].sum() < len(i): v.append(f(y[i], p[i]))
        return np.percentile(v, [2.5, 97.5])
    def delong(y, p1, p2=None):
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        def comps(p):
            X, Y = p[pos], p[neg]
            psi = (X[:, None] > Y[None, :]) + 0.5 * (X[:, None] == Y[None, :])
            return psi.mean(), psi.mean(1), psi.mean(0)
        a1, v10_1, v01_1 = comps(p1)
        if p2 is None:
            return a1, np.var(v10_1, ddof=1) / len(pos) + np.var(v01_1, ddof=1) / len(neg)
        a2, v10_2, v01_2 = comps(p2)
        s10 = np.cov(np.vstack([v10_1, v10_2])); s01 = np.cov(np.vstack([v01_1, v01_2]))
        var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / len(pos) + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / len(neg)
        from scipy.stats import norm
        z = (a1 - a2) / np.sqrt(var); return a1 - a2, np.sqrt(var), 2 * norm.sf(abs(z))
    from scipy.stats import norm
    import statsmodels.api as sm
    S, DL, CAL, DCA, PRED = [], [], [], [], []
    for name in COHORTS:
        e = cache[name]; y = e["y"]
        for m, p in e["p"].items():
            lo, hi = boot(y, p, roc_auc_score); plo, phi = boot(y, p, average_precision_score)
            S.append({"cohort": name, "model": m, "n": len(y), "n_pcr": int(y.sum()), "auc": roc_auc_score(y, p),
                      "auc_lo": lo, "auc_hi": hi, "pr_auc": average_precision_score(y, p), "pr_lo": plo, "pr_hi": phi,
                      "pr_baseline": y.mean()})
            if m != "C_full" and "C_full" in e["p"] and m not in ("MP_only",):
                diff, se, pv = delong(y, e["p"]["C_full"], p)
                DL.append({"cohort": name, "comparison": f"C_full vs {m}", "auc_diff": diff, "lo": diff - 1.96 * se, "hi": diff + 1.96 * se, "p": pv})
        if "MP_only" in e["p"]:
            for m1, m2 in [("A_baseline", "MP_only"), ("A_plus_MP", "A_baseline"), ("T1_through_T1", "A_baseline"),
                           ("T1_plus_MP", "A_plus_MP"), ("T1_plus_MP", "T1_through_T1"), ("C_full", "MP_only")]:
                diff, se, pv = delong(y, e["p"][m1], e["p"][m2])
                DL.append({"cohort": name, "comparison": f"{m1} vs {m2}", "auc_diff": diff, "lo": diff - 1.96 * se, "hi": diff + 1.96 * se, "p": pv})
        if name in ("Pooled", "HR+/HER2-", "HER2+", "TNBC"):
            p = np.clip(e["p"]["C_full"], 1e-4, 1 - 1e-4); lp = np.log(p / (1 - p))
            slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
            inter = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
            CAL.append({"cohort": name, "n": len(y), "prevalence": y.mean(), "mean_predicted": p.mean(),
                        "cal_intercept": inter.params[0], "cal_int_lo": inter.conf_int()[0][0], "cal_int_hi": inter.conf_int()[0][1],
                        "cal_slope": slope.params[1], "cal_slope_lo": slope.conf_int()[1][0], "cal_slope_hi": slope.conf_int()[1][1],
                        "brier": brier_score_loss(y, p), "brier_null_prevalence": y.mean() * (1 - y.mean())})
            for t in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]:
                tp = ((p >= t) & (y == 1)).sum(); fp = ((p >= t) & (y == 0)).sum(); n = len(y)
                DCA.append({"cohort": name, "threshold": t, "nb_model": tp / n - fp / n * t / (1 - t),
                            "nb_treat_all": y.mean() - (1 - y.mean()) * t / (1 - t), "flagged": int((p >= t).sum())})
            for i in range(len(y)):
                PRED.append({"cohort": name, "Patient_ID": e["ids"][i], "fold": int(e["fold"][i]), "pCR": int(y[i]),
                             **{f"p_{m}": e["p"][m][i] for m in e["p"]}})
    # unpaired subtype comparisons (separate patients): z from independent DeLong variances
    sub_rows = []
    for s1, s2 in [("HR+/HER2-", "HER2+"), ("HR+/HER2-", "TNBC"), ("HER2+", "TNBC"), ("HR+/HER2+", "HR-/HER2+")]:
        a1, v1 = delong(cache[s1]["y"], cache[s1]["p"]["C_full"]); a2, v2 = delong(cache[s2]["y"], cache[s2]["p"]["C_full"])
        se = np.sqrt(v1 + v2); z = (a1 - a2) / se
        sub_rows.append({"comparison": f"{s1} vs {s2}", "auc_diff": a1 - a2, "lo": a1 - a2 - 1.96 * se, "hi": a1 - a2 + 1.96 * se, "p": 2 * norm.sf(abs(z))})
    TH = []
    for name, e in cache.get("nested_threshold", {}).items():
        y, pr = e["y"], e["pred"]
        def mets(y, pr):
            tp = ((pr == 1) & (y == 1)).sum(); tn = ((pr == 0) & (y == 0)).sum(); fp = ((pr == 1) & (y == 0)).sum(); fn = ((pr == 0) & (y == 1)).sum()
            return np.array([tp / (tp + fn), tn / (tn + fp), tp / max(tp + fp, 1), tn / max(tn + fn, 1)])
        est = mets(y, pr); r = np.random.RandomState(42)
        bs = np.array([mets(y[i], pr[i]) for i in (r.randint(0, len(y), len(y)) for _ in range(1000))])
        lo, hi = np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)
        TH.append({"cohort": name, "thresholds_by_fold": ";".join(f"{t:.3f}" for t in e["thr"]),
                   "sensitivity": est[0], "sens_lo": lo[0], "sens_hi": hi[0], "specificity": est[1], "spec_lo": lo[1], "spec_hi": hi[1],
                   "ppv": est[2], "ppv_lo": lo[2], "ppv_hi": hi[2], "npv": est[3], "npv_lo": lo[3], "npv_hi": hi[3],
                   "n_pred_pos": int(pr.sum()), "n_pred_neg": int((pr == 0).sum())})
    for nm, rows in [("model_summary", S), ("delong_paired", DL), ("subtype_comparisons_unpaired", sub_rows),
                     ("calibration", CAL), ("decision_curve", DCA), ("nested_threshold_metrics", TH), ("patient_level_predictions", PRED)]:
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{nm}.csv"), index=False)
    pd.set_option("display.width", 250)
    for nm in ["model_summary", "delong_paired", "subtype_comparisons_unpaired", "calibration", "nested_threshold_metrics"]:
        print("\n==", nm); print(pd.read_csv(os.path.join(OUT, f"{nm}.csv")).round(3).to_string(index=False))
