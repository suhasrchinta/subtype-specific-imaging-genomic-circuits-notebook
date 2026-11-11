"""
Version 2: mediation re-specified in temporal order.
  Exposure X  = pretreatment gene expression (GSE194040, core biopsy before NAC)
  Mediator M  = the imaging feature measured after the biopsy (DCE-MRI) or at baseline (IVIM D*)
  Outcome Y   = pCR at surgery
  Path a: OLS  M ~ X + covariates;  path b: logistic  Y ~ M + X + covariates
  Indirect effect = a*b (log-odds of pCR per SD of gene expression, via the imaging feature).
  No "proportion mediated" is reported (the OLS/logistic product is not a share of the total effect).
Covariates: age, experimental vs control arm (control = paclitaxel, or paclitaxel + trastuzumab
in HER2+), ESTIMATE score (stromal + immune ssGSEA, Yoshihara 2013; tumor-purity proxy; the
Affymetrix purity formula is not applied because GSE194040 is Agilent), and HR status in HER2+.
ERBB2 and GRB7 are co-amplified at 17q12 and are combined into one amplicon score (mean of z-scores).
Bootstrap: 5000 resamples stratified by pCR, percentile 95% CI, two-sided bootstrap p.
BH correction across the two tests. All results are exploratory (features, genes and
directions were selected in the same cohort).
"""
import os, numpy as np, pandas as pd, warnings
import statsmodels.api as sm
from scipy import stats
warnings.filterwarnings("ignore")
OUT = os.path.join(os.getcwd(), "results", "revision_r2")
d = pd.read_csv("datawith4visits.csv", skiprows=1).rename(columns={"CLINICAL-TRIAL-SUBJECT-ID": "Patient_ID"})
c = pd.read_csv("clinical_data.csv"); df = d.merge(c, on="Patient_ID"); assert len(df) == 384
df["sub"] = np.where(df.HER2 == 1, "HER2+", np.where(df.HR == 1, "HR+/HER2-", "TNBC"))
df["experimental_arm"] = (~df.Arm.isin(["Paclitaxel", "Paclitaxel + Trastuzumab"])).astype(int)
df["Patient_ID"] = df.Patient_ID.astype(str)

expr = pd.read_csv("GSE194040_ISPY2ResID_AgilentGeneExp_990_FrshFrzn_meanCol_geneLevel_n988.txt", sep="\t", index_col=0)
expr.columns = expr.columns.astype(str)
gs = pd.read_csv("data/estimate_gene_sets.csv")
# ESTIMATE ssGSEA (port of tidyestimate::estimate_score), computed on all 988 samples
R = expr.rank(axis=0); R = 10000 * R / len(R)
def ssgsea(col, genes):
    o = col.sort_values(ascending=False); hit = o.index.isin(genes).astype(float)
    w = (o.values ** 0.25) * hit
    return float(np.sum(np.cumsum(w) / w.sum() - np.cumsum(1 - hit) / (1 - hit).sum()))
est = pd.DataFrame({k: [ssgsea(R[s], set(gs[k].dropna())) for s in R.columns] for k in gs.columns}, index=R.columns)
est["ESTIMATE"] = est.sum(1); est.to_csv(os.path.join(OUT, "estimate_scores_gse194040.csv"))
cover = {k: int(expr.index.isin(gs[k].dropna()).sum()) for k in gs.columns}
z = lambda v: (v - v.mean()) / v.std(ddof=0)

ivim = pd.read_csv("data/acrin_ivim_features.csv"); ivim = ivim[(ivim.status == "ok") & (ivim.Timepoint == 0)]
ivim["Patient_ID"] = ivim.Patient_ID.str.replace("ACRIN-6698-", "", regex=False); ivim = ivim.rename(columns={"Dstar": "Dstar_T0"})
df = df.merge(ivim[["Patient_ID", "Dstar_T0"]], on="Patient_ID", how="left")
g_need = ["SCUBE2", "ERBB2", "GRB7", "FLT1"]
E = expr.loc[g_need].T; E.index.name = "Patient_ID"
df = df.merge(E.reset_index(), on="Patient_ID", how="left").merge(est[["ESTIMATE"]].rename_axis("Patient_ID").reset_index(), on="Patient_ID", how="left")

TESTS = [("SCUBE2", "HR+/HER2-", ["SCUBE2"], "LD_T3"),
         ("ERBB2/GRB7 amplicon", "HER2+", ["ERBB2", "GRB7"], "FTV_pch_T0_T1"),
         ]
# FLT1 -> IVIM D* was removed: verification showed the IVIM signal in ivim_extract.py is the
# mean of whole central image slices (no tumor ROI), so D* does not describe the tumor.

def paths(X, M, Y, C):
    Xa = sm.add_constant(np.column_stack([X] + C)) if C else sm.add_constant(X)
    a = sm.OLS(M, Xa).fit().params[1]
    Xb = sm.add_constant(np.column_stack([M, X] + C)) if C else sm.add_constant(np.column_stack([M, X]))
    fb = sm.Logit(Y, Xb).fit(disp=0, maxiter=200); b = fb.params[1]; cp = fb.params[2]
    Xc = sm.add_constant(np.column_stack([X] + C)) if C else sm.add_constant(X)
    ctot = sm.Logit(Y, Xc).fit(disp=0, maxiter=200).params[1]
    return a, b, cp, ctot

rows = []
for label, sub, genes, feat in TESTS:
    g = df[(df["sub"] == sub)].copy()
    g["X"] = z(g[genes].apply(z).mean(1)) if len(genes) > 1 else g[genes[0]]
    for adj in ("adjusted", "unadjusted"):
        cov = ["age", "experimental_arm", "ESTIMATE"] + (["HR"] if sub == "HER2+" else [])
        if label == "FLT1" and adj == "adjusted": cov = ["ESTIMATE"]   # n=27: one covariate only
        h = g.dropna(subset=["X", feat, "pCR", "ESTIMATE", "Age_at_Screening"]).copy(); h["age"] = h.Age_at_Screening
        if adj == "unadjusted": cov = []
        X = z(h.X.values); M = z(h[feat].values); Y = h.pCR.astype(int).values
        C = [z(h[k].values) if h[k].nunique() > 2 else h[k].values.astype(float) for k in cov]
        a, b, cp, ctot = paths(X, M, Y, C)
        rng = np.random.RandomState(42); pos, neg = np.where(Y == 1)[0], np.where(Y == 0)[0]; boot = []; fails = 0
        for _ in range(5000):
            i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            try:
                aa, bb, _, _ = paths(X[i], M[i], Y[i], [cc[i] for cc in C]); boot.append(aa * bb)
            except Exception:
                fails += 1
        boot = np.array(boot); ab = a * b
        p2 = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
        rho, rp = stats.spearmanr(h.X, h[feat])
        rows.append({"test": label, "subtype": sub, "imaging_feature": feat, "model": adj, "covariates": ";".join(cov) or "none",
                     "n": len(Y), "n_pcr": int(Y.sum()), "gene_feature_spearman_rho": rho, "gene_feature_p": rp,
                     "path_a": a, "path_b": b, "direct_c_prime": cp, "total_c": ctot, "indirect_ab": ab,
                     "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5), "p_two_sided": p2,
                     "boot_ok": len(boot), "boot_failed": fails})
        np.save(os.path.join(OUT, f"mediation_boot_{label.split('/')[0].split(' ')[0]}_{adj}.npy"), boot)
        print(rows[-1]["test"], adj, "n", len(Y), "ab", round(ab, 3), "CI", np.round(np.percentile(boot, [2.5, 97.5]), 3), "p", round(p2, 4), "fails", fails, flush=True)
Rr = pd.DataFrame(rows)
for adj in ("adjusted", "unadjusted"):
    m = Rr.model == adj; p = Rr.loc[m, "p_two_sided"].values; o = np.argsort(p); q = np.empty(len(p))
    q[o] = np.minimum(1, np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]); Rr.loc[m, "q_BH"] = q
Rr.to_csv(os.path.join(OUT, "mediation_temporal.csv"), index=False)
print("ESTIMATE gene coverage:", cover); pd.set_option("display.width", 250)
print(Rr[["test", "model", "n", "n_pcr", "gene_feature_spearman_rho", "gene_feature_p", "path_a", "path_b", "indirect_ab", "ci_lo", "ci_hi", "p_two_sided", "q_BH"]].round(4).to_string(index=False))
