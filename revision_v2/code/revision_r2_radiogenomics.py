"""
Version 2: radiogenomic correlations recomputed on the
verified 384-patient data, plus the exact feature-gene pair table for each circuit.
Level 1: pooled Spearman, 12 SHAP features x 6 curated gene sets, BH within feature x set
(same family definition as radiogenomics.py). Level 2: pooled genome-wide Spearman,
BH within each feature. Pair table: within-subtype Spearman and a within-subtype,
genome-wide BH q for that feature (the multiplicity actually faced when picking the gene).
"""
import os, re, numpy as np, pandas as pd
from scipy import stats
OUT = os.path.join(os.getcwd(), "results", "revision_r2")
src = open("code/radiogenomics.py").read()
GENE_SETS = eval(re.search(r"GENE_SETS = (\{.*?\n\})", src, re.S).group(1))
TWELVE = ["FTV_pch_T0_T1", "LD_T3", "LD_pch_T0_T1", "LD_pch_T0_T2", "LD_T2", "FTV_pch_T0_T2", "FTV_pch_T0_T3",
          "BPE_5slice_mean_T0", "BPE_5slice_mean_T1", "BPE_5slice_mean_T2", "LD_T0", "VOLUME_TUM_BLU_V10"]
d = pd.read_csv("datawith4visits.csv", skiprows=1).rename(columns={"CLINICAL-TRIAL-SUBJECT-ID": "Patient_ID"})
c = pd.read_csv("clinical_data.csv"); df = d.merge(c, on="Patient_ID"); df["Patient_ID"] = df.Patient_ID.astype(str)
df["sub"] = np.where(df.HER2 == 1, "HER2+", np.where(df.HR == 1, "HR+/HER2-", "TNBC"))
expr = pd.read_csv("GSE194040_ISPY2ResID_AgilentGeneExp_990_FrshFrzn_meanCol_geneLevel_n988.txt", sep="\t", index_col=0)
expr.columns = expr.columns.astype(str); expr = expr[~expr.index.duplicated()]
ivim = pd.read_csv("data/acrin_ivim_features.csv"); ivim = ivim[(ivim.status == "ok") & (ivim.Timepoint == 0)]
ivim["Patient_ID"] = ivim.Patient_ID.str.replace("ACRIN-6698-", "", regex=False)
df = df.merge(ivim[["Patient_ID", "Dstar"]].rename(columns={"Dstar": "Dstar_T0"}), on="Patient_ID", how="left")
m = df[df.Patient_ID.isin(expr.columns)].reset_index(drop=True)
G = expr[m.Patient_ID].T.reset_index(drop=True); G = G.loc[:, G.std() > 1e-10]
print("overlap n =", len(m), {s: int((m["sub"] == s).sum()) for s in ["HR+/HER2-", "HER2+", "TNBC"]}, "genes", G.shape[1])
def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = np.empty(len(p))
    q[o] = np.minimum(1, np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]); return q
def scan(x, Gm):
    """Spearman of x with every gene; genes with missing values use pairwise-complete data."""
    ok = ~np.isnan(x); Gk = Gm[ok]; n = int(ok.sum())
    full_cols = Gk.columns[Gk.notna().all()]; part_cols = Gk.columns[~Gk.notna().all()]
    xr = stats.rankdata(x[ok]); xr = (xr - xr.mean()) / xr.std()
    Gr = Gk[full_cols].rank().values; sd = Gr.std(0); keep = sd > 0
    Gr = (Gr[:, keep] - Gr[:, keep].mean(0)) / sd[keep]
    r = xr @ Gr / n; t = r * np.sqrt((n - 2) / (1 - r ** 2)); p = 2 * stats.t.sf(np.abs(t), n - 2)
    out = [pd.DataFrame({"gene": full_cols[keep], "r": r, "p": p})]
    rows = []
    for gname in part_cols:
        v = Gk[gname].values; mk = ~np.isnan(v)
        if mk.sum() > 10 and np.std(v[mk]) > 0:
            rr, pp = stats.spearmanr(x[ok][mk], v[mk]); rows.append((gname, rr, pp))
    out.append(pd.DataFrame(rows, columns=["gene", "r", "p"]))
    res = pd.concat(out, ignore_index=True); res["q"] = bh(res.p.values)
    return res, n
L1, L2 = [], []
for f in TWELVE:
    full, n = scan(m[f].values.astype(float), G)
    L2.append({"feature": f, "n": n, "genes_tested": len(full), "n_q_lt_0.05": int((full.q < 0.05).sum()),
               "min_q": full.q.min(), "top_gene": full.loc[full.p.idxmin(), "gene"]})
    full.to_csv(os.path.join(OUT, f"level2_pooled_{f}.csv"), index=False)
    for setname, genes in GENE_SETS.items():
        gg = [g for g in genes if g in G.columns]; sub, _ = scan(m[f].values.astype(float), G[gg])
        for _, r in sub.iterrows(): L1.append({"feature": f, "gene_set": setname, **r.to_dict(), "n": n})
L1 = pd.DataFrame(L1); L1.to_csv(os.path.join(OUT, "level1_pooled.csv"), index=False)
L2 = pd.DataFrame(L2); L2.to_csv(os.path.join(OUT, "level2_pooled_summary.csv"), index=False)
print("Level 1 significant (q<0.05):", int((L1.q < 0.05).sum()), "of", len(L1))
print(L1[L1.gene.isin(["ERBB2", "GRB7", "VEGFA", "FLT1", "ANGPT2"]) & (L1.q < 0.05)].round(4).to_string(index=False))
print(L2.round(4).to_string(index=False))
PAIRS = [("HR+/HER2-", "SCUBE2", "LD_T3"), ("HER2+", "ERBB2", "FTV_pch_T0_T1"), ("HER2+", "GRB7", "FTV_pch_T0_T1"),
         ("TNBC", "FLT1", "Dstar_T0"), ("TNBC", "ANGPT2", "BPE_5slice_mean_T0"), ("TNBC", "ANGPT2", "BPE_5slice_mean_T1"),
         ("TNBC", "ANGPT2", "BPE_5slice_mean_T2"), ("TNBC", "FLT1", "BPE_5slice_mean_T0"), ("TNBC", "FLT1", "BPE_5slice_mean_T1"),
         ("TNBC", "FLT1", "BPE_5slice_mean_T2"), ("HR+/HER2-", "SCUBE2", "FTV_pch_T0_T2"), ("HER2+", "ERBB2", "LD_T3"), ("HER2+", "GRB7", "LD_T3")]
P = []
for s, gene, f in PAIRS:
    h = m[m["sub"] == s].reset_index(drop=True); Gs = G[m["sub"].values == s].reset_index(drop=True)
    full, n = scan(h[f].values.astype(float), Gs); row = full[full.gene == gene].iloc[0]
    P.append({"subtype": s, "gene": gene, "feature": f, "n": n, "rho": row.r, "p": row.p, "q_genomewide_within_subtype": row.q})
P = pd.DataFrame(P); P.to_csv(os.path.join(OUT, "circuit_pairs_within_subtype.csv"), index=False); print(P.round(4).to_string(index=False))
