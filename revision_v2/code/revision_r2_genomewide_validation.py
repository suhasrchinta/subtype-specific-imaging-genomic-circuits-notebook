"""
Version 2: genome-wide concordance between the discovery pCR signature and
each de-duplicated validation cohort, with one discovery-defined rule for every
subtype x cohort combination.
Discovery (GSE194040 overlap with the imaging cohort): per subtype, per gene, signed
difference in mean expression (pCR minus non-pCR) and Mann-Whitney BH q.
Validation: probes mapped to gene symbols from the platform annotation (GPL96 via the
GSE25066 SOFT file, GPL570 via the GSE32646 SOFT file), averaged per gene; fold change =
mean pCR minus mean RD (log2 scale). Clinical receptor subtypes; GSE20194 restricted to
MAQC arrays not duplicated in GSE25066 (results/revision_r1/gse20194_dedup_map.csv).
Tiers (all defined in discovery only): all matched genes; discovery q<0.05; discovery top-200
by |difference|. Statistic: Spearman rho between discovery and validation differences.
P-value: 1000 permutations of the validation pCR labels (preserves gene-gene correlation).
"""
import os, numpy as np, pandas as pd
from scipy import stats
OUT = os.path.join(os.getcwd(), "results", "revision_r2")
exec(open("code/revision_r1_external_validation.py").read().split("# ---- load")[0].split('ROOT = ')[0])
load_src = open("code/revision_r1_external_validation.py").read()
exec(load_src[load_src.index("def load(path):"):load_src.index("def gpl570_probes")])
exec(load_src[load_src.index("def subtype(er, pr, her2):"):load_src.index("def rank_corr")])
def platform_map(soft):
    hdr = None; inside = False; rows = []
    for line in open(soft, encoding="utf-8", errors="replace"):
        if line.startswith("!platform_table_begin"): inside = True; continue
        if line.startswith("!platform_table_end"): break
        if not inside: continue
        p = line.rstrip("\n").split("\t")
        if hdr is None:
            hdr = [x.strip().lower() for x in p]; i_id = hdr.index("id"); i_s = [i for i, x in enumerate(hdr) if "gene symbol" in x or x == "symbol"][0]; continue
        if len(p) > i_s and p[i_s].strip() and "///" not in p[i_s]: rows.append((p[i_id].strip(), p[i_s].strip()))
    return pd.DataFrame(rows, columns=["probe", "gene"]).drop_duplicates("probe").set_index("probe").gene
def to_genes(X, pm):
    X = X.loc[X.index.intersection(pm.index)]; return X.groupby(pm.loc[X.index].values).mean()
# discovery
d = pd.read_csv("datawith4visits.csv", skiprows=1).rename(columns={"CLINICAL-TRIAL-SUBJECT-ID": "Patient_ID"})
c = pd.read_csv("clinical_data.csv"); df = d.merge(c, on="Patient_ID"); df["Patient_ID"] = df.Patient_ID.astype(str)
df["sub"] = np.where(df.HER2 == 1, "HER2+", np.where(df.HR == 1, "HR+/HER2-", "TNBC"))
expr = pd.read_csv("GSE194040_ISPY2ResID_AgilentGeneExp_990_FrshFrzn_meanCol_geneLevel_n988.txt", sep="\t", index_col=0)
expr.columns = expr.columns.astype(str); expr = expr[~expr.index.duplicated()]
m = df[df.Patient_ID.isin(expr.columns)]
def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = np.empty(len(p))
    q[o] = np.minimum(1, np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]); return q
DISC = {}
for s in ["HR+/HER2-", "HER2+", "TNBC"]:
    h = m[m["sub"] == s]; E = expr[h.Patient_ID]; y = h.pCR.values.astype(bool)
    diff = E.loc[:, y].mean(1) - E.loc[:, ~y].mean(1)
    p = np.array([stats.mannwhitneyu(r[y][~np.isnan(r[y])], r[~y][~np.isnan(r[~y])]).pvalue if np.isfinite(r).sum() > 10 else np.nan for r in E.values])
    t = pd.DataFrame({"diff": diff.values, "p": p}, index=E.index).dropna(); t["q"] = bh(t.p.values); DISC[s] = t
    print("discovery", s, "n", len(h), "genes", len(t), "q<0.05:", int((t.q < 0.05).sum()), flush=True)
# validation cohorts
pm96 = platform_map("data/GSE25066_family.soft"); pm570 = platform_map("data/GSE32646_family.soft")
c25, X25 = load("data/GSE25066_series_matrix.txt"); c20, X20 = load("data/GSE20194_series_matrix.txt"); c32, X32 = load("data/GSE32646_series_matrix.txt")
pos = lambda s: {"positive": "P", "negative": "N"}.get(str(s).strip().lower(), str(s).strip().upper())
c25["sub"] = [subtype(a, b, cc) for a, b, cc in zip(c25.er_status_ihc, c25.pr_status_ihc, c25.her2_status)]; c25["pcr"] = c25.pathologic_response_pcr_rd.map({"pCR": 1, "RD": 0})
c20["sub"] = [subtype(a, b, cc) for a, b, cc in zip(c20.er_status, c20.pr_status, c20["her2 status"])]; c20["pcr"] = c20.pcr_vs_rd.map({"pCR": 1, "RD": 0})
c32["sub"] = [subtype(pos(a), pos(b), pos(cc)) for a, b, cc in zip(c32["er status ihc"], c32["pr status ihc"], c32["her2 status fish"])]; c32["pcr"] = c32["pathologic response pcr ncr"].map({"pCR": 1, "nCR": 0})
dm = pd.read_csv("results/revision_r1/gse20194_dedup_map.csv", index_col=0); keep20 = dm[dm.maqc & ~dm.dup].index
V = {"GSE25066": (c25, to_genes(X25, pm96)), "GSE20194": (c20.loc[keep20], to_genes(X20, pm96)), "GSE32646": (c32, to_genes(X32, pm570))}
rows = []
for s in ["HR+/HER2-", "HER2+", "TNBC"]:
    D = DISC[s]; tiers = {"all_matched": D.index, "discovery_q_lt_0.05": D.index[D.q < 0.05], "discovery_top200": D["diff"].abs().nlargest(200).index}
    for coh, (cl, Xg) in V.items():
        h = cl[(cl["sub"] == s) & cl.pcr.notna()]; n, e = len(h), int(h.pcr.sum())
        if n < 15 or e < 5 or n - e < 5:
            rows.append({"subtype": s, "cohort": coh, "n": n, "n_pcr": e, "note": "not testable"}); continue
        Xv = Xg[h.index]; y = h.pcr.values.astype(bool); rng = np.random.RandomState(42)
        for tname, genes in tiers.items():
            g = [x for x in genes if x in Xv.index]
            if len(g) < 10: rows.append({"subtype": s, "cohort": coh, "tier": tname, "n": n, "n_pcr": e, "genes": len(g), "note": "too few genes"}); continue
            A = Xv.loc[g].values; dv = D.loc[g, "diff"].values
            fc = A[:, y].mean(1) - A[:, ~y].mean(1); rho = stats.spearmanr(dv, fc)[0]
            null = []
            for _ in range(1000):
                yp = rng.permutation(y); null.append(stats.spearmanr(dv, A[:, yp].mean(1) - A[:, ~yp].mean(1))[0])
            null = np.array(null)
            rows.append({"subtype": s, "cohort": coh, "tier": tname, "n": n, "n_pcr": e, "genes": len(g), "rho": rho,
                         "p_perm_two_sided": (1 + (np.abs(null) >= abs(rho)).sum()) / 1001, "p_naive_spearman": stats.spearmanr(dv, fc)[1]})
        print(s, coh, "done", flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "genomewide_concordance.csv"), index=False)
pd.set_option("display.width", 220); print(R.round(4).to_string(index=False))
