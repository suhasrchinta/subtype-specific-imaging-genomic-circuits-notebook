"""
Version 2: external validation rebuilt.

Changes from validate_gse*.py + combined_pvalue_analysis.py:
  1. Clinical receptor status everywhere. GSE25066 previously used the
     mRNA-based esr1_status / erbb2_status calls; now er_status_ihc,
     pr_status_ihc, her2_status (IHC/FISH). Indeterminate/NA excluded.
  2. GSE20194 restricted to the 230 MAQC training+validation arrays
     (drops 29 MDA_R replicate arrays and 19 MAQC_Q not-used arrays).
  3. Cross-cohort de-duplication: every GSE20194 / GSE32646 array is compared
     with every GSE25066 array by whole-profile Spearman correlation over the
     shared GPL96 probes, and by MDACC patient identifier. Arrays matching a
     GSE25066 sample are removed from GSE20194 (GSE25066 keeps them).
  4. Two-sided tests are primary. Fisher's method is reported on the
     two-sided p-values, and separately on direction-converted one-sided
     p-values (p/2 if observed direction matches expected, 1-p/2 otherwise),
     with BH correction across the five genes for each.
  5. Every subtype x cohort sample list is written out (GSM IDs).
Run: python3 code/revision_r1_external_validation.py
"""
import os, re, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = os.path.expanduser("~/Documents/research project")
if not os.path.isdir(ROOT):  # allow running from the project folder itself
    ROOT = os.getcwd()
DATA = os.path.join(ROOT, "data"); OUT = os.path.join(ROOT, "results", "revision_r1")
os.makedirs(OUT, exist_ok=True); os.makedirs(os.path.join(OUT, "sample_lists"), exist_ok=True)

GPL96 = {"SCUBE2": ["219197_s_at"], "ERBB2": ["210930_s_at", "216836_s_at"],
         "GRB7": ["210761_s_at"], "FLT1": ["204406_at", "210287_s_at", "222033_s_at"],
         "ANGPT2": ["205572_at", "211148_s_at"]}
TESTS = [("HR+/HER2-", "SCUBE2", -1), ("HER2+", "ERBB2", +1), ("HER2+", "GRB7", +1),
         ("TNBC", "FLT1", -1), ("TNBC", "ANGPT2", -1)]
DUP_RHO = 0.9556   # 99th pct of non-matching best correlations + 0.02 (see dedup output)

def load(path):
    r, ch, hdr, data, idx, inmat = {}, [], None, [], [], False
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("!series_matrix_table_begin"): inmat = True; continue
        if line.startswith("!series_matrix_table_end"): break
        if inmat:
            f = line.rstrip("\n").split("\t")
            if hdr is None: hdr = [x.strip('"') for x in f[1:]]; continue
            idx.append(f[0].strip('"'))
            data.append([float(x) if x not in ("", "null", "NA") else np.nan for x in f[1:]])
        elif line.startswith("!Sample_"):
            t = line.split("\t")[0]; v = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            if t == "!Sample_characteristics_ch1": ch.append(v)
            else: r.setdefault(t, v)
    ids = r["!Sample_geo_accession"]; cl = pd.DataFrame(index=ids)
    for v in ch:
        for i, s in enumerate(v):
            if ":" in s:
                k, val = s.split(":", 1); cl.loc[ids[i], k.strip().lower()] = val.strip()
    for t, name in (("!Sample_title", "title"), ("!Sample_source_name_ch1", "source"),
                    ("!Sample_description", "description")):
        if t in r: cl[name] = r[t]
    X = pd.DataFrame(data, index=idx, columns=hdr)
    if np.nanmax(X.values) > 100: X = np.log2(X.clip(lower=1))
    return cl, X

def gpl570_probes(soft):
    g2p = {g: [] for g in GPL96}; hdr = None; inside = False
    for line in open(soft, encoding="utf-8", errors="replace"):
        if line.startswith("!platform_table_begin"): inside = True; hdr = None; continue
        if line.startswith("!platform_table_end"): break
        if not inside: continue
        p = line.rstrip("\n").split("\t")
        if hdr is None:
            hdr = [c.strip().lower() for c in p]; i_id = hdr.index("id")
            i_sym = [i for i, c in enumerate(hdr) if "gene symbol" in c or c == "symbol"][0]; continue
        if len(p) <= max(i_id, i_sym): continue
        syms = [s.strip() for s in p[i_sym].split("///")]
        for g in g2p:
            if g in syms: g2p[g].append(p[i_id].strip())
    return g2p

def subtype(er, pr, her2):
    if her2 == "P": return "HER2+"
    if her2 != "N" or er not in ("P", "N") or pr not in ("P", "N"): return None
    return "TNBC" if (er == "N" and pr == "N") else "HR+/HER2-"

def rank_corr(A, B):
    Ra = A.rank(); Rb = B.rank()
    Ra = (Ra - Ra.mean()) / Ra.std(); Rb = (Rb - Rb.mean()) / Rb.std()
    return pd.DataFrame(Rb.values.T @ Ra.values / (len(Ra) - 1), index=B.columns, columns=A.columns)

# ---- load
c25, X25 = load(os.path.join(DATA, "GSE25066_series_matrix.txt"))
c20, X20 = load(os.path.join(DATA, "GSE20194_series_matrix.txt"))
c32, X32 = load(os.path.join(DATA, "GSE32646_series_matrix.txt"))
p570 = gpl570_probes(os.path.join(DATA, "GSE32646_family.soft"))

pos = lambda s: {"positive": "P", "negative": "N"}.get(str(s).strip().lower(), str(s).strip().upper())
c25["sub"] = [subtype(a, b, c) for a, b, c in zip(c25.er_status_ihc, c25.pr_status_ihc, c25.her2_status)]
c25["pcr"] = c25.pathologic_response_pcr_rd.map({"pCR": 1, "RD": 0})
c20["sub"] = [subtype(a, b, c) for a, b, c in zip(c20.er_status, c20.pr_status, c20["her2 status"])]
c20["pcr"] = c20.pcr_vs_rd.map({"pCR": 1, "RD": 0})
c32["sub"] = [subtype(pos(a), pos(b), pos(c)) for a, b, c in
              zip(c32["er status ihc"], c32["pr status ihc"], c32["her2 status fish"])]
c32["pcr"] = c32["pathologic response pcr ncr"].map({"pCR": 1, "nCR": 0})

# ---- GSE20194: MAQC arrays only
c20["maqc"] = c20.description.str.contains("MAQC_T|MAQC_V", regex=True)
log = {"GSE20194_arrays_total": len(c20), "GSE20194_MAQC_T_V": int(c20.maqc.sum()),
       "GSE20194_MDA_R_replicates": int(c20.description.str.contains("MDA_R").sum()),
       "GSE20194_MAQC_Q_not_used": int(c20.description.str.contains("MAQC_Q").sum())}

# ---- de-duplication against GSE25066 (expression + identifier)
common = X25.index.intersection(X20.index)
C20 = rank_corr(X25.loc[common], X20.loc[common])
c20["best25"] = C20.idxmax(1); c20["best_rho"] = C20.max(1)
c20["mid"] = c20.title.str.extract(r"_(M\d+)$")[0]
mid25 = c25.source.str.extract(r"MDACC-(M\d+)$")[0].dropna()
id_map = pd.Series(mid25.index, index=mid25.values)
id_map = id_map[~id_map.index.duplicated()]
c20["id25"] = c20.mid.map(id_map)
c20["dup_expr"] = c20.best_rho > DUP_RHO; c20["dup_id"] = c20.id25.notna()
c20["dup"] = c20.dup_expr | c20.dup_id
m = c20[c20.maqc]
log.update({"GSE20194_MAQC_dup_by_expression": int(m.dup_expr.sum()),
            "GSE20194_MAQC_dup_by_identifier": int(m.dup_id.sum()),
            "GSE20194_MAQC_dup_either": int(m.dup.sum()),
            "GSE20194_MAQC_id_and_expr_agree": int((m.dup_id & m.dup_expr & (m.id25 == m.best25)).sum()),
            "GSE20194_MAQC_independent": int((~m.dup).sum())})
log["GSE20194_MAQC_by_subtype_dup"] = pd.crosstab(m["sub"].fillna("unclassified"), m.dup).to_dict()

common32 = X25.index.intersection(X32.index)
C32 = rank_corr(X25.loc[common32], X32.loc[common32])
C32b = rank_corr(X20.loc[common32], X32.loc[common32])
log["GSE32646_max_rho_vs_GSE25066"] = round(float(C32.max(1).max()), 4)
log["GSE32646_max_rho_vs_GSE20194"] = round(float(C32b.max(1).max()), 4)
c20[["title", "source", "description", "sub", "pcr", "maqc", "best25", "best_rho",
     "mid", "id25", "dup_expr", "dup_id", "dup"]].to_csv(os.path.join(OUT, "gse20194_dedup_map.csv"))

# ---- gene expression
def genes(X, probes):
    return pd.DataFrame({g: X.loc[[p for p in ps if p in X.index]].mean() for g, ps in probes.items()})
G = {"GSE25066": genes(X25, GPL96), "GSE20194": genes(X20, GPL96), "GSE32646": genes(X32, p570)}
log["GSE32646_probes"] = p570
CL = {"GSE25066": c25, "GSE20194": c20[c20.maqc & ~c20.dup], "GSE32646": c32}
CL_FULL20 = c20[c20.maqc]

def test(cl, ge, sub, gene, cohort, tag):
    d = cl[(cl["sub"] == sub) & cl.pcr.notna()]
    x = ge.loc[d.index, gene]; y = d.pcr.astype(int)
    pd.Series(d.index).to_csv(os.path.join(OUT, "sample_lists", f"{cohort}_{tag}_{sub.replace('/','-').replace('+','pos').replace('-','neg')}.txt"), index=False, header=False)
    n, e = len(y), int(y.sum())
    row = {"subtype": sub, "gene": gene, "cohort": cohort, "set": tag, "n": n, "n_pcr": e}
    if n < 15 or e < 5 or n - e < 5:
        row["note"] = "not testable (n<15 or <5 per outcome group)"; return row
    rho, p = stats.spearmanr(x, y); z = np.arctanh(rho); se = 1 / np.sqrt(n - 3)
    auc = roc_auc_score(y, x)
    row.update({"rho": rho, "rho_lo": np.tanh(z - 1.96 * se), "rho_hi": np.tanh(z + 1.96 * se),
                "p_two_sided": p, "mwu_p": stats.mannwhitneyu(x[y == 1], x[y == 0]).pvalue,
                "auc_raw": auc})
    return row

rows = []
for sub, gene, exp in TESTS:
    for cohort in ("GSE25066", "GSE20194", "GSE32646"):
        rows.append({**test(CL[cohort], G[cohort], sub, gene, cohort, "primary"), "expected": exp})
    rows.append({**test(CL_FULL20, G["GSE20194"], sub, gene, "GSE20194", "all230_incl_dups"), "expected": exp})
R = pd.DataFrame(rows)

def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    q = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]; out = np.empty(m); out[o] = np.minimum(q, 1); return out

comb = []
P = R[(R.set == "primary") & R.p_two_sided.notna()]
for sub, gene, exp in TESTS:
    d = P[(P.gene == gene) & (P["subtype"] == sub)]
    p2 = d.p_two_sided.values
    p1 = np.where(np.sign(d.rho.values) == exp, p2 / 2, 1 - p2 / 2)
    comb.append({"subtype": sub, "gene": gene, "cohorts_tested": ";".join(d.cohort),
                 "k": len(d), "dir_consistent": int((np.sign(d.rho) == exp).sum()),
                 "fisher_two_sided": stats.combine_pvalues(p2, "fisher")[1] if len(d) >= 2 else np.nan,
                 "fisher_one_sided": stats.combine_pvalues(p1, "fisher")[1] if len(d) >= 2 else np.nan,
                 "sign_test_p": stats.binomtest(int((np.sign(d.rho) == exp).sum()), len(d), 0.5, "greater").pvalue if len(d) else np.nan})
Cb = pd.DataFrame(comb)
ok = Cb.fisher_two_sided.notna()
Cb.loc[ok, "q_two_sided_BH5"] = bh(Cb.loc[ok, "fisher_two_sided"])
Cb.loc[ok, "q_one_sided_BH5"] = bh(Cb.loc[ok, "fisher_one_sided"])

R.to_csv(os.path.join(OUT, "external_validation_per_cohort.csv"), index=False)
Cb.to_csv(os.path.join(OUT, "external_validation_combined.csv"), index=False)
json.dump(log, open(os.path.join(OUT, "dedup_log.json"), "w"), indent=1, default=str)
pd.set_option("display.width", 220)
print(json.dumps(log, indent=1, default=str))
print(R.round(4).to_string(index=False)); print(Cb.round(5).to_string(index=False))
