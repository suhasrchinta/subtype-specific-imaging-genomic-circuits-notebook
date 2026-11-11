"""Version 2 figures (all generated from results/revision_r1 and results/revision_r2)."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, roc_auc_score
R2 = "results/revision_r2"; FIG = os.path.join(R2, "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5, "axes.spines.top": False, "axes.spines.right": False})
OI = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#999999"]
P = pd.read_csv(os.path.join(R2, "patient_level_predictions.csv"))

# Figure 3: time-matched ROC, pooled
pp = P[P.cohort == "Pooled"]; y = pp.pCR.values
fig, ax = plt.subplots(figsize=(6.2, 5.4))
for col, lab, c, ls in [("p_MP_only", "MammaPrint High1/High2 (baseline)", OI[1], "-"), ("p_A_baseline", "Baseline MRI, T0 (4 features)", OI[7], "--"),
                        ("p_T1_through_T1", "MRI through T1 (12 features)", OI[2], "-"), ("p_T1_plus_MP", "MRI through T1 + MammaPrint", OI[4], "-"),
                        ("p_C_full", "Full-course MRI, T0 to T3 (28 features)", OI[5], "-")]:
    f, t, _ = roc_curve(y, pp[col]); ax.plot(f, t, color=c, ls=ls, lw=2, label=f"{lab}: AUC {roc_auc_score(y, pp[col]):.3f}")
ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1, ls=":"); ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
ax.legend(loc="lower right", frameon=False); ax.set_aspect("equal")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_time_matched_roc.png"), dpi=300); plt.close()

# Figure 4: calibration (A) and decision curve (B)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
cal = pd.read_csv(os.path.join(R2, "calibration.csv")).set_index("cohort")
for (name, c) in zip(["Pooled", "HR+/HER2-", "HER2+", "TNBC"], [OI[0], OI[1], OI[2], OI[3]]):
    q = P[P.cohort == name]; bins = pd.qcut(q.p_C_full, 5 if name != "Pooled" else 10, duplicates="drop")
    g = q.groupby(bins, observed=True).agg(pred=("p_C_full", "mean"), obs=("pCR", "mean"))
    lab = name.replace("HER2-", "HER2−")
    axs[0].plot(g.pred, g.obs, "o-", color=c, lw=1.6, ms=5, label=f"{lab}: slope {cal.loc[name,'cal_slope']:.2f}, intercept {cal.loc[name,'cal_intercept']:.2f}")
axs[0].plot([0, 1], [0, 1], color="#bbbbbb", ls=":"); axs[0].set_xlim(0, 1); axs[0].set_ylim(0, 1)
axs[0].set_xlabel("Mean predicted probability of pCR (bin)"); axs[0].set_ylabel("Observed pCR proportion"); axs[0].legend(frameon=False, loc="upper left", fontsize=9)
axs[0].set_title("A) Calibration (out-of-fold, full-course model)", loc="left")
dca = pd.read_csv(os.path.join(R2, "decision_curve.csv")); d = dca[dca.cohort == "Pooled"]
axs[1].plot(d.threshold, d.nb_model, color=OI[5], lw=2, label="Full-course MRI model"); axs[1].plot(d.threshold, d.nb_treat_all, color=OI[4], lw=1.6, ls="--", label="Assume all achieve pCR")
axs[1].axhline(0, color="black", lw=1, ls=":", label="Assume none achieve pCR"); axs[1].set_ylim(-0.1, 0.3); axs[1].set_xlim(0.1, 0.6)
axs[1].set_xlabel("Threshold probability of pCR"); axs[1].set_ylabel("Net benefit"); axs[1].legend(frameon=False); axs[1].set_title("B) Decision curve, pooled cohort (n=384)", loc="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_calibration_dca.png"), dpi=300); plt.close()

# Figure 5: tested imaging-gene pairs (A, B) and external gene-outcome replication (C)
d = pd.read_csv("datawith4visits.csv", skiprows=1).rename(columns={"CLINICAL-TRIAL-SUBJECT-ID": "Patient_ID"})
c = pd.read_csv("clinical_data.csv"); df = d.merge(c, on="Patient_ID"); df["Patient_ID"] = df.Patient_ID.astype(str)
expr = pd.read_csv("GSE194040_ISPY2ResID_AgilentGeneExp_990_FrshFrzn_meanCol_geneLevel_n988.txt", sep="\t", index_col=0, usecols=lambda s: True)
expr.columns = expr.columns.astype(str); E = expr.loc[["SCUBE2", "ERBB2", "GRB7"]].T; E.index.name = "Patient_ID"
df = df.merge(E.reset_index(), on="Patient_ID")
z = lambda v: (v - v.mean()) / v.std(ddof=0)
pairs = pd.read_csv(os.path.join(R2, "circuit_pairs_within_subtype.csv"))
fig = plt.figure(figsize=(11.5, 8.2)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
ax = fig.add_subplot(gs[0, 0]); h = df[df.HER2 == 1].copy(); h["amp"] = z(h[["ERBB2", "GRB7"]].apply(z).mean(1))
for v, col, lab in [(0, OI[7], "No pCR"), (1, OI[5], "pCR")]:
    s = h[h.pCR == v]; ax.scatter(s.amp, s.FTV_pch_T0_T1, s=22, color=col, alpha=0.8, label=lab, edgecolor="none")
r = pairs[(pairs.gene == "ERBB2") & (pairs.feature == "FTV_pch_T0_T1")].iloc[0]; r2 = pairs[(pairs.gene == "GRB7") & (pairs.feature == "FTV_pch_T0_T1")].iloc[0]
ax.set_xlabel("ERBB2/GRB7 amplicon score (z)"); ax.set_ylabel("FTV percent change, T0 to T1 (%)")
ax.set_title(f"A) HER2+ (n={len(h)}): ERBB2 ρ={r.rho:.2f}, q={r.q_genomewide_within_subtype:.3f};\n     GRB7 ρ={r2.rho:.2f}, q={r2.q_genomewide_within_subtype:.2f}", loc="left", fontsize=11)
ax.legend(frameon=False, loc="upper right")
ax = fig.add_subplot(gs[0, 1]); h = df[(df.HER2 == 0) & (df.HR == 1)].dropna(subset=["SCUBE2"])
for v, col, lab in [(0, OI[7], "No pCR"), (1, OI[5], "pCR")]:
    s = h[h.pCR == v]; ax.scatter(s.SCUBE2, s.LD_T3, s=22, color=col, alpha=0.8, label=lab, edgecolor="none")
r = pairs[(pairs.gene == "SCUBE2") & (pairs.feature == "LD_T3")].iloc[0]
ax.set_xlabel("SCUBE2 expression (log2, pretreatment)"); ax.set_ylabel("Longest diameter at T3 (cm)")
ax.set_title(f"B) HR+/HER2− (n={len(h)}): SCUBE2 ρ={r.rho:.2f}, q={r.q_genomewide_within_subtype:.2f}", loc="left", fontsize=11); ax.legend(frameon=False)
ax = fig.add_subplot(gs[1, :]); V = pd.read_csv("results/revision_r1/external_validation_per_cohort.csv"); V = V[(V.set == "primary") & V.rho.notna()]
order = [("SCUBE2", "HR+/HER2−"), ("ERBB2", "HER2+"), ("GRB7", "HER2+"), ("FLT1", "TNBC"), ("ANGPT2", "TNBC")]
cols = {"GSE25066": OI[4], "GSE20194": OI[1], "GSE32646": OI[3]}; ypos = 0; ticks = []
for gname, sub in order:
    rows = V[V.gene == gname]
    for _, rr in rows.iterrows():
        ax.errorbar(rr.rho, ypos, xerr=[[rr.rho - rr.rho_lo], [rr.rho_hi - rr.rho]], fmt="o", color=cols[rr.cohort], ms=6, capsize=3, lw=1.5)
        ax.text(0.62, ypos, f"{rr.cohort}  n={int(rr.n)} ({int(rr.n_pcr)} pCR)  p={rr.p_two_sided:.3g}", va="center", fontsize=9)
        ticks.append((ypos, f"{gname} ({sub})")); ypos -= 1
    ypos -= 0.6
ax.set_yticks([t[0] for t in ticks]); ax.set_yticklabels([t[1] for t in ticks]); ax.axvline(0, color="black", lw=1)
ax.set_xlim(-0.65, 1.15); ax.set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]); ax.set_xlabel("Spearman ρ between pretreatment expression and pCR (95% CI)")
ax.set_title("C) External gene-outcome replication in de-duplicated cohorts (clinical receptor subtypes)", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_imaging_gene_evidence.png"), dpi=300); plt.close()

# Figure 6: mediation bootstrap distributions
M = pd.read_csv(os.path.join(R2, "mediation_temporal.csv")); M = M[M.model == "adjusted"]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (lab, key, ttl) in zip(axs, [("SCUBE2", "SCUBE2", "A) HR+/HER2−: SCUBE2 → LD at T3 → pCR"), ("ERBB2/GRB7 amplicon", "ERBB2", "B) HER2+: amplicon → FTV change T0 to T1 → pCR")]):
    b = np.load(os.path.join(R2, f"mediation_boot_{key}_adjusted.npy")); r = M[M.test == lab].iloc[0]
    ax.hist(b, bins=80, color=OI[2], edgecolor="white", lw=0.3); ax.axvline(0, color="black", lw=1.2)
    ax.axvline(r.ci_lo, color=OI[5], ls="--", lw=1.4); ax.axvline(r.ci_hi, color=OI[5], ls="--", lw=1.4); ax.axvline(r.indirect_ab, color=OI[0], lw=2)
    ax.set_xlabel("Indirect effect a×b (log-odds of pCR per SD of expression)"); ax.set_ylabel("Bootstrap resamples (of 5,000)")
    ax.set_title(ttl, loc="left", fontsize=11)
    left = key == "SCUBE2"
    ax.text(0.03 if left else 0.98, 0.95, f"n={int(r.n)} ({int(r.n_pcr)} pCR)\nestimate {r.indirect_ab:.3f}\n95% CI {r.ci_lo:.3f} to {r.ci_hi:.3f}\np={r.p_two_sided:.3f}, q={r.q_BH:.3f}", transform=ax.transAxes, ha="left" if left else "right", va="top", fontsize=9.5, bbox=dict(fc="white", ec="none", alpha=0.85))
    lo, hi = np.percentile(b, [0.5, 99.5]); ax.set_xlim(lo, hi)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_mediation_bootstrap.png"), dpi=300); plt.close()
print("ok", os.listdir(FIG))
