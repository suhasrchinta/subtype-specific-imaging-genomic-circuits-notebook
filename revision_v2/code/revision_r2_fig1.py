"""Version 2 Figure 1: study overview with revised numbers."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "DejaVu Sans"})
steps = [("Cohort", "I-SPY2/ACRIN-6698 via TCIA: n=384 (113 pCR)\nHR+/HER2− 162, HER2+ 90, TNBC 132; GSE194040 expression n=383"),
         ("Prediction", "Random Forest, 28 longitudinal DCE-MRI features, stratified 5-fold CV\nPooled AUC 0.799; time-matched comparison with MammaPrint"),
         ("Radiogenomics", "Imaging–expression correlations (pooled and within subtype)\nPathway context from preranked GSEA (exploratory, FDR<0.25)"),
         ("Exploratory mediation", "Pretreatment gene → later imaging feature → pCR\nAdjusted for age, arm, tumor purity; two tests, BH-corrected"),
         ("External check", "GSE25066, GSE20194 (de-duplicated), GSE32646\nClinical receptor subtypes; two-sided Fisher, BH across five genes")]
fig, ax = plt.subplots(figsize=(9, 5.4)); ax.set_xlim(0, 10); ax.set_ylim(0, len(steps) * 1.3); ax.axis("off")
for i, (h, t) in enumerate(steps):
    y = (len(steps) - 1 - i) * 1.3 + 0.15
    ax.add_patch(FancyBboxPatch((0.2, y), 9.6, 1.0, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#EAF2FA", ec="#0072B2", lw=1.4))
    ax.text(0.45, y + 0.5, h, fontsize=12, fontweight="bold", va="center", color="#0b3d66")
    ax.text(3.0, y + 0.5, t, fontsize=10.5, va="center")
    if i < len(steps) - 1: ax.annotate("", xy=(5, y - 0.27), xytext=(5, y + 0.0), arrowprops=dict(arrowstyle="-|>", color="#0072B2", lw=1.4))
fig.tight_layout(); fig.savefig("results/revision_r2/figures/fig1_overview.png", dpi=300)
