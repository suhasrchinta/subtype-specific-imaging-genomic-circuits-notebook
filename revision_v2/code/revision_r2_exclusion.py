"""
Version 2: included (n=384) vs excluded patients, using every patient in the
public TCIA I-SPY2 clinical table (n=985; pCR recorded for all 985). Reports counts,
standardized mean differences (SMD; |SMD|<0.1 negligible) and tests, overall and by subtype.
"""
import os, numpy as np, pandas as pd
from scipy import stats
OUT = os.path.join(os.getcwd(), "results", "revision_r2")
d = pd.read_csv("datawith4visits.csv", skiprows=1).rename(columns={"CLINICAL-TRIAL-SUBJECT-ID": "Patient_ID"})
c = pd.read_csv("clinical_data.csv"); c["included"] = c.Patient_ID.isin(d.Patient_ID).astype(int)
c["subtype"] = np.where(c.HER2 == 1, "HER2+", np.where(c.HR == 1, "HR+/HER2-", "TNBC"))
c["postmenopausal"] = c.menopausal_status.astype(str).str.lower().str.startswith("post").astype(int)
c["white"] = (c.Race.astype(str).str.lower() == "white").astype(int)
c["experimental_arm"] = (~c.Arm.isin(["Paclitaxel", "Paclitaxel + Trastuzumab"])).astype(int)
rows = []
def smd_cont(a, b): return (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
def smd_bin(a, b): p1, p2 = a.mean(), b.mean(); return (p1 - p2) / np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
I, E = c[c.included == 1], c[c.included == 0]
rows.append({"variable": "n", "included": len(I), "excluded": len(E)})
a, b = I.Age_at_Screening.dropna(), E.Age_at_Screening.dropna()
rows.append({"variable": "age, mean (SD)", "included": f"{a.mean():.1f} ({a.std():.1f})", "excluded": f"{b.mean():.1f} ({b.std():.1f})",
             "smd": smd_cont(a, b), "p": stats.mannwhitneyu(a, b).pvalue})
for v, lab in [("pCR", "pCR"), ("HR", "HR-positive"), ("HER2", "HER2-positive"), ("MP", "MammaPrint High2"),
               ("postmenopausal", "postmenopausal"), ("white", "race: White"), ("experimental_arm", "experimental arm")]:
    t = pd.crosstab(c.included, c[v]); p = stats.chi2_contingency(t)[1]
    rows.append({"variable": lab, "included": f"{int(I[v].sum())} ({100*I[v].mean():.1f}%)", "excluded": f"{int(E[v].sum())} ({100*E[v].mean():.1f}%)",
                 "smd": smd_bin(I[v], E[v]), "p": p})
t = pd.crosstab(c.included, c.subtype); rows.append({"variable": "subtype (3 groups)", "included": "; ".join(f"{k} {v}" for k, v in t.loc[1].items()),
             "excluded": "; ".join(f"{k} {v}" for k, v in t.loc[0].items()), "p": stats.chi2_contingency(t)[1]})
for s in ["HR+/HER2-", "HER2+", "TNBC"]:
    i, e = I[I.subtype == s].pCR, E[E.subtype == s].pCR
    rows.append({"variable": f"pCR in {s}", "included": f"{int(i.sum())}/{len(i)} ({100*i.mean():.1f}%)", "excluded": f"{int(e.sum())}/{len(e)} ({100*e.mean():.1f}%)",
                 "smd": smd_bin(i, e), "p": stats.chi2_contingency(pd.crosstab(np.r_[np.ones(len(i)), np.zeros(len(e))], np.r_[i, e]))[1]})
R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "exclusion_comparison.csv"), index=False); print(R.round(3).to_string(index=False))
