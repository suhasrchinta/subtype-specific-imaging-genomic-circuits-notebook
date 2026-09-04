# Subtype-Specific Imaging-Genomic Circuits — Reproducibility Notebook

## What this is
This repository contains the annotated Jupyter notebook (`Reproducibility_Notebook.ipynb`) documenting the full computational pipeline behind:

> *Subtype-Specific Imaging-to-Gene Circuits Predicting Pathologic Complete Response in Breast Cancer: A Longitudinal DCE-MRI, Radiogenomic, and Mediation-Based Analysis*
> Suhas Chinta, Keller Collegiate Academy

The notebook walks through every stage of the pipeline in order — data loading, IVIM extraction, the primary predictive model, SHAP interpretability, RFECV, GSEA pathway enrichment, radiogenomics, external validation across three independent GEO cohorts, mediation analysis, and the final significance lock — with the real code from each underlying script alongside a plain-language explanation of what it does and why, plus the exact locked result it produces.

All numbers in the notebook are checked against the final paper and match it as the golden source.

## How to read (and run) this repository
The notebook is a **reproducibility artifact, not a packaged pipeline**. Code cells are real excerpts from the project's original analysis scripts, included so a reader can see exactly how each locked result was produced — not a single script meant to be run top-to-bottom.

Running the notebook end-to-end will raise errors on several cells by design: some reference raw data files (patient-level DICOM series, `datawith4visits.csv`, `clinical_data.csv`, GEO series-matrix files) that are not included in this repository, for the size and data-use reasons described below. This is expected. Each cell is self-contained and documented well enough to be understood, checked, and adapted on its own, given access to the source data.

## Repository structure
```
.
├── README.md                      -- this file
├── LICENSE                        -- MIT license
├── requirements.txt                -- Python environment used for the analysis
└── Reproducibility_Notebook.ipynb  -- annotated pipeline walkthrough
```

## Requirements
The analysis was run locally in VS Code on Python 3.11.9. Install dependencies with:
```
pip install -r requirements.txt
```
See `requirements.txt` for the specific library versions used, including `gseapy==1.3.0` (version-pinned because pathway-name column handling changed across `gseapy` releases; see notebook §6 for the specific fix this required).

## Data
This project uses only public, de-identified research data:

- **I-SPY2 / ACRIN-6698** (imaging), via [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)
- **GSE194040** (pretreatment gene expression, discovery cohort), via [GEO](https://www.ncbi.nlm.nih.gov/geo/)
- **GSE25066, GSE20194, GSE32646** (external validation cohorts), via GEO
- **GDSC** (Genomics of Drug Sensitivity in Cancer), via the [GDSC data portal](https://www.cancerrxgene.org/)
- **DepMap/PRISM** (CRISPR dependency and drug-sensitivity data), via the [DepMap Portal](https://depmap.org/portal/)

Raw data files are not included in this repository due to size and data-use terms; the accession numbers and portal links above are sufficient to obtain them independently under each source's own access terms. Processing and QC steps (including known parsing bugs and their fixes) are documented in full inside the notebook.

## Status of analysis
All computational work is complete and locked, matching the final paper exactly. Wet-lab validation (SCUBE2 knockdown + paclitaxel sensitivity assay) was not performed as part of this study; it is proposed in the paper's Conclusion as a natural next step for functional follow-up, not as an open or pending result (see notebook §12).

## Reproducibility note
Code cells reflect the actual scripts used to generate each locked result. Where a script name is referenced but the original file was not separately preserved, this is stated explicitly in that cell rather than reconstructed silently. Every numeric result in the notebook is cross-checked against the paper as the single source of truth; any place where an earlier or superseded analysis run produced a different number is noted explicitly as superseded, rather than silently omitted.

## License
This repository is released under the [MIT License](LICENSE). The dataset accessions referenced above are governed by their respective sources' own terms of use.
