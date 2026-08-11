# Subtype-Specific Imaging-Genomic Circuits — Reproducibility Notebook

**Status:** Private repository. Will be switched to public upon Regeneron Science Talent Search (STS) submission.

## What this is

This repository contains the annotated Jupyter notebook (`STS_Reproducibility_Notebook.ipynb`) documenting the full computational pipeline behind:

> *Subtype-Specific Imaging-Genomic Circuits and Chemotherapy Response in Breast Cancer*
> Suhas Chinta, Keller Collegiate Academy

The notebook is a reproducibility artifact, not a new analysis. It walks through every stage of the pipeline in order — data loading, IVIM extraction, the primary predictive model, SHAP interpretability, RFECV, GSEA pathway enrichment, radiogenomics, external validation across three independent GEO cohorts, mediation analysis, and the final significance lock — with the real code from each underlying script alongside a plain-language explanation of what it does and why, plus the exact locked result it produces.

All numbers in the notebook are checked against the final paper and match it as the golden source.

## Data

This project uses only public, de-identified research data:

- **I-SPY2 / ACRIN-6698** (imaging), via [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)
- **GSE194040** (pretreatment gene expression, discovery cohort), via [GEO](https://www.ncbi.nlm.nih.gov/geo/)
- **GSE25066, GSE20194, GSE32646** (external validation cohorts), via GEO
- **GDSC** (Genomics of Drug Sensitivity in Cancer) and **DepMap/CCLE** (functional/expression validation)

Raw data files are not included in this repository due to size; accession numbers above are sufficient to obtain them independently. Processing and QC steps (including known parsing bugs and their fixes) are documented in full inside the notebook.

## Status of analysis

All computational work is complete and locked. One section (wet-lab validation, SCUBE2 knockdown + paclitaxel sensitivity assay at UT Southwestern) is pending and marked as a placeholder in the notebook until results are available.

## Reproducibility note

Code cells reflect the actual scripts used to generate each locked result. Where a script name is referenced but the original file was not separately preserved, this is stated explicitly in that cell rather than reconstructed silently.
