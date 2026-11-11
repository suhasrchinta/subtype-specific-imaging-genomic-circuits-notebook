# Subtype-Specific Imaging-Gene Associations and Pathologic Complete Response in Breast Cancer: Reproducibility Repository

## What this is
This repository contains the code, outputs, and annotated notebook behind:

> *Subtype-Specific Imaging–Gene Associations and Pathologic Complete Response in Breast Cancer: A Longitudinal DCE-MRI, Radiogenomic, and Exploratory Mediation Analysis*
> Suhas Chinta, Keller Collegiate Academy

Version 2.0.0 accompanies the revised version of the manuscript. It replaces version 1.0.0, which accompanied the original version under the earlier title "Subtype-Specific Imaging-to-Gene Circuits Predicting Pathologic Complete Response in Breast Cancer."

## Repository structure
```
.
|-- README.md                       this file
|-- LICENSE                         MIT license
|-- requirements.txt                Python environment for the version 2 analyses
|-- Reproducibility_Notebook.ipynb  annotated record of the version 1 pipeline, with a
|                                   "Version 2 status" note in every section
`-- revision_v2/
    |-- README.md                   run order, settings, seeds, and file descriptions
    |-- code/                       all version 2 analysis scripts
    |-- data/                       ESTIMATE gene signatures (the only bundled input)
    `-- results/                    all version 2 outputs, including fold assignments,
                                    patient-level predictions, the GSE20194
                                    de-duplication map, per-cohort sample identifiers,
                                    permutation null distributions, and figures
```

## Where to start
The authoritative results for the revised manuscript are produced by the scripts in `revision_v2/code/` and saved in `revision_v2/results/`. `revision_v2/README.md` lists the run order, the random seeds, the cross-validation settings, and what each output file contains. The notebook remains as a readable walkthrough of the version 1 pipeline; each section states whether its result was verified, corrected, superseded, or withdrawn in version 2, and Section 14 reports the version 2 results.

## Main corrections in version 2
- One version 1 run read a copy of the imaging file in which one patient's identifier was corrupted, which dropped that patient (383 instead of 384). All model results were recomputed on the verified 384-patient data.
- External validation now uses clinical receptor status in every cohort and removes the 187 GSE20194 patients duplicated in GSE25066. Only SCUBE2 replicates.
- Mediation was reformulated in temporal order (pretreatment gene, later imaging feature, pCR) with covariates and a tumor-purity adjustment, and is reported as exploratory.
- Feature selection and threshold tuning now run inside the training folds only.
- The IVIM diffusion analysis, the GDSC cell-line correlation, the 8-feature minimal panel, and the claim that MRI outperformed MammaPrint were withdrawn. The reasons are given in the notebook and in the revised manuscript.

## Requirements
Version 2 analyses were run on Python 3.10.12. Install dependencies with:
```
pip install -r requirements.txt
```
The version 1 analyses were run on Python 3.11.9 with the libraries named in the notebook. GSEA output was produced in version 1 with `gseapy==1.3.0` and is reused in version 2.

## Data
This project uses only public, de-identified research data:

- **I-SPY2 / ACRIN-6698** imaging features and clinical data, via [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)
- **GSE194040** (pretreatment gene expression), via [GEO](https://www.ncbi.nlm.nih.gov/geo/)
- **GSE25066, GSE20194, GSE32646** (external validation cohorts), via GEO
- **DepMap PRISM Repurposing** (drug viability) and expression data, via the [DepMap Portal](https://depmap.org/portal/)

Raw data files are not included because of size and data-use terms; the accession numbers above are sufficient to obtain them. `revision_v2/README.md` lists the exact file names each script expects.

## Status
Computational work for the revised manuscript is complete. Wet-lab validation was not performed; it is proposed in the paper only as future work.

## License
This repository is released under the [MIT License](LICENSE). The datasets referenced above are governed by their respective sources' terms of use.
