# Reproducibility archive, version 2.0.0

Code and outputs for "Subtype-Specific Imaging-Gene Associations and Pathologic Complete Response in Breast Cancer: A Longitudinal DCE-MRI, Radiogenomic, and Exploratory Mediation Analysis" (revised manuscript).

## Input data (public; not redistributed here)
Place these in the project root (scripts are run from the project root):
- `datawith4visits.csv`: the TCIA multi-feature MRI file (Multi-feature-MRI-NACT-Data.xlsx, sheet datawith4visits) exported to CSV; the first line is the sheet name, so it is read with `skiprows=1`. 384 patients, 28 features, identical cell for cell to the TCIA file.
- `clinical_data.csv`: TCIA ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx exported to CSV (985 patients).
- `GSE194040_ISPY2ResID_AgilentGeneExp_990_FrshFrzn_meanCol_geneLevel_n988.txt` (GEO GSE194040).
- `data/GSE25066_series_matrix.txt`, `data/GSE25066_family.soft`, `data/GSE20194_series_matrix.txt`, `data/GSE32646_series_matrix.txt`, `data/GSE32646_family.soft` (GEO).
- `data/acrin_ivim_features.csv` is read only to document why the IVIM analysis was removed.
- `data/estimate_gene_sets.csv`: ESTIMATE stromal and immune signatures (Yoshihara et al. 2013), included.

## Settings
- All models: RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42); StratifiedKFold(n_splits=5, shuffle=True, random_state=42). Fold assignments and out-of-fold predictions for every patient and model: `results/revision_r2/patient_level_predictions.csv` (columns cohort, Patient_ID, fold, pCR, p_<model>).
- Bootstrap CIs: 1,000 patient resamples (RandomState(42)), percentile method. Mediation: 5,000 resamples stratified by pCR.
- Permutation test: 1,000 label permutations generated from RandomState(42); full 5-fold cross-validation repeated for each; p = (1 + count of permuted AUC >= observed) / 1001.

## Scripts (run from the project root, in this order)
1. `revision_r1_external_validation.py`: clinical receptor subtypes, GSE20194 MAQC filtering and de-duplication against GSE25066 (expression and identifier), per-cohort tests, Fisher combination (two-sided and one-sided), BH across five genes. Outputs in `results/revision_r1/` including `gse20194_dedup_map.csv`, `dedup_log.json`, and `sample_lists/` (GSM identifiers for every subtype x cohort analysis).
2. `revision_r1_smote_loocv.py`: SMOTE inside training folds and leave-one-out CV.
3. `revision_r2_core_models.py preds | repeat | threshold | summarize`: matched-timepoint models, repeated CV, nested Youden thresholds, AUC/PR-AUC CIs, DeLong tests, calibration, decision curves.
4. `revision_r2_nested_rfecv.py COHORT FOLD|full|summarize`: RFECV inside outer training folds; selection frequencies.
5. `revision_r2_shap_table.py`: out-of-fold SHAP ranks and stability.
6. `revision_r2_radiogenomics.py`: pooled Level 1 and Level 2 correlations and within-subtype pair statistics.
7. `revision_r2_level5.py`: imaging vs nested genomic models with paired DeLong tests.
8. `revision_r2_mediation_temporal.py`: gene -> imaging -> pCR mediation with covariates and ESTIMATE purity.
9. `revision_r2_genomewide_validation.py`: discovery-defined gene sets and label-permutation p-values.
10. `revision_r2_exclusion.py`: included vs excluded patients with standardized mean differences.
11. `revision_r2_figures.py`, `revision_r2_fig1.py`: manuscript figures.
Permutation tests were run with `perm.py` (included as `revision_r2_permutation.py`).

## Notes
- Earlier analyses (version 1.0.0) included one run on a copy of the imaging file in which patient 111881's identifier was corrupted, which silently dropped that patient (383 patients, 89 HER2+). All version 2 results use the intact 384-patient file.
- The IVIM/diffusion analysis from version 1 was withdrawn: `ivim_extract.py` averaged whole central image slices rather than a tumor region.
