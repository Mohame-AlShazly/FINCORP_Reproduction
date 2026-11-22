# FINCORP results — evidence inventory

These files document previously performed research. **No new experimental runs or invented results are included.** Raw folders are retained unchanged from the supplied files.

## Evidence

- `Reproduction_Screenshots/`: screenshots E1–E5, Exp1–Exp5, hardware and model images.
- `E1_EVIDENCE_VARIANTS.csv`: three distinct recorded E1 variants. They do **not** establish three independent experiments. The final-report E1 figures are not the same as either screenshot. Do not present them as a single reconciled run.
- `table_01_reproduction_E1_E5_SCREENSHOT_E1_VALUES.csv`: E1 from `E1.png`, E2–E5 from existing report data. The E2–E5 values require screenshot cross-check in the submitted repository. Batch-level notebook F1, **not global test macro-F1**.
- `table_01_reproduction_E1_E5_PREVIOUS_REPORT_VALUES.csv`: original summary transcription preserved for traceability; contains the different final-report E1. Do not submit competing result tables as if both are definitive.
- `table_02_published_vs_E5_REPORT_TRANSCRIPTION.csv`: descriptive paper vs E5 comparison, not matched test protocol. The paper reports **73.77%** severity accuracy, not 75.77%.
- `table_03_distilroberta_V1_V5_VALIDATION.csv`: validation comparisons, n=628, not held-out test results.
- `table_04_CPU_SVM_HELD_OUT_TEST.csv`: reported held-out SVM values, n=1,885; validate from original underlying outputs.
- `CPU_SVM_Results/`: source predictions, results and confusion matrices.
- `DistilRoBERTa_Results/`: histories, configurations, predictions and matrices; checkpoint configurations are not model weights.
- `ARCHIVED_EXPERIMENT_LOG_UNMODIFIED.md`: original dated log, including historical paths and V2 reference error (preserved as evidence).
- `EXPERIMENT_LOG_FILE_INDEX.md`: corrected, local-relative locations.
- `SOURCE_MANIFEST.csv`: original inventory (source hashes/provenance); may include superseded filenames from the prior supplement.

## Reproduction requirements not proven by this archive

The professor expects the GitHub README to provide one executable command for every reported result, installation instructions, a version-pinned environment, tests and a reduced run if necessary. A `results/` archive cannot substitute for those executable files.

## Evidence cautions

The original reproduction notebook may reuse trainable encoder state across experiment blocks, and scores batches rather than pooling test predictions. CPU SVM results and DistilRoBERTa validation results are **different evaluation protocols**; do not rank them as though they used the same held-out test set. The E1 conflict must be resolved against the actual saved notebook before final-report submission.
