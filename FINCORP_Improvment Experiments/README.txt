FINCORP CPU EXPERIMENTS — CODE & RECORDED RESULTS
=================================================

This ZIP collects the CPU experiments previously run in this conversation.
They are exploratory severity-classification baselines, NOT the published authors' RoBERTa training runs.

Folder structure:
  code/            five tested Python scripts (including exploratory NBSVM and ensemble trials)
  outputs/         saved validation/test summaries, predictions, confusion matrices and reports
  data/FINCORP.csv original CSV from the authors' FINCORP-main.zip
  FINCORP-main.zip original repository archive for use by the existing scripts

Reported test results (accuracy / global macro-F1):
  Standard SVM:           81.49% / 58.88%
  Best balanced SVM:      82.23% / 61.52%
  Two-stage SVM:          81.86% / 61.16%
  New tuned SVM:          81.38% / 60.21%
  Severity-adjusted ensemble: 81.11% / 60.92%

These represent separate comparison runs on the same documented 60/10/30 stratified split, seed=40.
The selection was driven by validation macro-F1. None of these CPU models beats the
previously reported RoBERTa scores (which also used a different evaluation method).

RUNNING THE SCRIPTS:
  pip install numpy pandas scipy scikit-learn
  The original scripts assume /mnt/data/FINCORP-main.zip for their input and /mnt/data/
  for output. On Kaggle or locally, edit `out=Path('/mnt/data')` and the archive path
  accordingly, or adapt them to the folder where you extracted this ZIP.
  Scripts are included as executed; no claims of a rerun were made when packaging.

The results are preserved files from previous experiments; model weights were not saved.
Original research paper: Singh et al. (2024), Complaint and Severity Identification From Online Financial Content.
