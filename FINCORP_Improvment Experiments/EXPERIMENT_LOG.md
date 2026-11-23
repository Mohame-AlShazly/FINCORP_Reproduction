# FINCORP — Reproduction and Improvement Log



## Dataset Verification and Class Distribution

**Date:** 2026-11-11 06:25

**Dataset:** FINCORP.csv  
**Total records:** 6,281  
**Columns:** Domain, Complaint/ Opinion, Complaint Label, Severity level, Sentiment, Emotion

### Severity Class Distribution

| Severity Class | Records | Percentage |
|---|---:|---:|
| Non-Complaint | 3,132 | 49.86% |
| No explicit reproach | 1,548 | 24.65% |
| Blame | 875 | 13.93% |
| Disapproval | 604 | 9.62% |
| Accusation | 122 | 1.94% |

### Figure 1 — Severity Label Distribution

**File:** `outputs/severity_label_distribution.png`

![FINCORP Severity Label Distribution](outputs/severity_label_distribution.png)

**Academic caption:** Figure 1. Distribution of severity
labels in the FINCORP dataset (N = 6,281). The pronounced
class imbalance, particularly the underrepresentation of the
Accusation category, motivates the investigation of class-aware
training strategies for severity classification.

### Research Implications

- The dataset contains five severity classes.
- Accusation is the smallest class, with 122 examples.
- Class imbalance may contribute to lower macro-F1 despite high accuracy.
- Class-weighted loss and focal loss will be evaluated as possible improvements.
- All class weights must be calculated using training data only.
- Improvement must be measured on a fixed held-out test set.


## DistilRoBERTa weighted CE v1
- Best epoch: 5
- Validation macro-F1: 0.6002
- Validation accuracy: 0.7962
- Training history: `outputs/distilroberta_weighted_v1/training_history.csv`
- Best model: `outputs/distilroberta_weighted_v1/best_model/`
- Test set: not evaluated

## DistilRoBERTa unweighted CE v2
- Best epoch: 4
- Validation macro-F1: 0.6026
- Validation accuracy: 0.8344
- Training history: `Tests & Reproduction Results/DistilRoBERTa_Results/Training_History/distilroberta_unweighted_v2/training_history.csv`
- Best model: `Tests & Reproduction Results/DistilRoBERTa_Results/Evaluation_Metrics/distilroberta_unweighted_v2/best_model/ (configuration only; model weights not uploaded)`
- Test set: not evaluated

## Experiment V3 — Accusation Threshold Adjustment
- Model: Saved unweighted DistilRoBERTa V2
- Training: None; inference only
- Baseline validation accuracy: 0.8344
- Baseline validation macro-F1: 0.6026
- Selected threshold: 0.050
- Adjusted validation accuracy: 0.8217
- Adjusted validation macro-F1: 0.6261
- Test set: Not evaluated
- Note: Threshold tuned on validation data; validation gains are exploratory.

## Experiment V4 — Targeted Accusation Weighting
- Architecture: DistilRoBERTa sequence classifier
- Change: Accusation loss weight = 2.5; others = 1.0
- Best epoch: 5
- Validation accuracy: 0.8010
- Validation macro-F1: 0.6185
- Test set: untouched
- Checkpoint: `outputs/distilroberta_targeted_v4/best_model/`

## Experiment V5 — External-to-FINCORP Transfer
- External dataset records after filtering: 1232
- Exact FINCORP overlaps removed: 0
- Stage 1: Four-class external severity training
- Stage 1 best macro-F1: 0.5707
- Stage 2: Five-class FINCORP fine-tuning
- Stage 2 best epoch: 4
- FINCORP validation accuracy: 0.8328
- FINCORP validation macro-F1: 0.5987
- Test set: not evaluated
- Checkpoints: `outputs/distilroberta_external_transfer_v5/`
- Limitation: Exact normalized text overlaps removed; semantic near-duplicates not yet audited.
