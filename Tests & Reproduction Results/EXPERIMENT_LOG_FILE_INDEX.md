# Experiment log and verified file locations

The original experiment log is preserved as `ARCHIVED_EXPERIMENT_LOG_UNMODIFIED.md`. Its `outputs/` paths refer to the original working directory, not this submission folder. Those paths should not be used as relative links here.

| Experiment | Training history in this package | Evaluation evidence |
|---|---|---|
| V1 weighted | `DistilRoBERTa_Results/Training_History/distilroberta_weighted_v1/training_history.csv` | `DistilRoBERTa_Results/Evaluation_Metrics/distilroberta_weighted_v1/` |
| V2 unweighted | `DistilRoBERTa_Results/Training_History/distilroberta_unweighted_v2/training_history.csv` | `DistilRoBERTa_Results/Evaluation_Metrics/distilroberta_unweighted_v2/` |
| V3 threshold | No independent training; uses V2 predictions | `DistilRoBERTa_Results/Evaluation_Metrics/distilroberta_threshold_v3/` |
| V4 targeted | `DistilRoBERTa_Results/Training_History/distilroberta_targeted_v4/training_history.csv` | `DistilRoBERTa_Results/Evaluation_Metrics/distilroberta_targeted_v4/` |
| V5 transfer | `DistilRoBERTa_Results/Training_History/distilroberta_external_transfer_v5/stage2_fincorp/training_history.csv` | `DistilRoBERTa_Results/Evaluation_Metrics/distilroberta_external_transfer_v5/` |

**Correction:** The upstream log's V2 training and model locations mistakenly reference V1. The V2 training history is in the separate unweighted folder above. Only configuration files for `best_model/` are present in this package; full checkpoint weights have not been supplied.

All V1–V5 scores in the summary are validation-set results, not held-out neural test results.
