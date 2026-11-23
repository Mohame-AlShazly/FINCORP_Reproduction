# FINCORP — Reproduction and Severity Classification Experiments

**AI504 Assignment 2 — Reproduction and Validation of an NLP Research Study**

This repository contains the saved FINCORP reproduction notebook, improvement scripts, experiment results, and supporting evidence. Archived metrics have not all been independently rerun in a fresh GPU environment.

## 1. Paper and research question

A. Singh, R. Bhatia, and S. Saha, “Complaint and Severity Identification From Online Financial Content,” *IEEE Transactions on Computational Social Systems*, vol. 11, no. 1, pp. 660–670, 2024. DOI: https://doi.org/10.1109/TCSS.2022.3215528

Authors' public source code and FINCORP corpus: https://github.com/RohanBh23/FINCORP

**Claim examined:** The authors report that their four-task RoBERTa/BiGRU model jointly predicting complaint, severity, sentiment, and emotion achieves **93.78% complaint macro-F1** and **75.43% severity macro-F1** on FINCORP and improves over alternative task configurations. The repository's original-code runs and separate severity-classification experiments help examine reproducibility and class-imbalance effects. They do **not** constitute a matched replication of every published ablation.

## 2. Repository contents and provenance

The following paths match the GitHub repository.

```text
.
├── README.md
├── FINCORP_Reproduction/
│   ├── Database/FINCORP.csv
│   ├── MTL Model Experiments/Quad_Task_(train_test_split).ipynb
│   ├── MTL Model Experiments/Quad Task (Cross Validation).ipynb
│   └── README.md
├── FINCORP_Improvment Experiments/
│   ├── code/
│   ├── data/FINCORP.csv
│   ├── FINCORP-main.zip
│   ├── EXPERIMENT_LOG.md
│   └── requirements.txt
├── Tests/
│   ├── tests/test_fincorp_sanity.py
│   ├── README_TESTS.md
│   └── TEST_RUN_RESULTS.txt
└── Tests & Reproduction Results/
    ├── CPU_SVM_Results/
    ├── DistilRoBERTa_Results/
    ├── Reproduction_Screenshots/
    └── table_*.csv
```

**Reused from the authors:** FINCORP dataset, original quad-task architecture, tokenizer/model approach and training notebook. **Group work:** two reported Colab executions of the original five sequential experiment blocks and their archived outputs; separately executed CPU severity comparisons (SVM, hierarchical, exploratory models); GPU-accelerated Google Colab DistilRoBERTa severity experiments V1–V5 and result interpretation. A compact reproduction notebook is uploaded in this repository. The original full Colab runtime history is not archived here.

## 3. Data and preprocessing

- **FINCORP released CSV:** 6,281 rows, six columns: `Domain`, `Complaint/ Opinion`, `Complaint Label`, `Severity level`, `Sentiment`, `Emotion`. The publication refers to 6,282 tweets; the downloadable CSV is the data actually analyzed here.
- **Main Colab notebook:** Reads a CSV with the *literal* name `Fincorp.csv`, shuffles its rows, uses RoBERTa tokenization with maximum length 200, and creates train/test then train/validation partitions (30% test; 15% of remaining data for validation, seed 42). Because the original dataset file is named `FINCORP.csv`, upload a copy renamed **`Fincorp.csv`** to the Colab working directory or update the notebook's `read_csv` statements consistently.
- **CPU scripts:** Use the original FINCORP data inside `FINCORP-main.zip`, change links/handles to placeholders, create stratified train/validation/test sets (seed 40), and extract word/character TF-IDF features **using training text only**.
- **Neural extension:** Uses five severity classes on a fixed 3,768/628/1,885 train/validation/test split. V5 additionally uses 1,232 complaint-only external severity examples. The neural experiments here are accompanied by saved configurations and evaluations, but their complete executable training scripts were not found in the provided improvement ZIP.

## 4. Environment and hardware

### Main reproduction and additional DistilRoBERTa experiments — Google Colab (GPU)

The archived authors' notebook imports TensorFlow/Keras, Transformers, NumPy, pandas and scikit-learn; it starts with `pip install transformers`. **This installation alone is not a reproducible pinned environment.** The group reports executing both the original quad-task sequence and the later DistilRoBERTa neural experiments on Google Colab GPUs. Exact versions for the submitted reproduction and the Colab GPU model must still be copied from its own saved runtime log. The original notebook also references `tensorflow.keras.layers.experimental.preprocessing.TextVectorization`, which may require an older compatible TensorFlow/Keras environment or an explicitly documented compatibility edit.

Before recording an official reproduction, save the environment details from Colab:

```python
import sys, tensorflow as tf, transformers, sklearn, numpy, pandas
print('Python:', sys.version)
print('TensorFlow:', tf.__version__)
print('Transformers:', transformers.__version__)
print('scikit-learn:', sklearn.__version__)
print('NumPy:', numpy.__version__)
print('pandas:', pandas.__version__)
print('GPUs:', tf.config.list_physical_devices('GPU'))
```

For the full GPU model and memory, save the output of `!nvidia-smi` in a Colab cell.

### Separate CPU/SVM experiments — Python

Required imports seen in the supplied CPU scripts: `numpy`, `pandas`, `scipy`, `scikit-learn`. Install the packages using:

```bash
python -m pip install numpy pandas scipy scikit-learn
```

**Version pins:** Not verified against the originally executed environment; do not assign guessed versions. The dependency files list libraries used by each phase, but the exact historical Colab versions have not been confirmed, so version pins must be verified before the final submission.

## 5. Reproduce the original multitask notebook

1. Open Google Colab, select a GPU runtime, and upload the **actual group reproduction notebook** from `FINCORP_Reproduction/`. The authors' reference notebook is `FINCORP_Reproduction/MTL Model Experiments/Quad_Task_(train_test_split).ipynb`.
2. Upload `FINCORP.csv` as **`Fincorp.csv`** into the same Colab runtime working directory, *unless your group notebook has already corrected that reference*. The original notebook contains more than one `read_csv('Fincorp.csv')` statement.
3. Confirm the Colab runtime and packages are compatible with the notebook imports and RoBERTa download. Execute its setup, loading, tokenization, model creation, training, and evaluation cells in the sequence used for the archived group runs.
4. **Execute the five original quad-task experiment blocks in their existing notebook order, sequentially, as the authors’ notebook is structured.** The group reports running the complete procedure twice to check consistency. Record the outputs for all five blocks in each execution where available; the E1–E5 table documents the archived five-block sequence. Do **not** reset or independently rebuild the five blocks unless explicitly presenting a separate, new test: the original notebook reuses a trainable RoBERTa encoder object and successive blocks may inherit updated encoder weights. Reproducing the sequence is distinct from five independent seeded trials.
5. Save the output cells/screenshots for each block and preserve the full notebook execution order and run history. The team reported two complete Colab executions. The archived screenshots support the recorded outputs, but the historical full GPU sessions were not independently rerun.
6. Record complaint, severity, sentiment and emotion accuracy and F1. **The printed F1 is an average of per-batch macro-F1**, not a global macro-F1 computed from all test-row predictions. The archived evaluation also does not score all 1,885 test records. This limits direct equivalence to the publication.

The archived author notebook uses RoBERTa `roberta-base`, BiGRU(100), batch size 32, Adam learning rate 1e-5, loss weights complaint/severity/sentiment/emotion = 1/1/0.5/0.5, and repeated blocks of 3 epochs at 150 steps per epoch. These are settings visible in the original source notebook; verify changes in the group's separate notebook.

### Archived main reproduction outcome

The final report records the following E5 scores (accuracy / notebook batch-averaged F1, percentages):

| Task | Accuracy | Notebook F1 |
|---|---:|---:|
| Complaint | 98.52 | 98.41 |
| Severity | 86.84 | 72.35 |
| Sentiment | 81.74 | 72.01 |
| Emotion | 61.79 | 40.34 |

**E1 evidence is not uniform:** The final report, an earlier draft, and different E1 screenshots contain differing E1 values. Retain the separate original screenshots and identify the notebook execution that corresponds to the final report before using E1 as a definitive result. Do not silently replace evidence.

## 6. Reproduce the group's CPU severity baselines

These scripts are included in `FINCORP_Improvment Experiments/code/`. **As originally executed, they assume the input archive at `/mnt/data/FINCORP-main.zip` and write to `/mnt/data/`.** On a new machine, adjust those hardcoded paths or execute in an environment with the same paths before running. The ZIP already contains `FINCORP-main.zip`.

Run from a shell **after** setting up the expected paths and installing dependencies:

```bash
python 'FINCORP_Improvment Experiments/code/FINCORP_severity_cpu_experiment.py'
python 'FINCORP_Improvment Experiments/code/FINCORP_hierarchical_severity_experiment.py'
python 'FINCORP_Improvment Experiments/code/FINCORP_severity_new_search.py'
python 'FINCORP_Improvment Experiments/code/FINCORP_ensemble_trial.py'
python 'FINCORP_Improvment Experiments/code/FINCORP_nbsvm_trial.py'
```

The files above are genuine provided scripts; the commands are **not claimed to have been verified end-to-end in a fresh clone**. They represent separate exploratory analyses, not the original four-task neural model.

### Archived CPU held-out results

All percentages below are from saved test comparisons, evaluated on 1,885 records (global five-class macro-F1):

| Model | Test accuracy | Test macro-F1 |
|---|---:|---:|
| Standard linear SVM (C=1) | 81.49 | 58.88 |
| Balanced linear SVM (C=0.3) | 82.23 | 61.52 |
| Hierarchical SVM | 81.86 | 61.16 |
| Further-tuned SVM | 81.38 | 60.21 |
| Severity-adjusted ensemble | 81.11 | 60.92 |

See `Tests & Reproduction Results/CPU_SVM_Results/` for available test predictions, confusion matrices, and classification reports. Some exploratory analyses are documented with summaries rather than complete per-row evidence.

## 7. Additional neural severity experiments V1–V5

**Actual platform:** Google Colab, GPU runtime (reported by the group). These are **separate neural severity experiments**, not the five sequential original quad-task model blocks. In Colab, each experiment was run with its corresponding configuration; V3 reused V2 predictions and required no new training. The exact GPU model, saved notebook cells and end-to-end training time require confirmation from the corresponding original Colab files/logs.

The uploaded project includes metrics, configurations, validation predictions and training histories in `Tests & Reproduction Results/DistilRoBERTa_Results/`. The group confirms that the V1–V5 DistilRoBERTa experiments were **run in Google Colab with GPU acceleration**. The uploaded `FINCORP_Improvment Experiments(1).zip` includes CPU/SVM `.py` scripts, FINCORP/external datasets, a log and CPU outputs, **but it does not include the executed V1–V5 Colab notebooks or GPU training scripts**. Their validation metrics, histories and other evidence are separately archived under `Tests & Reproduction Results/DistilRoBERTa_Results/`. Add the original V1–V5 Colab notebook(s), as executed, to make these experiments rerunnable; do not represent the CPU `.py` scripts as their GPU training code.

| Run | Modification | Validation accuracy | Global validation macro-F1 |
|---|---|---:|---:|
| V1 | Broad class weighting | 79.62% | 60.02% |
| V2 | Unweighted cross-entropy | 83.44% | 60.26% |
| V3 | Accusation decision threshold (0.05) applied to V2 probabilities | 82.17% | 62.61% |
| V4 | Accusation class weight of 2.5 | 80.10% | 61.85% |
| V5 | External four-class pretraining → FINCORP fine-tuning | 83.28% | 59.87% |

**These are scores on the shared validation set (628 rows), not held-out neural test results.** The reserved neural test split was not evaluated in the provided records. V3 is a post-processing change to V2, not a separately trained model.

## 8. Results, checks and expected execution time

Run the seven static sanity checks from the repository root:

```bash
python -m unittest discover -s Tests/tests -v
```

They passed against an extracted local copy of the uploaded data, notebook, and CPU scripts in the actual GitHub folder layout. This does not replace a fresh GPU reproduction.


- Reproduction screenshots: `Tests & Reproduction Results/Reproduction_Screenshots/`
- Archived CPU test predictions and evaluations: `Tests & Reproduction Results/CPU_SVM_Results/`
- DistilRoBERTa experiment histories and metrics: `Tests & Reproduction Results/DistilRoBERTa_Results/`
- Transcribed numerical tables and source reconciliation notes: `Tests & Reproduction Results/`

**Result verification:** Compare model outputs to the archived predictions and confusion matrices, using global macro-F1 for the CPU experiments. Compare original notebook outputs only under their documented batch-averaging evaluation rule. The professor's instructions require a command for each reported result and a fresh-clone test. Such a full execution check has **not been performed here**.

**Expected runtime:** The full original training time and GPU memory were not established for the group's submitted run. Some CPU experiment logs provide per-candidate fitting times, but there is no verified whole-pipeline runtime. For GPU runs that take more than two hours, the professor requests a reduced execution finishing in under 30 minutes and the original full-run logs; neither should be claimed until demonstrated.

**Further checks before tagging:** Verify the notebook on a fresh Google Colab runtime, exact library versions and hardware, and commands for each reported result. The seven static sanity checks are available in `Tests/`.

## 9. Known differences and research interpretation

- Published dataset size is 6,282; released CSV size is 6,281.
- Published nominal split/seed differs from the original notebook (paper 60/10/30, seed 40; source notebook approximately 59.5/10.5/30, seed 42 without stratification).
- Source notebook uses five severity and seven emotion classes, whereas parts of the paper's methods refer to four and six output neurons.
- Original notebook uses batchwise-averaged F1 rather than global macro-F1 and can reuse the encoder between training blocks.
- CPU and DistilRoBERTa severity experiments use different architectures and/or evaluation partitions from the original multitask model; their scores are not a direct head-to-head measurement of an improved original model.
- Accusation remains difficult: class imbalance makes accuracy insufficient as a sole success measure.

The reproduction establishes that the released implementation can produce the archived predictions and reported scores in the group’s recorded execution. It **does not independently establish the paper’s claimed multitask-versus-ablation advantage** under a fully matched protocol. The additional experiments explore a narrower severity problem.

