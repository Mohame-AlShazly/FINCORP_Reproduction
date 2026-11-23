# FINCORP sanity checks

Run from the repository root:

```bash
python -m unittest discover -s Tests/tests -v
```

The **10 tests** use only the Python standard library. They check:

- FINCORP dataset dimensions, nonempty values, class distributions, and complaint/severity consistency.
- Exact equality of original and improvement dataset copies.
- Original E1–E5 notebook structure and the five `model.fit` blocks.
- Python syntax of the CPU improvement scripts.
- V1–V5 DistilRoBERTa notebook JSON structure and Python cell syntax (allowing Colab setup magics).
- The presence of runnable V1, V2, V3, V4, and V5 code.
- The documented V3 dependency on V2 validation predictions and the V5 external-to-FINCORP transfer stages.

The existing GitHub Actions workflow runs the suite automatically on pushes and pull requests; no separate workflow changes are required.

These are **static integrity checks**, not full model executions. Passing them does **not** establish correct training, checkpoint loading, model accuracy, or historical result reproducibility. The new notebook should be tested on a GPU, with fresh predictions compared against the archived metrics. Test source: `Tests/tests/test_fincorp_sanity.py`.
