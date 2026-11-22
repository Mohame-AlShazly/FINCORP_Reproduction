# FINCORP sanity checks

Copy the `tests/` folder into the **root** of your FINCORP repository, alongside `README.md` and the numbered project folders. Run from that root:

```bash
python -m unittest discover -s tests -v
```

Uses only Python's standard library; no installation or GPU is needed. It checks the released FINCORP CSV's shape, columns, missing values, label counts, complaint/severity alignment, byte-for-byte agreement with the improvement dataset, the original holdout notebook's readable structure and its five `model.fit` blocks, and Python syntax of the uploaded CPU scripts.

**Scope:** These are sanity checks, not model reproduction, GPU training, published metric verification, or a substitute for executable V1–V5 notebooks. Some checks intentionally detect modifications to the exact files provided in the author's and improvement archives. If your folder names differ, update the path constants at the beginning of the test file rather than renaming data arbitrarily.
