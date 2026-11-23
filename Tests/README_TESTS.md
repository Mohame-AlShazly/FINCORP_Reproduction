# FINCORP sanity checks

Run from the repository root:

```bash
python -m unittest discover -s Tests/tests -v
```

The seven tests require only the Python standard library. They verify the released dataset shape, missing values, label distributions, consistency between dataset copies, the five model.fit blocks in the reproduction notebook, and Python syntax of the uploaded CPU experiment scripts.

These are static sanity checks, **not** a replacement for GPU training, published-result validation, or reproducibility of the V1–V5 notebooks. The script is kept at `Tests/tests/test_fincorp_sanity.py` and expects this repository's actual directory names.
