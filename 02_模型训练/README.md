# 2. Model training and tuning

**Responsible: 茅人方.** Train the majority reference, RBF-SVM and random forest on the same features and split.

`experiment.py` is the current training entry. It loads the first stage's full feature matrix, excludes source overlap, fits preprocessing only on training data, selects hyperparameters by validation Macro-F1, and saves models and per-clip predictions.

- Training: folds 1-8 after excluding 79 overlapping-source clips (7,000 clips).
- Validation: fold 9 (816 clips), also used for model selection.
- Fold 10: reserved (837 clips), not used by this main development experiment.
- SVM search: 9 combinations of C and gamma; selected C=10, gamma=scale.
- Forest search: 4 combinations; selected 200 trees, max_depth=20, min_samples_leaf=1.

Inspect `results/metrics.csv`, `search_results.json`, `protocol.json`, `split_manifest.csv` and `predictions.csv`. Trained `.joblib` binaries are regenerated locally and excluded from Git.

These scores are validation results, not independent test estimates. See the [root README](../README.md) for commands and limitations.
