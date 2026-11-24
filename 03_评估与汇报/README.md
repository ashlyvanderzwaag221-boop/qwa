# 3. Evaluation and reporting

**Responsible: 程志阳.** Check reproducibility, inspect per-class errors, communicate results and plan noise experiments.

- `tests/`: nine checks covering data/index consistency, source overlap, complete feature coverage, RBF configuration, training-only scaling and baseline arithmetic.
- `analyze_results.py`: generate per-class tables and four figures from recorded results; does not need raw audio.
- `verify_results.py`: reload locally trained models and compare their predictions, metrics, split and feature hash with saved evidence. Run feature extraction and training first.
- `results/verification.json`: recorded verification from the original local experiment.
- `reports/Milestone_Report.pdf`: one-page English milestone report.
- `reports/figures/`: class distribution, validation comparison and two confusion matrices.

The repository includes the report and numerical evidence. Slides, speaker notes, local archives, caches and temporary logs remain in the local course workspace.

The third report adds clean ten-fold results for all three models, a fold-9 SVM class-weight comparison and a recording-level audit of the milestone errors. The progress directory contains the verification scripts, fold settings, confusion matrices and diagnostic results. [Third-report evidence and commands](evidence/progress/README.md)

The next experiments will log individual outer-fold predictions and compare performance under added white noise and alternative time-based features. [Current results and remaining work](../README.md)
