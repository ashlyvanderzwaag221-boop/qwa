# 3. Evaluation and reporting

**Responsible: 程志阳.** Check reproducibility, inspect per-class errors, communicate results and plan noise experiments.

- `tests/`: nine checks covering data/index consistency, source overlap, complete feature coverage, RBF configuration, training-only scaling and baseline arithmetic.
- `analyze_results.py`: generate per-class tables and four figures from recorded results; does not need raw audio.
- `verify_results.py`: reload locally trained models and compare their predictions, metrics, split and feature hash with saved evidence. Run feature extraction and training first.
- `results/verification.json`: recorded verification from the original local experiment.
- `reports/Milestone_Report.pdf`: one-page English milestone report.
- `reports/figures/`: class distribution, validation comparison and two confusion matrices.

The repository includes the report and numerical evidence. Slides, speaker notes, local archives, caches and temporary logs remain in the local course workspace.

Full learned-model ten-fold evaluation, alternative features, class weighting/oversampling and added-white-noise experiments are future work. Do not describe their expected benefits as completed results. See the [root README](../README.md) for the current protocol and commands.
