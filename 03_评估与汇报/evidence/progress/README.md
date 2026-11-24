# Third project report: experiment evidence

The clean-audio results cover ten outer test folds. The class-weight comparison and source-level error audit use the original milestone validation split (fold 9).

| File | Contents |
|---|---|
| [tenfold_results.json](tenfold_results.json) | Settings, accuracy, macro-F1 and the 10-class confusion matrix for each model on each outer fold. |
| [tenfold_summary.json](tenfold_summary.json) | Metrics independently recomputed from the matrices, plus sample counts and source exclusions by fold. |
| [verify_tenfold.py](verify_tenfold.py) | Checks clip coverage, the feature archive and recording-source separation, then recalculates the scores. |
| [fold9_diagnostics.json](fold9_diagnostics.json) | Counts by source recording and results of the controlled class-weight comparison. |
| [fold9_diagnostics.py](fold9_diagnostics.py) | Groups saved validation predictions by source and reproduces the two SVM fits under identical conditions. |

## Check the error trace from a fresh clone

Run the following from the repository root with Python. The source audit reads the committed [milestone predictions](../../../02_模型训练/results/predictions.csv) and uses only the Python standard library.

```powershell
python "03_评估与汇报/evidence/progress/fold9_diagnostics.py" --project-root .
```

In the output, `predicted_drilling` is 43 and `errors_from_two_largest_contributors` is 38. Recordings 75743 and 101729 contributed 22 and 16 of those errors.

## Refit the models and verify the scores

Use the project environment from the [root README](../../../README.md). Run the data download, metadata check and feature extraction steps there to create `01_数据处理与特征/features/features.npz` and its receipt. With that environment active, run these Python commands from the repository root.

```powershell
python "00_公共支持/pipeline.py" train --mode final --out rerun_tenfold.json
python "03_评估与汇报/evidence/progress/verify_tenfold.py" --project . --results rerun_tenfold.json --out rerun_tenfold_summary.json
python "03_评估与汇报/evidence/progress/fold9_diagnostics.py" --project-root . --class-weights --out rerun_fold9_diagnostics.json
```

The report table gives each outer fold equal weight. Pooled scores in the JSON combine all predictions before calculating metrics and can differ from the table. The fold-9 comparisons are development diagnostics.

