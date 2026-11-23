"""Independently validate UrbanSound8K outer-test scores from saved confusion matrices."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def matrix_scores(matrix):
    correct = matrix.diagonal()
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    per_class_f1 = np.divide(2 * correct, denominator, out=np.zeros(10), where=denominator != 0)
    return float(correct.sum() / matrix.sum()), float(per_class_f1.mean())


def verify(project, results_path):
    data_stage = project / '01_数据处理与特征'
    manifest_path = data_stage / 'results' / 'manifest.csv'
    feature_path = data_stage / 'features' / 'features.npz'
    with manifest_path.open(encoding='utf-8', newline='') as stream:
        meta = list(csv.DictReader(stream))
    ids = {r['clip_id'] for r in meta}
    with np.load(feature_path, allow_pickle=False) as features:
        require(len(meta) == len(ids) == 8732 and set(features['ids']) == ids, 'Clip ID mismatch')
        require(features['X'].shape == (8732, 40) and np.isfinite(features['X']).all(), 'Invalid features')
    digest = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    receipt = read_json(data_stage / 'features' / 'feature_receipt.json')
    require(digest == receipt['sha256'], 'Feature artifact does not match extraction receipt')
    source = np.array([r['fsID'] for r in meta])
    fold = np.array([int(r['fold']) for r in meta])
    y = np.array([int(r['classID']) for r in meta])
    raw = read_json(results_path)
    require(raw['mode'] == 'final', 'Expected full outer-test evaluation')
    models = ['majority', 'random_forest', 'svm']
    index = {(r['test_fold'], r['model']): r for r in raw['results']}
    require(len(index) == len(raw['results']) == 30, 'Incomplete outer-fold results')
    folds = []
    by_model = {m: [] for m in models}
    pooled = {m: np.zeros((10, 10), dtype=int) for m in models}
    for k in range(1, 11):
        val_fold = k % 10 + 1
        test = fold == k
        other = ~test
        available = other & ~np.isin(source, source[test])
        val = available & (fold == val_fold)
        train = available & (fold != val_fold) & ~np.isin(source, source[val])
        groups = [set(source[mask]) for mask in (train, val, test)]
        require(all(groups[a].isdisjoint(groups[b]) for a, b in [(0, 1), (0, 2), (1, 2)]), 'Shared source in tuning split')
        require(set(source[available]).isdisjoint(groups[2]), 'Shared source in refit split')
        folds.append(dict(test_fold=k, validation_fold=val_fold, n_test=int(test.sum()),
                          n_inner_train=int(train.sum()), n_inner_val=int(val.sum()),
                          n_refit=int(available.sum()), n_purged_at_refit=int((other & ~available).sum()),
                          purged_refit_source_ids=sorted(set(source[other & ~available].tolist()))))
        actual = np.bincount(y[test], minlength=10)
        majority_class = np.bincount(y[available], minlength=10).argmax()
        majority_matrix = np.zeros((10, 10), dtype=int)
        majority_matrix[:, majority_class] = actual
        for model in models:
            row = index[k, model]
            require(row['validation_fold'] == val_fold and row['n'] == int(test.sum()), 'Wrong fold or size')
            cm = np.asarray(row['confusion_matrix'])
            require(cm.shape == (10, 10) and np.issubdtype(cm.dtype, np.integer) and (cm >= 0).all(), 'Invalid matrix')
            require(np.array_equal(cm.sum(axis=1), actual), 'Per-class support mismatch')
            if model == 'majority':
                require(np.array_equal(cm, majority_matrix), 'Majority benchmark disagrees with training labels')
            accuracy, macro_f1 = matrix_scores(cm)
            require(abs(accuracy - row['accuracy']) < 1e-12 and abs(macro_f1 - row['macro_f1']) < 1e-12, 'Saved score mismatch')
            pooled[model] += cm
            by_model[model].append([accuracy, macro_f1])
    summaries = {}
    for model in models:
        values = np.asarray(by_model[model])
        matrix = pooled[model]
        aggregate_accuracy, aggregate_f1 = matrix_scores(matrix)
        top_errors = matrix.copy()
        np.fill_diagonal(top_errors, 0)
        actual, predicted = np.unravel_index(top_errors.argmax(), matrix.shape)
        require(matrix.sum() == 8732, 'Incomplete evaluation coverage')
        summaries[model] = dict(mean_accuracy=float(values[:, 0].mean()), mean_macro_f1=float(values[:, 1].mean()),
                                sample_sd_accuracy=float(values[:, 0].std(ddof=1)), sample_sd_macro_f1=float(values[:, 1].std(ddof=1)),
                                min_fold_macro_f1=float(values[:, 1].min()), max_fold_macro_f1=float(values[:, 1].max()),
                                pooled_accuracy=aggregate_accuracy, pooled_macro_f1=aggregate_f1,
                                largest_confusion=dict(actual_id=int(actual), predicted_id=int(predicted), count=int(matrix[actual, predicted]),
                                                       actual_support=int(matrix[actual].sum())),
                                parameter_counts=dict(Counter(json.dumps(index[k, model]['parameters'], sort_keys=True) for k in range(1, 11))))
    return dict(evaluation='Equal-weight mean of ten outer test folds; sample SD uses ddof=1.',
                n_clips=8732, n_models=3, n_folds=10, no_shared_train_val_test_sources=True,
                feature_sha256=digest, training_code_sha256=hashlib.sha256((project/'00_公共支持'/'pipeline.py').read_bytes()).hexdigest(),
                results_sha256=hashlib.sha256(results_path.read_bytes()).hexdigest(), folds=folds, models=summaries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--results', type=Path, default=Path(__file__).with_name('tenfold_results.json'))
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('tenfold_summary.json'))
    args = parser.parse_args()
    summary = verify(args.project, args.results)
    args.out.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    for name, metrics in summary['models'].items():
        print(f"{name}: mean accuracy={metrics['mean_accuracy']:.6f}, mean macro-F1={metrics['mean_macro_f1']:.6f}")
