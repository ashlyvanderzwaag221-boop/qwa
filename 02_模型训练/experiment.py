"""Reproducible development experiment. Fold 10 is never used for fitting or scoring.

Tune on fold 9 after training on folds 1-8, with source-ID overlap removed.
Scores are model-selection validation scores, NOT independent test estimates.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '00_公共支持'))
from project_paths import DATA_RESULTS, FEATURES_DIR, TRAIN_RESULTS, MODELS_DIR
from pipeline import CONFIG, LABELS, read_rows, save_json

# Fixed before looking at the validation results; select by macro-F1.
GRIDS = {
    'rbf_svm': [{'kernel': 'rbf', 'C': c, 'gamma': g}
                for c in [0.1, 1, 10] for g in ['scale', 0.01, 0.1]],
    'random_forest': [{'n_estimators': 200, 'max_depth': depth, 'min_samples_leaf': leaf}
                      for depth in [None, 20] for leaf in [1, 2]],
}


def load_features(manifest_path, features_path):
    rows = read_rows(manifest_path)
    if len(rows) != 8732 or len({r['clip_id'] for r in rows}) != 8732:
        raise ValueError('Incomplete or duplicate official manifest')
    data = np.load(features_path, allow_pickle=False)
    ids, X = data['ids'].tolist(), data['X']
    if len(ids) != len(set(ids)) or set(ids) != {r['clip_id'] for r in rows}:
        raise ValueError('Incomplete or duplicate audio features; full dataset required')
    if X.shape != (len(ids), 40) or not np.isfinite(X).all():
        raise ValueError('Invalid feature shape or nonfinite values')
    lookup = {r['clip_id']: r for r in rows}
    meta = [lookup[k] for k in ids]
    return X, np.array([r['classID'] for r in meta]), meta


def development_split(meta):
    fold = np.array([r['fold'] for r in meta])
    group = np.array([r['fsID'] for r in meta])
    validation = fold == 9
    held_out = fold == 10
    blocked = set(group[validation]) | set(group[held_out])
    purged = (fold <= 8) & np.isin(group, list(blocked))
    train = (fold <= 8) & ~purged
    if set(group[train]) & set(group[validation | held_out]):
        raise ValueError('Source recording leakage')
    if not train.any() or not validation.any():
        raise ValueError('Empty training or validation split')
    return train, validation, held_out, purged


def make_model(name, parameters):
    if name == 'rbf_svm':
        return make_pipeline(StandardScaler(), SVC(cache_size=512, **parameters))
    if name == 'random_forest':
        return RandomForestClassifier(random_state=CONFIG['seed'], n_jobs=4, **parameters)
    if name == 'majority':
        return DummyClassifier(strategy='most_frequent')
    raise ValueError(name)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scores(y, prediction):
    return {'accuracy': float(accuracy_score(y, prediction)),
            'macro_f1': float(f1_score(y, prediction, labels=list(range(10)), average='macro', zero_division=0))}


def run(manifest_path, features_path, output, models_dir):
    output, models_dir = Path(output), Path(models_dir)
    output.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    X, y, meta = load_features(manifest_path, features_path)
    train, val, held, purged = development_split(meta)
    roles = np.full(len(meta), 'excluded_source_overlap', dtype='<U30')
    roles[train], roles[val], roles[held] = 'train', 'validation', 'held_out_unused'
    split_rows = [{**r, 'role': role} for r, role in zip(meta, roles)]
    write_csv(output / 'split_manifest.csv', split_rows)
    protocol = {
        'scope': 'Development model-selection validation; NOT independent test or final 10-fold results',
        'train_folds': list(range(1, 9)), 'validation_fold': 9, 'held_out_fold': 10,
        'n_train': int(train.sum()), 'n_validation': int(val.sum()),
        'n_held_out_unused': int(held.sum()), 'n_excluded_source_overlap': int(purged.sum()),
        'excluded_source_ids': sorted({r['fsID'] for r, flag in zip(meta, purged) if flag}),
        'selection_metric': 'macro_f1', 'search_grids': GRIDS,
        'feature_config': CONFIG,
        'feature_file_sha256': hashlib.sha256(Path(features_path).read_bytes()).hexdigest(),
        'split_file_sha256': hashlib.sha256((output / 'split_manifest.csv').read_bytes()).hexdigest(),
        'python': platform.python_version(),
        'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'librosa', 'scikit-learn', 'soundfile', 'joblib']},
    }
    save_json(output / 'protocol.json', protocol)
    print(f"Training: {train.sum()}; validation: {val.sum()}; held out unused: {held.sum()}; source exclusions: {purged.sum()}", flush=True)
    all_search, metrics, predictions = [], [], []
    with threadpool_limits(limits=1):
        for name in ['majority', 'rbf_svm', 'random_forest']:
            best = None
            grid = GRIDS.get(name, [{}])
            for parameters in grid:
                begun = time.perf_counter()
                model = make_model(name, parameters).fit(X[train], y[train])
                fit_seconds = time.perf_counter() - begun
                prediction = model.predict(X[val])
                evaluation = scores(y[val], prediction)
                trial = {'model': name, 'parameters': parameters, **evaluation, 'fit_seconds': fit_seconds}
                all_search.append(trial)
                save_json(output / 'search_results.json', all_search)
                print(f'{name} {parameters}: accuracy={evaluation["accuracy"]:.4f}, macro_f1={evaluation["macro_f1"]:.4f}', flush=True)
                if best is None or evaluation['macro_f1'] > best['macro_f1']:
                    best = {**trial, 'fitted': model, 'prediction': prediction}
            model, prediction = best.pop('fitted'), best.pop('prediction')
            joblib.dump(model, models_dir / f'{name}.joblib', compress=3)
            report = classification_report(y[val], prediction, labels=list(range(10)), target_names=LABELS, output_dict=True, zero_division=0)
            cm = confusion_matrix(y[val], prediction, labels=list(range(10)))
            record = {**best, 'n_validation': int(val.sum()), 'scope': protocol['scope'],
                      'classification_report': report, 'confusion_matrix': cm.tolist()}
            metrics.append(record)
            for r, truth, predicted in zip([r for r, flag in zip(meta, val) if flag], y[val], prediction):
                predictions.append({'model': name, 'clip_id': r['clip_id'], 'fold': r['fold'],
                                    'fsID': r['fsID'], 'actual_id': int(truth), 'predicted_id': int(predicted),
                                    'actual': LABELS[truth], 'predicted': LABELS[predicted],
                                    'correct': bool(truth == predicted)})
            save_json(output / 'metrics.json', {'protocol': protocol, 'models': metrics})
            write_csv(output / 'predictions.csv', predictions)
    write_csv(output / 'metrics.csv', [{'model': r['model'], 'accuracy': r['accuracy'], 'macro_f1': r['macro_f1'],
                                      'n_validation': r['n_validation'], 'parameters': json.dumps(r['parameters'])} for r in metrics])
    save_json(models_dir / 'feature_config.json', CONFIG)
    print('Development experiment saved. Fold 10 was not fitted or scored.', flush=True)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=DATA_RESULTS / 'manifest.csv')
    parser.add_argument('--features', default=FEATURES_DIR / 'features.npz')
    parser.add_argument('--out', default=TRAIN_RESULTS)
    parser.add_argument('--models', default=MODELS_DIR)
    args = parser.parse_args()
    run(args.manifest, args.features, args.out, args.models)
