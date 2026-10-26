"""Independent acceptance check against saved predictions, models, split and features."""
import csv
import json
from pathlib import Path
import sys
import hashlib
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '00_公共支持'))
from project_paths import DATA_RESULTS, FEATURES_DIR, TRAIN_RESULTS, MODELS_DIR, EVAL_RESULTS, TRAIN_STAGE
sys.path.insert(0, str(TRAIN_STAGE))
from experiment import load_features, development_split
from pipeline import save_json

result = json.loads((TRAIN_RESULTS / 'metrics.json').read_text())
X, y, rows = load_features(DATA_RESULTS / 'manifest.csv', FEATURES_DIR / 'features.npz')
train, val, held, excluded = development_split(rows)
assert not (train & val).any() and not (train & held).any()
assert not {r['fsID'] for r, f in zip(rows, train) if f} & {r['fsID'] for r, f in zip(rows, val | held) if f}
with (TRAIN_RESULTS / 'predictions.csv').open(encoding='utf-8') as stream:
    saved = list(csv.DictReader(stream))
checks = []
for record in result['models']:
    model = joblib.load(MODELS_DIR / (record['model'] + '.joblib'))
    prediction = model.predict(X[val])
    evidence = [r for r in saved if r['model'] == record['model']]
    assert [r['clip_id'] for r in evidence] == [r['clip_id'] for r, f in zip(rows, val) if f]
    np.testing.assert_array_equal(prediction, [int(r['predicted_id']) for r in evidence])
    np.testing.assert_array_equal(y[val], [int(r['actual_id']) for r in evidence])
    assert abs(accuracy_score(y[val], prediction) - record['accuracy']) < 1e-12
    assert abs(f1_score(y[val], prediction, labels=list(range(10)), average='macro', zero_division=0) - record['macro_f1']) < 1e-12
    np.testing.assert_array_equal(confusion_matrix(y[val], prediction, labels=list(range(10))), record['confusion_matrix'])
    matrix = np.array(record['confusion_matrix'], dtype=float)
    denominators = matrix.sum(axis=0) + matrix.sum(axis=1)
    manual_f1 = np.divide(2 * np.diag(matrix), denominators, out=np.zeros(10), where=denominators != 0).mean()
    assert abs(np.trace(matrix) / matrix.sum() - record['accuracy']) < 1e-12
    assert abs(manual_f1 - record['macro_f1']) < 1e-12
    if record['model'] == 'rbf_svm':
        assert model.named_steps['svc'].kernel == 'rbf'
        np.testing.assert_allclose(model.named_steps['standardscaler'].mean_, X[train].astype('float64').mean(axis=0), rtol=1e-10, atol=1e-10)
    checks.append({'model': record['model'], 'saved_model_reproduces_predictions': True, 'metrics_recomputed': True})
assert hashlib.sha256((FEATURES_DIR / 'features.npz').read_bytes()).hexdigest() == result['protocol']['feature_file_sha256']
save_json(EVAL_RESULTS / 'verification.json', {'passed': True, 'n_features': len(X), 'models': checks,
    'source_overlap_train_vs_evaluation': 0, 'fold10_scored': False, 'training_only_scaler_verified': True})
print(json.dumps({'passed': True, 'models_verified': len(checks), 'validation_examples_each': int(val.sum())}))
