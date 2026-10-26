"""Evaluation guards: source leakage, full coverage, RBF configuration and scaling."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '00_公共支持'))
from project_paths import DATA_DIR, DATA_RESULTS, TESTS_DIR, TRAIN_STAGE
sys.path.insert(0, str(TRAIN_STAGE))
from experiment import development_split, load_features, make_model
from pipeline import ROOT, read_rows


class MilestoneContracts(unittest.TestCase):
    def test_no_source_overlap_or_fold10_in_development(self):
        rows = read_rows(DATA_RESULTS / 'manifest.csv')
        train, validation, held, excluded = development_split(rows)
        groups = lambda mask: {r['fsID'] for r, keep in zip(rows, mask) if keep}
        self.assertFalse(groups(train) & groups(validation | held))
        self.assertEqual(int(validation.sum()), 816)
        self.assertEqual(int(held.sum()), 837)
        self.assertTrue(excluded.any())
        self.assertEqual(int((train | validation | held | excluded).sum()), 8732)

    def test_partial_cache_rejected_before_any_training(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'features.npz'
            np.savez(file, X=np.zeros((1, 40)), ids=['101415-3-0-2'])
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                load_features(DATA_RESULTS / 'manifest.csv', file)

    def test_rbf_and_training_only_scaler(self):
        train = np.array([[0, 0], [2, 1], [5, 4], [8, 6]], dtype=float)
        validation = np.array([[1000, 1000]], dtype=float)
        model = make_model('rbf_svm', {'kernel': 'rbf', 'C': 1, 'gamma': 'scale'})
        model.fit(train, [0, 0, 1, 1])
        self.assertEqual(model.named_steps['svc'].kernel, 'rbf')
        before = model.named_steps['standardscaler'].mean_.copy()
        model.predict(validation)
        np.testing.assert_array_equal(before, train.mean(axis=0))
        np.testing.assert_array_equal(before, model.named_steps['standardscaler'].mean_)


if __name__ == '__main__':
    unittest.main()
