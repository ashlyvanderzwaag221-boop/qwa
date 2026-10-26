import unittest,sys,tempfile,json,argparse,contextlib,io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '00_公共支持'))
from project_paths import DATA_DIR, DATA_RESULTS, TESTS_DIR, TRAIN_STAGE
sys.path.insert(0, str(TRAIN_STAGE))
from pipeline import train,label_baseline,ROOT
class EvaluationContracts(unittest.TestCase):
 def test_partial_features_cannot_produce_benchmark(self):
  import numpy as np
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'features.npz';np.savez(f,X=np.zeros((1,40)),ids=np.array(['101415-3-0-2']))
   a=argparse.Namespace(manifest=DATA_RESULTS/'manifest.csv',features=f,mode='development',out=Path(d)/'scores.json')
   with self.assertRaisesRegex(ValueError,'Incomplete'):train(a)
   self.assertFalse(a.out.exists())
 def test_sklearn_baseline_matches_independent_arithmetic(self):
  with tempfile.TemporaryDirectory() as d:
   a=argparse.Namespace(manifest=DATA_RESULTS/'manifest.csv',out=Path(d)/'scores.json')
   with contextlib.redirect_stdout(io.StringIO()):label_baseline(a)
   actual=json.loads(a.out.read_text());expected=json.loads((TESTS_DIR/'majority_baseline_reference.json').read_text())
   self.assertAlmostEqual(actual['mean_accuracy'],expected['mean_accuracy'],places=12)
   self.assertAlmostEqual(actual['mean_macro_f1'],expected['mean_macro_f1'],places=12)
if __name__=='__main__':unittest.main()
