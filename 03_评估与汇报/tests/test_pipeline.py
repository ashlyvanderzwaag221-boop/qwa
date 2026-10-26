import unittest,sys,json,tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '00_公共支持'))
from project_paths import DATA_DIR, DATA_RESULTS, TESTS_DIR, TRAIN_STAGE
sys.path.insert(0, str(TRAIN_STAGE))
from pipeline import manifest,feature_vector,ROOT
class DataContracts(unittest.TestCase):
 def test_manifest_totals_and_folds(self):
  rows=manifest(DATA_DIR/'official_index.json')
  self.assertEqual(len(rows),8732);self.assertEqual(set(r['fold'] for r in rows),set(range(1,11)))
  self.assertEqual(len({r['clip_id'] for r in rows}),8732)
 def test_index_checksum(self):
  import hashlib
  self.assertEqual(hashlib.md5((DATA_DIR/'official_index.json').read_bytes()).hexdigest(),'1c4940e08c1305c49b592f3d9c103e6f')
 def test_known_overlap_is_visible(self):
  rows=manifest(DATA_DIR/'official_index.json');g={}
  for r in rows:g.setdefault(r['fsID'],set()).add(r['fold'])
  self.assertEqual(len([v for v in g.values() if len(v)>1]),5)
 def test_silence_rejected_and_stereo_resampled(self):
  import numpy as np,soundfile as sf
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'clip.wav';sf.write(p,np.zeros(8000),8000)
   with self.assertRaises(ValueError):feature_vector(p)
   t=np.arange(8000)/8000;sf.write(p,np.column_stack([np.sin(2*np.pi*440*t),np.sin(2*np.pi*660*t)]),8000)
   v=feature_vector(p);self.assertEqual(v.shape,(40,));self.assertTrue(np.isfinite(v).all())
if __name__=='__main__':unittest.main()
