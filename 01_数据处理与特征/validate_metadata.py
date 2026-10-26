"""Compare original metadata with the independently checksummed Soundata index."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '00_公共支持'))
from project_paths import DATA_DIR, DATA_RESULTS
DATA_RESULTS.mkdir(parents=True, exist_ok=True)
from pipeline import manifest, save_json

index = DATA_DIR / 'official_index.json'
metadata = DATA_DIR / 'UrbanSound8K/metadata/UrbanSound8K.csv'
assert hashlib.md5(index.read_bytes()).hexdigest() == '1c4940e08c1305c49b592f3d9c103e6f'
assert hashlib.md5(metadata.read_bytes()).hexdigest() == '5015c37ce7d54c0bb8cd40a0b5ee2074'
rows = manifest(index)
with metadata.open(encoding='utf-8', newline='') as stream:
    original = list(csv.DictReader(stream))
lookup = {Path(r['slice_file_name']).stem: r for r in original}
assert len(original) == len(lookup) == len(rows) == 8732
for row in rows:
    source = lookup[row['clip_id']]
    for field in ['fold', 'classID', 'fsID', 'class']:
        assert str(row[field]) == source[field], (row['clip_id'], field)
    row.update({key: source[key] for key in ['start', 'end', 'salience']})
with (DATA_RESULTS / 'manifest.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
groups = defaultdict(set)
for row in rows:
    groups[row['fsID']].add(row['fold'])
audit = {'samples': len(rows), 'classes': 10, 'recording_groups': len(groups),
         'class_counts': dict(Counter(r['class'] for r in rows)),
         'fold_counts': dict(Counter(r['fold'] for r in rows)),
         'recording_groups_crossing_folds': {k: sorted(v) for k, v in groups.items() if len(v) > 1},
         'metadata_and_index_agree': True,
         'metadata_md5': hashlib.md5(metadata.read_bytes()).hexdigest(),
         'index_md5': hashlib.md5(index.read_bytes()).hexdigest()}
save_json(DATA_RESULTS / 'data_audit.json', audit)
print(json.dumps(audit, indent=2))
