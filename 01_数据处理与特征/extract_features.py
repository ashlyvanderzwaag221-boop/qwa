"""Verify all official WAVs and cache 40-dimensional MFCC features (CPU only)."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import soundfile as sf
from threadpoolctl import threadpool_limits
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '00_公共支持'))
from project_paths import DATA_DIR, DATA_RESULTS, FEATURES_DIR
from pipeline import CONFIG, feature_vector, read_rows, save_json


def process_one(item):
    row, audio, cache = item
    path, dest = audio / row['path'], cache / (row['clip_id'] + '.npy')
    try:
        if hashlib.md5(path.read_bytes()).hexdigest() != row['md5']:
            raise ValueError('Official audio checksum mismatch')
        info = sf.info(path)
        vector = np.load(dest, allow_pickle=False) if dest.exists() else feature_vector(path)
        if vector.shape != (40,) or not np.isfinite(vector).all():
            raise ValueError('Invalid feature vector')
        if not dest.exists():
            np.save(dest, vector)
        return row['clip_id'], vector, {
            'clip_id': row['clip_id'], 'sample_rate': info.samplerate,
            'channels': info.channels, 'duration': info.duration, 'frames': info.frames,
            'subtype': info.subtype, 'fold': row['fold'], 'class': row['class'],
            'md5_verified': True,
        }, None
    except Exception as exc:
        return row['clip_id'], None, None, str(exc)


def run(manifest_path, audio, cache, workers=6):
    rows, audio, cache = read_rows(manifest_path), Path(audio), Path(cache)
    missing = [r['path'] for r in rows if not (audio / r['path']).is_file()]
    if missing:
        raise ValueError(f'{len(missing)} official audio files are missing; finish download first')
    cache.mkdir(parents=True, exist_ok=True)
    config_path = cache / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != CONFIG:
        raise ValueError('Feature settings changed; use a new cache directory')
    save_json(config_path, CONFIG)
    vectors, ids, headers, failures = [], [], [], []
    begun = time.perf_counter()
    # Warm up lazy imports/JIT once before concurrent librosa calls.
    feature_vector(audio / rows[0]['path'])
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as executor:
        for i, (clip, vector, header, error) in enumerate(executor.map(process_one, [(r, audio, cache) for r in rows]), 1):
            if error:
                failures.append({'clip_id': clip, 'error': error})
            else:
                ids.append(clip)
                vectors.append(vector)
                headers.append(header)
            if i % 250 == 0 or i == len(rows):
                print(f'{i}/{len(rows)} verified and processed; failures={len(failures)}; {time.perf_counter()-begun:.1f}s', flush=True)
    audit = {'expected': len(rows), 'succeeded': len(ids), 'failed': failures,
             'feature_dimensions': 40, 'config': CONFIG, 'seconds': time.perf_counter() - begun,
             'sample_rate_counts': dict(Counter(h['sample_rate'] for h in headers)),
             'channel_counts': dict(Counter(h['channels'] for h in headers)),
             'subtype_counts': dict(Counter(h['subtype'] for h in headers)),
             'duration_min_seconds': min(h['duration'] for h in headers),
             'duration_max_seconds': max(h['duration'] for h in headers),
             'duration_mean_seconds': float(np.mean([h['duration'] for h in headers])),
             'total_duration_hours': sum(h['duration'] for h in headers) / 3600,
             'audio_headers': headers}
    save_json(cache / 'audio_audit.json', audit)
    if failures or len(ids) != 8732:
        raise ValueError('Incomplete feature extraction; inspect audio_audit.json')
    np.savez_compressed(cache / 'features.npz', X=np.stack(vectors), ids=np.array(ids))
    save_json(cache / 'feature_receipt.json', {
        'shape': [len(ids), 40], 'all_finite': True, 'all_audio_md5_verified': True,
        'sha256': hashlib.sha256((cache / 'features.npz').read_bytes()).hexdigest(),
    })
    print('Complete feature matrix saved:', cache / 'features.npz', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=DATA_RESULTS / 'manifest.csv')
    parser.add_argument('--audio', default=DATA_DIR / 'UrbanSound8K')
    parser.add_argument('--cache', default=FEATURES_DIR)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    run(args.manifest, args.audio, args.cache, args.workers)
