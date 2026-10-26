"""Resume the official archive using checked HTTP ranges; call downloader to verify/extract."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '00_公共支持'))
from project_paths import DATA_DIR
from download_dataset import URL, SIZE, MD5
CHUNK = 16 * 1024 * 1024


def fetch_range(root, start, end):
    dest = root / f'{start:012d}-{end:012d}.chunk'
    length = end - start + 1
    if dest.exists() and dest.stat().st_size == length:
        return dest
    partial = dest.with_suffix('.partial')
    for attempt in range(8):
        done = partial.stat().st_size if partial.exists() else 0
        if done == length:
            partial.replace(dest)
            return dest
        if done > length:
            raise ValueError('Range partial is larger than expected')
        req = urllib.request.Request(URL, headers={
            'Range': f'bytes={start + done}-{end}',
            'User-Agent': 'UrbanSound8K-course-project/1.0',
            'Accept-Encoding': 'identity',
        })
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                expected = f'bytes {start + done}-{end}/{SIZE}'
                if response.status != 206 or response.headers.get('Content-Range') != expected:
                    raise ValueError(f'Invalid HTTP range: {response.status}, {response.headers.get("Content-Range")}')
                with partial.open('ab') as f:
                    while True:
                        b = response.read(256 * 1024)
                        if not b:
                            break
                        f.write(b)
            if partial.stat().st_size != length:
                raise IOError('Connection ended before requested range completed')
            partial.replace(dest)
            return dest
        except Exception as exc:
            print(f'Range {start}: attempt {attempt + 1} failed: {exc}', flush=True)
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', default=DATA_DIR)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    root = Path(args.destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'UrbanSound8K.tar.gz'
    part = root / 'UrbanSound8K.tar.gz.part'
    chunkdir = root / 'download_chunks'
    chunkdir.mkdir(exist_ok=True)
    if not archive.exists():
        prefix = part.stat().st_size if part.exists() else 0
        if prefix > SIZE:
            raise ValueError('Existing partial file is larger than expected')
        ranges = [(s, min(s + CHUNK, SIZE) - 1) for s in range(prefix, SIZE, CHUNK)]
        cached = sum((chunkdir / f'{start:012d}-{end:012d}.chunk').stat().st_size
                     for start, end in ranges
                     if (chunkdir / f'{start:012d}-{end:012d}.chunk').is_file()
                     and (chunkdir / f'{start:012d}-{end:012d}.chunk').stat().st_size == end - start + 1)
        paths = {}
        completed = prefix
        begun = time.monotonic()
        print(f'Resuming from {prefix / 1e6:.1f} MB; {len(ranges)} ranges, {args.workers} workers', flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(fetch_range, chunkdir, start, end): (start, end) for start, end in ranges}
            for job in as_completed(jobs):
                start, end = jobs[job]
                paths[start] = job.result()
                completed += end - start + 1
                elapsed = time.monotonic() - begun
                print(f'{completed / 1e6:.1f}/{SIZE / 1e6:.1f} MB; {max(0, completed - prefix - cached) / max(elapsed, .01) / 1e6:.2f} MB/s new transfer', flush=True)
        with part.open('ab') as f:
            for start, end in ranges:
                with paths[start].open('rb') as source:
                    shutil.copyfileobj(source, f, 4 * 1024 * 1024)
        if part.stat().st_size != SIZE:
            raise ValueError('Full archive size failed; files retained')
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'download_dataset.py'), '--destination', str(root)], check=True)


if __name__ == '__main__':
    main()
