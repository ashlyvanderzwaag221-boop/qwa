"""Download official archive with resume support, checksum verification and safe extraction."""
import argparse,urllib.request,hashlib,tarfile,time,re,json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '00_公共支持'))
from project_paths import DATA_DIR
URL='https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz?download=1'
SIZE=6023741708
MD5='9aa69802bbf37fb986f71ec1483a196e'
def main():
 p=argparse.ArgumentParser();p.add_argument('--destination',default=DATA_DIR);a=p.parse_args();root=Path(a.destination).resolve();root.mkdir(parents=True,exist_ok=True)
 archive=root/'UrbanSound8K.tar.gz';part=root/'UrbanSound8K.tar.gz.part'
 if not archive.exists():
  start=part.stat().st_size if part.exists() else 0
  if start>SIZE:raise ValueError('Partial file exceeds expected size')
  if start<SIZE:
   req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-'} if start else {})
   with urllib.request.urlopen(req,timeout=60) as r:
    if start and r.status==206:
     if not r.headers.get('Content-Range','').startswith(f'bytes {start}-'):raise ValueError('Unexpected resume offset')
    elif start:start=0
    with part.open('ab' if start else 'wb') as f:
     n=start;last=time.time()
     while True:
      b=r.read(4*1024*1024)
      if not b:break
      f.write(b);n+=len(b)
      if time.time()-last>=10:print(f'{n/1e6:.0f} / {SIZE/1e6:.0f} MB',flush=True);last=time.time()
  if part.stat().st_size!=SIZE:raise ValueError('Incomplete download; rerun to resume')
  archive_candidate=part
 else:archive_candidate=archive
 h=hashlib.md5()
 with archive_candidate.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 if h.hexdigest()!=MD5:raise ValueError('Archive checksum mismatch; file retained for inspection')
 if archive_candidate==part:part.replace(archive)
 print('Official archive MD5 verified. Extracting...',flush=True)
 count=0
 with tarfile.open(archive,'r:gz') as t:
  for m in t:
   target=(root/m.name).resolve()
   if not target.is_relative_to(root) or m.issym() or m.islnk():raise ValueError('Unsafe archive member')
   if m.isdir():target.mkdir(parents=True,exist_ok=True)
   elif m.isfile():
    target.parent.mkdir(parents=True,exist_ok=True)
    with t.extractfile(m) as src,target.open('wb') as dst:
     for b in iter(lambda:src.read(1024*1024),b''):dst.write(b)
    count+=1
    if count%1000==0:print(f'Extracted {count} files',flush=True)
 (root/'download_receipt.json').write_text(json.dumps({'url':URL,'bytes':SIZE,'md5':MD5,'verified':True,'extracted_files':count},indent=2),encoding='utf-8')
 print('Verified and extracted:',root/'UrbanSound8K')
if __name__=='__main__':main()
