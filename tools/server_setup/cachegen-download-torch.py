import os
import concurrent.futures
import hashlib
import time
from pathlib import Path
REPO = Path(os.environ.get('CACHEGEN_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
import requests

URL='https://download.pytorch.org/whl/cu121/torch-2.3.1%2Bcu121-cp310-cp310-linux_x86_64.whl'
EXPECTED='f0deb5d2f932a68ed54625ba140eddbf2af22be978ee19b9b63c986add6425b2'
root=REPO / '.venv/wheelhouse'
root.mkdir(exist_ok=True)
parts=root/'torch-parts'
parts.mkdir(exist_ok=True)
wheel=root/'torch-2.3.1+cu121-cp310-cp310-linux_x86_64.whl'
s=requests.Session();s.trust_env=False
r=s.get(URL,headers={'Range':'bytes=0-0'},timeout=30)
r.raise_for_status()
assert r.status_code==206 and len(r.content)==1, 'Range requests unsupported'
total=int(r.headers['Content-Range'].split('/')[-1])
block=16*1024*1024

def download(i):
 start=i*block;end=min(total,start+block)-1
 path=parts/f'{i:04d}'
 if path.exists() and path.stat().st_size==end-start+1:return
 for attempt in range(5):
  try:
   session=requests.Session();session.trust_env=False
   with session.get(URL,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(20,60)) as response:
    response.raise_for_status()
    assert response.status_code==206 and response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
    with path.open('wb') as stream:
     for chunk in response.iter_content(1024*1024):stream.write(chunk)
   assert path.stat().st_size==end-start+1, 'Truncated chunk'
   print('Completed chunk',i,flush=True)
   return
  except (requests.RequestException,OSError,AssertionError):
   if attempt==4:raise
   time.sleep(2*(attempt+1))

count=(total+block-1)//block
print('Downloading official Torch wheel bytes:',total,'chunks:',count,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(download,range(count)))
h=hashlib.sha256()
with wheel.open('wb') as output:
 for i in range(count):
  with (parts/f'{i:04d}').open('rb') as stream:
   for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk);output.write(chunk)
assert wheel.stat().st_size==total
assert h.hexdigest()==EXPECTED, 'Official SHA-256 mismatch'
for path in parts.iterdir():path.unlink()
parts.rmdir()
print('TORCH_OFFICIAL_SHA256_VERIFIED',EXPECTED,flush=True)
