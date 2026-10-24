import os
import hashlib
import json
from pathlib import Path
REPO = Path(os.environ.get('CACHEGEN_REPO_ROOT', Path(__file__).resolve().parents[2])).resolve()
root=Path(os.environ.get('HF_HOME', REPO / '.venv/huggingface')) / 'hub/models--mistral-community--Mistral-7B-v0.2'
revision='80cf49b18de6354fd8a6d1e81c177dd830e05eea'
snapshot=root/'snapshots'/revision
expected={
'model-00001-of-00003.safetensors':'fbebdaa87478f5b1f9d01539963874db56958ff8cf8d801e465d78ce24328069',
'model-00002-of-00003.safetensors':'79a5f871a0fbb7919f263d78caff0d4d3b7a1d444552db3011ada70f48fd5ee4',
'model-00003-of-00003.safetensors':'3db96f068de2a5b5947692954c252a5f74a1c8ad7d87ea950b5e396a7ddeca27',
}
files=[]
for name,digest in expected.items():
 path=snapshot/name
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==digest, f'SHA-256 mismatch: {name}'
 files.append({'file':name,'bytes':path.stat().st_size,'sha256':digest})
 print('Verified',name,flush=True)
for name in ['config.json','generation_config.json','model.safetensors.index.json','tokenizer.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json']:
 assert (snapshot/name).is_file(), f'Missing {name}'
(root/'refs').mkdir(exist_ok=True)
(root/'refs'/'main').write_text(revision)
(REPO / '.venv/mistral-download-verified.json').write_text(json.dumps({'model':'mistral-community/Mistral-7B-v0.2','revision':revision,'files':files,'status':'verified'},indent=2)+'\n')
print('MISTRAL_DOWNLOAD_VERIFIED',flush=True)
