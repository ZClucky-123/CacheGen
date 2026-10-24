import json,subprocess,sys,time
from pathlib import Path
repo=Path(__file__).resolve().parents[1]
state=repo/'.local/launches/tiny-v2-final-check.json'
while True:
    data=json.loads(state.read_text())
    if 'exit_code' in data:break
    time.sleep(3)
if data['exit_code']:
    raise SystemExit(data['exit_code'])
command=[sys.executable,'-m','experiments.background','start','--name','mistral-v2-small-final','--','--model','mistral','--config','experiments/configs/small.json','--run-id','v2-small-final-20261024']
with (repo/'.local/final-sequence.log').open('a') as log:
    result=subprocess.run(command,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
(repo/'.local/final-sequence.json').write_text(json.dumps({'tiny_exit_code':data['exit_code'],'mistral_launch_exit_code':result.returncode,'command':command},indent=2)+'\n')
raise SystemExit(result.returncode)
