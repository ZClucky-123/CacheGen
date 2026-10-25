import json,os,signal,sqlite3,subprocess,sys,time
from pathlib import Path
run=Path('results/mistral/minimal/v3-final-03')
for _ in range(3600):
    if not (run/'manifest.json').exists():
        time.sleep(.5)
        continue
    m=json.loads((run/'manifest.json').read_text())
    if m['status']=='initializing':
        time.sleep(.5)
        continue
    progress=m.get('progress',{})
    if m['status'] != 'running':
        raise SystemExit('Validation stopped before controlled recovery: '+m['status'])
    if progress.get('stage')=='ttft' and progress.get('completed',0)>=80:
        pid=m['runner_pid']
        command=Path(f'/proc/{pid}/cmdline').read_bytes()
        if b'experiments.suite' not in command or b'v3-final-03' not in command:
            raise SystemExit('Runner PID no longer belongs to this verification')
        db=sqlite3.connect('file:'+str(run/'records.sqlite3')+'?mode=ro',uri=True)
        prior=dict(db.execute('select key,sha256 from observations'))
        db.close()
        prepared={str(p.relative_to(run)):json.loads(p.read_text()) for p in (run/'pending-prepared').rglob('prepared.json')}
        Path('.local/v3-recovery-before-03.json').write_text(json.dumps({'run':str(run),'runner_pid':pid,'prior_observations':prior,'prepared':prepared,'progress':progress},indent=2)+'\n')
        os.kill(pid,signal.SIGTERM)
        print('Controlled TERM sent; prior durable rows:',len(prior),flush=True)
        break
    time.sleep(.5)
else:
    raise SystemExit('Timed out waiting for TTFT')
for _ in range(240):
    launch=json.loads(Path('.local/launches/mistral-v3-final-03.json').read_text())
    if launch['status']=='interrupted' and launch.get('exit_code')==143:
        subprocess.check_call([sys.executable,'-m','experiments.background','start','--name','mistral-v3-resume-03','--','--resume',str(run)])
        break
    if launch['status']=='failed':
        raise SystemExit('Interrupted worker failed unexpectedly')
    time.sleep(.5)
else:
    raise SystemExit('Timed out waiting for checkpoint shutdown')
