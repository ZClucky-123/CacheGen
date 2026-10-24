"""Show progress without attaching to tmux or touching the GPU."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

root=Path('/root/cachegen-repro/latest')
manifest=json.loads((root/'manifest.json').read_text())
started=datetime.fromisoformat(manifest['started_utc'])
end=datetime.fromisoformat(manifest.get('finished_utc',datetime.now(timezone.utc).isoformat()))
result={'run_directory':str(root.resolve()),'status':manifest['status'],
        'started_china_time':started.astimezone(timezone(timedelta(hours=8))).isoformat(),
        'elapsed_minutes':round((end-started).total_seconds()/60,1),'progress':manifest['progress']}
if (root/'quality.json').exists():
    data=json.loads((root/'quality.json').read_text())
    result['natural_contexts']=f"{len(data['continuation'])}/30"
    result['qa_questions']=f"{len(data['qa'])}/10"
if (root/'ttft.json').exists():
    data=json.loads((root/'ttft.json').read_text())
    count=sum(len(r['measurements']) for r in data['cases'])
    count+=len(data.get('current_case',{}).get('measurements',[]))
    result['ttft_measurements']=f'{count}/540'
if (root/'artifacts.json').exists():result['artifacts']=json.loads((root/'artifacts.json').read_text())['status']
if (root/'exit_code').exists():result['exit_code']=int((root/'exit_code').read_text())
print(json.dumps(result,ensure_ascii=False,indent=2))
