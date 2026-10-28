"""Seal completed stage1 artifacts; verify before spending GPU time on stage2."""
import argparse
import json
from pathlib import Path
from tools.paper_compression import provenance, sha256, save


def manifest(root):
    sources = provenance()
    paths = []
    for phase in ('insights', 'profiles', 'validation', 'test'):
        folder = root / phase
        status = json.loads((folder / 'status.json').read_text())
        if status['status'] != 'completed':
            raise ValueError('Incomplete stage1 phase: ' + phase)
        if json.loads((folder / 'source_hashes.json').read_text()) != sources:
            raise ValueError('Source code differs from stage1 phase: ' + phase)
        paths.extend(p for p in folder.rglob('*') if p.is_file())
    for phase in ('validation-plots', 'test-plots'):
        folder = root / phase
        for length in (1024, 2048, 4096):
            for suffix in ('png', 'pdf'):
                if not (folder / f'curve-{length}.{suffix}').is_file():
                    raise ValueError('Stage1 plot missing')
        paths.extend(p for p in folder.rglob('*') if p.is_file())
    if not (root / 'test-plots/encoding-error.pdf').is_file():
        raise ValueError('Stage1 encoding-error plot missing')
    return {'format': 'paper-core-stage1-v1', 'source_hashes': sources,
            'files': {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('seal', 'verify'))
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    path = args.root / 'stage1-complete.json'
    current = manifest(args.root)
    if args.action == 'seal':
        if path.exists():
            raise FileExistsError(path)
        save(path, current)
    elif json.loads(path.read_text()) != current:
        raise ValueError('Stage1 artifacts changed after completion')
    print('STAGE1', args.action.upper(), 'OK:', args.root, flush=True)


if __name__ == '__main__':
    main()
