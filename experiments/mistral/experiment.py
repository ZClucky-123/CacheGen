"""Compatibility entrypoint. All current measurements use experiments.suite."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.scoring import normalize, score_qa
from experiments.engine import Engine, fresh_cache, timed, validate_blob
from experiments.suite import main as suite_main


def main():
    args = sys.argv[1:]
    if '--resume' not in args:
        args = ['--model', 'mistral'] + args
    return suite_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
