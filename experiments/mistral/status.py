"""Compatibility entrypoint; read status without loading ML dependencies."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(["status", "mistral", *sys.argv[1:]]))
