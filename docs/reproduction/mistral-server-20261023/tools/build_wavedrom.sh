#!/usr/bin/env bash
set -euo pipefail
build_temp=$(mktemp -d /tmp/cachegen-wavedrom-build.XXXXXX)
/home/aaa/CacheGen/.cuda-12.1/bin/python -m venv "$build_temp"
"$build_temp/bin/python" -m pip install --index-url https://pypi.org/simple pip==24.0 setuptools==69.5.1 wheel==0.43.0 packaging==24.1 setuptools-scm==8.1.0 tomli==2.0.1
"$build_temp/bin/python" -m pip wheel --no-deps --no-build-isolation --wheel-dir /home/aaa/CacheGen/.venv/wheelhouse /home/aaa/CacheGen/.venv/wheelhouse/wavedrom-2.0.3.post3.tar.gz
/home/aaa/CacheGen/.venv/bin/python -m pip install --no-deps /home/aaa/CacheGen/.venv/wheelhouse/wavedrom-2.0.3.post3-*.whl
printf 'WAVEDROM_BUILD_CONFLICT_RESOLVED\n'
