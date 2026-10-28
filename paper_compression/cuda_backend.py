"""Isolated 8bit-alphabet build of the repository's arithmetic CUDA kernels.

Never replaces torchac_cuda or edits vendor files. Construction/compilation is
lazy and is allowed only when the caller actually requests CUDA on the server.
"""
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import re


def sources():
    root = Path(__file__).resolve().parents[1] / 'LMCache/third_party/torchac_cuda'
    converted = {}
    for name in ('torchac_kernel_enc_new.cu', 'torchac_kernel_dec_new.cu'):
        original = (root / name).read_text()
        text, count = re.subn(r'#define MAX_LP (48|64)\b', '#define MAX_LP 257', original)
        if count != 1 or '#define MAX_THREAD_PER_BLOCK 128' not in text:
            raise ValueError('Vendor CUDA layout changed; review the 8bit adaptation')
        text = text.replace('#define MAX_THREAD_PER_BLOCK 128', '#define MAX_THREAD_PER_BLOCK 32')
        text, count = re.subn(r'        case (64|128):\n            LAUNCH_[A-Z_]+\(\1\);\n            break;\n', '', text)
        expected = 2 if 'enc_' in name else 4
        if count != expected:
            raise ValueError('Unexpected CUDA dispatch; refusing an unreviewed patch')
        text = text.replace('#define MAX_LP 257', 'namespace paper_ac256 {\n#define MAX_LP 257', 1)
        converted[name] = text + '\n} // namespace paper_ac256\n'
    header = (root / 'torchac_kernel.cuh').read_text()
    if 'const int precision = 16;' not in header:
        raise ValueError('Vendor header layout changed')
    converted['torchac_kernel.cuh'] = header.replace('const int precision = 16;',
        'namespace paper_ac256 {\nconst int precision = 16;', 1) + '\n} // namespace paper_ac256\n'
    converted['bindings.cpp'] = '''#include "torchac_kernel.cuh"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_fast_new", &paper_ac256::encode_cuda_new);
    m.def("decode_fast_prefsum", &paper_ac256::decode_cuda_prefsum);
}
'''
    return converted


@lru_cache(maxsize=1)
def extension():
    import torch
    from torch.utils.cpp_extension import load
    if not torch.cuda.is_available():
        raise RuntimeError('The paper CUDA codec requires a GPU; no automatic CPU substitution')
    files = sources()
    identity = hashlib.sha256(''.join(files[k] for k in sorted(files)).encode()).hexdigest()[:16]
    folder = Path(os.environ.get('TORCH_EXTENSIONS_DIR', str(Path.home() / '.cache/torch_extensions'))) / ('paper_ac256_' + identity)
    folder.mkdir(parents=True, exist_ok=True)
    for name, contents in files.items():
        path = folder / name
        if not path.exists() or path.read_text() != contents:
            path.write_text(contents)
    return load(name='paper_ac256_' + identity,
                sources=[str(folder / n) for n in ('bindings.cpp', 'torchac_kernel_enc_new.cu', 'torchac_kernel_dec_new.cu')],
                extra_include_paths=[str(folder)], build_directory=str(folder), verbose=True)
