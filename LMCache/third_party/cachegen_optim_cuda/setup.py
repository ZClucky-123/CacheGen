from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).resolve().parent
setup(name='cachegen_optim_cuda', version='2.0.0', ext_modules=[CUDAExtension(
    'cachegen_optim_cuda', [str(root / f) for f in
        ('bindings.cpp','cdf_codec.cu','mixed_bits.cu','restore_kv.cu')],
    extra_compile_args={'cxx':['-O3'], 'nvcc':['-O3','--fmad=false','--ftz=false','--prec-div=true']})],
    cmdclass={'build_ext':BuildExtension})
