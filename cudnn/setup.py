import torch.cuda
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from torch.utils.cpp_extension import CUDA_HOME

setup(name='convtbcglu',
      ext_modules=[CUDAExtension('convtbcglu', ['convtbcglu.cpp'])],
      cmdclass={'build_ext': BuildExtension})
