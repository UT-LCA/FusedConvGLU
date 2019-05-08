from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='conv_glu',
      #ext_modules=[CppExtension('conv_glu', ['conv_glu.cpp'])],
      #cmdclass={'build_ext': BuildExtension})
      ext_modules=[
        CUDAExtension('conv_glu_cuda', [
          'conv_glu.cpp',
          'conv_glu_kernel.cu',
        ],
        extra_compile_args={'cxx': ['-I/root/cutlass'],
                            'nvcc': ['-I/root/cutlass']})
      ],
      cmdclass={
          'build_ext': BuildExtension
      })
