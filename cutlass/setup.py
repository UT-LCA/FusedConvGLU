from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='convtbcglu',
      ext_modules=[CppExtension('convtbcglu', ['convtbcglu.cpp'])],
      cmdclass={'build_ext': BuildExtension})
