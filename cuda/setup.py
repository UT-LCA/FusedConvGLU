from setuptools import setup
#from torch.utils.cpp_extension import CppExtension, BuildExtension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

#setup(name='convtbcglu',
#      ext_modules=[CppExtension('convtbcglu', ['convtbcglu.cpp'])],
#      cmdclass={'build_ext': BuildExtension})
			 
setup(
    name='convtbcglu',
    ext_modules=[
        CUDAExtension('convtbcglu_cuda', [
            'convtbcglu.cpp',
            'convtbcglu_cuda.cu',
        ]
		)  
		
    ],
    cmdclass={
        'build_ext': BuildExtension
    })