from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

extra_objects = []
extra_compile_args = {"cxx": ['-O3']}
extra_include_paths = []
setup(
    name='topt',
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name='topt._c',
            sources=['topt/register.cpp', 'topt/fusion_pass.cpp', 'topt/fusion_adddiv.cpp', 'topt/compiler.cpp', 'topt/topt_compiler.cpp'],
            extra_objects=extra_objects,
            extra_compile_args=extra_compile_args,
            include_dirs=extra_include_paths
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
