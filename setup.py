from setuptools import setup, Extension
import pybind11
import sys

extra_compile_args = []
extra_link_args = []

if sys.platform.startswith('win'):
    # Windows (MSVC)
    extra_compile_args = ['/O2', '/openmp', '/arch:AVX2']
else:
    # Linux/macOS (GCC/Clang)
    extra_compile_args = ['-O3', '-fopenmp', '-march=native', '-ffast-math', '-funroll-loops']
    extra_link_args = ['-fopenmp']

ext_modules = [
    Extension(
        'agent_sim',
        ['agent_sim.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    ),
]

setup(
    name='agent_sim',
    version='0.9',
    description='C++ accelerated multi-step agent simulation with internal normalization',
    ext_modules=ext_modules,
    install_requires=['pybind11'],
)