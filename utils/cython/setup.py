'''
python setup.py build_ext -i
to compile
'''

# setup.py
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from distutils.sysconfig import get_python_lib
import numpy

setup(
    name='mesh_core_cython',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("mesh_core_cython",
                           sources=["mesh_core_cython.pyx", "mesh_core.cpp"],
                           language='c++',
                           include_dirs=[get_python_lib(), numpy.get_include()])],
)
