"""
Created on Sun Jun  7 10:10:44 2020

@author: andrewg, copied from Langtangen
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules=[
	Extension('_fe_cy_omp',
			  ['fe_cy_omp.pyx'],
			  extra_compile_args = ["-O3", "-ffast-math"],
			  extra_link_args = ["-fopenmp"],
			  include_dirs = [numpy.get_include()],)
]

setup(name='OpenMP FE step',
	  ext_modules = ext_modules,
      cmdclass={'build_ext':build_ext}
)