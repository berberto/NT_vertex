# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy


ext_modules=[
	Extension('_centroids_cy',
			  ['centroids_cy.pyx'],
			  extra_compile_args = ["-O3", "-ffast-math", "-mavx"],
			  include_dirs = [numpy.get_include()],
	),
	Extension('_fe_cy',
			  ['fe_cy.pyx'],
			  extra_compile_args = ["-O3", "-ffast-math", "-mavx"],	# "-march=native"
			  include_dirs = [numpy.get_include()],
	),
	Extension('_fe_cy_omp',
			  ['fe_cy_omp.pyx'],
			  extra_compile_args = ["-fopenmp", "-O3", "-ffast-math", "-mavx"],
			  extra_link_args = ["-fopenmp"],
			  include_dirs = [numpy.get_include()],
	)
]


setup(name='FE step modules',
	  ext_modules = cythonize(ext_modules,language_level=3.6),
      cmdclass={'build_ext':build_ext}
)