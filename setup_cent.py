#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 10:10:44 2020

@author: andrewg, copied from Langtangen
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

# import numpy

ext_modules=[
	Extension('_centroids_cy',
			  ['centroids_cy.pyx'],
			  # extra_compile_args = ["-xhost"], # , "-O3", "-ffast-math"],
			  # extra_link_args = ["-xhost", "-fopenmp"],
			  # include_dirs = [numpy.get_include()],
	)
]


setup(name='FE step',
	  ext_modules = cythonize(ext_modules,language_level=3.6),
      cmdclass={'build_ext':build_ext}
)