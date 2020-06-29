#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 10:10:44 2020

@author: andrewg, copied from Langtangen
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(name='fe_attempt',
      ext_modules=[Extension('_centroids_cy', ['centroids_cy.pyx'],include_dirs = [numpy.get_include()],)],
      cmdclass={'build_ext':build_ext}
)