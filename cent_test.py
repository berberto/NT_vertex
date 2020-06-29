#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:24:23 2020

@author: andrewg
"""

from cells_extra import cells_setup
from _centroids_cy import centroids_cy, centroid_cy, cents_cy, c_view
import time
from Finite_Element import centroids2
import numpy as np
from numba import jit
#from mesh import _edge_lookup


def cen(c):
    """
    Args:
        c is a cells object
    Returns:
        The list of centroids.
    """
    return centroids_cy(c.mesh.vertices, c.mesh.edges.next, c.mesh._edge_lookup, c.mesh.n_face)

def cen2(c):
    """
    Args:
        c is a cells object
    Returns:
        The list of centroids.
    """
    return cents_cy(c.mesh.vertices.astype(np.float64), c.mesh.edges.next.astype(np.intc), c.mesh._edge_lookup.astype(np.intc), int(c.mesh.n_face))#.astype(np.intc)


if  __name__ == "__main__":
    c = cells_setup(size=[100,6])
    t1 =time.time()

    #a = centroids_cy(c.mesh.vertices, c.mesh.edges.next, c.mesh._edge_lookup, c.mesh.n_face)
    a = cen(c)
    t2 =time.time()

    print("cython 1 ", t2 -t1)

    t3=time.time()
    b = centroids2(c)
    t4 =time.time()
    print("normal ",t4 - t3)

    t5=time.time()
    df = cen2(c)
    t6 =time.time()
    print("cython 2" , t6 - t5)

    print("cython is ", (t4 - t3)/(t6 - t5), " times faster.")

    print(a[1], b[1],df[1])

    print(c_view(c.mesh.vertices))
