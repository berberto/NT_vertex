#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:27:08 2020

@author: andrewg
"""

#jit centroids.
import numpy as np
cimport numpy as np
import cython
from numba import jit
import os
from cells_extra import cells_setup
import time







@cython.boundscheck(False)
@cython.wraparound(False)
def centroids_cy(np.ndarray vertices, np.ndarray nxt , np.ndarray el, int n_face):
    """
    vertices is the array of vertices a cells object
    n_face is the number of faces
    nxt is the array next from mesh
    el is an array.  If el[i]<0, face i is dead.  Otherwise el[i] is an edge in 
    the boundary of face i.
    """
    cdef np.ndarray cents = -1.0*np.empty((n_face,2), dtype=float) 
    for  i in range(n_face):
        cents[i][0] , cents[i][1] = centroid_cy(i, vertices, nxt , el)
    return cents

@cython.boundscheck(False)
@cython.wraparound(False)  
def centroid_cy(int i, np.ndarray vertices, np.ndarray nxt, np.ndarray el):
    cdef int x, count
    if el[i]<0 :#if face i is dead
        return -1.0,-1.0
    else:
        bdy = []
        x = el[i]
        count=0
        bdy.append(x)
        x = nxt[x]
        while x!=el[i]:
            bdy.append(x)
            x = nxt[x]
            count+=1
            if count > 100:
                os._exit(1)
    bdy = np.array(bdy)
    return np.sum(vertices[0][bdy])/len(bdy), np.sum(vertices[1][bdy])/len(bdy) 

cdef centroid_cy_v2(int i, double[:,:] vertices, double[:,:] cent, int[::1] nxt, int[::1] el ):
    cdef int x, count=1
    cdef double x_coord=0.0, y_coord=0.0
    if el[i]<0 :#if face i is dead
        cent[i][0] = -1.0
        cent[i][1] = -1.0
    else:
        x = el[i]        
        x_coord += vertices[0][x]
        y_coord += vertices[1][x]
        x = nxt[x]
        while x!=el[i]:
            x_coord += vertices[0][x]
            y_coord += vertices[1][x]
            count+=1
            x = nxt[x] 
        cent[i][0] = x_coord / count
        cent[i][1] = y_coord / count
            
def cents_cy(np.ndarray vert, np.ndarray nx , np.ndarray e, int n_face):
    cdef double[:,:] vertices = vert
    cdef int[::1] nxt = nx
    cdef int[::1] el = e
    cdef int i = 0
    cdef np.ndarray cents = np.empty((n_face,2), dtype=np.double)
    cdef double[:,:] c = cents
    while (i < n_face):
        centroid_cy_v2(i,vertices, c, nxt, el)
        i+=1
    return cents
        
def c_view(np.ndarray vert):
    cdef double[:,:] vertices = vert
    return vertices[0][0]     
          


