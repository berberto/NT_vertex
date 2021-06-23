#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import numpy as np
import os


def division_axis(mesh,face_id,rand):
    """Choose a random division axis (given as a pair of boundary edges to bisect) for the given cell.
    
    The first edge is chosen randomly from the bounding edges of the cell with probability proportional 
    to edge length. The second edge is then fixed to be n_edge/2 from the first. 
    """
    edges = mesh.boundary(face_id) #was mesh.boundary
    if edges==[-1]: ##was if edges==[-1]
        print('here')
        os._exit(1)
    p = np.cumsum(mesh.length[edges])
    e0 = p.searchsorted(rand.rand()*p[-1])
    return edges[e0],edges[e0-len(edges)/2] 

def mod_division_axis(mesh,face_id):
    """Choose a random division axis (given as a pair of boundary edges to bisect) for the given cell.
    
    The first edge is chosen randomly from the bounding edges of the cell with probability proportional 
    to edge length. The second edge is then fixed to be n_edge/2 from the first. 
    """
    edges = mesh.boundary_liv(face_id) #list
    l = len(edges)
    rand_ind = np.random.random_integers(0, l-1) #index that will have a node as a midpt
    approx_opp_ind = (rand_ind + int(l/2)) % l # index of opposite edge
    return edges[rand_ind],edges[approx_opp_ind]  
