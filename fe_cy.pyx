#cython: boundscheck=False, wraparound=False, nonecheck=False
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:07:58 2020

@author: andrewg
"""

import numpy as np
import scipy
cimport numpy as np
import time
import cython
from libc.math cimport fabs
# from _mat_Solve import cabbage
from scipy.linalg.cython_lapack cimport dgesv, dgelsd
from scipy.linalg import cho_factor, cho_solve

def ev_cy_trivial(np.ndarray new_verts):
    return np.zeros(len(new_verts))

def ev_cy_sparse(np.ndarray old_verts, np.ndarray new_verts, np.ndarray old_cents,  np.ndarray new_cents, np.ndarray old_con, np.ndarray nx,np.ndarray f_by_e, np.ndarray  e_t_n, np.ndarray f_t_n, np.ndarray f , int n_edge , double v, double dt ):
    cdef int m = len(old_con)
    cdef np.ndarray a = np.zeros((m,m), dtype = np.float_) 
    cdef double[:,:]A = a #memoryview of a
    cdef np.ndarray b_vect = np.zeros(m, dtype = np.float_)#bv stands for b vector
    cdef double[::1] bv = b_vect #memoryview of b_vect
    cdef double[:,:] ov = old_verts
    cdef double[:,:] nv = new_verts
    cdef double[:,:] oc = old_cents
    cdef double[:,:] nc = new_cents
    cdef double[::1] o_con = old_con
    cdef int[::1] nxt = nx #next
    cdef int[::1] fbe = f_by_e # face by edge
    cdef int[::1] etn = e_t_n #edges to nodes
    cdef int[::1] ftn = f_t_n#faces to nodes
    cdef double[3][2] nds #coords of triangle in loop
    cdef double[:,:] nodes = nds #memoryview
    cdef double[3][2] prev_nds #previous coords of triangle 
    cdef double[:,:] prev_nodes = prev_nds  #memoryview
    cdef double[::1] s_fn = f # memoryview of source
    cdef double d #abs value of determinant of mat
    cdef double old_d #determinant (abs value of)
    cdef double[2][2] mat #matrix to be used in  update of A
    cdef double[:,:] Mat = mat # memoryview of mat
    cdef double[3][2] nP # to store nab_Phi
    cdef double [:,:] nab_Phi  = nP #memoryview 
    cdef int[3] nd_id #node ids for triangle
    cdef int[::1] node_ids = nd_id
    cdef int e=0 #to index the loop over edges
    cdef int i=0 #index 
    cdef int j=0
    for e in range(n_edge):
        set_up_nodes(nodes , nv, nc , nxt, fbe, e)
        set_up_nodes(prev_nodes , ov, oc , nxt, fbe, e)
        set_node_ids(node_ids, nxt, fbe, etn, ftn , e)
        M_c( nodes , Mat )
        d = just_det(nodes)
        old_d = just_det( prev_nodes )
        nabPhi2_c( Mat , nab_Phi)
        for i in range(3):
            bv[node_ids[i]]+=b_c2(i, d, old_d, s_fn, old_con , fbe , e ,node_ids ,dt) 
            for j in range(3):
                A[node_ids[i]][node_ids[j]]+=I_c(i,j,d)+K_c(i,j,d,nab_Phi,v)+W_c(i,j,d,nab_Phi,nodes, prev_nodes)
    
    return scipy.sparse.linalg.spsolve(scipy.sparse.coo_matrix(a),b_vect)


def ev_cy(np.ndarray old_verts, np.ndarray new_verts, np.ndarray old_cents,  np.ndarray new_cents, np.ndarray old_con, np.ndarray nx,np.ndarray f_by_e, np.ndarray  e_t_n, np.ndarray f_t_n, np.ndarray f , int n_edge , double v, double dt ):
    cdef int m = len(old_con)
    cdef np.ndarray a = np.zeros((m,m), dtype = np.float_) 
    cdef double[:,:]A = a #memoryview of a
    cdef np.ndarray b_vect = np.zeros(m, dtype = np.float_)#bv stands for b vector
    cdef double[::1] bv = b_vect #memoryview of b_vect
    cdef double[:,:] ov = old_verts
    cdef double[:,:] nv = new_verts
    cdef double[:,:] oc = old_cents
    cdef double[:,:] nc = new_cents
    cdef double[::1] o_con = old_con
    cdef int[::1] nxt = nx #next
    cdef int[::1] fbe = f_by_e # face by edge
    cdef int[::1] etn = e_t_n #edges to nodes
    cdef int[::1] ftn = f_t_n#faces to nodes
    cdef double[3][2] nds #coords of triangle in loop
    cdef double[:,:] nodes = nds #memoryview
    cdef double[3][2] prev_nds #previous coords of triangle 
    cdef double[:,:] prev_nodes = prev_nds  #memoryview
    cdef double[::1] s_fn = f # memoryview of source
    cdef double d #abs value of determinant of mat
    cdef double old_d #determinant (abs value of)
    cdef double[2][2] mat #matrix to be used in  update of A
    cdef double[:,:] Mat = mat # memoryview of mat
    cdef double[3][2] nP # to store nab_Phi
    cdef double [:,:] nab_Phi  = nP #memoryview 
    cdef int[3] nd_id #node ids for triangle
    cdef int[::1] node_ids = nd_id
    cdef int e=0 #to index the loop over edges
    cdef int i=0 #index 
    cdef int j=0
    for e in range(n_edge):
        set_up_nodes(nodes , nv, nc , nxt, fbe, e)
        set_up_nodes(prev_nodes , ov, oc , nxt, fbe, e)
        set_node_ids(node_ids, nxt, fbe, etn, ftn , e)
        M_c( nodes , Mat )
        d = just_det(nodes)
        old_d = just_det( prev_nodes )
        nabPhi2_c( Mat , nab_Phi)
        for i in range(3):
            bv[node_ids[i]]+=b_c2(i, d, old_d, s_fn, old_con , fbe , e ,node_ids ,dt) 
            for j in range(3):
                A[node_ids[i]][node_ids[j]]+=I_c(i,j,d)+dt*K_c(i,j,d,nab_Phi,v)+W_c(i,j,d,nab_Phi,nodes, prev_nodes)

    return np.linalg.solve(a,b_vect)


    

cpdef void set_up_nodes(double[:,:] nds , double[:,:] verts, double[:,:] cents , int[::1] nxt, int[::1] fbe, int e) nogil:
    nds[0,0] = verts[e,0] #used transposed version of cells.mesh.vertices
    nds[0,1] = verts[e,1]
    nds[1,0] = verts[nxt[e],0] #used transposed version of cells.mesh.vertices
    nds[1,1] = verts[nxt[e],1]
    nds[2,0] = cents[fbe[e],0] #used transposed version of cells.mesh.vertices
    nds[2,1] = cents[fbe[e],1]


cpdef void set_node_ids(int[::1] node_ids, int[::1] nxt, int[::1] fbe, int[::1] etn, int[::1] ftn , int e) nogil:
    node_ids[0] = etn[e]
    node_ids[1] = etn[nxt[e]]
    node_ids[2] = ftn[fbe[e]]

    
cpdef void set_red_f(double[::1] reduced_f, double[::1] s_fn, int[::1] fbe, int e) nogil:
    reduced_f[0] = 0.0
    reduced_f[1] = 0.0
    reduced_f[2] = s_fn[fbe[e]]
    
    
         


cpdef void M_c( double [:, :] nodes , double [:, :] Mat ):
    """
    Sets the values in the matrix Mat.
    Args:
        nodes is a list of three coordinates.
    Return:
        FE map matrix.
    """
    Mat[0,0] = nodes[1,0] - nodes[0,0]
    Mat[0,1] = nodes[2,0] - nodes[0,0]
    Mat[1,0] = nodes[1,1] - nodes[0,1]
    Mat[1,1] = nodes[2,1] - nodes[0,1]
    #print "Mc", Mat
    

cdef double just_det( double [:, :] nodes ):
    """
    Returns the absolute value of the determinant of the matrix M derived
    from the triangle.
    
    Args:
        reference to an array of three coordinates which form a triangle.
    Return:
        FE map matrix.
    """
    return fabs((nodes[1][0] - nodes[0][0])*(nodes[2][1] - nodes[0][1]) - (nodes[2][0] - nodes[0][0])*(nodes[1][1] - nodes[0][1]))
      
    

    
cdef double I_c(int i, int j, double d):
    """
    Args:
        d is the absolute value of the determinant of the map matrix M
    """
    if ( i - j ):
        return (1.0/24)*d
    else:
        return (1.0/12)*d


    


cdef nabPhi2_c(double [:,:] M , double [:,:] nab_Phi):
    """
    Sets the values of nab_Phi.
    Args:
        M is the FE matrix map.
    """
    cdef double dm = 1.0/(M[0][0]*M[1][1] -  M[0][1]*M[1][0])
    cdef double[2][2] inv_t
    inv_t[0][0] = dm * M[1][1] #inv_t is inverse transpose of M
    inv_t[1][1] = dm * M[0][0]
    inv_t[0][1] = -dm * M[1][0] #forgot minus
    inv_t[1][0] = -dm * M[0][1] #forgot minus
    nab_Phi[0][0] = -inv_t[0][0] - inv_t[0][1]
    nab_Phi[0][1] = -inv_t[1][0] - inv_t[1][1]
    nab_Phi[1][0] = inv_t[0][0]  
    nab_Phi[1][1] = inv_t[1][0]  
    nab_Phi[2][0] = inv_t[0][1]
    nab_Phi[2][1] = inv_t[1][1]
        



cdef double K_c(int i, int j, double d, double[:,:] nabPhi, double v):
    """
    FROM FiniteElement
    i,j are indices of triangle.  So, i,j belong to {0,1,2}.
    M is a matrix
    d is the absolute value of the determinant of M
    v is a constant (the diffusion coefficient)
    nabPhi is the output from New_nabPhi.
    This is a contribution to A[triangle[i],triangle[j]] when updating the 
    matrix with the part of the integral from triangle.
    """   
    return (0.5)*v*(nabPhi[i][0]*nabPhi[j][0]+nabPhi[i][1]*nabPhi[j][1])*d        



cdef double W_c(int i,int j,double d, double[:,:] nabPhi, double[:,:] nodes,  double[:,:] previous_nodes):
    """
    Args:
        i,j in {0,1,2}
        d is the absolute value of det(M), where M is the matrix map
        nabPhi is the output from nabPhi.
        nodes is a list of three node coordinates corresponding to a triangle
        previous_nodes are the positions of the three nodes at the previous time step.
        
    """
    cdef double[2] P0, P1, P2, dummy
    P0[0] = nodes[0][0]-previous_nodes[0][0] 
    P0[1] = nodes[0][1]-previous_nodes[0][1]
    P1[0] = nodes[1][0]-previous_nodes[1][0]
    P1[1] = nodes[1][1]-previous_nodes[1][1]
    P2[0] = nodes[2][0]-previous_nodes[2][0]
    P2[1] = nodes[2][1]-previous_nodes[2][1]
    dummy[0] = P0[0] + P1[0] + P2[0] + (nodes[j][0]-previous_nodes[j][0])
    dummy[1] = P0[1] + P1[1] + P2[1] + (nodes[j][1]-previous_nodes[j][1])
    return (1.0 /24)*d*(nabPhi[i][0]*dummy[0]+nabPhi[i][1]*dummy[1])


cdef b_c(int i, double d, double d_old, double[::1] f, double[::1] old_alpha, double dt): 
    """
    Args:
            i is an index of triangle.  That is, i belongs to {0,1,2}.
            triangle is a list of three node indices.
            d is the determinant of M.
            d_old is the determinant of M from the last time step.
            f is the source vector of length = len(nodes).
            f is the vector coefficients of the source expressed in terms of hat functions.
            old_alpha is the vector of coefficients alphas from the previous time step.
            dt is the time step
    """
    cdef double dummy
    dummy = I_c(i,0,d)*f[0]
    dummy +=I_c(i,1,d)*f[1]
    dummy +=I_c(i,2,d)*f[2]
    dummy +=I_c(i,0,d_old)*old_alpha[0]
    dummy +=I_c(i,1,d_old)*old_alpha[1]
    dummy +=I_c(i,2,d_old)*old_alpha[2]
    return dummy

cdef b_c2(int i, double d, double d_old, double[::1] f, double[::1] old_alpha, int[::1] fbe , int e ,  int[::1]node_ids ,  double dt): 
    """
    Args:
            i is an index of triangle.  That is, i belongs to {0,1,2}.
            triangle is a list of three node indices.
            d is the determinant of M.
            d_old is the determinant of M from the last time step.
            f is the source vector of length = len(nodes).
            f is the vector coefficients of the source expressed in terms of hat functions.
            old_alpha is the vector of coefficients alphas from the previous time step.
            dt is the time step
    """
    cdef double dummy=0.0
    dummy = I_c(i,0,d)*0 #always zero
    dummy +=I_c(i,1,d)*0
    dummy +=I_c(i,2,d)*f[fbe[e]] #index was 
    dummy +=I_c(i,0,d_old)*old_alpha[node_ids[0]]
    dummy +=I_c(i,1,d_old)*old_alpha[node_ids[1]]
    dummy +=I_c(i,2,d_old)*old_alpha[node_ids[2]]
    return dummy

