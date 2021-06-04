#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:07:58 2020

@author: andrewg
"""

import time

import numpy as np
import scipy
from numba import jit
from scipy.sparse import coo_matrix
from FE_transitions import T1, rem_collapsed, divide
from cells_extra import cells_setup, add_IKNM_properties, ready_to_divide, cells_evolve


@jit(nopython=True)
def get_M(nodes):
    """
    Args:
      nodes is a list of three coordinates.
    Return:
      FE map matrix.
    """
    return (nodes[1:] - nodes[0]).transpose((1, 2, 0))


@jit(nopython=True)
def det2d_stack(X):
    """
    Calculates the determinant of the 2x2 matrix for the 2nd and 3rd dims (i.e. with a matrix (n,2,2))
    """
    det2d = X[:, 0, 0] * X[:, 1, 1] - X[:, 1, 0] * X[:, 0, 1]
    return det2d


@jit(nopython=True)
def inv2d_stack(M,det):
    """
    Inverse of matrix, row-wise

    """
    N = ((1 / det) * np.stack((np.row_stack((M[:, 1, 1], -M[:, 1, 0])), np.row_stack((-M[:, 0, 1], M[:, 0, 0]))))).T
    return N

@jit(nopython=True)
def get_nabla_Phi(M,d):
    """
    Args:
      M is the FE matrix map.
    Returns:
      Forgotten what this is.
    """
    N = inv2d_stack(M,d)
    nabP_q = N[:,0]
    nabP_r = N[:,1]
    nabP_p = -nabP_q - nabP_r
    return np.stack((nabP_p, nabP_q, nabP_r))

@jit(nopython=True)
def get_I(abs_d):
    """
    Args:
      d is the absolute value of the determinant of the map matrix M
    """
    return (np.expand_dims(np.expand_dims(abs_d,1),2)/np.array(((12,24,24),(24,12,24),(24,24,12)))).T



@jit(nopython=True)
def get_b(new_I, old_I,f, old_alpha, dt):
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
    dummy = new_I*f*dt + old_I*old_alpha
    return dummy[:,0]+dummy[:,1]+dummy[:,2]

@jit(nopython=True)
def get_K(new_abs_det, nabla_Phi, diff_coef):
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
    K_out = 0.5 * diff_coef * np.expand_dims(np.expand_dims(new_abs_det,1),0)*nabla_Phi*np.expand_dims(nabla_Phi,1)
    K_out = K_out[:,:,:,0]+K_out[:,:,:,1] # the inner product.
    return K_out
    ##note that this is a symmetric matrix, so is storing 2x the amount of data it needs to. Could be optimized further.


@jit(nopython=True)
def get_W(new_abs_det, nabla_Phi, new_nodes, prev_nodes):
    """
    Args:
      i,j in {0,1,2}
      d is the absolute value of det(M), where M is the matrix map
      nabPhi is the output from nabPhi.
      nodes is a list of three node coordinates corresponding to a triangle
      previous_nodes are the positions of the three nodes at the previous time step.

    """

    P = new_nodes - prev_nodes
    sum_P = P[0]+P[1]+P[2]
    dummy = sum_P + P
    W_out = np.expand_dims(np.expand_dims(new_abs_det, 1), 0) * dummy * np.expand_dims(nabla_Phi, 1)
    W_out = (W_out[:,:,:,0] + W_out[:,:,:,1])/24
    return W_out


class FE_vtx(object):
    def __init__(self, cells, centroids, concentration, edges_to_nodes, faces_to_nodes):
        # default parameters: will be set in the "evolve" method,
        # and specified triangle by triangle (when they are locally defined)
        self.parameters = {
            "diff_coeff": 1.,
            "degr_rate": 0.,
            "delta_t": 1.,
            "prod_rate": 0.
        }
        self.cells = cells
        self.centroids = centroids
        self.concentration = concentration
        self.edges_to_nodes = edges_to_nodes
        self.faces_to_nodes = faces_to_nodes

    @property
    def concentration_triangles (self):
        nxt = self.cells.mesh.edges.next
        f_by_e = self.cells.mesh.face_id_by_edge
        e_to_n = self.edges_to_nodes
        f_to_n = self.faces_to_nodes
        return np.array([
                self.concentration[e_to_n],         # c_{e,0}
                self.concentration[e_to_n[nxt]],    # c_{e,1}
                self.concentration[f_to_n[f_by_e]], # c_{e,2}
            ]).T

    @property
    def n_face(self):
        return self.cells.mesh.n_face

    @property
    def properties(self):
        return self.cells.properties

    def evolve(self, diff_coef, prod_rate, degr_rate, dt, vertex=True, move=True, morphogen=True, diff_rates=None,
               diff_adhesion=None):
        """
        Performs one step of the FE method. Computes the new cells object itself.
        Uses np.linalg.solve
        Args:
          new_cells is the new cells object after movement
          v is the diffusion coefficient
          prod_rate is the morphogen production rate.
          dt is the time step
        -
        """
        m = len(self.concentration)
        nxt = self.cells.mesh.edges.next
        f_by_e = self.cells.mesh.face_id_by_edge
        old_verts = self.cells.mesh.vertices.T
        old_cents = self.centroids
        n_e = self.cells.mesh.edges.ids.size

        if move:  # move: use vertex model forces if True, else only expansion
            new_cells = cells_evolve(self.cells, dt, vertex=vertex, diff_rates=diff_rates, diff_adhesion=diff_adhesion)
            new_verts = new_cells.mesh.vertices.T
            new_cents = new_cells.mesh.centres.T
        else:
            new_cells = self.cells
            new_verts = old_verts
            new_cents = old_cents
        #perform the FE step if morphogen is True, otherwise set concentration to 0
        if morphogen:
            f = self.cells.properties['source'] * prod_rate  # source

            ##define the connectivity from the mesh via node ids
            node_id_tri = np.array((self.edges_to_nodes, self.edges_to_nodes[nxt], self.faces_to_nodes[f_by_e]))
            node_id_mat = np.stack((node_id_tri,node_id_tri,node_id_tri))
            node_id_flat,node_idT_flat = node_id_mat.ravel(), (node_id_mat.transpose(1,0,2)).ravel()

            new_nodes = np.array((new_verts, new_verts[nxt], new_cents[f_by_e]))
            prev_nodes = np.array((old_verts, old_verts[nxt], old_cents[f_by_e]))

            reduced_f = f[f_by_e]
            reduced_f = np.stack((reduced_f,reduced_f,reduced_f))
            old_alpha = self.concentration[node_id_tri]
            new_M = get_M(new_nodes)
            old_M = get_M(prev_nodes)
            new_det = det2d_stack(new_M)
            new_abs_det = np.abs(new_det)
            old_det = det2d_stack(old_M)
            old_abs_det = np.abs(old_det)
            nabla_Phi = get_nabla_Phi(new_M,new_det)
            new_I = get_I(new_abs_det)
            old_I = get_I(old_abs_det)
            b_tri = get_b(new_I, old_I, reduced_f, old_alpha, dt) #checked
            K_tri = get_K(new_abs_det, nabla_Phi, diff_coef) #checked
            W_tri = get_W(new_abs_det, nabla_Phi, new_nodes, prev_nodes)
            A_tri = (1. + degr_rate*dt)*new_I + dt*K_tri + W_tri
            b_vec = coo_matrix((b_tri.ravel(),(node_id_tri.ravel(),np.zeros(node_id_tri.size,dtype=np.int64))),shape=(m,1))
            A_mat = coo_matrix((A_tri.ravel(),(node_idT_flat,node_id_flat)),shape=(m,m))
            A_mat = A_mat.tocsr() #this performs the summations of repeated entries
            self.concentration = scipy.sparse.linalg.spsolve(A_mat,b_vec)
        else:
            self.concentration = np.zeros(m)

        self.cells = new_cells
        self.centroids = new_cents


    # this is never used anywhere
    def transitions(self, division=True):
        if ready is None:
            ready = ready_to_divide(self.cells)
        c_by_e = self.concentration[self.edges_to_nodes]
        c_by_c = self.concentration[self.faces_to_nodes]
        self.cells = T1(self.cells)  # perform T1 transitions - "neighbour exchange"
        self.cells, c_by_e = rem_collapsed(self.cells, c_by_e)  # T2 transitions-"leaving the tissue"
        if division:
            self.cells, c_by_e, c_by_c = divide(self.cells, c_by_e, c_by_c, ready)
        self.centroids = self.cells.mesh.centres.T  # centroids2(self.cells)
        eTn = self.cells.mesh.edges.ids // 3
        n = max(eTn)
        cTn = np.cumsum(~self.cells.empty()) + n
        con_part = c_by_e[::3]
        cent_part = c_by_c[~self.cells.empty()]
        self.concentration = np.hstack([con_part, cent_part])
        self.edges_to_nodes = self.cells.mesh.edges.ids // 3
        self.faces_to_nodes = cTn


def build_FE_vtx(cells, concentration_by_edge=None, concentration_by_face=None):
    cents = cells.mesh.centres.T  # centroids2(cells)
    eTn = cells.mesh.edges.ids // 3
    n = max(eTn)
    fTn = np.cumsum(~cells.empty()) + n  # we just care about the living faces getting the correct node index
    if concentration_by_face is None or concentration_by_edge is None:
        m = fTn[-1] + 1
        con = np.zeros(m)
    return FE_vtx(cells, cents, con, eTn, fTn)


def build_FE_vtx_from_scratch(size=None, vm_parameters=None, source_data=None, cluster_data=None, differentiation=True):
    cells = cells_setup(size, vm_parameters, source_data, cluster_data, differentiation=differentiation)
    add_IKNM_properties(cells)
    cents = cells.mesh.centres.T  # centroids2(cells)
    eTn = cells.mesh.edges.ids // 3
    n = max(eTn)
    fTn = np.cumsum(~cells.empty()) + n  # we just care about the living faces getting the correct node index
    m = fTn[-1] + 1
    con = np.zeros(m)
    return FE_vtx(cells, cents, con, eTn, fTn)

    # returns an FE_vtx object
