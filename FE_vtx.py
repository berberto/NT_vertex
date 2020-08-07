#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:07:58 2020

@author: andrewg
"""

import sys
import numpy as np
import scipy
from scipy.sparse import coo_matrix
from cells_extra import cells_setup,add_IKNM_properties, ready_to_divide, cells_evolve
from Finite_Element import centroids2
from FE_transitions import T1, rem_collapsed, divide
from _fe_cy import ev_cy, ev_cy_sparse, ev_cy_trivial
# from _fe_cy_omp import ev_cy
from cent_test import cen2
import concurrent.futures
from multiprocessing import Array ,Process
import matplotlib.pyplot as plt
import time
import ctypes

def M(nodes):
  """
  Args:
    nodes is a list of three coordinates.
  Return:
    FE map matrix.
  """
  va = [nodes[1][0] - nodes[0][0] , nodes[1][1] - nodes[0][1]] 
  vb = [nodes[2][0] - nodes[0][0] , nodes[2][1] - nodes[0][1]]
  M=np.array([va , vb])
  return M.T    
  
  
def I(i,j,d):
  """
  Args:
    d is the absolute value of the determinant of the map matrix M
  """
  if ( i - j ):
    return (1.0/24)* np.abs(d)
  else:
    return (1.0/12)* np.abs(d) 
  
def nabPhi(M):
  """
  Args:
    M is the FE matrix map.
  Returns:
    Forgotten what this is.
  """
  N = np.linalg.inv(M).T
  #nabP_p = np.matmul(N , np.array([ -1,-1]))
  #nabP_p = -1*N[:,0]-1*N[:,1]
  #nabP_q = np.matmul(N , np.array([1,0]))
  nabP_q = N[:,0]
  #nabP_r = np.matmul(N , np.array([0,1]))
  nabP_r = N[:,1]
  nabP_p = -nabP_q - nabP_r
  return nabP_p, nabP_q , nabP_r   
 
def nabPhi2(M):
  """
  Args:
    M is the FE matrix map.
  Returns:
    Forgotten what this is.
  """
  N = np.linalg.inv(M).T
  nabP_p = np.matmul(N , np.array([-1,-1]))
  nabP_q = np.matmul(N , np.array([1,0]))
  nabP_r = np.matmul(N , np.array([0,1]))
  return nabP_p, nabP_q , nabP_r    
  
def K(i,j,d,nabPhi,v):
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
  return (1.0/2)*v*np.inner(nabPhi[i],nabPhi[j])*d        
 
def W(i,j,d,nabPhi,nodes, previous_nodes):
  """
  Args:
    i,j in {0,1,2}
    d is the absolute value of det(M), where M is the matrix map
    nabPhi is the output from nabPhi.
    nodes is a list of three node coordinates corresponding to a triangle
    previous_nodes are the positions of the three nodes at the previous time step.
    
  """
  P0 = (nodes[0]-previous_nodes[0]).T
  P1 = (nodes[1]-previous_nodes[1]).T
  P2 = (nodes[2]-previous_nodes[2]).T
  dummy = P0 + P1 + P2 + (nodes[j]-previous_nodes[j]).T
  return (1.0 / 24 )*d*np.inner(nabPhi[i].T, dummy)
 
def b(i,d,d_old,f,old_alpha,dt): 
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
  dummy =  I(i,0,d) * f[0] * dt
  dummy += I(i,1,d) * f[1] * dt
  dummy += I(i,2,d) * f[2] * dt
  dummy += I(i,0,d_old) * old_alpha[0]
  dummy += I(i,1,d_old) * old_alpha[1]
  dummy += I(i,2,d_old) * old_alpha[2]
  return dummy
 

#
#   EDIT: TRIANGLE CLASS
#         handling triangular elements and integrals therein
#
canonicalBasisGrads = np.array([
              [-1,-1],
              [1,0],
              [0,1]]) # shape (3,2)

canonicalBasisInnerProd = 1./24.*np.array([
              [2,1,1],
              [1,2,1],
              [1,1,2]]) # shape (3,3)

class FE_Triangle(object):
  """
  FE_Triangle class: handles information about linear triangular elements,
  computing the local finite element matrix and vector

  """
  def __init__(self,nodes_ids=np.arange(3).astype(int),nodes=np.array([[0,0],[1,0],[0,1]]),nodes_vel=np.zeros((3,2)),parameters={"diff_coeff": 1.,"degr_rate": 0.,"delta_t": 1.,"prod_rate": 0.},source=0):
    # parameters -- better way of passing them?
    self.parameters = parameters  # dictionary
    self.parameters["prod_rate"] *= source

    self.nodes = nodes            # float (3,2): coordinates of the nodes
    self.nodes_ids = nodes_ids      # int (3,): global indices of the nodes
    self.jacobian = np.array([nodes[1]-nodes[0], nodes[2]-nodes[0]],dtype=float).T # float (2,2)
    self.det = np.linalg.det(self.jacobian)      # determinant of the Jacobian
    self.transf = 1./self.det*np.array([
            [self.jacobian[1,1],-self.jacobian[1,0]],
            [-self.jacobian[0,1],self.jacobian[0,0]]
          ]) # inverse transpose of Jacobian
    # velocities as rows
    self.nodes_vel = nodes_vel
    # gradients as columns
    self.gradients = np.matmul(self.transf, canonicalBasisGrads.T) #(2,2)x(2,3)=(2,3)
    # element FE matrix and vector
    self.mass_matrix = np.zeros((3,3))
    self.matrix = np.zeros((3,3))
    self.vector = np.zeros(3)


  def _elementIntegrals(self):
    self.mass_matrix = np.zeros((3,3))
    self.matrix = np.zeros((3,3))
    self.vector = np.zeros(3)
    D  = self.parameters["diff_coeff"]
    k  = self.parameters["degr_rate"]
    f  = self.parameters["prod_rate"]
    dt = self.parameters["delta_t"]

    # "mass" integral
    self.mass_matrix = self.det * canonicalBasisInnerProd
    self.matrix += (1. + k*dt) * self.mass_matrix
    # diffusion
    self.matrix += D * dt / 2. * self.det * np.matmul(self.gradients.T, self.gradients)
    # fictitious advection
    aux = np.matmul(self.gradients.T, self.nodes_vel.T) # (3,2)x(2,3)
    aux = np.matmul(aux, canonicalBasisInnerProd) # (3,3)
    self.matrix += dt * self.det * aux

    # source term
    self.vector += dt * f * self.det / 6. * np.ones(3)  # if source constant over the element
    # self.vector += dt * np.matmul(self.mass_matrix, f_vec) # if source has different values at 
#
#   END EDIT
#


class FE_vtx(object):
  def __init__(self,cells,centroids,concentration,edges_to_nodes,faces_to_nodes,triangles=None):
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
    self.triangles = triangles
    
    nNodes = len(self.concentration)
    self.mass_matrix = np.zeros((nNodes,nNodes),dtype=float)
    self.matrix = np.zeros((nNodes,nNodes),dtype=float)
    self.vector = np.zeros(nNodes,dtype=float)



  #
  #   EDIT: NEW METHODS
  #
  # define FE_Triangle objects from vertices, centroids and edges
  def _triangles(self, verP, cenP, verV, cenV, parameters):
    verts = verP
    vets_vel = verV
    cents = cenP
    cents_vel = cenV
    edges = self.cells.mesh.edges
    nxt = self.cells.mesh.edges.next
    f_by_e = self.cells.mesh.face_id_by_edge
    f = self.cells.properties['source']
    
    #
    # CREATING A NEW LIST OF TRIANGLES ALL THE TIMES,
    # WE MAY JUST EDIT THE LIST OF THE TRIANGLES (PERHAPS FASTER -- NOT DONE).
    # IF SO, WE NEED TO MAKE SURE THAT THE LIST OF TRIANGLES
    # IS TAKEN CARE OF BY "TRANSITIONS" AS WELL.
    # SO WE MAY JUST REDEFINE "TRIANGLES" ONLY WHEN
    # A TRANSITION OCCURS (DIVISION, T1, T2)
    #

    # set constant production rate everywhere on the producing cells
    # DIFFERENT FROM PREVIOUS SOLUTIONS WHERE PRODUCTION RATES WERE NON ZERO
    # ONLY AT THE CENTROIDS: THIS SHOULD LEAVE EVERYTHING UNAFFECTED,
    # WE MIGHT ONLY NEED TO DIVIDE THE PRODUCTION RATE BY 3 TO GET THE SAME
    # TOTAL AMOUNT OF MORPHOGEN PRODUCED

    # pass the arguments for constructing the triangle as dictionary
    triangle_properties = [{
            "nodes_ids": np.array([self.edges_to_nodes[e],self.edges_to_nodes[nxt[e]],self.faces_to_nodes[f_by_e[e]]],dtype=int),
            "nodes": np.array([verts[e],verts[nxt[e]],cents[f_by_e[e]]],dtype=float),
            "nodes_vel": np.array([verts[e],verts[nxt[e]],cents[f_by_e[e]]],dtype=float),
            "parameters": parameters,
            "source": f[f_by_e[e]]
      } for e in edges.ids]

    triangles = [FE_Triangle(**triangle_properties[e]) for e in edges.ids]

    for tr in triangles:
      tr._elementIntegrals()

    return triangles


  # assembly te FE matrix and vector
  def _assembly(self):
    nNodes = len(self.concentration)
    self.matrix = np.zeros((nNodes,nNodes),dtype=float)
    self.vector = np.zeros(nNodes,dtype=float)

    # contribution to FE vector from concentration at previous step
    self.vector = np.matmul(self.mass_matrix, self.concentration)

    # update the FE matrix and vector
    for tr in self.triangles:
      # select tables of indices
      ii,jj=np.meshgrid(tr.nodes_ids,tr.nodes_ids)
      # add element matrix and vector to correct entries
      self.mass_matrix[ii,jj] += tr.mass_matrix
      self.matrix[ii,jj] += tr.matrix
      self.vector[tr.nodes_ids] += tr.vector


  # solve the FE step
  def _solve(self):
    return scipy.linalg.solve(self.matrix, self.vector)


  def evolve_new(self, diff_coeff, prod_rate, degr_rate, dt, expansion=None, evolve_vertex=True):
    self.parameters["diff_coeff"] = diff_coeff
    self.parameters["prod_rate"] = prod_rate
    self.parameters["degr_rate"] = degr_rate
    self.parameters["delta_t"] = dt

    old_verts = self.cells.mesh.vertices.T
    old_cents = self.centroids

    if evolve_vertex:
      new_cells = cells_evolve(self.cells,dt,expansion=expansion)[0]
      new_verts = new_cells.mesh.vertices.T
      new_cents = centroids2(new_cells)
      verts_vel = new_cells.mesh.velocities.T
      cents_vel = (new_cents - old_cents)/dt
    else:
      new_cells = self.cells
      new_verts = old_verts
      new_cents = old_cents
      verts_vel = np.zeros(np.shape(old_verts))
      cents_vel = np.zeros(np.shape(old_cents))

    self.triangles = self._triangles(new_verts, new_cents, verts_vel, cents_vel, self.parameters)

    self._assembly()
    self.concentration = self._solve()
  #
  #   END EDIT
  #

  def evolve(self,v,prod_rate,degr_rate,dt,expansion=None,evolve_vertex=True):
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
    bv = np.zeros(m,dtype=float) #bv stands for b vector
    A = np.zeros((m,m),dtype=float)
    nxt=self.cells.mesh.edges.next
    f_by_e = self.cells.mesh.face_id_by_edge
    old_verts = self.cells.mesh.vertices.T
    old_cents = self.centroids
    
    if evolve_vertex:
      new_cells = cells_evolve(self.cells,dt,expansion=expansion)[0]
      new_verts = new_cells.mesh.vertices.T
      new_cents = centroids2(new_cells)
      verts_vel = new_cells.mesh.velocities.T
      cents_vel = (new_cents - old_cents)/dt
    else:
      new_cells = self.cells
      new_verts = old_verts
      new_cents = old_cents
      verts_vel = np.zeros(np.shape(old_verts))
      cents_vel = np.zeros(np.shape(old_cents))
    
    f = self.cells.properties['source']*prod_rate #source
    count=0
    rows=[]
    cols=[]
    entries=[]
    for e in self.cells.mesh.edges.ids: #modify if cells.mesh.edges.geometry is cylindrical
      new_nodes = [new_verts[e] ,new_verts[nxt[e]], new_cents[f_by_e[e]]]
      prev_nodes = [old_verts[e] ,old_verts[nxt[e]], old_cents[f_by_e[e]]]
      node_id_tri = np.array([self.edges_to_nodes[e],self.edges_to_nodes[nxt[e]] , self.faces_to_nodes[f_by_e[e]] ],dtype=int)
      reduced_f = f[f_by_e[e]]*np.ones(3) # [0,0,f[f_by_e[e]]]
      old_alpha = self.concentration[node_id_tri]
      new_M = M(new_nodes)
      old_M = M(prev_nodes)
      d = np.abs(np.linalg.det(new_M))
      d_old = np.abs(np.linalg.det(old_M))
      nabla_Phi = nabPhi(new_M)
      for i in range(3):
        bv[node_id_tri[i]] += b(i,d,d_old,reduced_f,old_alpha,dt)
        for j in range(3):
          A[node_id_tri[i],node_id_tri[j]] += (1. + degr_rate*dt)*I(i,j,d)+ dt*K(i,j,d,nabla_Phi,v)+ W(i,j,d,nabla_Phi,new_nodes, prev_nodes)
          # rows+=[node_id_tri[i]]  # can it be that node_id_tr
          # cols+=[node_id_tri[j]]
          # entries+=[(1.+ degr_rate*dt)*I(i,j,d)+dt*K(i,j,d,nabla_Phi,v)+W(i,j,d,nabla_Phi,new_nodes, prev_nodes)]
    
    # rows=np.array(rows).astype(int)
    # cols=np.array(cols).astype(int)
    # entries=np.array(entries).astype(float)
    # A = coo_matrix((entries,(rows,cols)), shape=(m,m))
    # self.concentration = scipy.sparse.linalg.spsolve(A,bv)
    
    self.concentration = scipy.linalg.solve(A,bv)
    self.cells = new_cells
    self.centroids = new_cents


  def evolve_cy(self,v,prod_rate,degr_rate,dt,expansion=None,evolve_vertex=True):
    """
    Performs one step of the FE method. Computes the new cells object itself.
    Uses np.linalg.solve
    Args:
      new_cells is the new cells object after movement
      v is the diffusion coefficient
      prod_rate is the morphogen production rate.
      dt is the time step
    
    """
    nxt=self.cells.mesh.edges.next
    f_by_e = self.cells.mesh.face_id_by_edge
    old_verts = self.cells.mesh.vertices.T
    old_cents = self.centroids
    
    if evolve_vertex:
      new_cells = cells_evolve(self.cells,dt,expansion=expansion)[0]
      new_verts = new_cells.mesh.vertices.T
      new_cents = cen2(new_cells)#centroids2(new_cells)
    else:
      new_cells = self.cells
      new_verts = old_verts
      new_cents = old_cents

    f = self.cells.properties['source']*prod_rate #source
    n_edge = self.cells.mesh.edges.ids[-1]+1
    # self.concentration = ev_cy_trivial(new_verts.astype(np.float64))
    self.concentration = ev_cy(old_verts.astype(np.float64), new_verts.astype(np.float64), old_cents.astype(np.float64),new_cents.astype(np.float64), self.concentration.astype(np.float64), nxt.astype(np.intc) ,f_by_e.astype(np.intc), self.edges_to_nodes.astype(np.intc), self.faces_to_nodes.astype(np.intc), f.astype(np.float64) , np.intc(n_edge) , np.float64(v), np.float64(dt) )
    # self.concentration = ev_cy_sparse(old_verts.astype(np.float64), new_verts.astype(np.float64), old_cents.astype(np.float64),new_cents.astype(np.float64), self.concentration.astype(np.float64), nxt.astype(np.intc) ,f_by_e.astype(np.intc), self.edges_to_nodes.astype(np.intc), self.faces_to_nodes.astype(np.intc), f.astype(np.float64) , np.intc(n_edge) , np.float64(v), np.float64(dt) )
    self.cells = new_cells
    self.centroids = new_cents


  def transitions(self,ready=None):
    if ready is None:
      ready = ready_to_divide(self.cells)
    c_by_e = self.concentration[self.edges_to_nodes]
    c_by_c = self.concentration[self.faces_to_nodes]
    self.cells = T1(self.cells) #perform T1 transitions - "neighbour exchange"
    self.cells,c_by_e = rem_collapsed(self.cells,c_by_e) #T2 transitions-"leaving the tissue"
    self.cells,c_by_e, c_by_c = divide(self.cells,c_by_e,c_by_c,ready)
    self.centroids = centroids2(self.cells)
    eTn = self.cells.mesh.edges.ids//3
    n = max(eTn)
    cTn=np.cumsum(~self.cells.empty())+n
    con_part=c_by_e[::3]
    cent_part = c_by_c[~self.cells.empty()]
    self.concentration = np.hstack([con_part,cent_part])
    self.edges_to_nodes = self.cells.mesh.edges.ids//3
    self.faces_to_nodes = cTn

    
    
def build_FE_vtx(cells,concentration_by_edge=None,concentration_by_face=None):
  cents = centroids2(cells)
  eTn = cells.mesh.edges.ids/3
  n = max(eTn)
  fTn=np.cumsum(~cells.empty())+n #we just care about the living faces getting the correct node index
  if concentration_by_face is None or concentration_by_edge is None:
    m = fTn[-1]+1
    con = np.zeros(m)
  return FE_vtx(cells,cents,con,eTn,fTn)

def build_FE_vtx_from_scratch(size=None, vm_parameters=None,source_data=None,cluster_data=None):
  cells = cells_setup(size, vm_parameters,source_data,cluster_data)
  add_IKNM_properties(cells)
  cents = centroids2(cells)
  eTn = cells.mesh.edges.ids//3
  n = max(eTn)
  fTn=np.cumsum(~cells.empty())+n #we just care about the living faces getting the correct node index
  m=fTn[-1]+1
  con = np.zeros(m)
  return FE_vtx(cells,cents,con,eTn,fTn)
  
    
   #returns an FE_vtx object
  
