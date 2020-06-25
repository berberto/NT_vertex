#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:07:58 2020

@author: andrewg
"""

import numpy as np
import scipy
from cells_extra import cells_setup,add_IKNM_properties, ready_to_divide, cells_evolve
from Finite_Element import centroids2
from FE_transitions import T1, rem_collapsed, divide
# from _fe_cy_v3 import ev2, ev3, ev4,ev5
# from cent_test import cen2
import concurrent.futures
from multiprocessing import Array ,Process
import time
import ctypes

class FE_vtx(object):
    def __init__(self,cells,centroids,concentration,edges_to_nodes,faces_to_nodes):
        self.cells=cells
        self.centroids=centroids
        self.concentration = concentration
        self.edges_to_nodes = edges_to_nodes
        self.faces_to_nodes = faces_to_nodes
        
    def evolve(self,new_cells,v,prod_rate,dt):
        """
        Performs one step of the FE method.  Uses np.linalg.solve which is slow.
        Args:
            new_cells is the new cells object after movement
            v is the diffusion coefficient
            prod_rate is the morphogen production rate.
            dt is the time step
        
        """
        m = len(self.concentration)
        A = np.zeros((m,m))
        bv = np.zeros(m) #bv stands for b vector
        nxt=self.cells.mesh.edges.__next__
        f_by_e = self.cells.mesh.face_id_by_edge
        old_verts = self.cells.mesh.vertices.T
        old_cents = self.centroids
        new_verts = new_cells.mesh.vertices.T
        new_cents = centroids2(new_cells)
        f = self.cells.properties['source']*prod_rate #source
        count=0
        for e in self.cells.mesh.edges.ids: #modify if cells.mesh.edges.geometry is cylindrical
            #l=[e,nxt[e],f_by_e[e]] #maybe don't need this
            new_nodes = [new_verts[e] ,new_verts[nxt[e]], new_cents[f_by_e[e]]]
            prev_nodes = [old_verts[e] ,old_verts[nxt[e]], old_cents[f_by_e[e]]]
            node_id_tri = [self.edges_to_nodes[e],self.edges_to_nodes[nxt[e]] , self.faces_to_nodes[f_by_e[e]] ]
            #print "node_id_tri", node_id_tri
            reduced_f = [0,0,f[f_by_e[e]]]
            old_alpha = self.concentration[np.array(node_id_tri)]
            new_M = M(new_nodes)
            old_M=M(prev_nodes)
            d = np.abs(np.linalg.det(new_M))
            d_old = np.abs(np.linalg.det(old_M))           
            nabla_Phi = nabPhi(new_M)
            for i in range(3):
                bv[node_id_tri[i]] += b(i,d,d_old,reduced_f,old_alpha,dt)
                for j in range(3):
                    A[node_id_tri[i],node_id_tri[j]]+=I(i,j,d)+K(i,j,d,nabla_Phi,v)+W(i,j,d,nabla_Phi,new_nodes, prev_nodes)
            #count+=1  MATRIX A IS SYMMETRIC.
            #print count
            #if count ==300:
                #print I(i,j,d),K(i,j,d,nabla_Phi,v),W(i,j,d,nabla_Phi,new_nodes, prev_nodes)
                #print old_alpha
        #for i in range(m):
         #   for j in range(m):
           #     print A[i][j]-A[j][i]
        #print np.shape(A), np.shape(bv)
        #print "shape",np.shape(A)
        #print A[:1]
        #print A[:2], "A2"
        #print scipy.sparse.issparse(A), " that A is sparse"
        A[np.where(A < 1.0e-8)] = 0
        self.concentration = np.linalg.solve(A,bv)#could change to scipy.linalg.solve(A,bv,assume_a='sym')
        self.cells = new_cells
        self.centroids = new_cents
        #return A,bv
        
    def evolve2(self,v,prod_rate,dt):
        """
        Performs one step of the FE method. Computes the new cells object itself.
        Uses np.linalg.solve
        Args:
            new_cells is the new cells object after movement
            v is the diffusion coefficient
            prod_rate is the morphogen production rate.
            dt is the time step
        
        """
        m = len(self.concentration)
        A = np.zeros((m,m))
        bv = np.zeros(m) #bv stands for b vector
        nxt=self.cells.mesh.edges.__next__
        f_by_e = self.cells.mesh.face_id_by_edge
        old_verts = self.cells.mesh.vertices.T
        old_cents = self.centroids
        new_cells = cells_evolve(self.cells,dt)[0]
        new_verts = new_cells.mesh.vertices.T
        new_cents = centroids2(new_cells)
        f = self.cells.properties['source']*prod_rate #source
        count=0
        for e in self.cells.mesh.edges.ids: #modify if cells.mesh.edges.geometry is cylindrical
            #l=[e,nxt[e],f_by_e[e]] #maybe don't need this
            #t1=time.time()
            new_nodes = [new_verts[e] ,new_verts[nxt[e]], new_cents[f_by_e[e]]]
            prev_nodes = [old_verts[e] ,old_verts[nxt[e]], old_cents[f_by_e[e]]]
            node_id_tri = [self.edges_to_nodes[e],self.edges_to_nodes[nxt[e]] , self.faces_to_nodes[f_by_e[e]] ]
            #print "node_id_tri", node_id_tri
            reduced_f = [0,0,f[f_by_e[e]]]
            old_alpha = self.concentration[np.array(node_id_tri)]
            new_M = M(new_nodes)
            old_M=M(prev_nodes)
            d = np.abs(np.linalg.det(new_M))
            d_old = np.abs(np.linalg.det(old_M))           
            nabla_Phi = nabPhi(new_M)
            for i in range(3):
                bv[node_id_tri[i]] += b(i,d,d_old,reduced_f,old_alpha,dt)
                for j in range(3):
                    A[node_id_tri[i],node_id_tri[j]]+=I(i,j,d)+K(i,j,d,nabla_Phi,v)+W(i,j,d,nabla_Phi,new_nodes, prev_nodes)
            #t2=time.time()
            #print t2 - t1
            #count+=1  MATRIX A IS SYMMETRIC.
            #print count
            #if count ==300:
                #print I(i,j,d),K(i,j,d,nabla_Phi,v),W(i,j,d,nabla_Phi,new_nodes, prev_nodes)
                #print old_alpha
        #for i in range(m):
         #   for j in range(m):
           #     print A[i][j]-A[j][i]
        #print np.shape(A), np.shape(bv)
        #print "shape",np.shape(A)
        #print A[:1]
        #print A[:2], "A2"
        #print A
        #print scipy.sparse.issparse(A), " that A is sparse"
        self.concentration = np.linalg.solve(A,bv) #could change to scipy.linalg.solve(A,bv,assume_a='sym')
        self.cells = new_cells
        self.centroids = new_cents
        #return A, bv
        
    def mat_and_vect(self,v,prod_rate,dt):
        """
        Returns the FE matrix and vector for the purposes of checking.
        
        """
        m = len(self.concentration)
        A = np.zeros((m,m))
        bv = np.zeros(m) #bv stands for b vector
        nxt=self.cells.mesh.edges.__next__
        f_by_e = self.cells.mesh.face_id_by_edge
        old_verts = self.cells.mesh.vertices.T
        old_cents = self.centroids
        new_cells = cells_evolve(self.cells,dt)[0]
        new_verts = new_cells.mesh.vertices.T
        new_cents = centroids2(new_cells)
        f = self.cells.properties['source']*prod_rate #source
        count=0
        for e in self.cells.mesh.edges.ids: #modify if cells.mesh.edges.geometry is cylindrical
            #l=[e,nxt[e],f_by_e[e]] #maybe don't need this
            #t1=time.time()
            new_nodes = [new_verts[e] ,new_verts[nxt[e]], new_cents[f_by_e[e]]]
            prev_nodes = [old_verts[e] ,old_verts[nxt[e]], old_cents[f_by_e[e]]]
            node_id_tri = [self.edges_to_nodes[e],self.edges_to_nodes[nxt[e]] , self.faces_to_nodes[f_by_e[e]] ]
            #print "node_id_tri", node_id_tri
            reduced_f = [0,0,f[f_by_e[e]]]
            old_alpha = self.concentration[np.array(node_id_tri)]
            new_M = M(new_nodes)
            old_M=M(prev_nodes)
            d = np.abs(np.linalg.det(new_M))
            d_old = np.abs(np.linalg.det(old_M))           
            nabla_Phi = nabPhi(new_M)
            for i in range(3):
                bv[node_id_tri[i]] += b(i,d,d_old,reduced_f,old_alpha,dt)
                for j in range(3):
                    A[node_id_tri[i],node_id_tri[j]]+=I(i,j,d)+K(i,j,d,nabla_Phi,v)+W(i,j,d,nabla_Phi,new_nodes, prev_nodes)
            return A, bv
        
        
    def evolve_original(self,v,prod_rate,dt):
        """
        Performs one step of the FE method. Computes the new cells object itself.
        Uses np.linalg.solve
        Args:
            new_cells is the new cells object after movement
            v is the diffusion coefficient
            prod_rate is the morphogen production rate.
            dt is the time step
        
        """
        m = len(self.concentration)
        A = np.zeros((m,m))
        bv = np.zeros(m) #bv stands for b vector
        nxt=self.cells.mesh.edges.__next__
        f_by_e = self.cells.mesh.face_id_by_edge
        old_verts = self.cells.mesh.vertices.T
        old_cents = self.centroids
        new_cells = cells_evolve(self.cells,dt)[0]
        new_verts = new_cells.mesh.vertices.T
        new_cents = centroids2(new_cells)
        f = self.cells.properties['source']*prod_rate #source
        count=0
        for e in self.cells.mesh.edges.ids: #modify if cells.mesh.edges.geometry is cylindrical
            #l=[e,nxt[e],f_by_e[e]] #maybe don't need this
            #t1=time.time()
            new_nodes = [new_verts[e] ,new_verts[nxt[e]], new_cents[f_by_e[e]]]
            prev_nodes = [old_verts[e] ,old_verts[nxt[e]], old_cents[f_by_e[e]]]
            node_id_tri = [self.edges_to_nodes[e],self.edges_to_nodes[nxt[e]] , self.faces_to_nodes[f_by_e[e]] ]
            #print "node_id_tri", node_id_tri
            reduced_f = [0,0,f[f_by_e[e]]]
            old_alpha = self.concentration[np.array(node_id_tri)]
            new_M = M(new_nodes)
            old_M=M(prev_nodes)
            d = np.abs(np.linalg.det(new_M))
            d_old = np.abs(np.linalg.det(old_M))           
            nabla_Phi = nabPhi(new_M)
            for i in range(3):
                bv[node_id_tri[i]] += b(i,d,d_old,reduced_f,old_alpha,dt)
                for j in range(3):
                    A[node_id_tri[i],node_id_tri[j]]+=I(i,j,d)+K(i,j,d,nabla_Phi,v)+W(i,j,d,nabla_Phi,new_nodes, prev_nodes)
            #t2=time.time()
            #print t2 - t1
            #count+=1  MATRIX A IS SYMMETRIC.
            #print count
            #if count ==300:
                #print I(i,j,d),K(i,j,d,nabla_Phi,v),W(i,j,d,nabla_Phi,new_nodes, prev_nodes)
                #print old_alpha
        #for i in range(m):
         #   for j in range(m):
           #     print A[i][j]-A[j][i]
        #print np.shape(A), np.shape(bv)
        #print "shape",np.shape(A)
        #print A[:1]
        #print A[:2], "A2"
        #print A
        #print scipy.sparse.issparse(A), " that A is sparse"
        self.concentration = np.linalg.solve(A,bv) #could change to scipy.linalg.solve(A,bv,assume_a='sym')
        self.cells = new_cells
        self.centroids = new_cents
    
    def evolve_v2(self,v,prod_rate,dt):
        """
        DOESN'T WORK.
        
        Attempt at using a shared array and parallel processing
        to update the matrix.  
        Performs one step of the FE method. Computes the new cells object itself.
        Args:
            new_cells is the new cells object after movement
            v is the diffusion coefficient
            prod_rate is the morphogen production rate.
            dt is the time step
        """
        con=self.concentration
        m = len(con)
        A = np.zeros((m,m)) #set up as Array from multiprocessing
        bv = np.zeros(m) #bv stands for b vector, set up as Array from multiprocessing
        nxt=self.cells.mesh.edges.__next__
        f_by_e = self.cells.mesh.face_id_by_edge
        old_verts = self.cells.mesh.vertices.T
        old_cents = self.centroids
        new_cells = cells_evolve(self.cells,dt)[0]
        new_verts = new_cells.mesh.vertices.T
        new_cents = centroids2(new_cells) #replace with the cython version.
        f = self.cells.properties['source']*prod_rate #source
        ftn = self.faces_to_nodes
        etn = self.edges_to_nodes
        bv = Array(ctypes.c_double, m)        
        A = Array(ctypes.c_double, m*m)
        processes=[]
        for e in self.cells.mesh.edges.ids: 
            p = Process(target=updater, args=(A,bv,m,e, new_verts, new_cents, old_verts, old_cents, nxt, f_by_e, etn, ftn, con, f, v, dt))
            p.start()
            processes.append(p)
        for proc in processes:
            proc.join()
        self.concentration = scipy.sparse.linalg.spsolve(A,bv)#could change to scipy.linalg.solve(A,bv,assume_a='sym')
        self.cells = new_cells
        self.centroids = new_cents

    # def evolve_cy(self,v,prod_rate,dt):
    #     """
    #     Performs one step of the FE method. Computes the new cells object itself.
    #     Uses np.linalg.solve
    #     Args:
    #         new_cells is the new cells object after movement
    #         v is the diffusion coefficient
    #         prod_rate is the morphogen production rate.
    #         dt is the time step
        
    #     """
    #     nxt=self.cells.mesh.edges.__next__
    #     f_by_e = self.cells.mesh.face_id_by_edge
    #     old_verts = self.cells.mesh.vertices.T
    #     old_cents = self.centroids
    #     new_cells = cells_evolve(self.cells,dt)[0]
    #     new_verts = new_cells.mesh.vertices.T
    #     new_cents = cen2(new_cells)#centroids2(new_cells)
    #     f = self.cells.properties['source']*prod_rate #source
    #     n_edge = self.cells.mesh.edges.ids[-1]+1
    #     #ev2(np.ndarray old_verts, np.ndarray new_verts, np.ndarray old_cents,  np.ndarray new_cents, np.ndarray old_con, np.ndarray nx,np.ndarray f_by_e, np.ndarray  e_t_n, np.ndarray f_t_n, np.ndarray f , int n_edge , double v, double dt ):
    #     self.concentration = ev5(old_verts.astype(np.float64), new_verts.astype(np.float64), old_cents.astype(np.float64),new_cents.astype(np.float64), self.concentration.astype(np.float64), nxt.astype(np.intc) ,f_by_e.astype(np.intc), self.edges_to_nodes.astype(np.intc), self.faces_to_nodes.astype(np.intc), f.astype(np.float64) , np.intc(n_edge) , np.float64(v), np.float64(dt) )
    #     self.cells = new_cells
    #     self.centroids = new_cents
        
    def transitions(self,ready=None):
        if ready is None:
            ready = ready_to_divide(self.cells)
        c_by_e = self.concentration[self.edges_to_nodes]
        c_by_c = self.concentration[self.faces_to_nodes]
        self.cells = T1(self.cells) #perform T1 transitions - "neighbour exchange"
        self.cells,c_by_e = rem_collapsed(self.cells,c_by_e) #T2 transitions-"leaving the tissue"
        self.cells,c_by_e, c_by_c = divide(self.cells,c_by_e,c_by_c,ready)
        self.centroids = centroids2(self.cells)
        eTn = self.cells.mesh.edges.ids/3
        n = max(eTn)
        cTn=np.cumsum(~self.cells.empty())+n
        con_part=c_by_e[::3]
        cent_part = c_by_c[~self.cells.empty()]
        self.concentration = np.hstack([con_part,cent_part])
        self.edges_to_nodes = self.cells.mesh.edges.ids/3
        self.faces_to_nodes = cTn
        
    # def transitions_faster(self,ready=None):
    #     if ready is None:
    #         ready = ready_to_divide(self.cells)
    #     c_by_e = self.concentration[self.edges_to_nodes]
    #     c_by_c = self.concentration[self.faces_to_nodes]
    #     self.cells = T1(self.cells) #perform T1 transitions - "neighbour exchange"
    #     self.cells,c_by_e = rem_collapsed(self.cells,c_by_e) #T2 transitions-"leaving the tissue"
    #     self.cells,c_by_e, c_by_c = divide(self.cells,c_by_e,c_by_c,ready)
    #     self.centroids = cen2(self.cells)
    #     eTn = self.cells.mesh.edges.ids/3
    #     n = max(eTn)
    #     cTn=np.cumsum(~self.cells.empty())+n
    #     con_part=c_by_e[::3]
    #     cent_part = c_by_c[~self.cells.empty()]
    #     self.concentration = np.hstack([con_part,cent_part])
    #     self.edges_to_nodes = self.cells.mesh.edges.ids/3
    #     self.faces_to_nodes = cTn


def updater(A,bv,dim, e, new_verts, new_cents, old_verts, old_cents, nxt, f_by_e, etn, ftn, con, f, v, dt):
    new_nodes = [new_verts[e] ,new_verts[nxt[e]], new_cents[f_by_e[e]]]
    prev_nodes = [old_verts[e] ,old_verts[nxt[e]], old_cents[f_by_e[e]]]
    node_id_tri = [etn[e],etn[nxt[e]] , ftn[f_by_e[e]] ]
    #print "node_id_tri", node_id_tri
    reduced_f = [0,0,f[f_by_e[e]]]
    old_con = con[np.array(node_id_tri)]
    new_M = M(new_nodes)
    old_M=M(prev_nodes)
    d = np.abs(np.linalg.det(new_M))
    d_old = np.abs(np.linalg.det(old_M))           
    nabla_Phi = nabPhi(new_M)
    update_matrix(A,dim,node_id_tri,d,nabla_Phi,v,new_nodes,prev_nodes)
    update_vect(bv,dim,node_id_tri,d,d_old, reduced_f, old_con, dt)
    

def update_matrix(A,dim ,node_id_tri,d,nabla_Phi,v,new_nodes,prev_nodes):
    with A.get_lock(): # synchronize access
        a = np.frombuffer(A.get_obj())
        aa = a.reshape(dim,dim)
        for i in range(3):
            for j in range(3):
                aa[node_id_tri[i]][node_id_tri[j]]+=I(i,j,d)+K(i,j,d,nabla_Phi,v)+W(i,j,d,nabla_Phi,new_nodes, prev_nodes)
                
def update_vect(bv,dim,node_id_tri,d,old_d, red_f, old_con, dt):
    with bv.get_lock():
        c = np.frombuffer(bv.get_obj())
        cc=c.reshape(dim)
        for i in range(3):
            c[node_id_tri[i]] += b(i,d,old_d,red_f,old_con,dt)
            
    

        
def evolve_modified(fe_vtx,v,prod_rate,dt):
        """
        Performs one step of the FE method. Computes the new cells object itself.
        Args:
            new_cells is the new cells object after movement
            v is the diffusion coefficient
            prod_rate is the morphogen production rate.
            dt is the time step
        
        """
        m = len(fe_vtx.concentration)
        A = np.zeros((m,m))
        bv = np.zeros(m) #bv stands for b vector
        nxt=fe_vtx.cells.mesh.edges.__next__
        f_by_e = fe_vtx.cells.mesh.face_id_by_edge
        old_verts = fe_vtx.cells.mesh.vertices.T
        old_cents = fe_vtx.centroids
        new_cells = cells_evolve(fe_vtx.cells,dt)[0]
        new_verts = new_cells.mesh.vertices.T
        new_cents = centroids2(new_cells)
        f = fe_vtx.cells.properties['source']*prod_rate #source
        count=0
        for e in fe_vtx.cells.mesh.edges.ids: #modify if cells.mesh.edges.geometry is cylindrical
            #l=[e,nxt[e],f_by_e[e]] #maybe don't need this
            new_nodes = [new_verts[e] ,new_verts[nxt[e]], new_cents[f_by_e[e]]]
            prev_nodes = [old_verts[e] ,old_verts[nxt[e]], old_cents[f_by_e[e]]]
            node_id_tri = [fe_vtx.edges_to_nodes[e],fe_vtx.edges_to_nodes[nxt[e]] , fe_vtx.faces_to_nodes[f_by_e[e]] ]
            #print "node_id_tri", node_id_tri
            reduced_f = [0,0,f[f_by_e[e]]]
            old_alpha = fe_vtx.concentration[np.array(node_id_tri)]
            new_M = M(new_nodes)
            old_M=M(prev_nodes)
            d = np.abs(np.linalg.det(new_M))
            d_old = np.abs(np.linalg.det(old_M))           
            nabla_Phi = nabPhi(new_M)
            for i in range(3):
                bv[node_id_tri[i]] += b(i,d,d_old,reduced_f,old_alpha,dt)
                for j in range(3):
                    A[node_id_tri[i]][node_id_tri[j]]+=I(i,j,d)+K(i,j,d,nabla_Phi,v)+W(i,j,d,nabla_Phi,new_nodes, prev_nodes)
            #count+=1  MATRIX A IS SYMMETRIC.
            #print count
            #if count ==300:
                #print I(i,j,d),K(i,j,d,nabla_Phi,v),W(i,j,d,nabla_Phi,new_nodes, prev_nodes)
                #print old_alpha
        #for i in range(m):
         #   for j in range(m):
           #     print A[i][j]-A[j][i]
        #print np.shape(A), np.shape(bv)
        #print "shape",np.shape(A)
        #print A[:1]
        #print A[:2], "A2"
        fe_vtx.concentration = np.linalg.solve(A,bv)#could change to scipy.linalg.solve(A,bv,assume_a='sym')
        fe_vtx.cells = new_cells
        fe_vtx.centroids = new_cents
        return fe_vtx

def edge_to_updates(nxt,concentration ,new_verts,new_cents,old_verts,old_cents,edges_to_nodes, faces_to_nodes ,f, f_by_e,v,dt, e):
    new_nodes = [new_verts[e] ,new_verts[nxt[e]], new_cents[f_by_e[e]]]
    prev_nodes = [old_verts[e] ,old_verts[nxt[e]], old_cents[f_by_e[e]]]
    node_id_tri = [edges_to_nodes[e],edges_to_nodes[nxt[e]] , faces_to_nodes[f_by_e[e]] ]
            #print "node_id_tri", node_id_tri
    reduced_f = [0,0,f[f_by_e[e]]]
    old_alpha = concentration[np.array(node_id_tri)]
    new_M = M(new_nodes)
    old_M=M(prev_nodes)
    d = np.abs(np.linalg.det(new_M))
    d_old = np.abs(np.linalg.det(old_M))           
    nabla_Phi = nabPhi(new_M)
    b_updates=[]
    A_updates=[]
    for i in range(3):
        b_updates.append([node_id_tri[i],b(i,d,d_old,reduced_f,old_alpha,dt)])
        for j in range(3):
            A_updates.append([node_id_tri[i] , node_id_tri[j],I(i,j,d)+K(i,j,d,nabla_Phi,v)+W(i,j,d,nabla_Phi,new_nodes, prev_nodes)])
    return b_updates, A_updates
        
        
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
    eTn = cells.mesh.edges.ids/3
    n = max(eTn)
    fTn=np.cumsum(~cells.empty())+n #we just care about the living faces getting the correct node index
    m=fTn[-1]+1
    con = np.zeros(m)
    return FE_vtx(cells,cents,con,eTn,fTn)
    
        
     #returns an FE_vtx object
    
    
    
    
        

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
    #nabP_p = np.matmul(N , np.array([-1,-1]))
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
    dummy = I(i,0,d)*f[0]
    dummy +=I(i,1,d)*f[1]
    dummy +=I(i,2,d)*f[2]
    dummy +=I(i,0,d_old)*old_alpha[0]
    dummy +=I(i,1,d_old)*old_alpha[1]
    dummy +=I(i,2,d_old)*old_alpha[2]
    return dummy


    
