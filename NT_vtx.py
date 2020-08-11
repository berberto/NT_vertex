#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 19:21:31 2020

@author: andrewg
"""

from FE_vtx import  build_FE_vtx, build_FE_vtx_from_scratch
from GeneRegulatoryNetwork import GRN_full_basic, build_GRN_full_basic
from FE_transitions import divide, T1, rem_collapsed
from Finite_Element import centroids2
from cells_extra import ready_to_divide
from cent_test import cen2
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys
import dill



class NT_vtx(object):
    def __init__(self, FE_vtx, GRN):
        self.FE_vtx = FE_vtx
        self.GRN = GRN
        
    def evolve(self,diff_coeff, prod_rate,bind_rate,deg_rate,time,dt, expansion=None, vertex=True, move=True):
        sig_input = self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes]
        self.FE_vtx.evolve(diff_coeff, prod_rate, deg_rate, dt, expansion=expansion, vertex=vertex, move=move)
        self.GRN.evolve(time , dt , sig_input , bind_rate)
        self.GRN.lost_morphogen[self.FE_vtx.cells.properties['source'].astype(bool)]=0.0 # no binding at source
        self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes] = self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes]-self.GRN.lost_morphogen
        neg = np.where(self.FE_vtx.concentration < 0)[0]
        self.FE_vtx.concentration[neg]=0 #reset any negative concentration values to zero.
        
    def evolve_fast(self,diff_coeff, prod_rate,bind_rate,deg_rate,time,dt, expansion=None, vertex=True, move=True):
        sig_input = self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes]
        self.FE_vtx.evolve_cy(diff_coeff,prod_rate,dt, expansion=expansion, vertex=vertex, move=move)
        self.GRN.evolve_ugly(time , dt , sig_input , bind_rate)
        self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes] = self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes]-self.GRN.lost_morphogen
        neg = np.where(self.FE_vtx.concentration < 0)[0]
        self.FE_vtx.concentration[neg]=0 #reset any negative concentration values to zero.
    
    def evolve_original(self,diff_coeff, prod_rate,bind_rate,deg_rate,time,dt):
        sig_input = self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes]
        self.FE_vtx.evolve_original(diff_coeff,prod_rate,dt)
        self.GRN.evolve(time , dt , sig_input , bind_rate)
        self.FE_vtx.concentration=self.FE_vtx.concentration - dt*deg_rate*self.FE_vtx.concentration
        self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes] = self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes]-self.GRN.lost_morphogen
        neg = np.where(self.FE_vtx.concentration < 0)[0]
        self.FE_vtx.concentration[neg]=0 #reset any negative concentration values to zero.
        
    def transitions(self,ready=None,division=True):
        # values of concentration at nodes on cell boundaries (roots of edges)
        c_by_e = self.FE_vtx.concentration[self.FE_vtx.edges_to_nodes]
        # values of concentration at nodes on cell centroids
        c_by_c = self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes]
        if division:
            if ready is None:
                ready = ready_to_divide(self.FE_vtx.cells)
            # interpolation at new nodes is done in the 'divide' function
            # CHECK!
            self.FE_vtx.cells,c_by_e, c_by_c = divide(self.FE_vtx.cells,c_by_e,c_by_c,ready)
            self.GRN.division(ready)

        # perform T1 transitions - "neighbour exchange"
        self.FE_vtx.cells = T1(self.FE_vtx.cells)
        # performe T2 transitions-"leaving the tissue"
        # by removing small edges from cells through T1 transitions, some cells will become triangular and collapse to a point, which becomes a new node
        # c_by_e is the vector of concentrations given the index of the edge: only that is affected by the T1 transitions, while centroids are untouched 
        self.FE_vtx.cells,c_by_e = rem_collapsed(self.FE_vtx.cells,c_by_e)

        # compute the new centroids
        # HOW ABOUT THE INTERPOLATION WHERE CELLS HAVE DIVIDED?
        self.FE_vtx.centroids = centroids2(self.FE_vtx.cells)

        # calculate the number of nodes associated to edges
        # this changes if there are boundaries
        eTn = self.FE_vtx.cells.mesh.edges.ids//3

        # number of nodes associated to vertices
        n = max(eTn)

        # counts the living cells (the 'Cells' object in 'FE_vtx' class also contains info about dead cells)
        # counting starts from 'n', i.e. the number of cell-boundary nodes
        cTn=np.cumsum(~self.FE_vtx.cells.empty())+n

        # take che concentration at nodes which are roots of edges
        # take every 3, because each node has 3 edges coming out
        # THESE HAVE BEEN AFFECTED BY THE T1 TRANSITIONS
        con_part=c_by_e[::3]


        cent_part = c_by_c[~self.FE_vtx.cells.empty()]
        self.FE_vtx.concentration = np.hstack([con_part,cent_part])
        self.FE_vtx.edges_to_nodes = self.FE_vtx.cells.mesh.edges.ids//3
        self.FE_vtx.faces_to_nodes = cTn

    def transitions_faster(self,ready=None):
        if ready is None:
            ready = ready_to_divide(self.FE_vtx.cells)
        c_by_e = self.FE_vtx.concentration[self.FE_vtx.edges_to_nodes]
        c_by_c = self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes]
        self.FE_vtx.cells,c_by_e, c_by_c = divide(self.FE_vtx.cells,c_by_e,c_by_c,ready)
        self.GRN.division(ready)
        self.FE_vtx.cells = T1(self.FE_vtx.cells) #perform T1 transitions - "neighbour exchange"
        self.FE_vtx.cells,c_by_e = rem_collapsed(self.FE_vtx.cells,c_by_e) #T2 transitions-"leaving the tissue"
        self.FE_vtx.centroids = cen2(self.FE_vtx.cells)
        eTn = self.FE_vtx.cells.mesh.edges.ids//3
        n = max(eTn)
        cTn=np.cumsum(~self.FE_vtx.cells.empty())+n
        con_part=c_by_e[::3]
        cent_part = c_by_c[~self.FE_vtx.cells.empty()]
        self.FE_vtx.concentration = np.hstack([con_part,cent_part])
        self.FE_vtx.edges_to_nodes = self.FE_vtx.cells.mesh.edges.ids//3
        self.FE_vtx.faces_to_nodes = cTn
    
def build_NT_vtx_from_scratch(size=None, vm_parameters=None,source_data=None,cluster_data=None):
    fe_vtx  = build_FE_vtx_from_scratch(size, vm_parameters,source_data,cluster_data)
    n_face = fe_vtx.cells.mesh.n_face
    grn=build_GRN_full_basic(n_face)
    return NT_vtx(fe_vtx,grn)
