#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 19:21:31 2020

@author: andrewg
"""

from FE_vtx import  build_FE_vtx, build_FE_vtx_from_scratch
from GeneRegulatoryNetwork import GRN_full_basic, build_GRN_full_basic
from FE_transitions import divide, T1, rem_collapsed
from cells_extra import ready_to_divide, centroids2
from cent_test import cen2
from plotting import animate_surf_video_mpg, draw_cells, cells_state_video
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
        
    def evolve(self,v, prod_rate,bind_rate,deg_rate,time,dt, expansion=None, evolve_vertex=True):
        sig_input = self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes]
        self.FE_vtx.evolve(v,prod_rate,dt, expansion=expansion, evolve_vertex=evolve_vertex)
        self.GRN.evolve(time , dt , sig_input , bind_rate)
        self.GRN.lost_morphogen[self.FE_vtx.cells.properties['source'].astype(bool)]=0.0 # no binding at source
        self.FE_vtx.concentration=self.FE_vtx.concentration - dt*deg_rate*self.FE_vtx.concentration
        self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes] = self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes]-self.GRN.lost_morphogen
        neg = np.where(self.FE_vtx.concentration < 0)[0]
        self.FE_vtx.concentration[neg]=0 #reset any negative concentration values to zero.
        
    def evolve_fast(self,v, prod_rate,bind_rate,deg_rate,time,dt, expansion=None, evolve_vertex=True):
        sig_input = self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes]
        self.FE_vtx.evolve_cy(v,prod_rate,dt, expansion=expansion, evolve_vertex=evolve_vertex)
        self.GRN.evolve_ugly(time , dt , sig_input , bind_rate)
        self.FE_vtx.concentration=self.FE_vtx.concentration - dt*deg_rate*self.FE_vtx.concentration
        self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes] = self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes]-self.GRN.lost_morphogen
        neg = np.where(self.FE_vtx.concentration < 0)[0]
        self.FE_vtx.concentration[neg]=0 #reset any negative concentration values to zero.
    
    def evolve_original(self,v, prod_rate,bind_rate,deg_rate,time,dt):
        sig_input = self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes]
        self.FE_vtx.evolve_original(v,prod_rate,dt)
        self.GRN.evolve(time , dt , sig_input , bind_rate)
        self.FE_vtx.concentration=self.FE_vtx.concentration - dt*deg_rate*self.FE_vtx.concentration
        self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes] = self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes]-self.GRN.lost_morphogen
        neg = np.where(self.FE_vtx.concentration < 0)[0]
        self.FE_vtx.concentration[neg]=0 #reset any negative concentration values to zero.
        
    def transitions(self,ready=None):
        if ready is None:
            ready = ready_to_divide(self.FE_vtx.cells)
        c_by_e = self.FE_vtx.concentration[self.FE_vtx.edges_to_nodes]
        c_by_c = self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes]
        self.FE_vtx.cells,c_by_e, c_by_c = divide(self.FE_vtx.cells,c_by_e,c_by_c,ready)
        self.GRN.division(ready)
        self.FE_vtx.cells = T1(self.FE_vtx.cells) #perform T1 transitions - "neighbour exchange"
        self.FE_vtx.cells,c_by_e = rem_collapsed(self.FE_vtx.cells,c_by_e) #T2 transitions-"leaving the tissue"
        self.FE_vtx.centroids = centroids2(self.FE_vtx.cells)
        eTn = self.FE_vtx.cells.mesh.edges.ids//3
        n = max(eTn)
        cTn=np.cumsum(~self.FE_vtx.cells.empty())+n
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

def load (filename):
    with open(filename, "rb") as f:
        return dill.load(f)


if __name__ == "__main__":
    
    np.random.seed(1984)

    simulate = True
    plotting = True
    evolve_vertex = True

    # default values
    xsize=20
    ysize=10
    dt = .001
    tSym = 200. # total time 
    N_step = int(tSym/dt)
    N_frames = 1000

    # fixed expansion parameters
    expansion = np.ones(2)
    expansion *= np.log(5.)/2./N_step # (1 + ex)**2e5 ~ sqrt(5) (area 5x biger after 2e5 steps)
    anisotropy=0.   # must be in [0,1] -> 0 = isotropy, 1 = expansion only on x
    expansion *= np.array([1.+anisotropy, 1.-anisotropy])

    if len(sys.argv) > 1:
        N_step=int(sys.argv[1])
        if len(sys.argv) > 2:
            N_frames = int(sys.argv[2])
    path="outputs/%dx%d_%.0e"%(xsize, ysize, N_step)
    N_skip = max(N_step//N_frames, 1)

    if ~evolve_vertex:
        expansion *= 0.
        anisotropy 
        path += "_novertex"
    else:
        if anisotropy != 0.:
            path += "_anis_%.1e_%.1e"%tuple(expansion)

    print("N_step   =", N_step)
    print("N_frames =", N_frames)
    print("N_skip   =", N_skip)
    print("saving in / retrieving from  \"%s\"\n"%(path))
    # print("expansion = ", expansion)
    # print(np.prod(1+expansion)**2e5)
    # sys.exit()

    if simulate:
        os.system("mkdir -p "+path)
        print("build NT")
        neural_tube=build_NT_vtx_from_scratch(size = [xsize,ysize])
        t1=time.time()
        for k in range(N_step+1):
            # ttot = time.time()
            if k%N_skip == 0:
                # tdump = time.time()
                print(k)
                with open (path+"/%06d_nodes.pkl"%(k), "wb") as f:
                    dill.dump(np.vstack([
                        neural_tube.FE_vtx.cells.mesh.vertices.T[::3],
                        neural_tube.FE_vtx.centroids[~neural_tube.FE_vtx.cells.empty()]
                    ]), f)
                with open (path+"/%06d_conc.pkl"%(k), "wb") as f:
                    dill.dump(neural_tube.FE_vtx.concentration, f)
                with open (path+"/%06d_poni.pkl"%(k), "wb") as f:
                    dill.dump(neural_tube.GRN.poni_grn.state, f)
                with open (path+"/%06d_cells.pkl"%(k), "wb") as f:
                    dill.dump(neural_tube.FE_vtx.cells, f)
                # tdump = time.time() - tdump
                # ttot = time.time() - ttot
                # print(k, ttot, tdump)
            neural_tube.evolve(.2,.05,0.,0.,.0,dt,expansion=expansion) #(v, prod_rate,bind_rate,deg_rate,time,dt):
            neural_tube.transitions()
        t2 = time.time()
        print("took:", t2 - t1)


    if plotting:

        try:
            allfiles = os.listdir(path)

        except FileNotFoundError:
            print("path not found: ", path)

        allnodes = sorted([x for x in allfiles if "_nodes.pkl" in x])
        allconcs = sorted([x for x in allfiles if "_conc.pkl" in x])
        allponis = sorted([x for x in allfiles if "_poni.pkl" in x])
        allcells = sorted([x for x in allfiles if "_cells.pkl" in x])
        
        nodes_list = [load(path+"/"+file) for file in allnodes]
        concs_list = [load(path+"/"+file) for file in allconcs]
        ponis_list = [load(path+"/"+file) for file in allponis]
        cells_list = [load(path+"/"+file) for file in allcells]

        print(nodes_list[-1][0], concs_list[-1][0], ponis_list[-1][0],  N_step)
        cells_state_video(cells_list, ponis_list, path, path+"/state-vid")
        animate_surf_video_mpg(nodes_list,concs_list, path, path+"/surface-video")
    