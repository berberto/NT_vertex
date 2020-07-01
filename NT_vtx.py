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
from plotting import animate_surf_video_mpg, draw_cells
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys



class NT_vtx(object):
    def __init__(self, FE_vtx, GRN):
        self.FE_vtx = FE_vtx
        self.GRN = GRN
        
    def evolve(self,v, prod_rate,bind_rate,deg_rate,time,dt):
        sig_input = self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes]
        self.FE_vtx.evolve(v,prod_rate,dt)
        self.GRN.evolve(time , dt , sig_input , bind_rate)
        self.GRN.lost_morphogen[self.FE_vtx.cells.properties['source'].astype(bool)]=0.0 # no binding at source
        self.FE_vtx.concentration=self.FE_vtx.concentration - deg_rate*self.FE_vtx.concentration
        self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes] = self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes]-self.GRN.lost_morphogen
        neg = np.where(self.FE_vtx.concentration < 0)[0]
        self.FE_vtx.concentration[neg]=0 #reset any negative concentration values to zero.
        
    def evolve_fast(self,v, prod_rate,bind_rate,deg_rate,time,dt):
        sig_input = self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes]
        self.FE_vtx.evolve_cy(v,prod_rate,dt)
        self.GRN.evolve_ugly(time , dt , sig_input , bind_rate)
        self.FE_vtx.concentration=self.FE_vtx.concentration - deg_rate*self.FE_vtx.concentration
        self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes] = self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes]-self.GRN.lost_morphogen
        neg = np.where(self.FE_vtx.concentration < 0)[0]
        self.FE_vtx.concentration[neg]=0 #reset any negative concentration values to zero.
    
    def evolve_original(self,v, prod_rate,bind_rate,deg_rate,time,dt):
        sig_input = self.FE_vtx.concentration[self.FE_vtx.faces_to_nodes]
        self.FE_vtx.evolve_original(v,prod_rate,dt)
        self.GRN.evolve(time , dt , sig_input , bind_rate)
        self.FE_vtx.concentration=self.FE_vtx.concentration - deg_rate*self.FE_vtx.concentration
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

def cells_state_video(cells_history, poni_state_history, name_file):
    #time = 0
    #history=[tissue.cells]
    outputdir="images"
    if not os.path.exists(outputdir): # if the folder doesn't exist create it
        os.makedirs(outputdir)
    fig = plt.figure(); 
    ax = fig.add_subplot(111); 
    fig.set_size_inches(6,6); 
    i=0
    frames=[]
    final_width = cells_history[-1].mesh.geometry.width
    if hasattr(cells_history[-1].mesh.geometry,'height'):
        final_height = cells_history[-1].mesh.geometry.height
    else:
        final_height =max(np.abs(cells_history[-1].mesh.vertices[1]))
    for k in range(len(cells_history)):
        #tissue = normalised_simulate_tissue_step(surface_function,tissue,bind_rate,time,dt,expansion)
        set_colour_poni_state(cells_history[k],poni_state_history[k])
        draw_cells(cells_history[k],final_width,final_height, ax) #draw_cells(cells,final_width=None,final_height=None, ax=None)
        #drawShh4(nodes_array[i],alpha_array[i], z_low,z_high,final_length, height,ax) #drawShh4(nodes,alpha,z_low,z_high,final_length,width, ax=None):
        i=i+1
        frame="images/image%03i.png" % i
        fig.savefig(frame,dpi=500)
        frames.append(frame)  
        #print tissue.cells.properties['color']
        #history.append(copy.deepcopy(tissue))
        #print tissue.cells.mesh.geometry.width
    os.system("cd ")
    #os.system("cd /opt")
    os.system("ffmpeg -framerate 5/1 -i images/image%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p "+name_file+".mp4") #for Mac computer
    print((os.system("pwd")))
    #os.system("cd ")
    #os.system("cd Desktop/vertex_model/images")
    #os.system("cd images")
    #for frame in frames: os.remove(frame)  
    
      
def set_colour_poni_state(cells,poni_state):
    n_face = cells.mesh.n_face
    source =cells.properties['source']
    cells.properties['color']=np.ones((n_face, 3)) #to store RGB number for each face
    for k in range(n_face):
        m = np.argmax(poni_state[k])
        if source[k]==1:
            cells.properties['color'][k] = np.array([1,1,1]) #source
        elif m==0:
            cells.properties['color'][k] = np.array([0,0,1]) #Blue, pax high
        elif m==1:
            cells.properties['color'][k] = np.array([1,0,0]) #Red, Olig2 high
        elif m==2:
            cells.properties['color'][k] = np.array([0,0,1]) #Green, NKx22 high
        elif m==3:
            cells.properties['color'][k] = np.array([0,1,1]) # ?, Irx high 


if __name__ == "__main__":

    np.random.seed(1984)

    xsize=20
    ysize=10
    N_step = 1000
    if len(sys.argv) > 1:
        N_step=int(sys.argv[1])
    filename="outputs/test_%dx%d_%d"%(xsize, ysize, N_step)

    # print(filename)
    # sys.exit()

    nt5=build_NT_vtx_from_scratch(size = [xsize,ysize])
    print("build NT")

    # mesh = nt5.FE_vtx.cells.mesh
    # face_id_by_edge = mesh.face_id_by_edge
    # edges = mesh.edges
    # verts = mesh.vertices   # position of the vertices
    # nexts = edges.next      # indices of the 'starting' vertices
    # prevs = edges.prev      # indices of the 'arriving' vertices

    # print("faces")
    # print(face_id_by_edge)
    # print(type(face_id_by_edge))
    # print(np.shape(face_id_by_edge), "\n")

    # print("verts")
    # print(verts)
    # print(type(verts))
    # print(np.shape(verts), "\n")

    # print("prevs")
    # print(prevs)
    # print(type(prevs))
    # print(np.shape(prevs), "\n")

    # print("nexts")
    # print(nexts)
    # print(type(nexts))
    # print(np.shape(nexts), "\n")

    # print("xs\n", np.unique(verts[0]))
    # print("ys\n", np.unique(verts[1]))

    # topvs = 

    # # print("comp")
    # # comp = np.array([x for x in zip(prevs,nexts,verts[0],verts[1])])
    # # comp=np.sort(comp, )
    # # for x in comp:
    # #     print("%d\t%d\t%.2f\t%.2f\t"%x)
    # sys.exit()

    nodes_list = []
    concentration_list = []
    cells_list=[]
    poni_state_list=[]

    # for k in range(0,N_step,10):
    #     nodes_list += [np.load(filename+"_%04d_nodes.npy"%(k))]
    #     concentration_list += [np.load(filename+"_%04d_conc.npy"%(k))]
    #     # cells_list += [np.load(filename+"_%04d_cells.npy"%(k))]
    #     poni_state_list += [np.load(filename+"_%04d_poni.npy"%(k))]
    # sys.exit()

    t1=time.time()
    for k in range(N_step):
        nt5.evolve_fast(.2,.05,0.,0.,.02,.001) #(v, prod_rate,bind_rate,deg_rate,time,dt):
        nt5.transitions_faster()
        if k%10 == 0:  # append every 100 steps
            print(k)
            np.save(filename+"_%04d_nodes.npy"%(k), np.vstack([nt5.FE_vtx.cells.mesh.vertices.T[::3] , nt5.FE_vtx.centroids[~nt5.FE_vtx.cells.empty()]]))
            np.save(filename+"_%04d_conc.npy"%(k), nt5.FE_vtx.concentration)
            # np.save(filename+"_%04d_cells.npy"%(k), nt5.FE_vtx.cells)
            np.save(filename+"_%04d_poni.npy"%(k), nt5.GRN.poni_grn.state)
            nodes_list.append(np.vstack([nt5.FE_vtx.cells.mesh.vertices.T[::3] , nt5.FE_vtx.centroids[~nt5.FE_vtx.cells.empty()]]))
            concentration_list.append(nt5.FE_vtx.concentration)
            cells_list.append(nt5.FE_vtx.cells)
            poni_state_list.append(nt5.GRN.poni_grn.state)
            # continue
            # break
    t2 = time.time()
    print(nodes_list[-1][0], concentration_list[-1][0], poni_state_list[-1][0],  N_step, "\n took ", t2 - t1)

    cells_state_video(cells_list,poni_state_list, filename+"_state-vid")
    animate_surf_video_mpg(nodes_list,concentration_list, filename+"_surface-video")
