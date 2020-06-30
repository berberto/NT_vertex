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
import concurrent.futures



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

    nt5=build_NT_vtx_from_scratch(size = [6,6])

    print("build NT")

    nodes_list = []
    concentration_list = []
    cells_list=[]
    poni_state_list=[]
    N_step = 1000
    if len(sys.argv) > 1:
        N_step=int(sys.argv[1])
    t1=time.time()
    for k in range(N_step):
        nt5.evolve_fast(.2,.05,0.,0.,0.,.001) #(v, prod_rate,bind_rate,deg_rate,time,dt):
        nt5.transitions_faster()
        if k%100 == 0:  # append every 100 steps
            print(k)
            nodes_list.append(np.vstack([nt5.FE_vtx.cells.mesh.vertices.T[::3] , nt5.FE_vtx.centroids[~nt5.FE_vtx.cells.empty()]]))
            concentration_list.append(nt5.FE_vtx.concentration)
            cells_list.append(nt5.FE_vtx.cells)
            poni_state_list.append(nt5.GRN.poni_grn.state)
    t2 = time.time()
    print(nodes_list[-1][0], concentration_list[-1][0], poni_state_list[-1][0],  N_step, "\n took ", t2 - t1)
    cells_state_video(cells_list,poni_state_list, "state-vid")
    animate_surf_video_mpg(nodes_list,concentration_list, "surface-video") 
    #print np.argmax(poni_state_list[0])


"""
SOME SPEED TESTS

nt=build_NT_vtx_from_scratch(size = [150,6])
t1=time.time()
N=10
for k in range(N):
    nt.evolve(0.1,1.0,0.01,0.01,0,0.001)
    nt.transitions()
t2 = time.time()
print N, " steps took ", t2 - t1, " secs."

nt2=build_NT_vtx_from_scratch(size = [150,6])

t3=time.time()

for k in range(N):
    nt2.evolve_fast(0.1,1.0,0.01,0.01,0,0.001)
    nt2.transitions_faster()
t4 = time.time()
print N, " steps fast? took ", t4 - t3, " secs."

nt3=build_NT_vtx_from_scratch(size = [150,6])

t5=time.time()
for k in range(N):
    nt3.evolve_original(0.1,1.0,0.01,0.01,0,0.001)
    nt3.transitions()
t6 = time.time()
print N, " original took ", t6 - t5, " secs."

print "new is ", (t6 - t5)/(t4 - t3), " times faster than original."


t3=time.time()
for k in range(N):
    nt2.evolve_parallel(0.1,1.0,0.01,0.01,0,0.001)
    nt2.transitions()
t4 = time.time()
print N, " steps in parallel took ", t4 - t3, " secs."

nt3=build_NT_vtx_from_scratch()
t5=time.time()
for k in range(N):
    nt3.evolve_parallel_2(0.1,1.0,0.01,0.01,0,0.001) #doesn't seem faster than evolve parallel
    nt3.transitions()
t6 = time.time()
print N, " steps in parallel took ", t6 - t5, " secs."
"""

"""
def flow_video(NT_history, name_file):
"""
"""
    Creates a video of the morphogen flow.
    Args:
        NT_history is a list of Neural Tube objects.
        name_file is the name of the output video.
"""
"""
    nodes_array=[]
    concentration_array=[]
    for k in range(len(NT_history)):
        nodes = np.vstack([NT_history[k].cells.vertices.T[::3] , NT_history[k].centroids[~NT_history[k].cells.empty()]])
        nodes_array.append(nodes)
        concentration_array.append(NT_history[k].FE.concentration)
    animate_surf_video_mpg(nodes_array,concentration_array, name_file)  
"""