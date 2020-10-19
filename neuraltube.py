#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import dill
import numpy as np
from NT_vtx import build_NT_vtx, load_NT_vtx
from plotting import morphogen_video, cells_state_video
from options import *



if __name__ == "__main__":
    
    np.random.seed(1984)

    N_step = int(T_sim/dt)

    if expand: # default: False; change with --expand flag
        # expansion set to fixed values
        print("expansion set manually")
        expansion = np.ones(2)
        expansion *= 100.*np.log(5.)/2./(200./dt) # (1 + ex)**(200./dt) ~ sqrt(5) (area 5x biger after 2e5 steps)
        anisotropy=0.   # must be in [0,1] -> 0 = isotropy, 1 = expansion only on x
        expansion *= np.array([1.+anisotropy, 1.-anisotropy])
    else:
        # expansion through drag formula
        print("expansion set automatically")
        expansion = None


    # SET OUTPUT
    path = "outputs/"
    if not file_prefix is None: # default is None
        path += file_prefix+"_"

    path += "%dx%d_T%.0e_dt%.0e"%(xsize, ysize, T_sim, dt)

    if frame_every > 0.: # default: frame_every < 0; changed with --every flag
        N_skip = int(frame_every/dt)
        N_frames = int(T_sim/frame_every)
    else:
        N_skip = max(N_step//N_frames, 1)
    
    N_frames = min(N_frames,N_step)

    if move: # default: move=True, vertex=True, division=True, expansion=None
        if not expansion is None:
            if anisotropy != 0.:
                path += "_anis_%.1e_%.1e"%tuple(expansion)
            else:
                path += "_E%.1e"%(expansion[0])

        if not vertex:
            path += "_novtx"
        
        if not division:
            path += "_nodiv"

    elif not move:
        path += "_static"


    if cython:
        path += "_cy"


    if simulate:
      print("performing simulation")
    else:
      print("skip simulation")

    if plotting:
      print("plotting stuff")
    else:
      print("don't bother plotting")

    print("starting size = ", xsize, ysize, "\n")

    print("initialization time   = ", T_init)
    print("      ''       frames = ", int(T_init/frame_every))
    print("simulation time   = ", T_sim)
    print("   ''      frames = ", N_frames)

    print("\nsaving in / retrieving from  \"%s\"\n"%(path))
    if os.path.exists(path):
        print("path exists\n")

    if test_output:
        sys.exit()

    if simulate:
        os.system("mkdir -p "+path)

        if restart_file is None:
            print("Building NT object from scratch")
            neural_tube=build_NT_vtx(size = [xsize,ysize])
        
            # initialization
            print("Initialization: simulation of the vertex model only")
            for k in range(int(T_init/dt)):
                if k%N_skip == 0:
                    print("%2.1f/100   t = %.4f   frame = %d"%(k*dt/T_init*100., k*dt, int(k/N_skip)), end="\r")
                    with open (path+"/%06d_NT_init.pkl"%(k), "wb") as f:
                        dill.dump(neural_tube, f)

                neural_tube.evolve(diff_coef,prod_rate,bind_rate,degr_rate,.0,dt,
                    grn=False, morphogen=False)
                neural_tube.transitions(division=division)
            print("")
        else:
            # load from file
            print("Load from restart file: "+restart_file)
            neural_tube=load_NT_vtx(restart_file)
            print("")

        # selecting a random cell to leave the tissue (set its target area to 0)
        # "poisoned" is the idiotic name that should mean "differentiating"
        if not 'poisoned' in neural_tube.FE_vtx.cells.properties:
            n_cells = neural_tube.FE_vtx.cells.mesh.n_face
            cell_leaving = np.random.randint(n_cells)
            leaving = np.zeros(n_cells).astype(int)
            leaving[cell_leaving] = 1
            print("cell %d leaves the tissue"%(cell_leaving))
            neural_tube.FE_vtx.cells.properties['poisoned'] = leaving
        
        if not init_only:
            # simulation
            print("Simulation of the full model")
            for k in range(N_step+1):
                if k%N_skip == 0:
                    print("%2.1f/100   t = %.4f   frame = %d"%(k*dt/T_sim*100., k*dt, int(k/N_skip)), end="\r")
                    with open (path+"/%06d_NT.pkl"%(k), "wb") as f:
                        dill.dump(neural_tube, f)

                neural_tube.evolve(diff_coef,prod_rate,bind_rate,degr_rate,.0,dt,
                    expansion=expansion,vertex=vertex,move=move,morphogen=morphogen)
                neural_tube.transitions(division=division)
            print("")

    if plotting:

        try:
            allfiles = os.listdir(path)

        except FileNotFoundError:
            print("path not found: ", path)

        allNT = sorted([x for x in allfiles if "_NT.pkl" in x])
        # allNTinit = sorted([x for x in allfiles if "_NT_init.pkl" in x])
        NT_list = [load_NT_vtx(path+"/"+file) for file in allNT]
        # NTinit_list = [load_NT_vtx(path+"/"+file) for file in allNTinit]
        nodes_list = [
                    np.vstack([
                        nt.FE_vtx.cells.mesh.vertices.T[::3],
                        nt.FE_vtx.centroids[~nt.FE_vtx.cells.empty()]
                    ]) for nt in NT_list]
        concs_list = [nt.FE_vtx.concentration   for nt in NT_list]
        ponis_list = [nt.GRN.poni_grn.state   for nt in NT_list]
        cells_list = [nt.FE_vtx.cells   for nt in NT_list]
        verts_list = [nt.FE_vtx.cells.mesh.vertices.T[::3] for nt in NT_list]

        cells_state_video(cells_list, ponis_list, path, path+"/video_cells")
        morphogen_video(nodes_list,concs_list, path, path+"/video_morphogen")#,zmin=0)#,zmax=10)
    
