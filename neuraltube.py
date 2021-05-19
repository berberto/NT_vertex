#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import dill
import numpy as np
from datetime import datetime

from NT_vtx import build_NT_vtx, load_NT_vtx
from plotting import morphogen_video, cells_state_video, combined_video
from options import (file_prefix, test_output, restart_file,
                T_sim, T_init, frame_every, init_only, dt, N_frames,
                simulate, plotting, from_last,
                vertex, morphogen, move, division,
                xsize,ysize, 
                degr_rate, prod_rate, diff_coef, bind_rate,
                Kappa, Gamma, Lambda, diff_adhesion,
                print_options
                )

if __name__ == "__main__":
    
    np.random.seed(1984)

    N_step = int(T_sim/dt)
    N_step_init = int(T_init/dt)


    # number of frames
    if frame_every > 0.: # default: frame_every < 0; changed with --every flag
        N_skip = int(frame_every/dt)
        N_frames = int(T_sim/frame_every)
    else:
        N_skip = max(N_step//N_frames, 1)
    N_frames = min(N_frames,N_step)

    time_id = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')

    # SET OUTPUT
    if file_prefix is not None: # default is None
        path = f"outputs/{file_prefix}"
    else:
        path = f"outputs/{time_id}"

    if morphogen:
        path += '_D:%.3f_k:%.3f_f:%.3f_b:%.3f'%(diff_coef, degr_rate, prod_rate, bind_rate)

    # debugging options (selectively remove part of the dynamics)
    if move: # default: move=True, vertex=True, division=True
        if not vertex:
            path += "_novtx"
        
        if not division:
            path += "_nodiv"
    else:
        path += "_static"

    print(f"\nsaving in / retrieving from  \"{path}\"")
    if os.path.exists(path):
        print("(path alread exists)\n")
    else:
        os.system("mkdir -p "+path)

    print_options(f"{path}/parameters.txt")

    print("starting size = ", xsize, ysize, "\n")

    print("initialization time   = ", T_init)
    print("      ''       frames = ", int(T_init/frame_every))
    print("simulation time   = ", T_sim)
    print("   ''      frames = ", N_frames)
    
    if test_output:
        exit(0)

    # start from the last saved step when option --continue is passed.
    #
    # note: different from --restart <file> option, where configuration
    #   file is used as initial condition at time 0, not at time contained
    #   in the name of the last saved file
    k_start = 0
    if from_last:
        last_file = [file for file in os.listdir(path) if '_NT.pkl' in file]
        last_file.sort() # sort the list of files, as os.listdir gives random order
        last_file = last_file[-1]
        k_start = int(last_file.split('_')[0])
        restart_file = f'{path}/{last_file}'
        print(f'\nInitial configuration from \"{restart_file}\"')

    if simulate:

        if restart_file is None:
            print('\nBuilding NT object from scratch')
            neural_tube=build_NT_vtx(size = [xsize,ysize])
        
            # initialization
            print('Initialization: simulation of the vertex model only')
            for k in range(N_step_init):
                if k%N_skip == 0:
                    print("%2.1f/100   t = %.4f   frame = %d"%(k*dt/T_init*100., k*dt, int(k/N_skip)), end="\r")
                    with open (path+"/%06d_NT_init.pkl"%(k), "wb") as f:
                        dill.dump(neural_tube, f)

                diff_rates=neural_tube.GRN.diff_rates.copy()
                neural_tube.evolve(diff_coef,prod_rate,bind_rate,degr_rate,.0,dt,
                    vertex=vertex,move=move,morphogen=False,
                    diff_rates=diff_rates,diff_adhesion=None)
                neural_tube.transitions(division=division)
                
                # set the source of morphogen to be few cells wide
                neural_tube.set_source_by_x(width=4)

            print('')
        else:
            print(f'Loading restart file \"{restart_file}\"')
            neural_tube=load_NT_vtx(restart_file)
            if 'leaving' not in neural_tube.properties:
                neural_tube.properties['leaving'] = np.zeros(len(neural_tube))
            print('')

        # selecting a random cell to leave the tissue (set its target area to 0)
        # if 'leaving' not in neural_tube.FE_vtx.cells.properties:
        #     n_cells = neural_tube.FE_vtx.cells.mesh.n_face
        #     cell_leaving = np.random.randint(n_cells)
        #     leaving = np.zeros(n_cells).astype(int)
        #     leaving[cell_leaving] = 1
        #     print("cell %d leaves the tissue"%(cell_leaving))
        #     neural_tube.FE_vtx.cells.properties['leaving'] = leaving
        
        if not init_only:
            # simulation
            print("Simulation of the full model")
            for k in range(k_start,N_step+1):
                if k%N_skip == 0:
                    print("%2.1f/100   t = %.4f   frame = %d"%(k*dt/T_sim*100., k*dt, int(k/N_skip)), end="\r")
                    with open (path+"/%06d_NT.pkl"%(k), "wb") as f:
                        dill.dump(neural_tube, f)

                leaving=neural_tube.properties['leaving']

                diff_rates=neural_tube.GRN.diff_rates.copy()
                neural_tube.evolve(diff_coef,prod_rate,bind_rate,degr_rate,.0,dt,
                    vertex=vertex,move=move,morphogen=morphogen,
                    diff_rates=diff_rates,diff_adhesion=diff_adhesion)
                neural_tube.transitions(division=division)
            print("")

    if plotting:

        try:
            allfiles = os.listdir(path)

        except FileNotFoundError:
            raise FileNotFoundError("path not found: ", path)

        allNT = sorted([x for x in allfiles if "_NT.pkl" in x])

        print(f'\n{len(allNT)} frames found')

        # allNTinit = sorted([x for x in allfiles if "_NT_init.pkl" in x])
        NT_list = [load_NT_vtx(path+"/"+file) for file in allNT]
        # NTinit_list = [load_NT_vtx(path+"/"+file) for file in allNTinit]
        nodes_list = [
                    np.vstack([
                        nt.FE_vtx.cells.mesh.vertices.T[::3],
                        nt.FE_vtx.centroids[~nt.FE_vtx.cells.empty()]
                    ]) for nt in NT_list]
        concs_list = [nt.FE_vtx.concentration   for nt in NT_list]
        ponis_list = [nt.GRN.state[:,-4:]   for nt in NT_list]
        cells_list = [nt.FE_vtx.cells   for nt in NT_list]
        verts_list = [nt.FE_vtx.cells.mesh.vertices.T[::3] for nt in NT_list]

        # cells_state_video(cells_list, ponis_list, path, path+"/video_cells")
        # morphogen_video(cells_list, nodes_list, concs_list, path, path+"/video_morphogen")#,zmin=0)#,zmax=10)
        combined_video(cells_list, nodes_list, concs_list, ponis_list, path, path+"/video_combined")#,zmin=0)#,zmax=10)
    
