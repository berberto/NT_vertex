#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import dill
import numpy as np
from NT_vtx import build_NT_vtx_from_scratch
from plotting import animate_surf_video_mpg, cells_state_video


def load (filename):
    with open(filename, "rb") as f:
        return dill.load(f)


if __name__ == "__main__":
    
    np.random.seed(1984)

    test_output=False
    simulate = False
    plotting = True
    evolve_vertex = False
    cython = False

    # default values
    xsize=20
    ysize=10
    dt = .0001
    tSym = 20. # total time 
    N_step = int(tSym/dt)
    N_frames = 100
    N_frames = min(N_frames,N_step)

    # parameters of the PDE
    degr_rate = 0.1
    prod_rate = 0.05
    diff_coef = 0.2
    bind_rate = 0.

    # fixed expansion parameters
    expansion = np.ones(2)
    expansion *= np.log(5.)/2./N_step # (1 + ex)**2e5 ~ sqrt(5) (area 5x biger after 2e5 steps)
    anisotropy=0.   # must be in [0,1] -> 0 = isotropy, 1 = expansion only on x
    expansion *= np.array([1.+anisotropy, 1.-anisotropy])


    # SET OUTPUT
    if len(sys.argv) > 1:
        N_step=int(sys.argv[1])
        if len(sys.argv) > 2:
            N_frames = int(sys.argv[2])
    path="outputs/%dx%d_N%.0e_dt%.0e"%(xsize, ysize, N_step, dt)
    N_skip = max(N_step//N_frames, 1)

    if not evolve_vertex:
        expansion *= 0.
        path += "_novertex"
    else:
        if anisotropy != 0.:
            path += "_anis_%.1e_%.1e"%tuple(expansion)

    if cython:
        path += "_cy"

    print("N_step   =", N_step)
    print("N_frames =", N_frames)
    print("N_skip   =", N_skip)
    print("saving in / retrieving from  \"%s\"\n"%(path))
    if os.path.exists(path):
        print("path exists\n")

    if test_output:
        sys.exit()

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
                with open (path+"/%06d_NT.pkl"%(k), "wb") as f:
                    dill.dump(neural_tube, f)

            if cython:
                neural_tube.evolve_fast(diff_coef,prod_rate,bind_rate,degr_rate,.0,dt,expansion=expansion,
                    evolve_vertex=evolve_vertex
                )
            else:
                neural_tube.evolve(diff_coef,prod_rate,bind_rate,degr_rate,.0,dt,expansion=expansion,
                    evolve_vertex=evolve_vertex
                )
            neural_tube.transitions()
        t2 = time.time()
        print("took:", t2 - t1)


    if plotting:

        try:
            allfiles = os.listdir(path)

        except FileNotFoundError:
            print("path not found: ", path)

        allNT = sorted([x for x in allfiles if "_NT.pkl" in x])
        NT_list = [load(path+"/"+file) for file in allNT]
        nodes_list = [
                    np.vstack([
                        nt.FE_vtx.cells.mesh.vertices.T[::3],
                        nt.FE_vtx.centroids[~nt.FE_vtx.cells.empty()]
                    ]) for nt in NT_list]
        concs_list = [nt.FE_vtx.concentration   for nt in NT_list]
        ponis_list = [nt.GRN.poni_grn.state   for nt in NT_list]
        cells_list = [nt.FE_vtx.cells   for nt in NT_list]
        verts_list = [nt.FE_vtx.cells.mesh.vertices.T[::3] for nt in NT_list]

        # print(nodes_list[-1][0], concs_list[-1][0], ponis_list[-1][0],  N_step)
        cells_state_video(cells_list, ponis_list, path, path+"/state-vid")
        animate_surf_video_mpg(nodes_list,concs_list, path, path+"/surface-video")#,zmin=0,zmax=10)
    
