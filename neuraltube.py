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

from argparse import ArgumentParser

parser = ArgumentParser(description='Run neuraltube simulation')

parser.add_argument('-t', dest='T', metavar='<val>', type=float, default=200.,
                help='Total time of the simulation (float). Default: 200.')
parser.add_argument('-s', dest='size', metavar=('<X>','<Y>'), type=int, nargs=2, default=(20,10),
                help='Initial tissue size (width and height) in cell numbers. Default: (20 10)')
parser.add_argument('--prefix', dest='file_prefix', metavar='<name>', type=str, default=None,
                help='Set prefix name (name of the simulation test). Default: none')
parser.add_argument('--frames', dest='frames', metavar='<num>', type=int, default=100,
                help='Number of frames to save and plot (int)')
parser.add_argument('--dt', '-d', dest='dt', metavar='<val>', type=float, default=.001,
                help='Time step (float). Default: 0.001')
parser.add_argument('--no-sim', dest='simulate', action='store_false', default='True',
                help='Do not simulate. Skip to plotting if target path and files exist')
parser.add_argument('--no-plot', dest='plotting', action='store_false', default='True',
                help='Do not perform plot')
parser.add_argument('--no-move', dest='move', action='store_false', default=True,
                help='Do not move cells at all, ie solve with static mesh.')
parser.add_argument('--no-vertex', dest='vertex', action='store_false', default=True,
                help='Do not simulate the vertex model. Still, the expansion could be set\
                different from zero, allowing cells to stretch, and divide')
parser.add_argument('--no-division', dest='division', action='store_false', default='True',
                help='Do not let cells divide')
parser.add_argument('--cython', dest='cython', action='store_true', default=False,
                help='Use cython version of FE code')
parser.add_argument('--dry', dest='test_output', action='store_true', default=False,
                help='Dry run. Test output path and options')

parser.add_argument('--diff-coef','-D', dest='diff_coef', metavar='<val>', type=float, default=.2,
                help='Diffusion coefficient. Default: 0.2')
parser.add_argument('--degr_rate','-k', dest='degr_rate', metavar='<val>', type=float, default=.1,
                help='Degradation rate. Default: 0.1')
parser.add_argument('--prod-rate','-f', dest='prod_rate', metavar='<val>', type=float, default=.05,
                help='Production rate. Default: 0.05')
parser.add_argument('--bind-rate','-b', dest='bind_rate', metavar='<val>', type=float, default=0.,
                help='Binding rate. Default: 0')

args = parser.parse_args()

# parameters of the PDE
degr_rate = args.degr_rate # default: 0.1
prod_rate = args.prod_rate # default: 0.05
diff_coef = args.diff_coef # default: 0.2
bind_rate = args.bind_rate # default: 0.

file_prefix = args.file_prefix

T_sim = args.T
dt = args.dt
N_frames = args.frames

simulate = args.simulate
plotting = args.plotting
(xsize,ysize) = args.size
test_output = args.test_output
vertex = args.vertex
cython = args.cython
move = args.move
division = args.division

if simulate:
  print("performing simulation")
else:
  print("skip simulation")

if plotting:
  print("plotting stuff")
else:
  print("don't bother plotting")


print("frames = ", N_frames)
print("tot time = ", T_sim)
print("starting size = ", xsize, ysize)


if __name__ == "__main__":
    
    np.random.seed(1984)

    N_step = int(T_sim/dt)
    N_frames = min(N_frames,N_step)


    # fixed expansion parameters
    expansion = np.ones(2)
    expansion *= np.log(5.)/2./(200./dt) # (1 + ex)**2e5 ~ sqrt(5) (area 5x biger after 2e5 steps)
    anisotropy=0.   # must be in [0,1] -> 0 = isotropy, 1 = expansion only on x
    expansion *= np.array([1.+anisotropy, 1.-anisotropy])


    # SET OUTPUT
    path = "outputs/"
    if not file_prefix is None:
        path += file_prefix+"_"

    path += "%dx%d_T%.0e_dt%.0e"%(xsize, ysize, T_sim, dt)
    N_skip = max(N_step//N_frames, 1)

    if not vertex:
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
        # t1=time.time()
        for k in range(N_step+1):
            if k%N_skip == 0:
                print(k)
                with open (path+"/%06d_NT.pkl"%(k), "wb") as f:
                    dill.dump(neural_tube, f)

            if cython:
                neural_tube.evolve_fast(diff_coef,prod_rate,bind_rate,degr_rate,.0,dt,
                    expansion=expansion,vertex=vertex
                )
            else:
                neural_tube.evolve(diff_coef,prod_rate,bind_rate,degr_rate,.0,dt,
                    expansion=expansion,vertex=vertex,move=move
                )
            neural_tube.transitions(division=division)
        # t2 = time.time()
        # print("took:", t2 - t1)


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

        cells_state_video(cells_list, ponis_list, path, path+"/state-vid")
        animate_surf_video_mpg(nodes_list,concs_list, path, path+"/surface-video")#,zmin=0)#,zmax=10)
    
