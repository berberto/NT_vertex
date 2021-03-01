#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

parser = ArgumentParser(description='Neural tube simulation')
parser.add_argument('--prefix', dest='file_prefix', metavar='<name>', type=str, default=None, help='Set prefix name (name of the simulation). Default: None')
parser.add_argument('--dry', dest='test_output', action='store_true', default=False, help='Dry run. Test output path and options')
parser.add_argument('--restart','-r', dest='restart_file', metavar='<file>', type=str, default=None, help='Restart from dill file storing a NT_vtx object: these are saved via "dill.dump(<NT obj>, <file>)"; NT object loaded as "<NT obj> = dill.dump(<file>)". The loaded NT object is used as initial condition for the full simulation (vertex + FE), and "thermalization" phase is skipped.')
parser.add_argument('--continue', dest='from_last', action='store_true', default=False, help='Continue run from last .pkl file in path.')

SIMpars = parser.add_argument_group(title='Simulation parameters')
SIMpars.add_argument('--dt', '-d', dest='dt', metavar='<val>', type=float, default=.001, help='Time step (float). Default: 0.001')
SIMpars.add_argument('-t', dest='T', metavar='<val>', type=float, default=200., help='Total time of the simulation (float). Default: 200.')
SIMpars.add_argument('--init', dest='T_init', metavar='<val>', type=float, default=.5, help='Initialization time: only relaxation of the vertex model (float). Default: .5')
SIMpars.add_argument('--init-only', dest='init_only', action='store_true', default=False, help='Run initialisation phase only (vertex model, no morphogen, no grn).')
SIMpars.add_argument('-s', dest='size', metavar=('<nX>','<nY>'), type=int, nargs=2, default=(20,10), help='Initial tissue size (width and height) in cell numbers. Default: (20 10)')
SIMpars.add_argument('--frames', dest='frames', metavar='<num>', type=int, default=100, help='Number of frames to save and plot (int). Overwritten by "--every". Default: 100')
SIMpars.add_argument('--every', dest='frame_every', metavar='<val>', type=float, default=-1, help='Simulation time between consecutive frames (float). Default: set by number of frames')
SIMpars.add_argument('--no-sim', dest='simulate', action='store_false', default=True, help='Do not simulate. Skip to plotting if target path and files exist')
SIMpars.add_argument('--no-plot', dest='plotting', action='store_false', default=True, help='Do not perform plot')
SIMpars.add_argument('--no-move', dest='move', action='store_false', default=True, help='Do not move cells at all, ie solve with static mesh.')
SIMpars.add_argument('--no-vertex', dest='vertex', action='store_false', default=True, help='Do not simulate the vertex model. Still, the expansion could be set different from zero, allowing cells to stretch, and divide')
SIMpars.add_argument('--no-morph', dest='morphogen', action='store_false', default=True, help='Simulate the vertex model only, concentration identically vanishing.')
SIMpars.add_argument('--no-divide', dest='division', action='store_false', default=True, help='Do not let cells divide')
SIMpars.add_argument('--cython', dest='cython', action='store_true', default=False, help='Use cython version of FE code')

FEpars = parser.add_argument_group(title='Morphogen parameters')
FEpars.add_argument('--diff-coef','-D', dest='diff_coef', metavar='<val>', type=float, default=.2, help='Diffusion coefficient. Default: 0.2')
FEpars.add_argument('--degr_rate','-k', dest='degr_rate', metavar='<val>', type=float, default=.1, help='Degradation rate. Default: 0.1')
FEpars.add_argument('--prod-rate','-f', dest='prod_rate', metavar='<val>', type=float, default=.05, help='Production rate. Default: 0.05')
FEpars.add_argument('--bind-rate','-b', dest='bind_rate', metavar='<val>', type=float, default=0., help='Binding rate. Default: 0')

#
#	not yet implemented --- need to change setup functions for cells
#
VTXpars = parser.add_argument_group(title='Vertex model parameters (UNEFFECTIVE)')
VTXpars.add_argument('--kappa','-K', dest='Kappa', metavar='<val>', type=float, default=1.0, help='Area elastic modulus. Default: 1.0')
VTXpars.add_argument('--gamma','-G', dest='Gamma', metavar='<val>', type=float, default=0.04, help='Global contractility parameter. Default: 0.04')
VTXpars.add_argument('--lambda','-L', dest='Lambda', metavar='<val>', type=float, default=.075, help='Line tension. Default: 0.05')
VTXpars.add_argument('--diff-adh', dest='diff_adhesion', metavar='<val>', type=float, default=None, help='Ratio between line tension at the boundary of floorplate region, and all the other line tensions (Lambda). Default: None (equivalent to 1., but skips the differential adhesion calculations)')

args = parser.parse_args()

file_prefix = args.file_prefix
test_output = args.test_output
restart_file = args.restart_file
from_last = args.from_last

T_sim = args.T
T_init = args.T_init
frame_every = args.frame_every
init_only = args.init_only
dt = args.dt
N_frames = args.frames

simulate = args.simulate
plotting = args.plotting
(xsize,ysize) = args.size
vertex = args.vertex
morphogen = args.morphogen
cython = args.cython
move = args.move
division = args.division

# parameters of the morphogen PDE
degr_rate = args.degr_rate # default: 0.1
prod_rate = args.prod_rate # default: 0.05
diff_coef = args.diff_coef # default: 0.2
bind_rate = args.bind_rate # default: 0.

# parameters of the vertex model
Kappa = args.Kappa # default: 1.0
Gamma = args.Gamma # default: 0.04
Lambda = args.Lambda # default: 0.075
diff_adhesion = args.diff_adhesion # default: None


def print_options (filename=None):
	print("---- optional argument values ----")
	for key, val in vars(args).items():
		print(f'{key} = {val}')
	print("----------------------------------")

	if filename is not None:
		with open(filename, 'w') as file:
			for key, val in vars(args).items():
				file.write(f'{key} = {val}\n')
			file.close()
