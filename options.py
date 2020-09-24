from argparse import ArgumentParser

parser = ArgumentParser(description='Run neuraltube simulation')

parser.add_argument('--dt', '-d', dest='dt', metavar='<val>', type=float, default=.001, help='Time step (float). Default: 0.001')
parser.add_argument('-t', dest='T', metavar='<val>', type=float, default=200., help='Total time of the simulation (float). Default: 200.')
parser.add_argument('--init', dest='T_init', metavar='<val>', type=float, default=.5, help='Initialization time: only relaxation of the vertex model (float). Default: .5')
parser.add_argument('--init-only', dest='init_only', action='store_true', default=False, help='Run initialisation phase only (vertex model, no morphogen, no grn).')
parser.add_argument('-s', dest='size', metavar=('<X>','<Y>'), type=int, nargs=2, default=(20,10), help='Initial tissue size (width and height) in cell numbers. Default: (20 10)')
parser.add_argument('--prefix', dest='file_prefix', metavar='<name>', type=str, default=None, help='Set prefix name (name of the simulation test). Default: none')
parser.add_argument('--frames', dest='frames', metavar='<num>', type=int, default=100, help='Number of frames to save and plot (int)')
parser.add_argument('--every', dest='frame_every', metavar='<val>', type=float, default=-1, help='Simulation time between consecutive frames (float). Default: set by number of frames')
parser.add_argument('--expand', dest='expand', action='store_true', default=False, help='Set expansion manually. Default is expansion through drag.')
parser.add_argument('--no-sim', dest='simulate', action='store_false', default=True, help='Do not simulate. Skip to plotting if target path and files exist')
parser.add_argument('--no-plot', dest='plotting', action='store_false', default=True, help='Do not perform plot')
parser.add_argument('--no-move', dest='move', action='store_false', default=True, help='Do not move cells at all, ie solve with static mesh.')
parser.add_argument('--no-vertex', dest='vertex', action='store_false', default=True, help='Do not simulate the vertex model. Still, the expansion could be set different from zero, allowing cells to stretch, and divide')
parser.add_argument('--no-morph', dest='morphogen', action='store_false', default=True, help='Simulate the vertex model only, concentration identically vanishing.')
parser.add_argument('--no-divide', dest='division', action='store_false', default=True, help='Do not let cells divide')
parser.add_argument('--cython', dest='cython', action='store_true', default=False, help='Use cython version of FE code')
parser.add_argument('--dry', dest='test_output', action='store_true', default=False, help='Dry run. Test output path and options')

parser.add_argument('--diff-coef','-D', dest='diff_coef', metavar='<val>', type=float, default=.2, help='Diffusion coefficient. Default: 0.2')
parser.add_argument('--degr_rate','-k', dest='degr_rate', metavar='<val>', type=float, default=.1, help='Degradation rate. Default: 0.1')
parser.add_argument('--prod-rate','-f', dest='prod_rate', metavar='<val>', type=float, default=.05, help='Production rate. Default: 0.05')
parser.add_argument('--bind-rate','-b', dest='bind_rate', metavar='<val>', type=float, default=0., help='Binding rate. Default: 0')

args = parser.parse_args()

# parameters of the PDE
degr_rate = args.degr_rate # default: 0.1
prod_rate = args.prod_rate # default: 0.05
diff_coef = args.diff_coef # default: 0.2
bind_rate = args.bind_rate # default: 0.

file_prefix = args.file_prefix

T_sim = args.T
T_init = args.T_init
frame_every = args.frame_every
init_only = args.init_only
dt = args.dt
N_frames = args.frames

simulate = args.simulate
plotting = args.plotting
(xsize,ysize) = args.size
test_output = args.test_output
vertex = args.vertex
morphogen = args.morphogen
cython = args.cython
expand = args.expand
move = args.move
division = args.division