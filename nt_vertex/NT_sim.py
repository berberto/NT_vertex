import numpy as np
import sys
import os
from datetime import datetime
import dill

from .NT_vtx import build_NT_vtx, load_NT_vtx
from .plotting import combined_video, plot_frame, snapshot
from .options import (file_prefix, test_output, restart_file, verbose,
                T_sim, T_init, frame_every, init_only, dt, N_frames,
                simulate, plotting, from_last,
                vertex, morphogen, move, division,
                xsize,ysize, 
                degr_rate, prod_rate, diff_coef, bind_rate, source_width,
                nucl_stiff, nucl_crowd, nucl_size, nucl_noise,
                Kappa, Gamma, Lambda, diff_adhesion,
                print_options
                )

# re-definition of print function:
# remains the same if verbose is True,
# does nothing otherwise
import builtins
def print(*args, **kwargs):
    if verbose:
        return builtins.print(*args, **kwargs)
    else:
        pass


class NT_simulation (object):
    def __init__(self, T_sim=T_sim, T_init=T_init, dt=dt,
            frame_every=frame_every, N_frames=N_frames,
            diff_coef=diff_coef,degr_rate=degr_rate,
            prod_rate=prod_rate, bind_rate=bind_rate,
            Kappa=Kappa, Gamma=Gamma, Lambda=Lambda,
            nucl_stiff=10., nucl_crowd=5., nucl_size=0.2, nucl_noise=0.01,
            simulate=simulate, plotting=plotting, init_only=init_only,
            from_last=from_last, restart_file=restart_file,
            dry=test_output,
            diff_adhesion=diff_adhesion,
            print_options=print_options,
            xsize=xsize, ysize=ysize,
            seed=None, t0=0.,
            path=None,
            prefix=file_prefix,
            outputdir='outputs'):

        self.seed = seed
        np.random.seed(self.seed)

        # simulation parameters
        self.t0=t0
        self.T_sim=T_sim
        self.T_init=T_init
        self.dt=dt
        self.xsize=xsize
        self.ysize=ysize
        self.frame_every=frame_every
        self.N_frames=N_frames
        self.simulate=simulate
        self.init_only=init_only
        self.plotting=plotting
        self.from_last=from_last
        self.restart_file=restart_file
        self.N_step = int(T_sim/dt)
        self.N_step_init = int(T_init/dt)
        self.time=t0
        # number of frames
        if self.frame_every > 0.: # default: frame_every < 0; changed with --every flag
            self.N_skip = int(self.frame_every/self.dt)
            self.N_frames = int(self.T_sim/self.frame_every)
        else:
            self.N_skip = max(self.N_step//self.N_frames, 1)
        self.N_frames = min(self.N_frames,self.N_step)

        # physical parameters - diffusion
        self.diff_coef=diff_coef
        self.degr_rate=degr_rate
        self.prod_rate=prod_rate
        self.bind_rate=bind_rate
        self.source_width=source_width
        # physical parameters - vertex
        self.Kappa=Kappa
        self.Gamma=Gamma
        self.Lambda=Lambda
        self.diff_adhesion=diff_adhesion
        # physical parameters - IKNM
        self.nucl_stiff=nucl_stiff
        self.nucl_crowd=nucl_crowd
        self.nucl_size=nucl_size
        self.nucl_noise=nucl_noise

        # debug options
        self.vertex=vertex
        self.morphogen=morphogen
        self.move=move
        self.division=division

        # output names
        self.prefix = prefix
        self.outputdir = outputdir

        # string with timestamp
        self.time_id = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')

        self.make_paths(path)

        print_options(self.path+"/parameters.txt")

        if dry:
            exit(0)

    def set_path(self, path=None):
        if path is not None:
            self.path = path
        else:
            if self.prefix is not None: # default is None
                self.path = f"{self.outputdir}/{self.prefix}"
            else:
                self.path = f"{self.outputdir}/{self.time_id}"

            if self.morphogen:
                self.path += '_D=%.3f_k=%.3f_f=%.3f_b=%.3f'%(diff_coef, degr_rate, prod_rate, bind_rate)

            # debugging options (selectively remove part of the dynamics)
            if self.move: # default: move=True, vertex=True, division=True
                if not self.vertex:
                    self.path += "_novtx"
                
                if not self.division:
                    self.path += "_nodiv"
            else:
                self.path += "_static"

    def make_paths(self, path=None):

        self.set_path(path)
        self.checkpoints_dir = self.path+"/checkpoints"
        self.stats_dir = self.path+"/stats"
        self.frames_dir = self.path+"/frames"
        self.plots_dir = self.path+"/plots"

        print(f"\nsaving in / retrieving from  \"{self.checkpoints_dir}\"")

        paths = [self.checkpoints_dir, self.stats_dir, self.frames_dir, self.plots_dir]
        for path in paths:
            if os.path.exists(path):
                print("Directory \""+path+"\" already exists")
            else:
                os.makedirs(path, exist_ok=True)

    def __call__ (self):
        self.run()
        self.video()

    def save (self, k, T=None, suffix="nt.pkl"):
        if k%self.N_skip == 0:
            print("%2.1f/100   t = %.4f   frame = %d"%(k*self.dt/T*100., k*self.dt, int(k/self.N_skip)), end="\r")
            outfile=(self.checkpoints_dir+"/"+_format_time(k*self.dt)+"_"+suffix)
            with open (outfile, "wb") as f:
                dill.dump(self.neural_tube, f)

    def load(self, files='main', index=None):
        try:
            allfiles = os.listdir(self.checkpoints_dir)

        except FileNotFoundError:
            raise FileNotFoundError("path not found: \""+self.checkpoints_dir+"\"")

        if files == 'init':
            allNT = sorted([x for x in allfiles if "_NT_init.pkl" in x])
            self.init_times = np.array([float(name.split("_NT")[0]) for name in allNT])
        elif files == 'main':
            allNT = sorted([x for x in allfiles if "_NT.pkl" in x])
            self.times = np.array([float(name.split("_NT")[0]) for name in allNT])
        elif files == 'both':
            allNT_init = sorted([x for x in allfiles if "_NT_init.pkl" in x])
            self.init_times = np.array([float(name.split("_NT")[0]) for name in allNT_init])
            allNT = sorted([x for x in allfiles if "_NT.pkl" in x])
            self.times = np.array([float(name.split("_NT")[0]) for name in allNT])
            allNT = (allNT_init + allNT).copy()
        else:
            raise ValueError(f"Invalid 'files' option '{files}'")

        if len(allNT) == 0:
            raise FileNotFoundError("No snapshots found in \""+self.checkpoints_dir+"\"")

        # print(f'\n{len(allNT)} frames found')

        if index is None:
            return [load_NT_vtx(self.checkpoints_dir+"/"+file) for file in allNT]
        else:
            return load_NT_vtx(self.checkpoints_dir+"/"+allNT[index])


    def initialize (self):

        # start from the last saved step when option --continue is passed.
        #
        # note: different from --restart <file> option, where configuration
        #   file is used as initial condition at time 0, not at time contained
        #   in the name of the last saved file
        self.k_start = 0
        if self.from_last:
            last_file = [file for file in os.listdir(self.path) if '_NT.pkl' in file]
            last_file.sort() # sort the list of files, as os.listdir gives random order
            last_file = last_file[-1]
            self.k_start = int(last_file.split('_')[0])
            self.restart_file = f'{self.path}/{last_file}'
            print(f'\nInitial configuration from \"{self.restart_file}\"')

        if self.restart_file is None:
            print('\nBuilding NT object from scratch')
            self.neural_tube=build_NT_vtx(size = [self.xsize,self.ysize])

            # set the source of morphogen to be few cells wide
            self.neural_tube.set_source_by_x(width=self.source_width)
        
            # initialization
            print('Initialization: simulation of the vertex model only')
            for k in range(self.N_step_init):
                self.save(k, T=T_init, suffix="NT_init.pkl")

                diff_rates=self.neural_tube.GRN.diff_rates.copy()
                self.neural_tube.evolve(self.diff_coef,self.prod_rate,self.bind_rate,self.degr_rate,
                    self.time,self.dt,
                    nucl_stiff=self.nucl_stiff, nucl_crowd=self.nucl_crowd,
                    nucl_size=self.nucl_size, nucl_noise=self.nucl_noise,
                    vertex=self.vertex, move=self.move, morphogen=False,
                    diff_adhesion=self.diff_adhesion,
                    diff_rates=diff_rates)
                self.neural_tube.transitions(division=self.division)     
            print('')
        else:
            print(f'Loading restart file \"{self.restart_file}\"')
            self.neural_tube=load_NT_vtx(self.restart_file)
            if 'leaving' not in self.neural_tube.properties:
                self.neural_tube.properties['leaving'] = np.zeros(len(self.neural_tube))
            print('')


    def run(self):
        if self.simulate:

            self.initialize()

            if not self.init_only:
                # simulation
                print("Simulation of the full model")
                for k in range(self.k_start,self.N_step+1):
                    self.save(k, T=T_sim, suffix="NT.pkl")

                    leaving=self.neural_tube.properties['leaving']

                    diff_rates=self.neural_tube.GRN.diff_rates.copy()
                    self.neural_tube.evolve(self.diff_coef,self.prod_rate,self.bind_rate,self.degr_rate,
                        self.time,self.dt,
                        nucl_stiff=self.nucl_stiff, nucl_crowd=self.nucl_crowd,
                        nucl_size=self.nucl_size, nucl_noise=self.nucl_noise,
                        vertex=self.vertex, move=self.move, morphogen=self.morphogen,
                        diff_adhesion=self.diff_adhesion,
                        diff_rates=diff_rates)
                    self.neural_tube.transitions(division=self.division)
                    self.time += self.dt
                print("")

    def video(self, duration=60., files='main', log=False):
        if self.plotting:
            NT_list = self.load(files=files)
            combined_video(NT_list, filename=self.path+"/video_combined", duration=duration, log=log)
        else:
            print("skip plotting")


    def save_stats (self, files='main'):

        NT_list = self.load(files=files)
        last = NT_list[-1]
        N_cells = len(last.cells)
        state_ = np.nan * np.ones( last.cell_state.shape + self.times.shape ).astype(float)
        ages_ = np.nan * np.ones( (N_cells,) + self.times.shape ).astype(float)
        areas_ = np.nan * np.ones( (N_cells,) + self.times.shape ).astype(float)
        neigs_ = np.nan * np.ones( (N_cells,) + self.times.shape ).astype(int)
        zposn_ = np.nan * np.ones( (N_cells,) + self.times.shape ).astype(float)

        for (i,t), nt in zip(enumerate(self.times), NT_list):

            # get the cells that exist at any given time
            alive = np.where(~nt.cells.empty())[0]

            # save properties of alive cells (other entries are nan)
            state_[alive,:,i] = nt.cell_state[alive]
            ages_[alive,i] = nt.properties['age'][alive]
            zposn_[alive,i] = nt.properties['zposn'][alive]
            areas_[alive,i] = nt.mesh.area[alive]
            neigs_[alive,i] = nt.mesh.neighbours[alive]

        np.save(self.stats_dir+"/state.npy", state_)
        np.save(self.stats_dir+"/ages.npy", ages_)
        np.save(self.stats_dir+"/zposn.npy", zposn_)
        np.save(self.stats_dir+"/areas.npy", areas_)
        np.save(self.stats_dir+"/neigs.npy", neigs_)

        self.stats = {}
        self.stats['state'] = state_
        self.stats['ages'] = ages_
        self.stats['zposn'] = zposn_
        self.stats['areas'] = areas_
        self.stats['neigs'] = neigs_

    def plots(self):

        import matplotlib.pyplot as plt
        from matplotlib import use
        use('tkagg')

        imshow_kwargs = {'origin':'lower',
                  # 'extent': [xmin, xmax, ymin, ymax],
                  # 'vmin':0.,
                  # 'vmax':np.max(action_high[:,i])
                }

        fig, ax = plt.subplots()
        ax.imshow(self.stats['age'], **imshow_kwargs)

        plt.show()

    def analysis (self):
        self.save_stats()
        self.plots()
        
    
    # TO DEBUG
    def snapshot(self, time, files="main"):
        _ = self.load(files=files)
        if time < self.times.min() or time > self.times.max():
            print("Time outside the simulated time interval.\n",
                  "Clipping between min and max.")
        idx = np.argmin(np.abs(self.times - time))
        nt = self.load(files=files, index=idx)
        filename = self.frames_dir+"/"+_format_time(time)+"_frame.svg"
        snapshot(nt, filename=filename)


def _format_time (time):
    return "{:07.3f}".format(time)