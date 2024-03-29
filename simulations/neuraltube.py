# -*- coding: utf-8 -*-

from nt_vertex import NT_simulation


if __name__ == "__main__":

    # Instatiate a simulation object.
    # See nt_vertex.NT_sim for the available keyward arguments,
    # or run "python neuraltube.py --help" to set them
    # from command line.
    sim = NT_simulation(seed=1990)
    # sim = NT_simulation(seed=1990, path="outputs/test")

    # run simulation and save snapshots in "path"
    sim.run()

    sim.analysis()

    # gather saved simulations and produce a video
    sim.video(duration=60.) # duration in s.
 