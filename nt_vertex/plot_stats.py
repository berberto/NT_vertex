import os
import numpy as np
import dill
from matplotlib import use
import matplotlib.pyplot as plt
import sys
use('tkagg')


def plot_stats (path=None, outdir=None):
    
    try:
        if path is None:
            path = sys.argv[1]
        state_ = np.load(path+"/state.npy")
        ages_  = np.load(path+"/ages.npy")
        zposn_ = np.load(path+"/zposn.npy")
        areas_ = np.load(path+"/areas.npy")
        neigs_ = np.load(path+"/neigs.npy")
        with open(path+"/adjs.pkl", "w") as f:
            adjs_ = dill.load(f)
    except:
        print("Something wrong with the path")
        exit(1)

    if outdir is None:
        outdir = path

    '''
    number of alive cells at any given time:
    1. check the NON-NaN values in a matrix (like `ages_`, N_cells x N_times)
    2. in each column (that is time-step) count the number of NON-NaN values
    '''
    fig, ax = plt.subplots()
    ax.set_xlabel("Time step")
    ax.set_ylabel("Number of cells")
    alive_ = (~np.isnan(ages_)).astype(int)
    alive_ = np.sum(alive_, axis=0)
    ax.plot(alive_)
    plt.savefig(outdir+"/alive.png")

    '''
    total area of the tissue at any given time:
    sum NON-NaN values of `areas_` in each column (time step)
    '''
    fig, ax = plt.subplots()
    ax.set_xlabel("Time step")
    ax.set_ylabel("Total tissue area")
    areatot_ = np.nansum(areas_, axis=0)
    ax.plot(areatot_)
    plt.savefig(outdir+"/total_area.png")


    '''
    mean and variance of the distribution of the z-position of the nuclei
    '''
    fig, ax = plt.subplots()
    ax.set_xlabel("Time step")
    ax.set_ylabel("A-B nuclear position")
    z25_ = np.nanpercentile(zposn_, 25, axis=0)
    z50_ = np.nanpercentile(zposn_, 50, axis=0)
    z75_ = np.nanpercentile(zposn_, 75, axis=0)
    zmean_ = np.nanmean(zposn_, axis=0)
    zstd_ = np.nanstd(zposn_, axis=0)
    ax.plot(zmean_,label='mean')
    ax.plot(zstd_,label='std')
    ax.plot(z25_,ls='--', lw=1, label='25 %ile')
    ax.plot(z50_,ls='--', lw=1, label='50 %ile')
    ax.plot(z75_,ls='--', lw=1, label='75 %ile')
    ax.legend(loc='best')
    plt.savefig(outdir+"/AB_stats.png")


    '''
    distribution of z-position at selected times
    '''
    fig, ax = plt.subplots()
    ax.set_xlabel("Time step")
    ax.set_ylabel("A-B nuclear position")
    times_ = np.linspace(0,zposn_.shape[1], 10, endpoint=False).astype(int)
    z_distr = list(map( lambda x: x[~np.isnan(x)], np.take(zposn_, times_, axis=1).T ))
    ax.violinplot(z_distr, times_, widths=0.8*(times_[1]-times_[0]))
    plt.savefig(outdir+"/AB_violin.png")


    '''
    Target area as a function of time and space
    '''
    def _target_area (age, zpos, M=1.):
        eps = 0.05
        t1 = 0.9
        g = np.log((1-eps)/eps)/t1
        return M*(1 + zpos**2)/(1 + np.exp(- g * age))
        
    fig, ax = plt.subplots(1,2)
    ax[0].set_xlabel("Time")
    ax[1].set_xlabel("A-B position")
    ax[0].set_ylabel("Target area")
    ax[1].set_ylabel("Target area")
    times_ = np.arange(0,1,0.2)
    zs_ = np.arange(0,1,0.2)
    xs_ = np.linspace(0,1,100)
    for z in zs_:
        ax[0].plot(xs_, _target_area(xs_, z), label="{:.2f}".format(z))
    ax[0].legend(title="A-B position")
    for time in times_:
        ax[1].plot(xs_, _target_area(time, xs_), label="{:.2f}".format(time))
    ax[1].legend(title="Time")
    plt.savefig(outdir+"/target_area.png")


    # '''
    # Plot tables with existing cells
    # '''
    # fig, ax = plt.subplots()
    # im_ = ax.imshow(ages_, origin='lower')
    # fig.colorbar(im_)
    # plt.savefig(outdir+".png")

    # fig, ax = plt.subplots()
    # target_ = _target_area(ages_, zposn_)
    # im_ = ax.imshow(target_, origin='lower')
    # fig.colorbar(im_)
    # plt.savefig(outdir+"/ages_all.png")


    '''
    cell-cycle length
    '''
    # 1. find cells that have existing at any of the saved steps and dying at some point
    non_empty = np.where( np.any( ~np.isnan(ages_), axis=1 ) & np.isnan(ages_[:,-1]) )[0]
    # 2. for each of these cells, find the last point before disappearing
    last_not_NaN = lambda x: np.where( ~np.isnan(x) )[0][-1]
    times_death = np.array([last_not_NaN(age) for age in ages_[non_empty]])
    ages_death = ages_[non_empty, times_death]
    fig, ax = plt.subplots()
    ax.set_xlabel("Age at death")
    ax.set_ylabel("Density")
    ax.hist(ages_death, density=True, bins=30)
    plt.savefig(outdir+"/CC_hist.png")


    '''
    A-B nuclear dynamics (cells aligned at birth)
    '''
    # 1. find cells that are born within the observed time-frame
    non_empty = np.where( np.any( ~np.isnan(ages_), axis=1 ) & np.isnan(ages_[:,0]) )[0]
    # 2. for each of these, select only the part which is defined, and create a list of arrays
    zposn_aligned = [ zs[~np.isnan(zs)] for zs in zposn_[non_empty] ]
    fig, ax = plt.subplots()
    for i in np.random.randint(0,len(non_empty),size=10):
        ax.plot(zposn_aligned[i])
    plt.savefig(outdir+"/AB_sample_dyn.png")


    '''
    A-B nuclear dynamics for one given cell and its neighbours
    '''
    # 1. find cells that are born within the observed time-frame
    non_empty = np.where( np.any( ~np.isnan(ages_), axis=1 ) & np.isnan(ages_[:,0]) )[0]
    # 2. for each of these, select only the part which is defined, and create a list of arrays
    zposn_aligned = [ zs[~np.isnan(zs)] for zs in zposn_[non_empty] ]
    times_birth = [ np.where(~np.isnan(zs))[0][0] for zs in zposn_[non_empty] ]
    for n,i in enumerate(np.random.randint(0,len(non_empty),size=10)):
        z_cell = zposn_aligned[i]
        t_start = times_birth[i]
        fig, ax = plt.subplots()
        ax.set_xlim([t_start,t_start+len(z_cell)])
        ax.set_ylim([0,1])
        ax.plot(z_cell, lw=2)
        # find cells neighbouring the one selected, at any time
        neighbours = np.array([]).astype(int)
        for t in len(self.times):
            neighbours = np.hstack((neighbours,np.where(adjs_[t][1] == i)[0]))
        neighbours = np.unique(neighbours)
        for j in neighbours:
            ax.plot(zposn_[j], lw=1) # !!!!
        plt.savefig(outdir+f"/AB_sample_neigh_{n}.png")



    # data = [ages_, zposn_, areas_, neigs_]
    # fig, ax = plt.subplots(1,len(data))
    # im = []
    # for data_, ax_ in zip(data, ax.ravel()):
    #     im_ = ax_.imshow(data_, origin='lower')
    #     im.append(im_)
    #     fig.colorbar(im_, ax=ax_)
    # plt.savefig(outdir+"/various_all.png")


if __name__ == "__main__":

    plot_stats()