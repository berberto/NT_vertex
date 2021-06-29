# Tissue mechanics and morphogen transport in neural tube patterning

#### Content

Main file: `simulations/neuraltube.py`

Main sources, in `nt_vertex`:
- `NT_sim`: definition of the `NT_simulation` class, which contains methods to set up output files, input parameters, run simulation, save and load checkpoints;
- `NT_vtx.py`: full neural tube object;
- `FE_vtx.py`: solving diffusion-degradation equation for morphogen with finite element method on a growing vertex model;
- `GeneRegulatoryNetwork.py`: signalling and grn dynamics for individual cells.
- `mesh.py`: `Mesh` object (and other auxiliary objects), containing the information about the vertex model geometry.
- `cells.py`: `Cells` object, containing a `Mesh` as attribute, and other properties about cells.


#### Setup

You will need to have installed 
- `python>=3.7.9`, with
	- `numpy`
	- `pandas`
	- `pip`
	- `scipy`
	- `seaborn`
	- `numba`
	- `matplotlib`
	- `dill`
- `ffmpeg` (as backend for videos)

A conda environment satisfying these requirements can be created from the environment file provided:
```bash
conda env create -f environment.yml
```

Activate the environment and install the package in it:
```bash
conda activate vertex
make
```


#### Usage

An example file which runs a simulation and produces a video of it, is
`simulations/neuraltube.py`

You can copy `neuraltube.py` where you like.

After changing to the directory that contains the simulation script (here is `simulations`), you can check all the parameters that can be set from command line:
```bash
cd simulations
python neuraltube.py --help
```

For instance, a basic simulation setting a tag for the output directory, total simulation time (separately for initialization and full simulation), time-step and time between frames:
```
python neuraltube.py \
	[--prefix <tag for output directory>]
	[-t  <total simulation time>]
	[--init  <initialization time>]
	[--dt  <time step>]
	[--every  <time b/w frames>]
```

The same options can be passed to the `NT_simulation` constructor as keyword arguments. In this case, the command-line options are overwritten.
The equivalent is obtained with
```python
sim = NT_simulation(
		prefix="tag_for_output_directory">,
		T_sim=<total simulation time>,
		T_init=<initialization time>,
		dt=<time step>,
		frame_every=<time b/w frames>
	)
```
in `neuraltube.py`.