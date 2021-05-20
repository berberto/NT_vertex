# Tissue mechanics and morphogen transport in neural tube patterning

#### Content

Main file: `neuraltube.py`

Main sources:
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

Activate the environment and you are ready to run it:
```bash
conda activate vertex
```

#### Usage

Basic simulation setting a tag for the output directory, total simulation time (separately for initialization and full simulation), time-step and time between frames:
```
python neuraltube.py [--prefix <tag for output directory>]
					 [-t  <total simulation time>]
					 [--every  <time b/w frames>]
					 [--init  <initialization time>]
					 [--dt  <time step>]
```

To show all possible options:
```bash
python neuraltube.py -h
```
