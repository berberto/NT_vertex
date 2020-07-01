# Tissue mechanics and morphogen transport in neural tube patterning

1. `NT_vtx.py`: full neural tube object;
2. `FE_vtx.py`: solving diffusion-degradation equation for morphogen with finite element method on a growing vertex model;
3. `GeneRegulatoryNetwork.py`: signalling and grn dynamics for individual cells.
4. `fe_cy{,_omp}.pyx`: cython implementation of finite element methods {, with openMP} 
5. Unused routines inside `evolution_routines.py`


### Comments

1. `FE_vtx.py`:
	- FE matrix filled in coordinate sparse format (can it be that node_id_tri[i (and j)] are the same for a different edge? Tested and it seems correct. No gain in speed anyway.
	- FE matrix is not symmetric (max difference ~ 1.e-2)
	- cython code with the sparse solver doesn't seem to work (tested by Graeme)

2. `fe_cy_v3.py` cleaned up and renamed as `fe_cy.pyx`
	- evolution function now called `ev_cy`

2. `fe_cy_omp.pyx`:
	- got it running, by adding `with gil` for inner loops over triangle vertices, but it's slower

4. `FE_transitions.py` corrected by Graeme, replaced the old one

5. `setup.py`:
	- put all the cython compilation scripts inside here
	- all shared objects compiled with `python setup.py build_ext --inplace`
	- deleted all the `setup_*.py` files and the bash script `setup.sh`

3. Ran with 200'000, saving snapshot every 10, in about 3 hrs

### To do

1. `NT_vtx.py`:
	- use `dill` for dumping session to disc (restart file)

1. `FE_vtx.py`:
	- maybe make the `build_FE_vtx`/`build_FE_vtx_from_scratch` routines members of the `FE_vtx` class?

2. `fe_cy_omp.pyx`:
	- rewriting the part to construct the matrix with a loop over nodes, rather than edges?
		- save the triplets of `node_id_tri` before entering the loop (can be vectorized)
		- iterates over nodes (can it be vectorized?)
	- check that I'm using the right cython function

3. `plotting`:
	- move functions to produce videos there?
	- `try...except` for `ffmpeg` first

4. Related to slurm issue, figure out which flags are needed in compilation

5. `setup_*`:
	- where to put the `-xhost` flag?

6. **Everywhere** (all the necessary classes):
	- add routines to save in binary intermediate configs
	- add constructors that take info from files


### Issues

1. `NT_vtx.py`:
	- `transitions_faster()` is actually slower ?

2. `fe_cy_omp.pyx`:
	- now getting to compile with `-fopenmp` flag, but giving `Fatal Python error: PyThreadState_Get: no current thread`. Loop over edges not vectorizable?

3. `setup_*`:
	- still getting the `"Using deprecated NumPy API, disable it by #defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"` error... 
	- bunch of warnings with uninitialized variables inside `#pragma` directives
