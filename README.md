# Tissue mechanics and morphogen transport in neural tube patterning

1. `NT_vtx.py`: full neural tube object;
2. `FE_vtx.py`: solving diffusion-degradation equation for morphogen with finite element method on a growing vertex model;
3. `GeneRegulatoryNetwork.py`: signalling and grn dynamics for individual cells.
4. `fe_cy{,_omp}.pyx`: cython implementation of finite element methods {, with openMP} 
5. Unused routines inside `evolution_routines.py`


### Comments

1. `NT_vtx`:
	- `deg_rate`: shouldn't it be multiplied by `dt`?

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

4. Corrected `expansion`: including anisotropy (or not) -> arbitrary growth factor

5. 


### To do

1. `FE_vtx.py`:
	- maybe make the `build_FE_vtx`/`build_FE_vtx_from_scratch` routines members of the `FE_vtx` class?

2. `fe_cy_omp.pyx`:
	- rewriting the part to construct the matrix with a loop over nodes, rather than edges?
		- save the triplets of `node_id_tri` before entering the loop (can be vectorized)
		- iterates over nodes (can it be vectorized?)
	- check that I'm using the right cython function
	- maybe SIMD enable some of the functions? how does that work with the `gil` stuff?

3. `plotting`:
	- move functions to produce videos there?
	- `try...except` for `ffmpeg` first

4. Related to slurm issue, figure out which flags are needed in compilation

6. **Everywhere** (all the necessary classes):
	- add routines to save in binary intermediate configs
	- add constructors that take info from files

### NEXT STUFF

1. `forces.py`:
	- check forces (comment for MAC?) `by_face`, check mesh objects? `cells_extra` update property methods

2. setup `expansion`.. why does it become negative? (see `run_select`)

3. check `add_IKNM_properties`, random age initialization

4. run "thermalization" phase of vm before initializing FE object (`build_FE_vtx`)

5. `source_data` in `cells_setup`, includes width of shh producing strip

6. check how to delete property from cells (like 'age', so cells divide only based on area and not on time)

7. check new concentration at centroids on division (should not lose/add material)

8. FE for cylinder

9. Full evolution but without expansion


### Issues

1. solution breaks down if expansion is set: running simulations with expansion set to 0 in both directions

1. `NT_vtx.py`:
	- `transitions_faster()` is actually slower ?

2. `fe_cy_omp.pyx`:
	- now getting to compile with `-fopenmp` flag, but giving `Fatal Python error: PyThreadState_Get: no current thread`. Loop over edges not vectorizable?

3. `setup_*`:
	- still getting the `"Using deprecated NumPy API, disable it by #defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"` warning... 
	- bunch of warnings with uninitialized variables inside `#pragma` directives. Looked at the `fe_cy_omp.c` output, and seems to be normal: they are auxiliary variables used for the vectorized loop.
