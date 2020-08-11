# Tissue mechanics and morphogen transport in neural tube patterning

1. `NT_vtx.py`: full neural tube object;
2. `FE_vtx.py`: solving diffusion-degradation equation for morphogen with finite element method on a growing vertex model;
3. `GeneRegulatoryNetwork.py`: signalling and grn dynamics for individual cells.
4. `fe_cy{,_omp}.pyx`: cython implementation of finite element methods {, with openMP} 
5. Unused routines inside `evolution_routines.py`


### Comments/Questions

1. Added comments to 'FE_transitions' and to 'FE_vtx.transition': **where is the interpolation to find concentrations at the new centroids upon cell division done? Is it in the `_add_edges` function?**

2. In 'NT_vtx', the function 'centroids2' was imported from 'cells_extra', where it was not defined!!! It was instead in 'Finite_Element'. **Please, let's clean up the code, removing duplicates of functions which are not used**



### To do

1. **Check the velocities at the nodes which are involved in T1 and T2 transitions. Might be that they explode there.**

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

6. **Everywhere** (all the necessary classes):
	- add constructors that take info from files

1. `forces.py`:
	- check forces (comment for MAC?) `by_face`, check mesh objects? `cells_extra` update property methods

2. setup `expansion`.. why does it become negative? (see `run_select`)

3. check `add_IKNM_properties`, random age initialization

4. run "thermalization" phase of vm before initializing FE object (`build_FE_vtx`)

5. `source_data` in `cells_setup`, includes width of shh producing strip

6. check how to delete property from cells (like 'age', so cells divide only based on area and not on time -- for testing)

7. check new concentration at centroids on division (should not lose/add material)

8. FE for cylinder


### Issues

- Finite element solution breaks down if the full vertex model dynamics, it is fine when growth (`expansion`) is set while excluding topological transitions. Cell division also taken care of not too bad.