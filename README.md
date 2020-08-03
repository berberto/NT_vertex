# Tissue mechanics and morphogen transport in neural tube patterning

1. `NT_vtx.py`: full neural tube object;
2. `FE_vtx.py`: solving diffusion-degradation equation for morphogen with finite element method on a growing vertex model;
3. `GeneRegulatoryNetwork.py`: signalling and grn dynamics for individual cells.
4. `fe_cy{,_omp}.pyx`: cython implementation of finite element methods {, with openMP} 
5. Unused routines inside `evolution_routines.py`


### Comments/Questions

1. `FE_vtx.py`:
	- FE matrix filled in coordinate sparse format (can it be that node_id_tri[i (and j)] are the same for a different edge? Tested and it seems correct. No gain in speed anyway.
	- FE matrix is not symmetric (max difference ~ 1.e-2)
	- cython code with the sparse solver doesn't seem to work (tested by Graeme)
	- `dt` was nowhere in the functions to build the finite element matrix and vector.

2. `fe_cy_omp.pyx`:
	- got it running, by adding `with gil` for inner loops over triangle vertices, but it's slower

3. Added argument `evolve_vertex` to `evolve` methods (for `NT_vtx` and `FE_vtx`), to enable/disable evolution of tissue

4. `NT_vtx`:
	- added check on whether to simulate/plot results.
	- all intermediate files are first checked, and then used for plotting (all of those present in the directory are used)

5. What is the **ALE** method?


### To do

1. Test when **expansion but not motion** is set.

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

#### Finite element edits (to do)

*Mesh refinement.*

If area of the triangle is too big,
- divide it into 4 triangles, by splitting each side into 2: this, however, requires splitting also the neighbouring triangles.
- divide it into 3 triangles by inserting at its centroid (baricentre).

Alternatively, if the length of an edge is too long, split it into 2: the triangle, and the adjacent one along that edge, will be divided into two triangles each
(SEE NOTES)

 

Finite element matrix-vector assembly. Instead of looping over edges, first construct an array of triangles and loop over triangles.
1. Define set of triangles for the mesh as the triplet of edges (vectors -- alternatively, with the coordinates of the 3 vertices, being careful to identify equivalent vertices for periodic b.c.)
2. 

### Issues

1. Finite element solution breaks down if:
	- tissue is not moving
	- tissue is moving but *NOT* expanding
	- **how about expansion but not motion?**

1. `NT_vtx.py`:
	- `transitions_faster()` is actually slower ?

2. `fe_cy_omp.pyx`:
	- now getting to compile with `-fopenmp` flag, but giving `Fatal Python error: PyThreadState_Get: no current thread`. Loop over edges not vectorizable?

3. `setup_*`:
	- still getting the `"Using deprecated NumPy API, disable it by #defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"` warning... 
	- bunch of warnings with uninitialized variables inside `#pragma` directives. Looked at the `fe_cy_omp.c` output, and seems to be normal: they are auxiliary variables used for the vectorized loop.

9. Full evolution but without expansion
	- concentration still diverges (though tissue doesn't grow with if `expansion` is zero)
	- diverges faster for the cython code (?)