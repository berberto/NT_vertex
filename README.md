# Tissue mechanics and morphogen transport in neural tube patterning

1. `NT_vtx.py`: full neural tube object;
2. `FE_vtx.py`: solving diffusion-degradation equation for morphogen with finite element method on a growing vertex model;
3. `GeneRegulatoryNetwork.py`: signalling and grn dynamics for individual cells.
4. `fe_cy_v3.pyx`: cython implementation of finite element methods, setup via `setup_fev3.py`
5. Unused routines inside `evolution_routines.py`


### Comments

1. `FE_vtx.py`:
	- FE matrix filled in coordinate sparse format (can it be that node_id_tri[i (and j)] are the same for a different edge? Tested and it seems correct. No gain in speed anyway.
	- FE matrix is not symmetric (max difference ~ 1.e-2)
	- cython code with the sparse solver doesn't seem to work (tested by Graeme)

### Issues

1. `NT_vtx.py`:
	- `transitions_faster()` is actually slower

2. `FE_vtx.py`:
	- `evolve-cy` gives `nan` for concentration (compilation issue?)

3. `fe_cy_omp.pyx`:
	- cannot use openMP yet, because of `Calling gil-requiring function not allowed without gil` (working on it)