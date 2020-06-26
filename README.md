# Tissue mechanics and morphogen transport in neural tube patterning

1. `NT_vtx.py`: full neural tube object;
2. `FE_vtx.py`: solving diffusion-degradation equation for morphogen with finite element method on a growing vertex model;
3. `GeneRegulatoryNetwork.py`: signalling and grn dynamics for individual cells.
4. `fe_cy_v3.pyx`: cython implementation of finite element methods, setup via `setup_fev3.py`


### Comments

1. `FE_vtx.py`:
	- line 60-61: determinant might be expensive. Can the old one be saved as member? How big is this matrix?
	- `transitions_faster` is the Cython version for the transitions? Does it work? Commented out for the moment.

2. `NT_vtx.py`:
	- `transitions_faster` (same as above)

3. `initialisationEd.py`:
	- getting the error below
	```
	vertices = vertex_positions[edge_data.vertex[order]].T.copy() #start 
	IndexError: index 4643721389214269440 is out of bounds for axis 0 with size 630
	```
	- is the function `_modified_build_mesh` useful at all?
	- commented out anything that was `_modified*`
	