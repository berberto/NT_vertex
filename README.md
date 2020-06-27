# Tissue mechanics and morphogen transport in neural tube patterning

1. `NT_vtx.py`: full neural tube object;
2. `FE_vtx.py`: solving diffusion-degradation equation for morphogen with finite element method on a growing vertex model;
3. `GeneRegulatoryNetwork.py`: signalling and grn dynamics for individual cells.
4. `fe_cy_v3.pyx`: cython implementation of finite element methods, setup via `setup_fev3.py`
5. Unused routines inside `evolution_routines.py`


### Comments

1. `FE_vtx.py`:
	- in `evolution` need to fill the FE matrix in sparse type from the beginning, to avoid conversion