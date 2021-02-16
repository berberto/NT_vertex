# Tissue mechanics and morphogen transport in neural tube patterning

Main classes:
1. `NT_vtx.py`: full neural tube object;
2. `FE_vtx.py`: solving diffusion-degradation equation for morphogen with finite element method on a growing vertex model;
3. `GeneRegulatoryNetwork.py`: signalling and grn dynamics for individual cells.


#### Usage

Basic simulation setting a tag for the output directory, total simulation time (separately for initialization and full simulation), time-step and time between frames:
```bash
python neuraltube.py [--prefix <tag>] [-t <tot time>]  [--every <time b/w frames>]  [--init <initialization time>]  [--dt <time step>]
```

To show all possible options:
```bash
python neuraltube.py -h
```

### Comments/Questions

1. Where should the daughters cells be added in GRN?

2. `property_update` for `divide`: when more than 1 is ready, what happens with `np.repeat`?

1. Added comments to 'FE_transitions' and to 'FE_vtx.transition': where is the interpolation to find concentrations at the new centroids upon cell division done? Is it in the `_add_edges` function?

2. The random seed is chosen at the beginning of the FULL (thermalization + FE sim) dynamics; when the `restart` option is used from an intermediate snapshot, the behaviour is unpredictable.



### What is to be done?

1. For cell type-specific differentiation,
    - add `differentiation` option (bool) to function building `NT_vtx` -- done, **to test**;
    - add "coin toss" for death at each time-step `cells_evolve` -- done, **to test**;
    - add `differentiation_rate` method in GRN class (could possibly add an auxiliary variable measuring how long Olig has been up) -- **placeholder only, to implement**

2. From `_T1` one should get information about the new positions of the extremes and the indices of the half-edges that are rotated. At the moment, it only returns a numpy array with the edges pointing to (and coming from) the neighbouring vertices, in order to avoid performing T1 transitions on those.

3. `FE_vtx.py`:
	- maybe make the `build_FE_vtx`/`build_FE_vtx_from_scratch` routines members of the `FE_vtx` class?

4. `fe_cy_omp.pyx`:
	- rewriting the part to construct the matrix with a loop over nodes, rather than edges?
		- save the triplets of `node_id_tri` before entering the loop (can be vectorized)
		- iterates over nodes (can it be vectorized?)
	- check that I'm using the right cython function
	- maybe SIMD enable some of the functions? how does that work with the `gil` stuff?

5. **Everywhere** (all the necessary classes):
	- add constructors that take info from files

6. `forces.py`:
	- check forces (comment for MAC?) `by_face`, check mesh objects? `cells_extra` update property methods

7. check `add_IKNM_properties`, random age initialization

8. `source_data` in `cells_setup`, includes width of shh producing strip (**note to self**: what should I do here?)

9. check how to delete property from cells (like 'age', so cells divide only based on area and not on time -- for testing) *It's a dictionary, so it should be done by, eg `properties.pop('age')`*

10. check new concentration at centroids on division (should not lose/add material)


### Issues/Troubleshooting

...