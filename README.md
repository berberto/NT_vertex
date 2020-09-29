# Tissue mechanics and morphogen transport in neural tube patterning

1. `NT_vtx.py`: full neural tube object;
2. `FE_vtx.py`: solving diffusion-degradation equation for morphogen with finite element method on a growing vertex model;
3. `GeneRegulatoryNetwork.py`: signalling and grn dynamics for individual cells.
4. `fe_cy{,_omp}.pyx`: cython implementation of finite element methods {, with openMP} 
5. Unused routines inside `evolution_routines.py`

#### Usage

```bash
python neuraltube.py [-t <tot time> | --every <time b/w frames> --init <initialization time --dt <time step>]
```

To show all possible options:
```bash
python neuraltube.py -h
```


### Comments/Questions

1. **Expansion** was negative only in the initialization period, then it becomes positive, so the tissue grows (check out its formula from the paper).

2. Added comments to 'FE_transitions' and to 'FE_vtx.transition': **where is the interpolation to find concentrations at the new centroids upon cell division done? Is it in the `_add_edges` function?**

3. In 'NT_vtx', the function 'centroids2' was imported from 'cells_extra', where it was not defined!!! It was instead in 'Finite_Element'. **Please, let's clean up the code, removing duplicates of functions which are not used**



### To do

1. From `_T1` one should get information about the new positions of the extremes and the indices of the half-edges that are rotated. At the moment, it only returns a numpy array with the edges pointing to (and coming from) the neighbouring vertices, in order to avoid performing T1 transitions on those.

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


### Issues/Troubleshooting

#### Problem with `_remove` and `_rem_collapsed`:

Running with
```
python neuraltube.py --init 10. -t 50. --every 0.1 --dt 0.005
```

`_remove` requires as positional argument the concentration-by-edge vector, which wasn't there. When added (at the same point where it was complaining about missing argument), it gave this.
```
Traceback (most recent call last):77
  File "neuraltube.py", line 134, in <module>
    neural_tube.transitions(division=division)
  File "/camp/lab/briscoej/working/alberto/vertex/NT_vtx.py", line 78, in transitions
    self.FE_vtx.cells,c_by_e = rem_collapsed(self.FE_vtx.cells,c_by_e)
  File "/camp/lab/briscoej/working/alberto/vertex/FE_transitions.py", line 203, in rem_collapsed
    while np.any(reverse[reverse[rotate[two_sided]]] != reverse[rotate[nxt[two_sided]]]):
IndexError: index 1958 is out of bounds for axis 1 with size 1956
```
There was a floating-point division at line 26, corrected to this:
```
    es = np.unique(edges//3*3)
```
The `//3*3` operation, presumably, is a "modulus 3" operator.

Still, error persists.


#### Repeated T1 transitions on same edge

With these settings,
```
python neuraltube.py --prefix debugRemAll --init 10. -t .3 --every 0.0005 --dt 0.0005
```
towards 0.3, one starts to see singularites developing where a T1 transition occurs frequently on the same edge.
In a T1, the line length goes from a value `l ~ eps` (tiny bit less `eps`), to a value `l' ~ (1 + 0.01)eps`. If the displacement, under the new forces, is larger than `0.01 eps`, then another T1 occurs.
Meaning that, perhaps `eps` is too large, or `dt` is too large.

One necessary thing is to **interpolate** when performing the T1.

Possible ways to cure this:
- time step `dt` smaller, in such a way that
- threshold on line length `eps` for T1 transition is too big, might have to be reduced compatibly with `dt`


#### Changed `mesh._T1` code

Commented 
New code for changing the coordinates of the edge's extremes
```python
for i in [0, 1]:
    dp = 0.5*(dv[i]+dw[i])
    dq = 0.5*(dv[i]-dw[i])
    vertices[i,e0] += dq
    vertices[i,e3] -= dq
    vertices[i,before] = vertices[i,after]  + np.array([dq, -dp, -dq, dp])
```

Old code
```python
for i in [0, 1]:
    dp = 0.5*(dv[i]+dw[i])
    dq = 0.5*(dv[i]-dw[i])
    v = vertices[i]
    v[before] = v.take(after) + np.array([dp, -dq, -dp, dq])
    v[e0] = v[e4] + dw[i]
    v[e3] = v[e1] - dw[i]
```

The old code was changing the indices in the correct way (correct topological change), but it was swapping the coordinates of the two extremes of the short edge. Now it seems fine. **How was this not causing problems? Why were cells printed fine at the end?**