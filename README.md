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
