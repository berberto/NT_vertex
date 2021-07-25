# Source code info

### IKNM dynamics

*Different options are implemented, but not accessible via parameters/options from command line or else. The source code should be edited to test them.*

The dynamics of the nucleus is defined in `update_zposn_and_A0` in `cells_extra.py`.

- `_active_force_z`: cell-autonomous part of IKNM, following a target position. This has two versions:

	1. the target position is given by the law described in Guerrero et al. as the actual position of the nucleus over time -- it recovers the dynamics of the original model in the limit of high stiffness.
	2. The target position is 0 during G1 and S phases (though it's a "soft" target), and 1 for G2 and M phases ("harder" target).
- `crowding_force`: pairwise interaction part of IKNM, due to crowding.

The target area is defined in `target_area` (a function of a `Cells` object). It calls `_target_area` (function of the A-B nuclear position and time) in turn. This also has two versions:

1. The original model, linear in time and quadratic in A-B position.
2. Logistic in time (saturating to 2 at the apical side in M phase), and same quadratic dependence in A-B position.
