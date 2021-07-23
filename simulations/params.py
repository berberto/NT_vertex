
import numpy as np

# nuclear dynamics parameters
k = np.array([5., 10., 20., 50.]) # "stiffness"
r = np.array([0., .3, 1., 3.]) # strength of crowding
e = np.array([0., 0.01]) # noise values
# morphogen parameters
f = np.array([.1])	# strength of source
d = np.array([.05]) # diffusion constant

kk, rr, ee, ff, dd = np.meshgrid(k, r, e, f, d)
k = kk.flatten()
r = rr.flatten() * k
e = ee.flatten()
f = ff.flatten()
d = dd.flatten()
pars = np.vstack([k,r,e,f,d]).T

np.savetxt("params.txt", pars, fmt=["%05.1f","%05.1f","%05.3f","%4.2f","%4.2f"], delimiter="  ")
