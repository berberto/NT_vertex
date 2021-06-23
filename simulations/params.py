
import numpy as np

f = np.array([.01, .02, .05, .1])
d = np.array([.001, .002, .005, .01, .02, .05])

ff,dd = np.meshgrid(f,d)
f = ff.flatten()
d = dd.flatten()

with open("params.txt", "w") as file:
	for x,y in zip(f,d):
		file.write("%.3f\t%.3f\n"%(x,y))

exit()
