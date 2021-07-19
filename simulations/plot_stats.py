import numpy as np
from matplotlib import use
import matplotlib.pyplot as plt
import sys
use('tkagg')

path = sys.argv[1]

# try:
state_ = np.load(path+"/state.npy")
ages_  = np.load(path+"/ages.npy")
zposn_ = np.load(path+"/zposn.npy")
areas_ = np.load(path+"/areas.npy")
neigs_ = np.load(path+"/neigs.npy")
# except:
# 	print("Something wrong with the path")
# 	exit()



fig, ax = plt.subplots()
ax.imshow(ages_, origin='lower')
plt.show()