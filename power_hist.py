#! /usr/bin/env python

from matplotlib.colors import LogNorm
import pylab as pl
import numpy as np

data=np.array(open("/Users/rmb/code/gpu-cpp/fissile_points").read().split(),dtype=float)
data=np.reshape(data,(-1,3))
print data

pl.hist2d(data[:,0], data[:,1], bins=40 )#norm=LogNorm())
pl.colorbar()
pl.show()