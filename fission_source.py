#! /usr/bin/env python

from matplotlib.colors import LogNorm
import pylab as pl
import numpy as np

data=np.array(open("/Users/rmb/code/gpu-cpp/fissile_points").read().split(),dtype=float)
data=np.reshape(data,(-1,3))

fig = pl.figure(figsize=(10,6))
ax = fig.add_subplot(1,1,1)
xmin = ymin = -2
xmax = ymax =  2
ax.hist2d(data[:,0], data[:,1], range=[[xmin, xmax], [ymin, ymax]], bins=256 , normed=True)#norm=LogNorm())
fig.colorbar(ax.get_images()[0], ax=ax, ticks=[ 0.0, 0.5, 1.0], cmap=pl.cm.jet, label='Relative Probability')  #, norm=pl.matplotlib.colors.Normalize(vmin=5, vmax=10))
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
ax.grid('on',color='k')
pl.show()