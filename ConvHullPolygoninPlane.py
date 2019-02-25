'''
A script which matches the test case set up in
ConvHullPolygoninPlane.m... for the most part.
'''
import numpy as np
from matplotlib import pyplot
from convex_hull import convex_hull

n = 50

np.random.seed(2718281828)
X = np.random.rand(n,2)

# Run the algorithm
Y,Wtil,Indices = convex_hull(X, 20, 10**-3, 10**-6)

bdry_pts = np.setdiff1d(np.arange(n), Indices)

# Visualize
fig,ax = pyplot.subplots(1,2, sharex=True, sharey=True, figsize=(10,5))
ax[0].scatter(X[:,0], X[:,1], s=40)
ax[1].scatter(X[:,0], X[:,1], s=40)

# Show points in interior vs boundary.
ax[0].scatter(X[bdry_pts,0], X[bdry_pts,1], c='r', s=20)
ax[0].scatter(X[bdry_pts,0], X[bdry_pts,1], c='r', s=20)

# Color by the norm of the Wtilde vectors.
NormVec = np.linalg.norm(Wtil, axis=1)
cax = ax[1].scatter(X[:,0], X[:,1], c=NormVec, cmap=pyplot.cm.plasma, edgecolor='k', s=60, vmin=0.)

fig.colorbar(cax)

fig.show()
