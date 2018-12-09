# CHSA
Convex Hull Stratification Algorithm Code

This repository contains MATLAB code related to the following article:

L. Ziegelmeier, M. Kirby, C. Peterson, "Stratifying High-Dimensional Data Based on Proximity to the Convex Hull Boundary," SIAM Review, Vol. 59, No. 2, pp. 346-365, 2017.

The file ConvexHull.m is the main piece of code which computes a weight vector for each point in a data set based on different preferences for a parameter that enforces convexity lambda and a parameter that enforces uniformity gamma

The file PDIPAQuad.m is called in the ConvexHull.m code. This solves the formulated quadratic program via the primal dual inter point algorithm. Note that this solves the full system of equations, however, reduced systems can speed up computations (although might induce instability).

The file ConvHullPolygoninPlane.m is an example use file of random toy data created in the plane, then colored according whether or not a point has an associated weight vector with a negative entry and then another figure which colors points according to magnitude of 2-norm.
