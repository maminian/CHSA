# About

This is a Python 3 implementation of the original Convex Hull Stratification Algorithm 
code, which is written in Matlab. Most of the code has been transliterated, with 
a few exceptions (some of which are critical to note):

1. Optional arguments included in some functions to turn off print statements and convergence 
information (by default);
2. Other optional arguments are included to expose internal parameters as part of the 
function call if desired, rather than needing to modify the python file;
3. Arrays, and the corresponding code underneath, has been switched to respect 
*row-major* format, which is the standard in Python. 

This is a fork of 

https://github.com/lziegelmeier/CHSA

which itself is based on the article:

L. Ziegelmeier, M. Kirby, C. Peterson, "Stratifying High-Dimensional Data Based on Proximity to the Convex Hull Boundary," SIAM Review, Vol. 59, No. 2, pp. 346-365, 2017.

