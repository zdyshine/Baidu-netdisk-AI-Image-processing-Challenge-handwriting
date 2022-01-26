#!/usr/bin/env python
"""Module providing functionality surrounding gaussian function.
"""
SVN_REVISION = '$LastChangedRevision: 16541 $'

import sys
import numpy

def gaussian2(size, sigma):
    """Returns a normalized circularly symmetric 2D gauss kernel array
    
    f(x,y) = A.e^{-(x^2/2*sigma^2 + y^2/2*sigma^2)} where
    
    A = 1/(2*pi*sigma^2)
    
    as define by Wolfram Mathworld 
    http://mathworld.wolfram.com/GaussianFunction.html
    """
    A = 1/(2.0*numpy.pi*sigma**2)
    x, y = numpy.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = A*numpy.exp(-((x**2/(2.0*sigma**2))+(y**2/(2.0*sigma**2))))
    return g

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = numpy.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = numpy.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def main():
    """Show simple use cases for functionality provided by this module."""
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    import pylab
    argv = sys.argv
    if len(argv) != 3:
        print >>sys.stderr, 'usage: python -m pim.sp.gauss size sigma'
        sys.exit(2)
    size = int(argv[1])
    sigma = float(argv[2])
    x, y = numpy.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    fig = pylab.figure()
    fig.suptitle('Some 2-D Gauss Functions')
    ax = fig.add_subplot(2, 1, 1, projection='3d')
    ax.plot_surface(x, y, fspecial_gauss(size, sigma), rstride=1, cstride=1, 
                    linewidth=0, antialiased=False, cmap=pylab.jet())
    ax = fig.add_subplot(2, 1, 2, projection='3d')
    ax.plot_surface(x, y, gaussian2(size, sigma), rstride=1, cstride=1, 
                    linewidth=0, antialiased=False, cmap=pylab.jet())
    pylab.show()
    return 0

if __name__ == '__main__':
    sys.exit(main())