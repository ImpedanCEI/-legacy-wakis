'''
Auxiliary functions for WarpX geometry definition:
- plot_implicit
- sphere
- cylinder
- cube
- prism
- translate
- union 
- intersect
- subtract
'''

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure #Requires: pip install scikit-image

def triang_implicit(fn, bbox=(-3,3)):
    ''' triangulates an implicit function
    fn: implicit function (plot where fn==0)
    bbox: the x,y,and z limits of plotting box
    ''' 
    bmin, bmax = bbox

    # Define the mesh 
    xl = np.linspace(bmin, bmax, 32)
    X, Y, Z = np.meshgrid(xl, xl, xl)
    F = fn(X, Y, Z)

    # Obtain vertex and faces with the marching cubes algorithm
    verts, faces, normals, values = measure.marching_cubes(F, 0, spacing=[np.diff(xl)[0]]*3 )
    verts -= bmax

    # Plot using trisurf
    fig = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color='b')

    # Set plot properties
    ax.set_zlim3d(bmin,bmax)
    ax.set_xlim3d(bmin,bmax)
    ax.set_ylim3d(bmin,bmax)
    ax.grid(True, color='gray', linewidth=0.2)
    ax.set(title='Geometry defined by the implicit function', 
            xlabel='z [m]',
            ylabel='x [m]',
            zlabel='y [m]'
            )
    plt.show()

def plot_implicit(fn, lims=(-2.5,2.5)*3, bbox=(-2.5,2.5)):
    ''' create a plot of an implicit function
    fn: implicit function (plot where fn==0)
    lims: limits of the geometry
    bbox: the x,y,and z limits of plotting box
    '''
    xmin, xmax, ymin, ymax, zmin, zmax = lims
    bmin, bmax = bbox

    fig = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    res = 64 # resolution of the contour
    slices = 20 # number of slices

    for x in np.linspace(xmin, xmax, slices): # plot contours in the YZ plane
        Y,Z = np.meshgrid(np.linspace(xmin, xmax, res),np.linspace(xmin, xmax, res))
        X = fn(x,Y,Z)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x', alpha=0.8, cmap='plasma' )

    for y in np.linspace(ymin, ymax, slices): # plot contours in the XZ plane
        X,Z = np.meshgrid(np.linspace(ymin, ymax, res),np.linspace(ymin, ymax, res))
        Y = fn(X,y,Z)
        cset = ax.contour(X, Y+y, Z, [y], zdir='y', alpha=0.8, cmap='plasma' )

    for z in np.linspace(zmin, zmax, slices): # plot contours in the XY plane
        X,Y = np.meshgrid(np.linspace(zmin, zmax, res),np.linspace(zmin, zmax, res))
        Z = fn(X,Y,z)
        cset = ax.contour(X, Y, Z+z, [z], zdir='z', alpha=0.8, cmap='plasma' )
        # [z] defines the only level to plot for this contour for this value of z    

    # must set plot limits because the contour will likely extend
    # way beyond the displayed level.  Otherwise matplotlib extends the plot limits
    # to encompass all values in the contour.
    ax.set_zlim3d(bmin,bmax)
    ax.set_xlim3d(bmin,bmax)
    ax.set_ylim3d(bmin,bmax)
    ax.grid(True, color='gray', linewidth=0.2)
    ax.set(title='Geometry defined by the implicit function', 
            xlabel='x [m]',
            ylabel='z [m]',
            zlabel='y [m]'
            )
    plt.show()

def sphere(x,y,z, r=2.0):
    return x**2 + y**2 + z**2 - r**2

def cylinder(x,y,z, r=2.0):
    return x*x + y*y - r*r

def cube(x,y,z, r=2.0):
    return np.maximum(np.abs(x)+np.abs(y)+np.abs(z))-r

def prism(x,y,z, w=1.0, h=1.0, L=3.0):
    return np.maximum(np.maximum(np.maximum(x-w/2,-w/2-x), np.maximum(y-h/2,-h/2-y)), np.maximum(z-L/2,-L/2-z))

def translate(fn,x,y,z):
    return lambda a,b,c: fn(x-a,y-b,z-c)

def union(*fns):
    return lambda x,y,z: np.min(
        [fn(x,y,z) for fn in fns], 0)

def intersect(*fns):
    return lambda x,y,z: np.max(
        [fn(x,y,z) for fn in fns], 0)

def subtract(fn1, fn2):
    return intersect(fn1, lambda *args:-fn2(*args))

def goursat_tangle(x,y,z):
    a,b,c = 0.0,-5.0,11.8
    return x**4+y**4+z**4+a*(x**2+y**2+z**2)**2+b*(x**2+y**2+z**2)+c
