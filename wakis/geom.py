'''
-------------------------
| WAKIS geometry module |
-------------------------
Auxiliary functions for WarpX geometry modelling.

Functions:
---------
- triang_implicit: triangulates the surface of an implicit function f(x,y,z) = 0
- eval_implicit: evaluates the expression parsed to the WarpX picmi.EmbeddedBoundary class
                 and transforms it to a f(x,y,z) function readable by triang_implicit
- next2power: return the closest 2**n power for a given number N

Requirements:
------------- 
pip install scikit-image, matplotlib, numpy

'''
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import axes3d
from skimage import measure 


def triang_implicit(fun, BC=None, bbox=(-3,3)):
    ''' 
    Triangulates the surface of an implicit function f(x,y,z) = 0
    
    Parameters:
    -----------
    - fun: implicit function (plot where fn==0)
    - BC: class containing the implicit function
    - bbox: the x,y,and z limits of plotting box

    Returns:
    --------
    - F: matrix with shape(32,32,32) that only has values 
         where the implicit function is close to 0 and 
         otherwise in float('nan') 
    ''' 
    # Define box limits
    bmin, bmax = bbox

    # Define the mesh 
    xl = np.linspace(bmin, bmax, 32)
    X, Y, Z = np.meshgrid(xl, xl, xl)

    #Define the field
    if BC == None:
        F = fun(X, Z, Y)

    else:
        F = fun(X, Z, Y, BC)

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

    return F

def eval_implicit(x, y, z, BC): 
    ''' 
    Evaluates the expression parsed to the WarpX picmi.EmbeddedBoundary class
    and transforms it to a f(x,y,z) function readable by triang_implicit.

    Parameters:
    -----------
    - x, y, z: variables to be defined by the plot function 
    - BC: custom class containing the implicit function and 
          parameters in a string + user defined kw arguments
          returned by picmi.EmbeddedBoundary
    '''

    # read the class
    implicit_class=BC.implicit_function.split(';')
    params=BC.user_defined_kw

    args=implicit_class[0:-1]      #assignements
    implicitf=implicit_class[-1]   #implicit function
    vals=np.zeros((len(args), len(x), len(y), len(z)))

    # introduce kw parameters 
    i=0
    for i in range(len(args)):
        for key in params.keys():  
            if key in args[i]:
                args[i]=args[i].replace(key, 'params[\''+key+'\']')
                

    # get the arguments value
    for i in range(len(args)):
        arg=args[i].split('=')
        vals[i, :, :, :]=eval(arg[1])
        implicitf=implicitf.replace(arg[0].strip(), 'vals['+str(i)+', :, :, :]')

    implicitf=implicitf.replace('max', 'np.maximum')
    
    return eval(implicitf) 

def next2power(n):
    ''' 
    returns the power of two (2**)
    closest to the number n given
    '''
    i = 1
    while i < n: 
        i *=2

    return i


def test_new_geometry():

    '''
    Test for creating new geometry for warpX
    '''
    from pywarpx import picmi

    # Define the geometry
    UNIT = 1e-3

    # width of the taper (x-dir)
    a = 50*UNIT

    # intial height of the taper (y-dir)
    b = 24*UNIT
    # final length of the taper (y-dir)
    target=12*UNIT

    # length of the straight part (z-dir)
    L1 = 15*UNIT
    # length of the inclined part (z-dir)
    L2 = 48*UNIT
    # length of the target part (z-dir)
    L3 = 15*UNIT
    # total length (z-dir)
    L = L1 + L2 + L3
    # Define mesh cells per direction. !![has to be a 2^3 power]
    nx = 64 
    ny = 64
    nz = 64

    # Define mesh resolution in x, y, z
    dh = 1.0*UNIT

    #----------------------------------------------------------------------------

    ##################################
    # Define the mesh
    ##################################

    # mesh bounds for domain. Last 10 cells are PML
    xmin = -nx*dh/2
    xmax = nx*dh/2
    ymin = -ny*dh/2
    ymax = ny*dh/2
    zmin = -nz*dh/2 
    zmax = nz*dh/2

    # mesh cell widths
    dx=(xmax-xmin)/nx
    dy=(ymax-ymin)/ny
    dz=(zmax-zmin)/nz

    # mesh arrays (center of the cell)
    x0=np.linspace(xmin, xmax, nx)+dx/2
    y0=np.linspace(ymin, ymax, ny)+dy/2
    z0=np.linspace(zmin, zmax, nz)+dz/2

    x, y, z = np.meshgrid(x0, y0, z0)

    # Define the implicit function for the boundary conditions
    BC = picmi.EmbeddedBoundary(
        implicit_function="w=a; h=b*(z>-Z)*(z<-Z+L1)+c*(z>Z-L3)*(z<Z)+((c-b)/L2*(z-(-Z+L1))+b)*(z>-L2/2)*(z<L2/2); max(max(x-w/2,-w/2-x),max(y-h/2,-h/2-y))",
        a=a, 
        b=b, 
        c=target, 
        Z=L/2.0,
        L1=L1, 
        L2=L2, 
        L3=L3
    )

    triang_implicit(fn=implicit_function, BC=BC, bbox=(-L/2,L/2))


if __name__ == "__main__":

    test_new_geometry()
    
 