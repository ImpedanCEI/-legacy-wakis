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
- execfile: executes a python script keeping the current local and global variables

Requirements:
------------- 
scikit-image, matplotlib, numpy

'''
import sys
import matplotlib.pyplot as plt
import numpy as np

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
    try:
        from mpl_toolkits.mplot3d import axes3d
        from skimage import measure 
    except ImportError:
        raise ImportError('To use this function ypu need to install "scikit-image" package v>= 0.17 ')


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

def execfile(filename, globals=None, locals=None):
    if globals is None:
        globals = sys._getframe(1).f_globals
    if locals is None:
        locals = sys._getframe(1).f_locals
    with open(filename, "r") as fh:
        exec(fh.read()+"\n", globals, locals)
   
