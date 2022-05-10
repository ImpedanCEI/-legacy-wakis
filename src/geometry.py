'''
-------------------------
| WAKIS geometry module |
-------------------------

Funtions:
---------
- triang_implicit: triangulates the surface of an implicit function f(x,y,z) = 0
- eval_implicit: evaluates the expression parsed to the WarpX picmi.EmbeddedBoundary class
                 and transforms it to a f(x,y,z) function readable by triang_implicit
- next2power: return the closest 2**n power for a given number N

Classes:
--------
- cube: defines the parameters of a cube (L)
    -- method fun(x,y,z) returns its implicit function
- sphere: defines the parameters of a sphere (r)
    -- method fun(x,y,z) returns its implicit function
- prism: defines the parameters of a prism. (w,h,L)
    -- method fun(x,y,z) returns its implicit function
- cylinder: defines the parameters of a cylinder. (r, h)
    -- method fun(x,y,z) returns its implicit function
- cone: defines the parameters of a cone. (r, h)
    -- method fun(x,y,z) returns its implicit function
- cone_frustum: defines the parameters of a truncated cone. (r1,r2, h)
    -- method fun(x,y,z) returns its implicit function
- operation: contains the methods
    -- substract(f1, f2)
    -- union(*fns)
    -- intersect(*fns)
    -- translate(fn,x,y,z)

Requirements:
------------- 
pip install scikit-image
pip install numpy-stl
pip install git+https://github.com/cpederkoff/stl-to-voxel.git

'''
import matplotlib.pyplot as plt
import numpy as np
import sympy as sm

from stl import mesh
from stltovoxel import convert_mesh
from mpl_toolkits.mplot3d import axes3d
from skimage import measure 

def stl_to_xyz(input_file_path, resolution=32, parallel=False):
    #Obtain mesh
    mesh_obj = mesh.Mesh.from_file(input_file_path)
    mesh_3d = np.hstack((mesh_obj.v0[:, np.newaxis], mesh_obj.v1[:, np.newaxis], mesh_obj.v2[:, np.newaxis]))

    vol, scale, shift = convert_mesh(mesh_3d, resolution=resolution, parallel=parallel)

    voxels = vol.astype(bool)
    surface = []
    for z in range(voxels.shape[0]):
        for y in range(voxels.shape[1]):
            for x in range(voxels.shape[2]):
                if voxels[z][y][x]:
                    point = (np.array([x, y, z]) / scale) + shift
                    surface.append(point)
    return np.array(surface), vol, voxels

def triang_voxel(F, xl):

    # Define box limits
    bmax = np.max(xl)
    bmin = np.min(xl)

    # Define the mesh 
    X, Y, Z = np.meshgrid(xl, xl, xl)

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

def goursat_tangle(x,y,z):
    a,b,c = 0.0,-5.0,11.8
    return x**4+y**4+z**4+a*(x**2+y**2+z**2)**2+b*(x**2+y**2+z**2)+c

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

class operation:
    def translate(fn,x,y,z):
        return lambda a,b,c: fn(x-a,y-b,z-c)
    def union(*fns):
        return lambda x,y,z: np.min(
            [fn(x,y,z) for fn in fns], 0)
    def subtract(fn1, fn2):
        return intersect(fn1, lambda *args:-fn2(*args))
    def intersect(*fns):
        return lambda x,y,z: np.max(
            [fn(x,y,z) for fn in fns], 0)

class cube:
    def __init__(self, L):
        self.L = L

    def fun(self,x,y,z):
        f = np.maximum(np.abs(x),np.maximum(np.abs(y),np.abs(z)))-self.L/2
        return f

class prism:
    def __init__(self, w, h, L):
        self.w = w
        self.h = h
        self.L = L

    def fun(self,x,y,z):
        ''' 
        Obtains the implicit function of a 3d infinite cylinder
        - x, y, z: sym variables or numeric arrays
        - r = float : radius. [value > 0]
        '''
        f = np.maximum(np.maximum(np.maximum(x-self.w/2,-self.w/2-x), \
            np.maximum(y-self.h/2,-self.h/2-y)), \
            np.maximum(z-self.L/2,-self.L/2-z))

        return f

class cylinder:
    def __init__(self, r, L, axis='z'):
        self.r = r
        self.L = L
        self.axis = axis

    def fun(self,x,y,z):
        ''' 
        Obtains the implicit function of a 3d cylinder
        - x, y, z: sym variables or numeric arrays
        - r = float : radius of the cylinder [value > 0]
        - L = float : heigth of the cylinder [value > 0]
        - axis = 'x', 'y' or 'z'. Sets the direction of longitudinal axis
        '''
        if self.axis == 'z':
            f = x*x + y*y - self.r*self.r 
            f[abs(z) > self.L/2] = None
        if self.axis == 'x':
            f = z*z + y*y - self.r*self.r 
            f[abs(x) > self.L/2] = None
        if self.axis == 'y':
            f = x*x + z*z - self.r*self.r 
            f[abs(y) > self.L/2] = None
        return f

class sphere:
    def __init__(self, r):
        self.r = r   

    def fun(self,x,y,z):
        ''' 
        Obtains the implicit function of a 3d sphere
        - x, y, z: sym variables or numeric arrays
        - r = float : radius of the sphere. [value > 0]
        '''
        f = x**2 + y**2 + z**2 - self.r**2
        return f

class cone:
    def __init__(self, r, h, axis='z', mirror=False):
        self.r = r
        self.h = h
        self.axis = axis
        self.mirror = mirror

    def fun(self,x,y,z):
        ''' 
        Obtains the implicit function of a 3d cone
        - x, y, z: sym variables or numeric arrays
        - r = float : radius of the cone [value > 0]
        - h = float : heigth of the cone [value > 0]
        - axis = 'x', 'y' or 'z'. Sets the direction of longitudinal axis
        - mirror = False or True : mirrors the orientation of the cone
        '''
        if self.axis == 'z':
            if self.mirror:
                z = -z
            c = self.r/self.h/2
            f =  (x**2 + y**2)/c -(z-self.h/2)**2
            f[abs(z) > self.h/2] = None
            f[abs(z) < -self.h/2] = None

        if self.axis == 'x':
            if self.mirror:
                x = -x
            c = self.r/self.h/2
            f =  (z**2 + y**2)/c -(x-self.h/2)**2
            f[abs(x) > self.h/2] = None
            f[abs(x) < - self.h/2] = None

        if self.axis == 'y':
            if self.mirror:
                y = -y
            c = self.r/self.h/2
            f =  (x**2 + z**2)/c -(y-self.h/2)**2
            f[abs(y) > self.h/2] = None
            f[abs(y) < - self.h/2] = None

        return f

class cone_frustum:
    def __init__(self, r1, r2, h, axis='z', mirror=False):
        self.r1 = r1
        self.r2 = r2
        self.h = h
        self.axis = axis
        self.mirror = mirror

    def fun(self,x,y,z):
        ''' 
        Obtains the implicit function of a 3d cone
        - x, y, z: sym variables or numeric arrays
        - r1 = float : initial radius of the cone [value > 0]
        - r2 = float : final radius of the cone [value > 0]
        - h = float : heigth of the cone [value > 0]
        - axis = 'x', 'y' or 'z'. Sets the direction of longitudinal axis
        - mirror = False or True : mirrors the orientation of the cone
        '''
        if self.axis == 'z':
            if self.mirror:
                z = -z
            H = self.h + (self.r1*self.h)/(self.r2-self.r1)
            c = self.r2/H/2
            f =  (x**2 + y**2)/c -(z-H/2)**2
            f[abs(z) > self.h/2] = None
            f[abs(z) < -self.h/2] = None
   
        if self.axis == 'x':
            if self.mirror:
                x = -x
            H = self.h + (self.r1*self.h)/(self.r2-self.r1)
            c = self.r2/H/2
            f =  (z**2 + y**2)/c -(x-H/2)**2
            f[abs(x) > self.h/2] = None
            f[abs(x) < - self.h/2] = None

        if self.axis == 'y':
            if self.mirror:
                y = -y
            H = self.h + (self.r1*self.h)/(self.r2-self.r1)
            c = self.r2/H/2
            f =  (x**2 + z**2)/c -(y-H/2)**2
            f[abs(y) > self.h/2] = None
            f[abs(y) < - self.h/2] = None

        return f        

class n_pyramid: #[TODO]
    def __init__(self, L, h, naxis='z', mirror=False):
        self.L = w
        self.h = h
        self.axis = axis
        self.mirror = mirror

if __name__ == "__main__":

    
    # Test classes
    cube = cube(L=2.0)
    prism = prism(w=1.0, h=2.0, L=3.0)
    cyl = cylinder(1, 3)
    sphere = sphere(2)
    cone = cone(1, 2, axis='y')
    cone_frustum = cone_frustum(1,2,2)

    fn = operation.intersect(cube.fun, cyl.fun)
    fn = operation.union(cube.fun, cyl.fun)
    fn = operation.translate(fn, 0, 0, 1)

    F=triang_implicit(fn, BC = None, bbox=(-3,3))
    
    '''
    UNIT = 1.e-3
    #Test stl to xyz conversion
    input_file_path='CRAB_CAV.stl'
    mesh_obj = mesh.Mesh.from_file(input_file_path)

    dh=10.0*UNIT

    total_W=mesh_obj.max_[0]*UNIT-mesh_obj.min_[0]*UNIT
    total_H=mesh_obj.max_[1]*UNIT-mesh_obj.min_[1]*UNIT
    total_L=mesh_obj.max_[2]*UNIT-mesh_obj.min_[2]*UNIT

    # number of cells needed. Last 10 cells are PML
    nx=next2power((total_W)/dh+10)
    ny=next2power((total_H)/dh+10)
    nz=next2power((total_L)/dh+10)

    # mesh bounds for domain 
    xmin = -nx*dh/2
    xmax = nx*dh/2
    ymin = -ny*dh/2
    ymax = ny*dh/2
    zmin = -nz*dh/2 
    zmax = nz*dh/2

    # mesh vectors
    x=np.linspace(xmin, xmax, nx)
    y=np.linspace(ymin, ymax, ny)
    z=np.linspace(zmin, zmax, nz)

    # conversion to voxels
    # Note: vol should be addressed with vol[z][y][x]
    resolution = int(np.min((nx,ny,nz))*total_L/np.maximum(total_W,total_H))
    mesh_3d = np.hstack((mesh_obj.v0[:, np.newaxis], mesh_obj.v1[:, np.newaxis], mesh_obj.v2[:, np.newaxis]))
    vol, scale, shift = convert_mesh(mesh_3d, resolution=resolution)  
    
    # create voxel grid with size of domain (nx,ny,nz)
    F=np.zeros((nx,ny,nz))

    # relative displacement between F and vol
    nnx=(nx-vol.shape[2])//2
    nny=(ny-vol.shape[1])//2
    nnz=(nz-vol.shape[0])//2

    # add vol to the voxel grid
    for k in range(vol.shape[0]):
        for j in range(vol.shape[1]):
            for i in range(vol.shape[2]):
                F[i+nnx,j+nny,k+nnz] = vol[k][j][i]

    # Obtain the points in the surface
    voxels = vol.astype(bool)
    surface = []
    for z in range(voxels.shape[0]):
        for y in range(voxels.shape[1]):
            for x in range(voxels.shape[2]):
                if voxels[z][y][x]:
                    point = (np.array([x, y, z]) / scale) + shift
                    surface.append(point)

    surface=np.array(surface)

    #mask = np.logical_not(F.astype(bool))
    #F[mask] = float('nan')

    # Plot every surface point
    fig = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], color='b')

    # Set plot properties
    #ax.set_zlim3d(xmin/UNIT,xmax/UNIT)
    #ax.set_xlim3d(ymin/UNIT,ymax/UNIT)
    #ax.set_ylim3d(zmin/UNIT,zmax/UNIT)
    ax.grid(True, color='gray', linewidth=0.2)
    ax.set(title='Geometry defined by the implicit function', 
            xlabel='z [m]',
            ylabel='x [m]',
            zlabel='y [m]'
            )
    plt.show()
    '''

    #surface, vol, voxels = stl_to_xyz('CRAB_CAV.stl', resolution=100)