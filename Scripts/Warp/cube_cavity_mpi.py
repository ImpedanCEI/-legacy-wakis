import numpy as np
import numpy.random as random
from warp import picmi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import time
import os
import scipy as sc  
from copy import copy
import pickle as pk

unit = 1e-3

##################################
# Define the geometry
##################################

# width of the rectangular beam pipe (x direction)
w_pipe = 15*unit
# height of the rectangular beam pipe (y direction)
h_pipe = 15*unit
# total length of the domain
L_pipe = 50*unit 
 
# width of the rectangular cavity (x direction)
w_cavity = 50*unit
# height of the rectangular beam pipe (y direction)
h_cavity = 50*unit
# length of each side of the beam pipe (z direction)
L_cavity = 30*unit 

##################################
# Define the mesh
##################################
# mesh cells per direction
nx = 110
ny = 110
nz = 200

# mesh bounds
xmin = -0.55*w_cavity
xmax = 0.55*w_cavity
ymin = -0.55*h_cavity
ymax = 0.55*h_cavity
zmin = -L_pipe
zmax = L_pipe

##################################
# Beam Setup
##################################
# generate the beam
beam = picmi.Species(particle_type = 'proton',
                     particle_shape = 'cubic',
                     name = 'beam')

##################################
# Setup the geometry
##################################
domain = picmi.warp.Box(1.5*(xmax-xmin), 1.5*(ymax-ymin), (zmax-zmin)) #removed the 1.5 in z direction
pipe = picmi.warp.Box(w_pipe, h_pipe, 2*L_pipe)
cavity = picmi.warp.Box(w_cavity, h_cavity, L_cavity)

conductors = domain - pipe - cavity

##################################
# Setupe the grid
##################################
lower_boundary_conditions = ['open', 'open', 'open']
upper_boundary_conditions = ['open', 'open', 'open']

# initiate the grid
grid = picmi.Cartesian3DGrid(number_of_cells = [nx, ny, nz],
                        lower_bound = [xmin, ymin, zmin],
                        upper_bound = [xmax, ymax, zmax],
                        lower_boundary_conditions = lower_boundary_conditions,
                        upper_boundary_conditions = upper_boundary_conditions)

##################################
# Setupe the solver
##################################
# set up a field smoother
smoother = picmi.BinomialSmoother(n_pass = [[1], [1], [1]],
                                  compensation = [[False], [False], [False]],
                                  stride = [[1], [1], [1]],
                                  alpha = [[0.5], [0.5], [0.5]])

# should we do divergence correction?
flag_correct_div = True

# set up the em solver 
# (it is also interesting to try the CKC solver) by setting method='CKC'
solver = picmi.ElectromagneticSolver(grid = grid,
                                     method = 'CKC',
                                     cfl = 0.80,
                                     source_smoother = smoother,
                                     warp_l_correct_num_Cherenkov = False,
                                     warp_type_rz_depose = 0,
                                     warp_l_setcowancoefs = True,
                                     warp_l_getrho = False,
                                     warp_conductors = conductors,
                                     warp_conductor_dfill = picmi.warp.largepos,
                                     warp_l_pushf = flag_correct_div)



##################################
# Setup the simulation object
##################################
sim = picmi.Simulation(solver = solver, verbose = 1,
                        warp_initialize_solver_after_generate = 1)

#this is needed to complete the simulation setup
sim.step(1)        

##################################
# Setup the beam injection
##################################
N=10**7
beam_layout = picmi.PseudoRandomLayout(n_macroparticles = N, seed = 3)

sim.add_species(beam, layout=beam_layout,
                initialize_self_field = solver=='EM')

# beam sigma in time and longitudinal direction
sigmat= 1.000000e-09/16.     #changed from /4 to /16
sigmaz = sigmat*picmi.constants.c 
# transverse sigmas.
sigmax = 2e-3 #changed from e-4 to e-3 
sigmay = 2e-3

# spacing between bunches
b_spac = 25e-9
# offset of the bunch centroid
t_offs = -5.332370636221942e-10   #like CST (-160 mm)
# number of bunches to simulate
n_bunches = 1

# beam energy
beam_gamma = 479. #increase to decrease space charge
beam_uz = beam_gamma*picmi.constants.c

# macroparticle info
bunch_charge = 1e-9
bunch_physical_particles  = int(bunch_charge/sc.constants.e)
bunch_macro_particles = N
bunch_w = bunch_physical_particles/bunch_macro_particles


bunch_rms_size            = [sigmax, sigmay, sigmaz]
bunch_rms_velocity        = [0.,0.,0.]
bunch_centroid_position   = [0,0,0.95*zmin]
bunch_centroid_velocity   = [0.,0.,beam_uz]

# time profile of a gaussian beam
def time_prof(t):
    val = 0
    sigmat = sigmaz/picmi.clight
    for i in range(0,n_bunches):
        val += bunch_macro_particles*1./np.sqrt(2*np.pi*sigmat*sigmat)*np.exp(-(t-i*b_spac+t_offs)*(t-i*b_spac+t_offs)/(2*sigmat*sigmat))*picmi.warp.top.dt
    return val

# auxiliary function for injection
def nonlinearsource():
    NP = int(time_prof(picmi.warp.top.time))
    x = random.normal(bunch_centroid_position[0],bunch_rms_size[0],NP)
    y = random.normal(bunch_centroid_position[1],bunch_rms_size[1],NP)
    z = bunch_centroid_position[2]
    vx = random.normal(bunch_centroid_velocity[0],bunch_rms_velocity[0],NP)
    vy = random.normal(bunch_centroid_velocity[1],bunch_rms_velocity[1],NP)
    vz = picmi.warp.clight*np.sqrt(1-1./(beam_gamma**2))
    beam.wspecies.addparticles(x=x,y=y,z=z,vx=vx,vy=vy,vz=vz,gi = 1./beam_gamma, w=bunch_w)

#set up the injection
picmi.warp.installuserinjection(nonlinearsource)

# define shortcuts
pw = picmi.warp
pw.winon()
em = solver.solver
step=pw.step


##################################
# Perform Simulation
##################################
tot_nsteps = 3000
t0 = time.time()

#pre-allocate timedep variables
Ez_t=[]
Ex_t=[]
Ey_t=[]
Bx_t=[]
By_t=[]
rho_t=[]
t=[]
dt=[]

#create output directories for txt
out_folder='out/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)


#define the integration path x2, y2
xtest=0.0*unit   #Assumes x2,y2 of the test particle in 0,0
ytest=0.0*unit
#---set up the vectors
x=np.linspace(xmin, xmax, nx+1)
y=np.linspace(ymin, ymax, ny+1)
z=np.linspace(zmin, zmax, nz+1)
dx=x[2]-x[1]
dy=y[2]-y[1]
dz=z[2]-z[1]
#---search for the index
ixtest=int((xtest-x[0])/dx)+1 #the +1 gives the correct cell
iytest=int((ytest-y[0])/dy)+1

#-------------#                             
#  time loop  #
#-------------#

for n_step in range(tot_nsteps):
    picmi.warp.step()
    #Extracting the electric field from all processors
    #---z direction
    Ez=em.gatherez()    #3D matrix
    wEz=beam.wspecies.getez()   #vector with N components (for each particle) - in particles.py 1054 "Returns the Ex field applied to the particles"
    #print(wEz) returns an empty vector when there is no beam
    #---x direction
    Ex=em.gatherex()    #3D matrix
    Bx=em.gatherbx()    #3D matrix
    wEx=beam.wspecies.getex()   #vector with N components (for each particle) - in particles.py 1054 "Returns the Ex field applied to the particles"
    #---y direction
    Ey=em.gatherey()    #3D matrix
    By=em.gatherby()    #3D matrix
    wEy=beam.wspecies.getey()   #vector with N components (for each particle) - in particles.py 1054 "Returns the Ex field applied to the particles"
    #get charge density
    rho=beam.wspecies.get_density()    #gathers the rho (electric density) of the beam rho[nx,ny,nz]

    if picmi.warp.me == 0:  #checks if we are in processor 0. Needed for mpi simulations

        #time vector
        t.append(picmi.warp.top.time)
        dt.append(picmi.warp.top.dt)

        #append the 1D electric field at (ixtest,iytest,z)
        #--- z direction
        Ez_t.append(Ez[ixtest,iytest,:]) #1D vector of len=tot_nsteps*nz 
        #--- x direction
        Ex_t.append(Ex[ixtest,iytest,:]) #1D vector of len=tot_nsteps*nz 
        #--- y direction
        Ey_t.append(Ey[ixtest,iytest,:]) #1D vector of len=tot_nsteps*nz 
        #append the 1D magnetic induction
        #--- x direction
        Bx_t.append(Bx[ixtest,iytest,:]) #1D vector of len=tot_nsteps*nz 
        #--- y direction
        By_t.append(By[ixtest,iytest,:]) #1D vector of len=tot_nsteps*nz
        #append the charge density at (ixtest,iytest,z)
        rho_t.append(rho[ixtest, iytest, :]) #1D vector of len=tot_nsteps*nz


#--------------------#                             
#  end of time loop  #
#--------------------#

#Calculate simulation time
t1 = time.time()
totalt = t1-t0
print('Run terminated in %ds' %totalt)

##################################
#           Save data
##################################

#--- create dictionary with the output data
data = { 'Ez' : Ez_t, #len(tot_nsteps*nz)
         'Ex' : Ex_t,
         'Ey' : Ey_t,
         'By' : By_t,
         'Bx' : By_t,
         'rho' : rho_t,
         't' : t, 
         'init_time' : -t_offs,
         'x' : x,
         'y' : y,
         'z' : z,
         'nt' : tot_nsteps,
         'nz' : nz,
         'w_cavity' : w_cavity,
         'h_cavity' : h_cavity,
         'L_cavity' : L_cavity,
         'w_pipe' : w_pipe,
         'h_pipe' : h_pipe,
         'L_pipe' : L_pipe,
         'sigmaz' : sigmaz,
         'xtest' : xtest,
         'ytest' : ytest
        }
# write the dictionary to a txt using pickle module
with open(out_folder+'out.txt', 'wb') as handle:
  pk.dump(data, handle)

#--- to read the dictionary type
# with open('out.txt', 'rb') as handle:
#   data = pickle.loads(handle.read())
#   print(data.keys())
#   x=data.get('x')


