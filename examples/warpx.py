'''
------------------------------
|  WAKIS+WarpX simulations   |
------------------------------
Contains auxiliary script to setup and run WarpX simulations

Usage:
------
Call it by wakis.execfile('warpx.py') from the input script

Requirements:
------------- 
pip install pywarpx, wakis, numpy, h5py

'''

import time
import pickle as pk  

import h5py
from pywarpx import picmi
from pywarpx import libwarpx, fields, callbacks
import pywarpx.fields as pwxf
import numpy.random as random

'''
# create output directory 
path=os.getcwd() + '/out_'+os.path.basename(__file__)+'/'

if not os.path.exists(path):
    os.mkdir(path)
'''
path=os.getcwd() + '/'


##################################
# Define the mesh
##################################

# Define mesh cells per direction. !![has to be divisible by 2**3]

nx = wakis.next2power(W/dh)
ny = wakis.next2power(H/dh) 
nz = wakis.next2power(L/dh)

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
x=np.linspace(xmin, xmax, nx)
y=np.linspace(ymin, ymax, ny)
z=np.linspace(zmin, zmax, nz)

# max grid size for mpi
max_grid_size_x = nx
max_grid_size_y = ny
max_grid_size_z = nz//NUM_PROC

# mask for the E field extraction
if flag_mask_pml:
    zmask = np.logical_and(z >= -L/2.0 - dz, z <= L/2 + dz)
else:
    zmask = z >= zmin   #no mask

#number of pml cells needed
#n_pml=10  #default: 10
n_pml=(nz-L/dz)//2 

#Injection position
z_inj=zmin+5*dz
#z_inj=zmin+n_pml/2*dz

##################################
# Beam Setup
##################################
# generate the beam
bunch = picmi.Species(particle_type='proton',
                     name = 'beam')

##########################
# numerics components
##########################
lower_boundary_conditions = ['dirichlet', 'dirichlet', 'open']
upper_boundary_conditions = ['dirichlet', 'dirichlet', 'open']

grid = picmi.Cartesian3DGrid(
    number_of_cells = [nx, ny, nz],
    lower_bound = [xmin, ymin, zmin],
    upper_bound = [xmax, ymax, zmax],
    lower_boundary_conditions = lower_boundary_conditions,
    upper_boundary_conditions = upper_boundary_conditions,
    lower_boundary_conditions_particles = ['absorbing', 'absorbing', 'absorbing'],
    upper_boundary_conditions_particles = ['absorbing', 'absorbing', 'absorbing'],
    moving_window_velocity = None,
    warpx_max_grid_size_x = max_grid_size_x,
    warpx_max_grid_size_y = max_grid_size_y,
    warpx_max_grid_size_z = max_grid_size_z,
)

flag_correct_div = False
flag_correct_div_pml = False
solver = picmi.ElectromagneticSolver(grid=grid, method='Yee', cfl=CFL,
                                     divE_cleaning = flag_correct_div,
                                     pml_divE_cleaning = flag_correct_div_pml,
                                     warpx_pml_ncell = n_pml,
                                     warpx_do_pml_in_domain = True,
                                     warpx_pml_has_particles = True,
                                     warpx_do_pml_j_damping = True, #Turned True for the pml damping
                                     warpx_adjusted_pml = True, 
                                     )


##########################
# simulation setup
##########################

# Obtain number of timesteps needed for the wake length

# time when the bunch enters the cavity
init_time = 5.332370636221942e-10 + (zmin+L/2)/c -z_inj/c #[s] injection time + PEC length - Injection length 

# timestep size
dt=CFL*(1/c)/np.sqrt((1/dx)**2+(1/dy)**2+(1/dz)**2)

# timesteps needed to simulate
max_steps=int((Wake_length+init_time*c+(zmax-zmin))/dt/c)

print('[WARPX][INFO] Timesteps to simulate = '+ str(max_steps) + ' with timestep dt = ' + str(dt))
print('[WARPX][INFO] Wake length = '+str(Wake_length/UNIT)+ ' mm')

sim = picmi.Simulation(
    solver = solver,
    max_steps = max_steps,
    warpx_embedded_boundary=embedded_boundary,
    particle_shape = 'cubic', #default: 'cubic'
    verbose = 1
)

beam_layout = picmi.PseudoRandomLayout(n_macroparticles = 0)

sim.add_species(bunch, layout=beam_layout)

sim.initialize_inputs()

##################################
# Setup the beam injection
##################################

# beam sigma in time and longitudinal direction
# defined by the user

# transverse sigmas.
sigmax = 2e-4
sigmay = 2e-4

# spacing between bunches
b_spac = 25e-9
# offset of the bunch centroid
t_offs = -init_time #+ 9*dz/c   #like CST (-160 mm) + non-vacuum cells from injection point
# number of bunches to simulate
n_bunches = 1

# beam energy
beam_gamma = 479.
beam_uz = beam_gamma*c
beam_beta = np.sqrt(1-1./(beam_gamma**2))

# macroparticle info
N=10**7
bunch_charge = q #beam charge in [C] defined by the user
bunch_physical_particles  = int(bunch_charge/e)
bunch_macro_particles = N
bunch_w = bunch_physical_particles/bunch_macro_particles

# Define the beam offset
ixsource=int((xsource-x[0])/dx)
iysource=int((ysource-y[0])/dy)
print('[WARPX][INFO] Beam center set to ('+str(round(x[ixsource]/UNIT,3))+','+str(round(y[iysource]/UNIT,3))+',z,t) [mm]')

bunch_rms_size            = [sigmax, sigmay, sigmaz]
bunch_rms_velocity        = [0.,0.,0.]
bunch_centroid_position   = [xsource, ysource, z_inj] #Always inject in position 5
bunch_centroid_velocity   = [0.,0.,beam_uz]

# time profile of a gaussian beam
def time_prof(t):
    val = 0
    sigmat =  sigmaz/c
    dt = libwarpx.libwarpx_so.warpx_getdt(0)
    for i in range(0,n_bunches):
        val += bunch_macro_particles*1./np.sqrt(2*np.pi*sigmat*sigmat)*np.exp(-(t-i*b_spac+t_offs)*(t-i*b_spac+t_offs)/(2*sigmat*sigmat))*dt
        #print(val)
    return val

# auxiliary function for injection
def nonlinearsource():
    t = libwarpx.libwarpx_so.warpx_gett_new(0)
    NP = int(time_prof(t))
    if NP>0:
        x = random.normal(bunch_centroid_position[0],bunch_rms_size[0],NP)
        y = random.normal(bunch_centroid_position[1],bunch_rms_size[1],NP)
        z = bunch_centroid_position[2]

        vx = np.zeros(NP)
        vy = np.zeros(NP)
        vz = np.ones(NP)*c*np.sqrt(1-1./(beam_gamma**2))

        beam_beta = np.sqrt(1-1./(beam_gamma**2))
        
        ux = np.zeros(NP)
        uy = np.zeros(NP)
        uz = beam_beta * beam_gamma * c

        libwarpx.add_particles(
            species_name='beam', x=x, y=y, z=z, ux=ux, uy=uy, uz=uz, w=bunch_w*np.ones(NP),
        )

callbacks.installparticleinjection(nonlinearsource)

print('[WARPX][INFO] Finished simulation setup')

##########################
# Simulation run
##########################
t0 = time.time()

if flag_logfile:
    # Create logfile
    sys.stdout = open(path+"log.txt", "w")

# Create h5 files overwriting previous ones

# Ez file
hf_name='Ez.h5'
if os.path.exists(path+hf_name):
    os.remove(path+hf_name)

hf_Ez = h5py.File(path+hf_name, 'w')

# rho file
hf_name='rho.h5'
if os.path.exists(path+hf_name):
    os.remove(path+hf_name)

hf_rho = h5py.File(path+hf_name, 'w')

# Define the integration path for test particle (xtest, ytest)
ixtest=int((xtest-x[0])/dx)
iytest=int((ytest-y[0])/dy)
print('[WARPX][INFO] Field will be extracted around ('+str(round(x[ixtest]/UNIT,3))+','+str(round(y[iytest]/UNIT,3))+',z,t) [mm]')

# Define number for datasets title
prefix=[]
for n_step in range(1, max_steps):
    prefix.append('0'*(5-int(np.log10(n_step))))

prefix=np.append('0'*5, prefix)

# Step by step running + saving data in hdf5 format
t=[]
rho_t=[]

# Perform the simulation --------------------------------------
print('[WARPX][PROGRESS] Starting simulation with a total of '+str(max_steps)+'timesteps...' )
for n_step in range(max_steps):

    print(n_step)
    sim.step(1)

    # Extract the electric field from all processors
    Ez = fields.EzWrapper().get_fabs(0,2,include_ghosts=False)[0]
    # Extract charge density
    rho = fields.JzWrapper().get_fabs(0,2,include_ghosts=False)[0]/(beam_beta*c)  #[C/m3]
    # Extraxt the timestep size
    dt = libwarpx.libwarpx_so.warpx_getdt(0)

    # append to t, rho lists
    t.append(n_step*dt)
    rho_t.append(rho[ixsource,iysource,:])

    # Save the 3D Ez matrix into a hdf5 dataset
    #hf_Ez.create_dataset('Ez_'+prefix[n_step]+str(n_step), data=Ez[:,:,zmask])

    # Saves the Ez field in a prism along the z axis 3 cells wide into a hdf5 dataset
    hf_Ez.create_dataset('Ez_'+prefix[n_step]+str(n_step), data=Ez[ixtest-1:ixtest+2 , 
                                                           iytest-1:iytest+2 , 
                                                           zmask] )
    hf_rho.create_dataset('rho_'+prefix[n_step]+str(n_step), data=rho[ixsource,iysource,:])

# Finish simulation --------------------------------------------

# Calculate simulation time
t1 = time.time()
totalt = t1-t0
print('[WARPX][PROGRESS] Run terminated in %ds' %totalt)

# Close the hdf5 files
hf_Ez.close()
hf_rho.close()

if flag_logfile:
    # Close logfile
    sys.stdout.close()

##########################
#    Generate output     # 
##########################

# Create dictionary with input data. SI UNITs: [m], [s], [C]
data = { 'init_time' : -t_offs, 
         't' : t,
         'x' : x[ixtest-1:ixtest+2],   
         'y' : y[iytest-1:iytest+2],
         'z' : z[zmask],
         'nt' : max_steps,
         'nx' : nx,
         'ny' : ny,
         'nz' : nz,
         'L' : L,    
         'sigmaz' : sigmaz,
         'xsource' : xsource,
         'ysource' : ysource,
         'xtest' : xtest,
         'ytest' : ytest,
         'q' : bunch_charge, 
         'unit' : UNIT,
         'x0' : x,
         'y0' : y,
         'z0' : z
        }

# write the input dictionary to a txt using pickle module
with open(path+'warpx.out', 'wb') as fp:
    pk.dump(data, fp)

print('[! OUT] warpx.out file succesfully generated') 