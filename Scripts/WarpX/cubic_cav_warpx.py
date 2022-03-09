'''
cubic_cav_warpx.py

Script to perform simulations of a cubic cavity in WarpX 
to obtain the wake potential and impedance

--- Define the simulation length based on the Wake length
--- Define the test and source positions of the beam
--- Performs the simulation with WarpX-PIC solver
--- Stores the electric field Ez and charge distribution in .h5 file
--- Stores the geometry and simulation input in a dictionary with pickle 

'''

from pywarpx import picmi
from pywarpx import libwarpx, fields, callbacks
import pywarpx.fields as pwxf
import numpy as np
import numpy.random as random
import scipy as sc
from scipy.constants import c, m_p, e
import matplotlib.pyplot as plt
import pickle as pk
import time
import os
import h5py
import sys

from geom_functions import plot_implicit, triang_implicit

t0 = time.time()

##################################
# User defined variables
##################################

# output path
path='/mnt/c/Users/elefu/Documents/CERN/GitHub/PyWake/Scripts/WarpX/'

# create output directory 
out_folder=path+'out/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# create logfile
sys.stdout = open(path+"out/log.txt", "w")

# simulation parameters
CFL=1.0             #Courant-Friedrichs-Levy criterion for stability
NUM_PROC = 1        #number of mpi processors wanted to use
unit = 1e-3         #conversion factor from input to [m]
Wake_length=2000*unit    #Wake potential length in s [m]

# flags
flag_add_diagnosis = False    #turn on and off warpx diagnostics

# beam center 
# ---[longitudinal impedance: beam center in 0,0]
# ---[dipolar impedance: beam center in a,0 or 0,a or a,a]
xsource = 3.0*unit
ysource = 3.0*unit 

# test particle center 
# ---[longitudinal impedance: test axis in 0,0]
# ---[quadrupolar impedance: test axis in a,0 or 0,a or a,a]
xtest = 0.0*unit   
ytest = 0.0*unit

##################################
# Define the geometry
##################################

# width of the rectangular beam pipe (x direction)
w_pipe = 15*unit
# height of the rectangular beam pipe (y direction)
h_pipe = 15*unit
# length of the pipe (bigger than the domain to resemble infinite length)
L_pipe = 50*unit

# width of the rectangular beam pipe (x direction)
w_cav = 50*unit
# height of the rectangular beam pipe (y direction)
h_cav = 50*unit
# length of the pipe (bigger than the domain to resemble infinite length)
L_cav = 30*unit


##################################
# Define the mesh
##################################
# mesh cells per direction. Has to be a 2^3 power
nx = 64 
ny = 64
nz = 256

# mesh bounds for domain. Last 10 cells are PML
xmin = -32*unit
xmax = 32*unit
ymin = -32*unit
ymax = 32*unit
zmin = -128*unit 
zmax = 128*unit

# mesh cell widths
dx=(xmax-xmin)/nx
dy=(ymax-ymin)/ny
dz=(zmax-zmin)/nz

# mesh arrays
x=np.linspace(xmin, xmax, nx)
y=np.linspace(ymin, ymax, ny)
z=np.linspace(zmin, zmax, nz)

# max grid size for mpi
max_grid_size_x = nx
max_grid_size_y = ny
max_grid_size_z = nz//NUM_PROC

##################################
# Beam Setup
##################################
# generate the beam
protn_mass = m_p
proton_charge = e
beam = picmi.Species(particle_type='proton',
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
                                     warpx_do_pml_in_domain = True,
                                     warpx_pml_has_particles = True,
                                     warpx_do_pml_j_damping = True) #Turned True for the pml damping

# Define the implicit function for the boundary conditions
flag_plot_geom= True
embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function="w=w_pipe+(w_cav-w_pipe)*(z<L_cav/2)*(z>-L_cav/2); h=h_pipe+(h_cav-h_pipe)*(z<L_cav/2)*(z>-L_cav/2); max(max(x-w/2,-w/2-x),max(y-h/2,-h/2-y))",
    w_cav=w_cav, 
    h_cav=h_cav, 
    L_cav=L_cav, 
    w_pipe=w_pipe, 
    h_pipe=h_pipe, 
)

# Plot the surface to check the geometry (x,z,y) are disposed so z is in the longitudinal direction
def implicit_function(x,z,y):
    w=w_pipe+(w_cav-w_pipe)*(z<L_cav/2)*(z>-L_cav/2)
    h=h_pipe+(h_cav-h_pipe)*(z<L_cav/2)*(z>-L_cav/2)
    return np.maximum(np.maximum(x-w/2,-w/2-x),np.maximum(y-h/2,-h/2-y))

if flag_plot_geom:
    #plot_implicit(fn=implicit_function, lims=(-w_cav/1.9,+w_cav/1.9,-h_cav/1.9,+h_cav/1.9,zmin,zmax), bbox=(zmin,zmax))
    triang_implicit(fn=implicit_function, bbox=(zmin,zmax))

##########################
# diagnostics
##########################

if flag_add_diagnosis:
    t_period = 1 #datadata saved every t_period timesteps
    field_diag = picmi.FieldDiagnostic(
        name = 'diag1',
        grid = grid,
        period = t_period, #field saved every 10 steps
        data_list = ['Ez', 'rho'],
        write_dir = 'out/diags',
        warpx_file_prefix = 'warpx_diag',
        #warpx_format = 'openpmd'
    )


##########################
# simulation setup
##########################

# obtain number of timesteps needed for the wake length
# time when the bunch enters the cavity
init_time = 5.332370636221942e-10 

# timestep size
dt=(1/c)/np.sqrt((1/dx)**2+(1/dy)**2+(1/dz)**2)

# timesteps needed to simulate
max_steps=int((Wake_length+init_time*c+(zmax-zmin))/dt/c)

print('Timesteps to simulate = '+ str(max_steps) + ' with timestep dt = ' + str(dt))
print('Wake length = '+str(Wake_length/unit)+ ' mm')

sim = picmi.Simulation(
    solver = solver,
    max_steps = max_steps,
    warpx_embedded_boundary=embedded_boundary,
    particle_shape = 'cubic',
    verbose = 1
)

if flag_add_diagnosis:
    sim.add_diagnostic(field_diag)

beam_layout = picmi.PseudoRandomLayout(n_macroparticles = 0)

sim.add_species(beam, layout=beam_layout)

sim.initialize_inputs()

##################################
# Setup the beam injection
##################################

# beam sigma in time and longitudinal direction
sigmat= 1.000000e-09/16.     #changed from /4 to /16
sigmaz = sigmat*picmi.constants.c         #[m]
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
bunch_charge = 1e-9
bunch_physical_particles  = int(bunch_charge/e)
bunch_macro_particles = N
bunch_w = bunch_physical_particles/bunch_macro_particles

# Define the beam offset
ixsource=int((xsource-x[0])/dx)
iysource=int((ysource-y[0])/dy)

bunch_rms_size            = [sigmax, sigmay, sigmaz]
bunch_rms_velocity        = [0.,0.,0.]
bunch_centroid_position   = [xsource,ysource,zmin+5*dz] #Always inject in the middle of the PML
bunch_centroid_velocity   = [0.,0.,beam_uz]

# time profile of a gaussian beam
def time_prof(t):
    val = 0
    sigmat = sigmaz/c
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

        #for simetric beam 
        # mask=np.logical_and(x>0, y>0)
        # xmirror=np.concatenate((+x[mask], -x[mask], -x[mask], +x[mask]))
        # ymirror=np.concatenate((y[mask], y[mask], -y[mask], -y[mask]))

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

##########################
# simulation run
##########################

if flag_add_diagnosis:
    sim.step(max_steps) #runs all the timesteps

if not flag_add_diagnosis: 
    # Step by step running + saving data in hdf5 format

    # Create h5 files overwriting previous ones
    #---Ez file
    hf_name='Ez.h5'

    if os.path.exists(out_folder+hf_name):
        os.remove(out_folder+hf_name)

    hf_Ez = h5py.File(out_folder+hf_name, 'w')

    #---rho file
    hf_name='rho.h5'

    if os.path.exists(out_folder+hf_name):
        os.remove(out_folder+hf_name)

    hf_rho = h5py.File(out_folder+hf_name, 'w')

    # Create empty lists to store arrays 
    Ez_t=[]
    Ex_t=[]
    Ey_t=[]
    Bx_t=[]
    By_t=[]
    rho_t=[]
    t=[]

    # Define the integration path for test particle (xtest, ytest)
    #---search for the index
    ixtest=int((xtest-x[0])/dx)
    iytest=int((ytest-y[0])/dy)
    #---print check
    print('Field will be extracted around ('+str(round(x[ixtest]/unit,3))+','+str(round(y[iytest]/unit,3))+',z,t) [mm]')
    #---save the n adjacent cells in each direction
    n_adj_cells=int(3) #number of adjacent cells to save


    # Perform the simulation
    for n_step in range(max_steps):

        print(n_step)
        sim.step(1)

        # Extract the electric field from all processors
        Ez = fields.EzWrapper().get_fabs(0,2,include_ghosts=False)[0]
        '''
        Ex = fields.ExWrapper().get_fabs(0,2,include_ghosts=False)[0]
        Ey = fields.EyWrapper().get_fabs(0,2,include_ghosts=False)[0]
        # Extract the magnetic field from all processors
        Bx = fields.BxWrapper().get_fabs(0,2,include_ghosts=False)[0]
        By = fields.ByWrapper().get_fabs(0,2,include_ghosts=False)[0]
        '''
        # Extract charge density
        rho = fields.JzWrapper().get_fabs(0,2,include_ghosts=False)[0]/(beam_beta*c)  #[C/m3]
        # Extraxt the timestep size
        dt = libwarpx.libwarpx_so.warpx_getdt(0)
        t.append(n_step*dt)

        # Store the 3D Ez matrix into a hdf5 dataset
        if n_step == 0:
            prefix='0'*5
            # Saves the Ez field in a prism along the z axis 3 cells wide
            hf_Ez.create_dataset('Ez_'+prefix+str(n_step), data=Ez[ixtest-n_adj_cells:ixtest+n_adj_cells+1 , iytest-n_adj_cells:iytest+n_adj_cells+1,nz//2-50:nz//2+51])
            hf_rho.create_dataset('rho_'+prefix+str(n_step), data=rho[ixsource,iysource,:])
            #hf_rho.create_dataset('rho_'+prefix+str(n_step), data=rho[ixsource-n_adj_cells:ixsource+n_adj_cells+1 , iysource-n_adj_cells:iysource+n_adj_cells+1,:])
        else:
            prefix='0'*(5-int(np.log10(n_step)))
            # Saves the Ez field in a prism along the z axis 3 cells wide
            hf_Ez.create_dataset('Ez_'+prefix+str(n_step), data=Ez[ixtest-n_adj_cells:ixtest+n_adj_cells+1 , iytest-n_adj_cells:iytest+n_adj_cells+1,nz//2-50:nz//2+51])
            hf_rho.create_dataset('rho_'+prefix+str(n_step), data=rho[ixsource,iysource,:])
            #hf_rho.create_dataset('rho_'+prefix+str(n_step), data=rho[ixsource-n_adj_cells:ixsource+n_adj_cells+1 , iysource-n_adj_cells:iysource+n_adj_cells+1,:])

        '''
        # Save field arrays [NOT WORKING]: saves the same value all the timesteps
        Ez_t.append(Ez[ixtest,iytest,:])
        Ex_t.append(Ex[ixtest,iytest,:])
        Ey_t.append(Ey[ixtest,iytest,:]) 
        Bx_t.append(Bx[ixtest,iytest,:])
        By_t.append(By[ixtest,iytest,:])
        rho_t.append(rho[ixtest,iytest,:])
        '''

    # Close the hdf5 files
    hf_Ez.close()
    hf_rho.close()


# Calculate simulation time
t1 = time.time()
totalt = t1-t0
print('Run terminated in %ds' %totalt)
sys.stdout.close()

##########################
#    Generate output     #
##########################

# Create dictionary with input data
input_data = { 'init_time' : -t_offs,
         't' : np.array(t),
         'x' : x,
         'y' : y,
         'z' : z,
         'tot_nsteps' : max_steps,
         'nx' : nx,
         'ny' : ny,
         'nz' : nz,
         'w_cavity' : w_cav,
         'h_cavity' : h_cav,
         'L_cavity' : L_cav,
         'w_pipe' : w_pipe,
         'h_pipe' : h_pipe,
         'L_pipe' : L_pipe,
         'sigmaz' : sigmaz,
         'xsource' : xsource,
         'ysource' : ysource,
         'xtest' : xtest,
         'ytest' : ytest,
         'ixtest' : ixtest,
         'iytest' : iytest,
         'n_adj_cells' : n_adj_cells,
        }

# write the input dictionary to a txt using pickle module
with open(out_folder+'input_data.txt', 'wb') as handle:
  pk.dump(input_data, handle)

'''
# Create dictionary with fields data [NOT WORKING]
field_data = {  'Ez' : np.transpose(np.array(Ez_t)),
                'Ex' : np.transpose(np.array(Ex_t)),
                'Ey' : np.transpose(np.array(Ey_t)),
                'Bx' : np.transpose(np.array(Bx_t)),
                'By' : np.transpose(np.array(By_t)),
                'rho' : np.transpose(np.array(rho_t)),
                't' : np.array(t),
            }


 # write the fields dictionary to a txt using pickle module
with open(out_folder+'field_data.txt', 'wb') as handle:
  pk.dump(field_data, handle)
'''