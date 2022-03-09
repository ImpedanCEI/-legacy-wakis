from pywarpx import picmi
import numpy as np
from pywarpx import libwarpx, fields, callbacks
import pywarpx.fields as pwxf
from scipy.constants import c, m_p, e
import numpy.random as random
import matplotlib.pyplot as plt
from geom_functions import plot_implicit, triang_implicit
import scipy as sc
import pickle as pk
import time
import os
import h5py

t0 = time.time()

# macros
path='/mnt/c/Users/elefu/Documents/CERN/GitHub/PyWake/Scripts/WarpX/'
max_steps = 3000
t_period = 1    #datadata saved every t_perio timesteps
NUM_PROC = 1    #number of mpi processors wanted to use
unit = 1e-3
flag_add_diagnosis=False    #turn on and off warpx diagnostics

##################################
# Define the geometry
##################################

# radius of the beam pipe 
r_pipe = 15*unit/2
# length of the pipe (bigger than the domain to resemble infinite length)
L_pipe = 50*unit

# radius of the beam cavity 
r_cav = 50*unit/2
# length of the beam cavity 
L_cav = 30*unit


##################################
# Define the mesh
##################################
# mesh cells per direction. Has to be a 2^3 power
nx = 64 
ny = 64
nz = 128

# mesh bounds for domain. Last 10 cells are PML
xmin = -32*unit
xmax = 32*unit
ymin = -32*unit
ymax = 32*unit
zmin = -64*unit 
zmax = 64*unit

# mesh cell widths
dx=(xmax-xmin)/nx
dy=(ymax-ymin)/ny
dz =(zmax-zmin)/nz

# shift the mesh to have (0,0,0) as a mesh point
xmin+=dx/2
xmax+=dx/2
ymin+=dy/2
ymax+=dy/2
zmin+=dz/2
zmax+=dz/2

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
                     #particle_shape = 'cubic',
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
solver = picmi.ElectromagneticSolver(grid=grid, method='ECT', cfl=1., #change solver to conformal (Yee --> ECT)
                                     divE_cleaning = flag_correct_div,
                                     pml_divE_cleaning = flag_correct_div_pml,
                                     warpx_do_pml_in_domain = True,
                                     warpx_pml_has_particles = True,
                                     warpx_do_pml_j_damping = True) #Turned True for the pml damping

# Define the implicit function for the boundary conditions
flag_plot_geom= True
embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function="r=(z<=(-L_cav/2))*r_pipe + (z>=(+L_cav/2))*r_pipe + (z>(-L_cav/2))*(z<(+L_cav/2))*r_cav; sqrt(x*x + y*y)-r;",
    r_cav=r_cav,  
    L_cav=L_cav, 
    r_pipe=r_pipe, 
)

# Plot the surface to check the geometry (x,z,y) are disposed so z is in the longitudinal direction
def implicit_function(x,z,y):
    r=(z<=(-L_cav/2))*r_pipe + (z>=(+L_cav/2))*r_pipe + (z>(-L_cav/2))*(z<(+L_cav/2))*r_cav
    return x*x + y*y - r*r

if flag_plot_geom:
    #plot_implicit(fn=implicit_function, lims=(zmin,zmax)*3, bbox=(zmin,zmax))
    triang_implicit(fn=implicit_function, bbox=(zmin,zmax))


##########################
# diagnostics 
##########################
#create output directories for txt
out_folder=path+'out/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

if flag_add_diagnosis:
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

sim = picmi.Simulation(
    solver = solver,
    max_steps = max_steps,
    warpx_embedded_boundary=embedded_boundary,
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
t_offs = -5.332370636221942e-10 #+ 9*dz/c   #like CST (-160 mm) + non-vacuum cells from injection point
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

xsource = 0.0*unit
ysource = 0.0*unit 

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
        print(val)
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

    # Define the integration path for test particle
    xtest=0.0*unit   #Default: test particle in 0,0
    ytest=0.0*unit
    #---search for the index
    ixtest=int((xtest-x[0])/dx) 
    iytest=int((ytest-y[0])/dy)
    #---print check
    print('Field will be extracted around ('+str(round(x[ixtest]/unit,3))+','+str(round(y[iytest]/unit,3))+',z,t) [mm]')
    #---save the n adjacent cells in each direction
    n_adj_cells=int(2) #number of adjacent cells to save


    # Perform the simulation
    for n_step in range(max_steps):

        sim.step(1)

        # Extract the electric field from all processors
        Ez = fields.EzWrapper().get_fabs(0,2,include_ghosts=False)[0]
        Ex = fields.ExWrapper().get_fabs(0,2,include_ghosts=False)[0]
        Ey = fields.EyWrapper().get_fabs(0,2,include_ghosts=False)[0]
        # Extract the magnetic field from all processors
        Bx = fields.BxWrapper().get_fabs(0,2,include_ghosts=False)[0]
        By = fields.ByWrapper().get_fabs(0,2,include_ghosts=False)[0]
        # Extract charge density
        rho = fields.JzWrapper().get_fabs(0,2,include_ghosts=False)[0]/(beam_beta*c)  #[C/m3]
        # Extraxt the timestep size
        dt = libwarpx.libwarpx_so.warpx_getdt(0)
        t.append(n_step*dt)

        # Store the 3D Ez matrix into a hdf5 dataset
        if n_step == 0:
            prefix='0'*5
            # Saves the Ez field in a prism along the z axis 3 cells wide
            hf_Ez.create_dataset('Ez_'+prefix+str(n_step), data=Ez[ixtest-n_adj_cells:ixtest+n_adj_cells+1 , iytest-n_adj_cells:iytest+n_adj_cells+1,:])
            hf_rho.create_dataset('rho_'+prefix+str(n_step), data=rho[ixtest,iytest,:])
        else:
            prefix='0'*(5-int(np.log10(n_step)))
            # Saves the Ez field in a prism along the z axis 3 cells wide
            hf_Ez.create_dataset('Ez_'+prefix+str(n_step), data=Ez[ixtest-n_adj_cells:ixtest+n_adj_cells+1 , iytest-n_adj_cells:iytest+n_adj_cells+1,:])
            hf_rho.create_dataset('rho_'+prefix+str(n_step), data=rho[ixtest,iytest,:])

        # Save field arrays [NOT WORKING]: saves the same value all the timesteps
        Ez_t.append(Ez[ixtest,iytest,:])
        Ex_t.append(Ex[ixtest,iytest,:])
        Ey_t.append(Ey[ixtest,iytest,:]) 
        Bx_t.append(Bx[ixtest,iytest,:])
        By_t.append(By[ixtest,iytest,:])
        rho_t.append(rho[ixtest,iytest,:])

    # Close the hdf5 files
    hf_Ez.close()
    hf_rho.close()


# Calculate simulation time
t1 = time.time()
totalt = t1-t0
print('Run terminated in %ds' %totalt)

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
         'r_cavity' : r_cav,
         'L_cavity' : L_cav,
         'r_pipe' : r_pipe,
         'L_pipe' : L_pipe,
         'sigmaz' : sigmaz,
         'xsource' : xsource,
         'ysource' : ysource,
         'ixtest' : ixtest,
         'iytest' : iytest,
         'n_adj_cells' : n_adj_cells,
        }

# Create dictionary with fields data [NOT WORKING]: saves the same value all the timesteps 
field_data = {  'Ez' : np.transpose(np.array(Ez_t)),
                'Ex' : np.transpose(np.array(Ex_t)),
                'Ey' : np.transpose(np.array(Ey_t)),
                'Bx' : np.transpose(np.array(Bx_t)),
                'By' : np.transpose(np.array(By_t)),
                'rho' : np.transpose(np.array(rho_t)),
                't' : np.array(t),
            }

# write the input dictionary to a txt using pickle module
with open(out_folder+'input_data.txt', 'wb') as handle:
  pk.dump(input_data, handle)

 # write the fields dictionary to a txt using pickle module
with open(out_folder+'field_data.txt', 'wb') as handle:
  pk.dump(field_data, handle)
