from pywarpx import picmi
import numpy as np
from pywarpx import libwarpx, fields, callbacks
from scipy.constants import c, m_p, e
import numpy.random as random
import matplotlib.pyplot as plt


max_steps = 1200
unit = 1e-3

##################################
# Define the geometry
##################################

# width of the rectangular beam pipe (x direction)
w_pipe = 15*unit
# height of the rectangular beam pipe (y direction)
h_pipe = 15*unit
# length of the pipe (bigger than the domain to resemble infinite length)
L_pipe = 100*unit

# width of the rectangular beam pipe (x direction)
w_cav = 50*unit
# height of the rectangular beam pipe (y direction)
h_cav = 50*unit
# length of the pipe (bigger than the domain to resemble infinite length)
L_cav = 75*unit


##################################
# Define the mesh
##################################
# mesh cells per direction
nx = 64
ny = 64
nz = 128

# mesh bounds for domain
xmin = -32*unit
xmax = 32*unit
ymin = -32*unit
ymax = 32*unit
zmin = -64*unit
zmax = 64*unit

dz = (zmax-zmin)/nz

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
    warpx_max_grid_size = 50
)


flag_correct_div = False
flag_correct_div_pml = False
solver = picmi.ElectromagneticSolver(grid=grid, method='Yee', cfl=1.,
                                     divE_cleaning = flag_correct_div,
                                     pml_divE_cleaning = flag_correct_div_pml,
                                     warpx_do_pml_in_domain = True,
                                     warpx_pml_has_particles = True,
                                     warpx_do_pml_j_damping = False)

embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function="w=w_pipe+(w_cav-w_pipe)*(z<L_cav/2)*(z>-L_cav/2); h=h_pipe+(h_cav-h_pipe)*(z<L_cav/2)*(z>-L_cav/2); max(max(x-w/2,-w/2-x),max(y-h/2,-h/2-y))",
    w_cav=w_cav, 
    h_cav=h_cav, 
    L_cav=L_cav, 
    w_pipe=w_pipe, 
    h_pipe=h_pipe, 
)

##########################
# diagnostics
##########################
field_diag = picmi.FieldDiagnostic(
    name = 'diag1',
    grid = grid,
    period = 10,
    data_list = ['E', 'B'],
    write_dir = 'diags',
    warpx_file_prefix = 'gaussian_beam_square_cav_plt'
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

sim.add_diagnostic(field_diag)

beam_layout = picmi.PseudoRandomLayout(n_macroparticles = 0)

sim.add_species(beam, layout=beam_layout)

sim.initialize_inputs()

#sim.write_input_file(file_name = 'inputs_from_PICMI_good')
#sim.initialize_warpx(mpi_comm=new_comm)

##################################
# Setup the beam injection
##################################

# beam sigma in time and longitudinal direction
sigmat= 1.000000e-09/16.     #changed from /4 to /16
sigmaz = 18.8e-3 #sigmat*picmi.constants.c         #[m]
# transverse sigmas.
sigmax = 2e-4
sigmay = 2e-4

# spacing between bunches
b_spac = 25e-9
# offset of the bunch centroid
t_offs = -5*sigmat #2.39e-8 + 5.85e-10
# number of bunches to simulate
n_bunches = 1

# beam energy
beam_gamma = 479.
beam_uz = beam_gamma*picmi.constants.c

# macroparticle info
bunch_physical_particles  = 2.5e11
bunch_w = 1e6
bunch_macro_particles = bunch_physical_particles/bunch_w


bunch_rms_size            = [sigmax, sigmay, sigmaz]
bunch_rms_velocity        = [0.,0.,0.]
bunch_centroid_position   = [0,0,zmin+5*dz] #Always inject in the middle of the PML
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

sim.step(max_steps)
