import numpy as np
import numpy.random as random
from warp import picmi
import matplotlib.pyplot as plt
import time
import os

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
L_cavity = 20*unit 

##################################
# Define the mesh
##################################
# mesh cells per direction
nx = 50
ny = 50
nz = 50

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
                     particle_shape = 'linear',
                     name = 'beam')

##################################
# Setup the geometry
##################################
domain = picmi.warp.Box(1.5*(xmax-xmin), 1.5*(ymax-ymin), 1.5*(zmax-zmin))
pipe = picmi.warp.Box(w_pipe, h_pipe, 2*L_pipe)
cavity = picmi.warp.Box(w_cavity, h_cavity, L_cavity)

conductors = domain - pipe - cavity

##################################
# Setupe the grid
##################################
lower_boundary_conditions = ['open', 'open', 'open']
upper_boundary_conditions = ['open', 'open', 'open']

# instantiate the grid
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
                                     method = 'Yee',
                                     cfl = 1.,
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
beam_layout = picmi.PseudoRandomLayout(n_macroparticles = 10**5, seed = 3)

sim.add_species(beam, layout=beam_layout,
                initialize_self_field = solver=='EM')

# beam sigma in time and longitudinal direction
sigmat= 1.000000e-09/4.
sigmaz = sigmat*299792458
# transverse sigmas.
sigmax = 2e-4
sigmay = 2e-4

# spacing between bunches
b_spac = 25e-9
# offset of the bunch centroid
t_offs = 2.39e-8 + 5.85e-10
# number of bunches to simulate
n_bunches = 1

# beam energy
beam_gamma = 479.
beam_uz = beam_gamma*picmi.constants.c

# macroparticle info
bunch_physical_particles  = 2.5e11
bunch_w = 1e8
bunch_macro_particles = bunch_physical_particles/bunch_w


bunch_rms_size            = [sigmax, sigmay, sigmaz]
bunch_rms_velocity        = [0.,0.,0.]
bunch_centroid_position   = [0,0,0.95*zmin]
bunch_centroid_velocity   = [0.,0.,beam_uz]

# time profile of a gaussian beam
def time_prof(t):
    val = 0
    sigmat = sigmaz/picmi.clight
    for i in range(1,n_bunches+1):
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

#set up the imnjection
picmi.warp.installuserinjection(nonlinearsource)

# define shortcuts
pw = picmi.warp
pw.winon()
em = solver.solver
step=pw.step


##################################
# Perform Simulation
##################################
tot_nsteps = 1000
t0 = time.time()

images_dir = 'images/'
if not os.path.exists(images_dir):
    os.mkdir(images_dir)

for n_step in range(tot_nsteps):
    picmi.warp.step()
    if n_step % 10 == 0:
        plt.imshow(em.gatherey()[int(ny/2),:,:], vmin=-2e6, vmax=2e6, 
                   extent=[zmin, zmax, ymin, ymax])
        plt.xlabel('z    [m]')
        plt.ylabel('y    [m]')
        plt.colorbar(label = 'Ey    [V/m]')
        plt.title('t = ' + str(picmi.warp.top.time) + ' s')
        plt.tight_layout()
        plt.savefig(images_dir + 'Ey_' + str(n_step) + '.png')
        plt.clf()

t1 = time.time()
totalt = t1-t0
print('Run terminated in %ds' %totalt)
