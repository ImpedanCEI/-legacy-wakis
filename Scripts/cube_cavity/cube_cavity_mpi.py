import numpy as np
import numpy.random as random
from warp import picmi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import time
import os
import scipy as sc  
from copy import copy

#get the name of the file to create the output folder
out_folder=os.path.basename("/home/edelafue/WarpBasics/Scripts/cube_cavity/cube_cavity_mpi.py").split('.')[0]

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
nx = 100
ny = 100
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
                                     method = 'Yee',
                                     cfl = 0.5,
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
N=10**5
beam_layout = picmi.PseudoRandomLayout(n_macroparticles = N, seed = 3)

sim.add_species(beam, layout=beam_layout,
                initialize_self_field = solver=='EM')

# beam sigma in time and longitudinal direction
sigmat= 1.000000e-09/16.     #changed from /4 to /16
sigmaz = sigmat*picmi.constants.c 
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
tot_nsteps = 10000
t0 = time.time()

#pre-allocate timedep variables
Ez_t=[]
Ex_t=[]
Ey_t=[]
Bx_t=[]
By_t=[]
w_fun=[]
t=[]

#create output directories
out_folder='images/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)
images_diry = out_folder+'Ey/'
if not os.path.exists(images_diry):
    os.mkdir(images_diry)
images_dirx = out_folder+'Ex/'
if not os.path.exists(images_dirx):
    os.mkdir(images_dirx)
images_dirz = out_folder+'Ez/'
if not os.path.exists(images_dirz):
    os.mkdir(images_dirz)

#creating patches for the conductors
rect1 = plt.Rectangle((xmin, ymin), #left lower corner of the rectangle
                     (L_pipe-L_cavity)/2.0, (w_cavity-w_pipe)/2.0, #length, width 
                     color='w',
                     alpha=1.0 )  
rect2 = plt.Rectangle((xmin, ymax), 
                     (L_pipe-L_cavity)/2.0, -(w_cavity-w_pipe)/2.0, #length, width 
                     color='w',
                     alpha=1.0 )  
rect3 = plt.Rectangle((xmax, ymax), 
                     -(L_pipe-L_cavity)/2.0, -(w_cavity-w_pipe)/2.0, #length, width 
                     color='w',
                     alpha=1.0 )  
rect4 = plt.Rectangle((xmax, ymin), 
                     -(L_pipe-L_cavity)/2.0, (w_cavity-w_pipe)/2.0, #length, width 
                     color='w',
                     alpha=1.0 )  

#define the integration path x2, y2
xtest=0.0   #Assumes x2,y2 of the test particle in 0,0
ytest=0.0
#---search for the index
x=np.linspace(xmin, xmax, nx)
y=np.linspace(ymin, ymax, ny)
tol=np.max(x)*1e-4                      #set tolerance
xindex=np.where(np.abs(x-xtest) < tol)  #indexes are in index[0]
yindex=np.where(np.abs(y-ytest) < tol) 
#---select the index
nxtest=xindex[0]
nytest=yindex[0]

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
    rho=beam.wspecies.get_density()    #gathers the rho (electric density) of the beam rho[nx,ny,nz]0
    #time vector
    t.append(picmi.warp.top.time)

    if picmi.warp.me == 0:  #checks if we are in processor 0. Needed for mpi simulations

        #time vector
        t.append(picmi.warp.top.time)

        #--- z direction
        Ez_t.append(Ez[nxtest,nytest,:]) #1D vector of len=tot_nsteps*nz 
        #--- x direction
        Ex_t.append(Ex[nxtest,nytest,:]) #1D vector of len=tot_nsteps*nx 
        #--- y direction
        Ey_t.append(Ey[nxtest,nytest,:]) #1D vector of len=tot_nsteps*ny 
        #append the 1D magnetic induction
        #--- x direction
        Bx_t.append(Bx[nxtest,nytest,:]) #1D vector of len=tot_nsteps*nx 
        #--- y direction
        By_t.append(By[nxtest,nytest,:]) #1D vector of len=tot_nsteps*ny
        #append the charge density at (0,0,z)
        rho_t.append(rho[nxtest, nytest, :]) #1D vector of len=tot_nsteps*ny


        #store the E field at particle position for each timestep
        #TO BE DONE. beam.wspecies.getex() only  while there is beam inside the cavity

        #2D plots of the electric field
        if n_step % 10 == 0:

            #Ez - x cut plot
            fig= plt.figure(1)
            ax=fig.gca()
            im=ax.imshow(em.gatherez()[int(ny/2),:,:], vmin=-2e6, vmax=2e6, extent=[zmin, zmax, ymin, ymax], cmap='jet')
            ax.set(title='t = ' + str(picmi.warp.top.time) + ' s',
               xlabel='z    [m]',
               ylabel='y    [m]',
               xlim=[xmin,xmax],
               ylim=[ymin,ymax])
            new_rect1=copy(rect1) #we need to create a new patch for each plot
            ax.add_patch(new_rect1)
            new_rect2=copy(rect2) 
            ax.add_patch(new_rect2)
            new_rect3=copy(rect3) 
            ax.add_patch(new_rect3)
            new_rect4=copy(rect4) 
            ax.add_patch(new_rect4)
            plt.colorbar(im, label = 'Ez    [V/m]')
            plt.tight_layout()
            plt.savefig(images_dirz + 'Ez_' + str(n_step) + '.png')
            plt.clf() 
        
            #Ey - x cut plot
            fig = plt.figure()
            plt.imshow(em.gatherey()[int(ny/2),:,:], vmin=-2e6, vmax=2e6, extent=[zmin, zmax, ymin, ymax])
            plt.xlabel('z    [m]')
            plt.ylabel('y    [m]')
            plt.colorbar(label = 'Ey    [V/m]')
            plt.title('t = ' + str(picmi.warp.top.time) + ' s')
            plt.tight_layout()
            plt.jet()
            plt.savefig(images_diry + 'Ey_' + str(n_step) + '.png')
            plt.clf() 
              
            #Ex - z cut plot
            fig2=plt.figure()
            plt.imshow(em.gatherex()[:,:,int(nz/2)], vmin=-2e6, vmax=2e6, extent=[xmin, xmax, ymin, ymax])
            plt.xlabel('x    [m]')
            plt.ylabel('y    [m]')
            plt.colorbar(label = 'Ex    [V/m]')
            plt.title('t = ' + str(picmi.warp.top.time) + ' s')
            plt.tight_layout()
            plt.jet()
            plt.savefig(images_dirx + 'Ex_' + str(n_step) + '.png')
            plt.clf()
            
#--------------------#                             
#  end of time loop  #
#--------------------#


#############################       
# Calculate wake function w #
#############################

#set up
dz=(zmax-zmin)/nz*np.ones(nz) 
WL=t[-1]*picmi.constants.c     #max wake length [m]
s=np.array(np.linspace(0, WL, tot_nsteps))/unit    #distance source-test particles [mm]

#search init time
z=np.linspace(zmin, zmax, tot_nsteps)
t_wf=(s+z)/picmi.constants.c/unit   #time for wake function [s]
tol=t[0]*1e-4                       #set tolerance
index=np.array(np.where(t-t_wf[0] > tol))   #time indexes are in index[0]

#reshape electric field
Ez=[]
Ez=np.reshape(Ez_t, (nz+1,tot_nsteps))      #array to matrix (z,t)

#Perform integral
q=1 #charge in eV

'''
wfun=np.zeros(tot_nsteps)
for i in range(tot_nsteps):  
    #print(i)
    print(Ez[0:100,i])
    wfun[i]=(1/q)*sum(Ez[0:100,i]*dz)
'''
wfun=[] 
for i in index[0]:  
    wfun.append((1/q)*np.sum(Ez[0:nz,i]*dz))
    #print(wfun)
wfun=np.array(wfun)

#plot and save result
fig2 = plt.figure(2, figsize=(6,4), dpi=200, tight_layout=True)
ax2=fig2.gca()
ax2.plot(s, wfun*unit, lw=1.5, color='g')
ax2.set(title='Longitudinal wake function',
        xlabel='s [mm]',
        ylabel='$w_{//}$')
fig2.savefig(out_folder + 'w_function' + '.png')

#dump into csv
np.savetxt("wfun.csv", wfun*unit, delimiter=",")


#############################        
#Calculate wake potential W #
#############################

#preallocate and reshape
ds=s[1]-s[0]
rho=[]
rho=np.reshape(rho_t, (nz+1,tot_nsteps))  
wpot_analytic=[]
wpot_rho=[]

#Perform the integral
for j in range(len(wfun)):  
    wpot_analytic.append((N/q)*np.sum(wfun[j]*np.exp((-z[j]**2)/(2*sigmaz))/(np.sqrt(2*np.pi)*sigmaz)*dz))
    wpot_rho.append((N/q)*np.sum(wfun[j]*rho[0:nz,j]*dz))
    #print(wpot)
wpot_analytic=np.array(wpot_analytic)
wpot_rho=np.array(wpot_rho)


#plot and save result
fig3 = plt.figure(3, figsize=(6,4), dpi=200, tight_layout=True)
ax3=fig3.gca()
ax3.plot(s, wpot_analytic*unit, lw=1.5, color='r', label="analytic")
ax3.plot(s, wpot_rho*unit, lw=1.5, color='b', label="$\rho(z,t)$")
ax3.set(title='Longitudinal wake potential',
        xlabel='s[mm]',
        ylabel='$Wp_{//}$')
ax3.legend(loc='best')
fig3.tight_layout()
fig3.savefig(out_folder + 'w_potential' + '.png')

#dump into csv
np.savetxt("wpot_analytic.csv", wpot_analytic, delimiter=",")
np.savetxt("wpot_rho.csv", wpot_rho, delimiter=",")


#######################        
# Calculate impedance #
#######################

#Using np.fff from numpy
freq_np=np.fft.fftfreq(len(s))
Zl_np=np.fft.fft(wfun)

#plot results and save
fig4 = plt.figure(4, figsize=(6,4), dpi=200, tight_layout=True)
plt.plot(freq_np, Zl_np.real, '-', color='red', lw=1.5, label='numpy.fft')
plt.xlabel('freq [GHz]')
plt.ylabel('$Z_{//}$')
plt.title('Longitudinal impedance')
plt.tight_layout()
plt.xlim((0.0, max(freq)))
plt.ylim((0.0, max(Zl_np)+0.2*max(Zl_np.real)))
plt.savefig('images/' + 'impedance' + '.png')

#dump into csv
np.savetxt("zl_np.csv", Zl_np, delimiter=",")

#Using scipy.fft [not working]
# freq_sc=sc.fft.fftfreq(len(s))
# Zl_sc=sc.fft.fft(wfun)
# np.savetxt("zl_sc.csv", Zl_sc, delimiter=",")
# plt.plot(freq_sc, Zl_sc.real, '-', color='blue', lw=1.5, label='scipy.fft')

#Calculate simulation time
t1 = time.time()
totalt = t1-t0
print('Run terminated in %ds' %totalt)

