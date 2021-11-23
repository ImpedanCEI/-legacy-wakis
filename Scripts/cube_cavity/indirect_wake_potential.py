'''
wake_potential.py

File for postprocessing warp simulations

--- Reads the out file with pickle module
--- Performs the indirect integration to obtain wake potential
--- Performs the fourier trnasform to obtain the impedance
--- Plots the results

'''

import numpy as np
from warp import picmi
import matplotlib.pyplot as plt
import time
import sys
import os
import scipy as sc  
from copy import copy
import pickle as pk

#to be able to use pyPIC the BIN path is needed
BIN = os.path.expanduser("/home/edelafue/PyCOMPLETE")
if BIN not in sys.path:
    sys.path.append(BIN)

import PyPIC.geom_impact_poly as poly 
import PyPIC.FiniteDifferences_Staircase_SquareGrid as PIC_FD

c=sc.constants.c

#--- to read the dictionary type
with open('out_nt2000/out.txt', 'rb') as handle:
  data = pk.loads(handle.read())
  print('stored variables')
  print(data.keys())

#--- retrieve the variables

Ez_t=data.get('Ez')
Ex_t=data.get('Ex')
Ey_t=data.get('Ey')
Bx_t=data.get('Bx')
By_t=data.get('By')
rho_t=data.get('rho')
x=data.get('x')
y=data.get('y')
z=data.get('z')
w_cavity=data.get('w_cavity')
h_cavity=data.get('h_cavity')
w_pipe=data.get('w_pipe')
h_pipe=data.get('h_pipe')
t=data.get('t')
nt=data.get('nt')
nz=data.get('nz')
sigmaz=data.get('sigmaz')
xtest=data.get('xtest')
ytest=data.get('ytest')

#reshape electric field
Ez=[]
Ez=np.reshape(Ez_t, (nz+1,nt))      #array to matrix (z,t)

######################
# 	Wake potential   #
######################

#---------------------------------------
# Set up the poisson solver from PyPIC #
#---------------------------------------

#--- set up z, t, dt, dz
z=np.array(z)
t=np.array(t)
dz=z[2]-z[1]
dt=t[2]-t[1]
zmax=np.max(z)
zmin=np.min(z)
init_time=(3*sigmaz)/c #time when the center of the bunch enters the cavity
dh=x[2]-x[1]	#resolution in the transversal plane

#--- set Wake_length, s
Wake_length=nt*dt*c - (zmax-zmin)
print('Max simulated time = '+str(round(t[-1]*1.0e9,4))+' ns')
print('Wake_length = '+str(Wake_length*1e3)+' mm')
ns_neg=int(6*sigmaz/c/dt)		#obtains the length of the negative part of s
ns_pos=int(Wake_length/(dt*c))	#obtains the length of the positive part of s
s=np.linspace(-6*sigmaz, 0, ns_neg) #sets the values for negative s
s=np.append(s, np.linspace(0, Wake_length,  ns_pos))

#--- initialize Wp 
Wake_potential=np.zeros_like(s)

#--- interpolate Ez so nz == nt
z_interp=np.linspace(zmin, zmax, nt)
Ez_interp=np.zeros((nt,nt))
dz_interp=z_interp[2]-z_interp[1]
n=0
for n in range(nt):
    Ez_interp[:, n]=np.interp(z_interp, z, Ez[:, n])

#--- define the limits for the poisson and the integral
l1=(w_cavity/2.0)         #[m]
l2=(w_cavity/2.0)         #[m] 
b1=h_pipe/2.0             #[m]
b2=h_pipe/2.0             #[m]
# define the rectangle size (aperture = half the area)
w_rect = w_pipe/2.0
h_rect = h_pipe/2.0
# find indexes for l1, l2, b1, b2
iz_l1=int((-l1-z_interp[0])/dz_interp)
iz_l2=int((l2-z_interp[0])/dz_interp)

#--- initialize variables
Ez_dt=np.zeros((nt,nt))  #time derivative of Ez
Ez_dz=np.zeros((nt,nt))  #z spatial derivative of Ez
phi_l1 = np.zeros((int((w_rect*2)/dh + 10 + 1), int((h_rect*2)/dh + 8 + 1), len(s)))  
phi_l2 = np.zeros((int((w_rect*2)/dh + 10 + 1), int((h_rect*2)/dh + 8 + 1), len(s)))  
integral = np.zeros(len(s))  #integral of ez between -l1, l2
t_s = np.zeros((nt, len(s)))

#--- obtain the derivatives
n=0
k=0
for n in range(nt-1):
    for k in range(nt-1):
        Ez_dz[k,n]= (Ez_interp[k+1, n] - Ez_interp[k, n])/dz_interp
        Ez_dt[k,n]= (Ez_interp[k, n+1] - Ez_interp[k, n])/dt


# s loop -------------------------------------#                                                           

for n in range(len(s)-1):    

    #-----------------------#
    # first poisson z=(-l1) #
    #-----------------------#

    t_l1=(-l1+s[n])/c-zmin/c-t[0]

    #--- obtain the rectangle size for -l1 
    # define the rectangle size (aperture = half the area)
    w_rect = w_pipe/2.0
    h_rect = h_pipe/2.0

    # PyPIC function to declare the implicit function for the conductors (this acts as BCs)
    PyPIC_chamber = poly.polyg_cham_geom_object({'Vx' : np.array([w_rect, -w_rect, -w_rect, w_rect]),
                                                 'Vy': np.array([h_rect, h_rect, -h_rect, -h_rect]),
                                                 'x_sem_ellip_insc' : 0.99*w_rect, #important to add this
                                                 'y_sem_ellip_insc' : 0.99*h_rect})
    # solver object
    picFD = PIC_FD.FiniteDifferences_Staircase_SquareGrid(chamb = PyPIC_chamber, Dh = dh, sparse_solver = 'PyKLU')
    phi_l1[:, :, n] = np.zeros_like(picFD.rho) 

    if t_l1 > 0.0:
        # find t index for -l1 and s[n]
        it_l1=int(t_l1/dt)

        # define the left side of the laplacian (driving term rho = 1/c*dEz/dt-dEz/dz)
        # rho = np.ones_like(picFD.rho)                                         #test rho to check the solver. Rho needs to be two dimensions?
        rho = (Ez_dt[iz_l1,it_l1]/c - Ez_dz[iz_l1, it_l1])*np.ones_like(picFD.rho)       #this rho is a constant evaluated at z=-l1, t=(s-l1)/c

        # solve the laplacian and obtain phi(0,0)
        picFD.solve(rho = rho)      #the dimensions are selected by pyPIC solver
        phi_l1[:, :, n] = picFD.phi.copy()

    #-----------------------#
    # second poisson z=(l2) #
    #-----------------------#

    t_l2=(l2+s[n])/c-zmin/c-t[0]

    #--- obtain the rectangle size for l2 
    # define the rectangle size (aperture = half the area)
    w_rect = w_pipe/2.0
    h_rect = h_pipe/2.0

    # PyPIC function to declare the implicit function for the conductors (this acts as BCs)
    PyPIC_chamber = poly.polyg_cham_geom_object({'Vx' : np.array([w_rect, -w_rect, -w_rect, w_rect]),
                                                 'Vy': np.array([h_rect, h_rect, -h_rect, -h_rect]),
                                                 'x_sem_ellip_insc' : 0.99*w_rect, #important to add this
                                                 'y_sem_ellip_insc' : 0.99*h_rect})
    # solver object
    picFD = PIC_FD.FiniteDifferences_Staircase_SquareGrid(chamb = PyPIC_chamber, Dh = dh, sparse_solver = 'PyKLU')
    phi_l1[:, :, n] = np.zeros_like(picFD.rho) 

    if t_l2>0.0:

        # find t index for l2 and s[n]
        it_l2=int(t_l2/dt)
        # define the left side of the laplacian (driving term rho = 1/c*dEz/dt-dEz/dz)
        # rho = np.ones_like(picFD.rho)                         #test rho to check the solver. Rho needs to be two dimensions?
        rho = (Ez_dt[iz_l2, it_l2]/c - Ez_dz[iz_l2, it_l2])*np.ones_like(picFD.rho)         #this rho is a constant size[nx,ny] evaluated at z=l2, t=(s+l2)/c

        # solve the laplacian and obtain phi(xtest,ytest)
        picFD.solve(rho = rho)      #rho has to be a matrix
        phi_l2[:, :, n] = picFD.phi.copy()   #phi[n] size [pyPIC_nx,pyPIC_ny]


    #-----------------------------#
    # integral between -l1 and l2 #
    #-----------------------------#

    #integral of (Ez(xtest, ytest, z, t=(s+z)/c))dz
    k=0
    for k in range(iz_l1, iz_l2):
    	t_s[k,n]=(z_interp[k]+s[n])/c-zmin/c-t[0]

    	if t_s[k,n]>0.0:
	        it=int(t_s[k,n]/dt)                 			#find index for t
	        integral[n]=integral[n]+(Ez_interp[k, it])*dz_interp   #compute integral


    #-----------------------#
    #      Obtain W(s)      #
    #-----------------------#

    # Define phi(x, y=b1) 
    # Phis indexes go from x(-w_rect,w_rect) and y(-h_rect,h_rect)
    # x direction is gridded with (w_rect*2)/dh + 10 ghost cells
    # y direction is gridded with (h_rect*2)/dh + 8 ghost cells
    # +info see: class PyPIC_Scatter_Gather(object):
    iy_b1=int((b1-picFD.bias_y)/dh)     
    iy_b2=int((b2-picFD.bias_y)/dh)
    ixtest_phi=int((xtest-picFD.bias_y)/dh)
    iytest_phi=int((ytest-picFD.bias_y)/dh)

    phi_b1=phi_l1[ixtest_phi, iy_b1, n] #phi(x=0, y=b1, s[n])
    phi_b2=phi_l2[ixtest_phi, iy_b1, n] #phi(x=0, y=b1, s[n])

    Wake_potential[n]=(phi_b1-phi_l1[ixtest_phi, iytest_phi, n])-integral[n]+(phi_l2[ixtest_phi, iytest_phi, n]-phi_b2)


#--- plot wake potential
q=1.6022e-7 #1 e- charge in pC
fig1 = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig1.gca()
ax.plot(s*1.0e3, Wake_potential*q, lw=1.2, color='orange', label='W_//(s)')
ax.set(title='Longitudinal Wake potential',
        xlabel='s [mm]',
        ylabel='$W_{//}$ [V/pC]',
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

#--- plot integral l1, l2

fig2 = plt.figure(2, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig2.gca()
ax.plot(s*1.0e3, integral*q, lw=1.2, color='r', label='W_//(s)')
ax.set(title='Longitudinal Wake potential - Integral(l1, l2)',
        xlabel='s [mm]',
        ylabel='$ W_{//} $ integral (l1, l2) [V/pC]',
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

#--- plot phi

fig3 = plt.figure(3, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig3.gca()
ax.plot(s*1.0e3, phi_l1[ixtest_phi, iytest_phi, :]*q, lw=1, color='r', label='Phi_l1(s)')
ax.plot(s*1.0e3, (phi_b1-phi_l1[ixtest_phi, iytest_phi, :])*q, lw=1, color='pink', label='Phi_b1-Phi_l1(s)', ls='--')
ax.plot(s*1.0e3, phi_l2[ixtest_phi, iytest_phi, :]*q, lw=1, color='b', label='Phi_l2(s)')
ax.plot(s*1.0e3, (phi_l2[ixtest_phi, iytest_phi, :]-phi_b2)*q, lw=1, color='cyan', label='Phi_l2(s)-Phi_b2', ls='--')
ax.set(title='Poisson solver results',
        xlabel='s [mm]',
        ylabel='$ Phi $ [V/pC]',
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()
