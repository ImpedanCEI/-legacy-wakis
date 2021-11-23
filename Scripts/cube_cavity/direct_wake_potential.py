'''
direct_wake_potential.py

File for postprocessing warp simulations

--- Reads the out file with pickle module
--- Performs the direct integration to obtain wake potential
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

#--- initialize variables
Ez_dt=np.zeros((nt,nt))  #time derivative of Ez
Ez_dz=np.zeros((nt,nt))  #z spatial derivative of Ez
t_s = np.zeros((nt, len(s)))


#-----------------------#
#      Obtain W(s)      #
#-----------------------#

# s loop -------------------------------------#                                                           

for n in range(len(s)-1):    

    
    #--------------------------------#
    # integral between zmin and zmax #
    #--------------------------------#

    #integral of (Ez(xtest, ytest, z, t=(s+z)/c))dz
    k=0
    for k in range(0, nz):
    	t_s[k,n]=(z_interp[k]+s[n])/c-zmin/c-t[0]

    	if t_s[k,n]>0.0:
	        it=int(t_s[k,n]/dt)                 			#find index for t
	        Wake_potential[n]=Wake_potential[n]+(Ez_interp[k, it])*dz_interp   #compute integral


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

