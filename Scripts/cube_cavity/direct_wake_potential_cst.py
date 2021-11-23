'''
wake_potential_cst.py

File for postprocessing CST simulations

--- Reads the out file from cube_cavity.py with pickle module
--- Reads the cst_out file from cst_to_dict.py to obtain Ez
--- Performs the indirect integration to obtain wake potential
--- Performs the fourier transform to obtain the impedance
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

#--- read the dictionary 
with open('out_nt2000/out.txt', 'rb') as handle:
  data = pk.loads(handle.read())
  print('cube_cavity stored variables')
  print(data.keys())

#--- retrieve the variables

x=data.get('x')
y=data.get('y')
z=data.get('z')
w_cavity=data.get('w_cavity')
h_cavity=data.get('h_cavity')
w_pipe=data.get('w_pipe')
h_pipe=data.get('h_pipe')
sigmaz=data.get('sigmaz')
xtest=data.get('xtest')
ytest=data.get('ytest')
zmax=np.max(z)
zmin=np.min(z)

#--- read the cst dictionary
with open('cst/cst_out.txt', 'rb') as handle:
  cst_data = pk.loads(handle.read())
  print('cst stored variables')
  print(cst_data.keys())

#--- retrieve the variables

Ez=cst_data.get('Ez')
t=cst_data.get('t')
nz=cst_data.get('nz')
nt=cst_data.get('nt')


######################
# 	Wake potential   #
######################

#---------------------------------------
# Set up the poisson solver from PyPIC #
#---------------------------------------

#--- set up z, t, dt, dz
z=np.linspace(zmin,zmax,nz)
t=np.array(t)
dz=z[2]-z[1]
dt=t[2]-t[1]

init_time=(3*sigmaz)/c #time when the center of the bunch enters the cavity
dh=x[2]-x[1]	#resolution in the transversal plane

#--- set Wake_length, s
Wake_length=nt*dt*c - (zmax-zmin)
print('Max simulated time = '+str(round(t[-1]*1.0e9,4))+' ns')
print('Wake_length = '+str(Wake_length*1e3)+' mm')
ns_neg=int(0/c/dt)		#obtains the length of the negative part of s
ns_pos=int(Wake_length/(dt*c))	#obtains the length of the positive part of s
s=np.linspace(0, 0, ns_neg) #sets the values for negative s
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
iz_l1=int((-l1-z_interp[0])/dz_interp)
iz_l2=int((l2-z_interp[0])/dz_interp)

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
    for k in range(iz_l1, iz_l2):
        t_s[k,n]=(z_interp[k]+s[n])/c-zmin/c-t[0]

        if t_s[k,n]>0.0:
            it=int(t_s[k,n]/dt)                             #find index for t
            Wake_potential[n]=Wake_potential[n]+(Ez_interp[k, it])*dz_interp   #compute integral


#--- plot wake potential
q=(1e-9)*1e12 # charge of the particle beam in pC
fig1 = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig1.gca()
ax.plot(s*1.0e3, Wake_potential/q, lw=1.2, color='orange', label='W_//(s)')
ax.set(title='Longitudinal Wake potential',
        xlabel='s [mm]',
        ylabel='$W_{//}$ [V/pC]',
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()



##########################
#   CST Wake potential   #
##########################

#--- read wake potential obtained from CST

Wake_potential_cst=[]
s_cst=[]
i=0

fname='Wake_potential'
with open('cst/cst_files/'+fname+'.txt') as f:
    for line in f:
        i+=1
        content = f.readline()
        columns = content.split()

        if i>1 and len(columns)>1:

            Wake_potential_cst.append(float(columns[1]))
            s_cst.append(float(columns[0]))

Wake_potential_cst=np.array(Wake_potential_cst) # in V/pC
s_cst=np.array(s_cst)  # in [mm]

#--- Plot comparison

q=(1e-9)*1e12 # charge of the particle beam in pC
fig4 = plt.figure(4, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig4.gca()
ax.plot(s*1.0e3+min(s_cst), Wake_potential/q, lw=1.2, color='orange', label='W_//(s) direct integration')
ax.plot(s_cst, Wake_potential_cst, lw=1.3, color='black', ls='--', label='W_//(s) CST')
ax.set(title='Longitudinal Wake potential',
        xlabel='s [mm]',
        ylabel='$W_{//}$ [V/pC]',
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

#--- Plot normalized comparison comparison

fig5 = plt.figure(5, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig5.gca()
ax.plot(s*1.0e3+min(s_cst), Wake_potential/np.max(Wake_potential), lw=1.2, color='orange', label='W_//(s) direct integration')
ax.plot(s_cst, Wake_potential_cst/np.max(Wake_potential_cst), lw=1.3, color='orange', ls='--', label='W_//(s) CST')
ax.set(title='Longitudinal Wake potential',
        xlabel='s [mm]',
        ylabel='$W_{//}$ [V/pC]',
        ylim=(-1.5,1.5)
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

