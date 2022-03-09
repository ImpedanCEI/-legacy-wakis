#! /usr/bin/env python

'''
File for postprocessing warpx simulations

--- Reads the out file with pickle module
--- Reads the dicts files with pickle module
--- Plots the Electric field and charge distribution
--- Compares with CST

'''

import yt
import os, sys
from scipy.constants import mu_0, pi, c, epsilon_0, e
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pickle
import h5py

out_folder='out/'

#------------------------------------#
#            1D variables            #
#------------------------------------#

# Read the input data dictionary
with open(out_folder+'input_data.txt', 'rb') as handle:
   input_data = pickle.loads(handle.read())
   print(input_data.keys())

#--- retrieve variables
x=input_data.get('x')
y=input_data.get('y')
z=input_data.get('z')
tot_nsteps=input_data.get('tot_nsteps')
init_time=input_data.get('init_time')
nx=input_data.get('nx')
ny=input_data.get('ny')
nz=input_data.get('nz')
w_cavity=input_data.get('w_cavity')
h_cavity=input_data.get('h_cavity')
L_cavity=input_data.get('L_cavity')
w_pipe=input_data.get('w_pipe')
h_pipe=input_data.get('h_pipe')
L_pipe=input_data.get('L_pipe')
sigmaz=input_data.get('sigmaz')
xsource=input_data.get('xsource')
ysource=input_data.get('ysource')
ixtest=input_data.get('ixtest')
iytest=input_data.get('iytest')

t=input_data.get('t')

#--- auxiliary variables
zmin=min(z)
zmax=max(z)
xmin=min(x)
xmax=max(x)
ymin=min(y)
ymax=max(y)
dx=x[2]-x[1]
dy=y[2]-y[1]
dz=z[2]-z[1]
dt=t[2]-t[1]

if len(z) > 128:
    x=np.linspace(xmin, xmax, nx)
    y=np.linspace(ymin, ymax, ny)
    z=np.linspace(zmin, zmax, nz)
    dx=x[2]-x[1]
    dy=y[2]-y[1]
    dz=z[2]-z[1]


# Read the fields data dictionary
with open(out_folder+'field_data.txt', 'rb') as handle:
   field_data = pickle.loads(handle.read())
   print(input_data.keys())

#--- retrieve variables len=[ncells x tot_nsteps]
Ez=field_data.get('Ez')
Ex=field_data.get('Ex')
Ey=field_data.get('Ey')
Bx=field_data.get('Bx')
By=field_data.get('By')
rho=field_data.get('rho') #[C/m3]

#--- auxiliary variables
charge_dist=rho*dx*dy # [C/m]


# Retrieve CST data from dictionary
cst_path='/mnt/c/Users/elefu/Documents/CERN/GitHub/PyWake/Scripts/CST/'
#--- read the cst dictionary
with open(cst_path+'cst_out.txt', 'rb') as handle:
  cst_data = pickle.loads(handle.read())
  print('cst stored variables')
  print(cst_data.keys())

#---Electric field
Ez_cst=cst_data.get('Ez')
t_cst = cst_data.get('t')
nz_cst=cst_data.get('nz')
nt_cst=cst_data.get('nt')
#---Charge distribution
charge_dist_cst=cst_data.get('charge_dist')
charge_dist_time=cst_data.get('charge_dist_time')
charge_dist_spectrum=cst_data.get('charge_dist_spectrum')
current=cst_data.get('current')
s_charge_dist=cst_data.get('s_charge_dist')
t_charge_dist=cst_data.get('t_charge_dist')
#---Wake potential
Wake_potential_cst=cst_data.get('Wake_potential_cst')
Wake_potential_interfaces=cst_data.get('Wake_potential_interfaces')
Wake_potential_testbeams=cst_data.get('Wake_potential_testbeams')
WPx_cst=cst_data.get('WPx_cst')
WPy_cst=cst_data.get('WPy_cst')
s_cst=cst_data.get('s_cst')
#---Impedance
Z_cst=cst_data.get('Z_cst')
Zx_cst=cst_data.get('Zx_cst')
Zy_cst=cst_data.get('Zy_cst')
freq_cst=cst_data.get('freq_cst')

#-------------------#
#       MOVIE       #
#-------------------#

'''
#--- loop with countours of the charge density
plt.ion()
for n in range(500):
    if n % 1 == 0:
        #--- Plot rho along z axis 
        fig2 = plt.figure(20, figsize=(6,4), dpi=200, tight_layout=True)
        ax=fig2.gca()
        ax.plot(np.array(z)*1.0e3, charge_dist, lw=1.2, color='r', label='Charge density from warp')
        ax.set(title='Charge density in t='+str(round(t[n]*1e9,2))+' ns | timestep '+str(n),
                xlabel='z [mm]',
                ylabel='$\lambda$(z) [C/m] ',
                xlim=(min(z*1e3)+3,max(z*1e3)),
                ylim=(0,2.2e-8) #max CST
                )
        ax.legend(loc='best')
        fig2.canvas.draw()
        fig2.canvas.flush_events()
        fig2.clf()
plt.close()
'''

#----------------------#
#   Compare with CST   #
#----------------------#

#--- Plot electric field Ez on axis 
fig1 = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig1.gca()
ax.plot(t*1.0e9, Ez[nz//2, :], color='g', label='Ez Warpx')
ax.plot(np.array(t_cst)*1.0e9, Ez_cst[int(nz_cst/2), :], lw=0.8, color='black', ls='--',label='Ez CST')
ax.set(title='Electric field at cavity center',
        xlabel='t [ns]',
        ylabel='E [V/m]',         
        ylim=(np.min(Ez_cst[int(nz_cst/2), :])*1.1,np.max(Ez_cst[int(nz_cst/2), :])*1.1),
        #xlim=(0,np.minimum(np.max(np.array(t[n])*1.0e9),np.max(t_cst*1.0e9)))
                )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

#--- Plot transverse electric field Ex Ey on axis 
fig10 = plt.figure(10, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig10.gca()
ax.plot(t*1.0e9, Ex[nx//2, :], color='r', label='Ex Warpx')
ax.plot(t*1.0e9, Ey[ny//2, :], color='b', label='Ey Warpx')
# No transverse field data from CST
# ax.plot(np.array(t_cst)*1.0e9, Ez_cst[int(nz_cst/2), :], lw=0.8, color='black', ls='--',label='Ez CST')
ax.set(title='Electric field at cavity center',
        xlabel='t [ns]',
        ylabel='E [V/m]',         
        ylim=(np.min(Ex[int(nx/2), :])*1.1,np.max(Ex[int(nx/2), :])*1.1),
        #xlim=(0,np.minimum(np.max(np.array(t[n])*1.0e9),np.max(t_cst*1.0e9)))
                )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

#--- Plot transverse magnetic field Bx By on axis 
fig10 = plt.figure(10, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig10.gca()
ax.plot(t*1.0e9, Bx[nx//2, :], color='m', label='Bx Warpx')
ax.plot(t*1.0e9, By[ny//2, :], color='c', label='By Warpx')
# No transverse field data from CST
# ax.plot(np.array(t_cst)*1.0e9, Ez_cst[int(nz_cst/2), :], lw=0.8, color='black', ls='--',label='Ez CST')
ax.set(title='Electric field at cavity center',
        xlabel='t [ns]',
        ylabel='E [V/m]',         
        ylim=(np.min(Bx[int(nx/2), :])*1.1,np.max(Bx[int(nx/2), :])*1.1),
        #xlim=(0,np.minimum(np.max(np.array(t[n])*1.0e9),np.max(t_cst*1.0e9)))
                )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

#--- Plot charge distribution (time)
fig2 = plt.figure(2, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig2.gca()
ax.plot(np.array(t)*1.0e9, charge_dist[nz//2, :]*c, color='r', label='$\lambda$(t) Warpx')
ax.plot(np.array(t_charge_dist)*1.0e9+0.2, charge_dist_time, lw=0.8, color='black', ls='--',label='$\lambda$(t) CST') #correct with -0.2
ax.set(title='Charge distribution at cavity center',
        xlabel='t [ns]',
        ylabel='$\lambda$(t) [C/s]',         
        ylim=(np.min(charge_dist_time)*1.1,np.max(charge_dist_time)*1.1),
        xlim=(0,np.minimum(np.max(np.array(t)*1.0e9),np.max(t_charge_dist*1.0e9)))
                )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

#--- Plot charge distribution (time) | normalized
fig3= plt.figure(3, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig3.gca()
factor=max(charge_dist_time)/max(charge_dist[nz//2, :]*c) #normalizing factor
ax.plot(np.array(t)*1.0e9, charge_dist[nz//2, :]*c*factor, color='r', label='$\lambda$(t) Warpx')
ax.plot(np.array(t_charge_dist)*1.0e9+0.2, charge_dist_time, lw=0.8, color='black', ls='--',label='$\lambda$(t) CST') #correct with -0.2
ax.set(title='Charge distribution at cavity center (normalizing factor = '+str(factor)+')',
        xlabel='t [ns]',
        ylabel='$\lambda$(t) [C/s]',         
        ylim=(np.min(charge_dist_time)*1.1,np.max(charge_dist_time)*1.1),
        xlim=(0,np.minimum(np.max(np.array(t)*1.0e9),np.max(t_charge_dist*1.0e9)))
                )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

#--- Plot charge distribution (distance) | normalized
fig4 = plt.figure(4, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig4.gca()
timestep=int((z[nz//2]/c+init_time+3*sigmaz/c)/dt)+1
ax.plot(z*1e3, charge_dist[:, timestep]*factor, color='r', label='$\lambda$(t) Warpx')
ax.plot(s_charge_dist*1e3, charge_dist_cst, lw=0.8, color='black', ls='--',label='$\lambda$(t) CST') #correct with -0.2
ax.set(title='Charge distribution at cavity center (normalizing factor = '+str(factor)+')',
        xlabel='z [mm]',
        ylabel='$\lambda$(z) [C/m]',         
        ylim=(np.min(charge_dist_cst)*1.1,np.max(charge_dist_cst)*1.1),
        xlim=(zmin,zmax)
                )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()
