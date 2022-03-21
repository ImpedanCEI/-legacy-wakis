#! /usr/bin/env python

'''
File for postprocessing warpx simulations

--- Reads the out file with pickle module
--- Reads the diags files with yt module
--- Plots the Electric field in the longitudinal direction
--- Obtains the frequency of the Electric field

'''

import yt
import os, sys
from scipy.constants import mu_0, pi, c, epsilon_0, e
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
import pickle
import h5py

out_folder='runs/out/'
#out_folder='runs/out_cub_cav_quadrupolar/'
flag_rho_3d = False

#------------------------------------#
#            1D variables            #
#------------------------------------#

# Read the input dictionary
with open(out_folder+'input_data.txt', 'rb') as handle:
   input_data = pickle.loads(handle.read())
   print(input_data.keys())

#--- retrieve variables
x=input_data.get('x')
y=input_data.get('y')
z=input_data.get('z')
nt=input_data.get('nt')
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


#------------------------------------#
#            3D variables            #
#------------------------------------#

#--- read the Ez.h5 file
hf_Ez = h5py.File(out_folder +'Ez.h5', 'r')
print('reading the h5 file '+ out_folder +'Ez.h5')
print('size of the file: '+str(round((os.path.getsize(out_folder+'Ez.h5')/10**9),2))+' Gb')
#get number of datasets
size_hf=0.0
dataset=[]
n_step=[]
for key in hf_Ez.keys():
    size_hf+=1
    dataset.append(key)
    n_step.append(int(key.split('_')[1]))

Ez_0=hf_Ez.get(dataset[0])
shapex=Ez_0.shape[0]  
shapey=Ez_0.shape[1] 
shapez=Ez_0.shape[2] 
z_Ez=z[len(z)//2-shapez//2:len(z)//2+shapez//2+1]
print('Ez field is stored in matrices '+str(Ez_0.shape)+' in '+str(int(size_hf))+' datasets')

# Extract field on axis Ez (z,t)
Ez_t=[]
Ez_t1=[]
Ez_t2=[]
Ez_t3=[]
for n in range(nt):
    Ez=hf_Ez.get(dataset[n]) # [V/m]
    Ez_t.append(np.array(Ez[shapex//2, shapey//2,:])) # [V/m]
    Ez_t1.append(np.array(Ez[-3, -3,:])) # [V/m]
    Ez_t2.append(np.array(Ez[-2, -2,:])) # [V/m]
    Ez_t3.append(np.array(Ez[-1, -1,:])) # [V/m]

Ez_t=np.transpose(np.array(Ez_t))
Ez_t1=np.transpose(np.array(Ez_t1))
Ez_t2=np.transpose(np.array(Ez_t2))
Ez_t3=np.transpose(np.array(Ez_t3))

#--- read the rho.h5 file
hf_rho = h5py.File(out_folder +'rho.h5', 'r')
print('reading the h5 file '+ out_folder +'rho.h5')
print('size of the file: '+str(round((os.path.getsize(out_folder+'rho.h5')/10**9),2))+' Gb')
#get number of datasets
size_rho_hf=0.0
dataset_rho=[]
n_step_rho=[]
for key in hf_rho.keys():
    size_rho_hf+=1
    dataset_rho.append(key)
    n_step_rho.append(int(key.split('_')[1]))

rho_0=hf_rho.get(dataset_rho[0]) #[C/m3]

print('Charge distribution map is stored in array of length '+str(rho_0.shape[0])+' in '+str(int(size_rho_hf))+' datasets')

# Extract charge distribution [C/m] lambda(z,t)
charge_dist=[]
for n in range(nt):
    rho=hf_rho.get(dataset_rho[n]) # [C/m3]
    if flag_rho_3d:
        rho_nx, rho_ny, rho_nz = rho.shape
        rho=rho[rho_nx//2, rho_ny//2, :]
    charge_dist.append(np.array(rho)*dx*dy) # [C/m]

charge_dist=np.transpose(np.array(charge_dist)) # [C/m]

# Create output dictionary
if os.path.exists(out_folder+'field_data.txt'):
    with open(out_folder+'field_data.txt', 'rb') as handle:
       field_data = pickle.loads(handle.read())
else:
    field_data = {}
#---add the new entries
field_data['Ez']=Ez_t
field_data['charge_dist']=charge_dist
#---update the dictionary
with open(out_folder+'field_data.txt', 'wb') as handle:
    pickle.dump(field_data, handle)

#-------------------#
#     2D  Plots     #
#-------------------#

'''
#--- loop with countours of electric field
plt.ion()
for n in range(nt):
    if n % 1 == 0:
        Ez=hf_Ez.get(dataset[n])
        #--- Plot Ez - x cut 
        fig1 = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
        ax=fig1.gca()
        im=ax.imshow(Ez[int(shapex/2),:,:], vmin=-3.e4, vmax=3.e4, extent=[zmin*1e3, zmax*1e3, ymin*1e3, ymax*1e3], cmap='jet')
        ax.set(title='Warpx Ez field, t = ' + str(round(t[n]*1e9,3)) + ' ns',
               xlabel='z    [mm]',
               ylabel='y    [mm]'
               )
        plt.colorbar(im, label = 'Ez    [V/m]')
        fig1.canvas.draw()
        fig1.canvas.flush_events()
        fig1.clf() 
plt.close()
''' 
'''
#--- loop with plot of the charge density
plt.ion()
for n in range(500):
    if n % 1 == 0:
        rho=hf_rho.get(dataset_rho[n]) # [C/m3]
        charge_dist=np.array(rho)*dx*dy # [C/m]
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

# Retrieve CST data from dictionary
cst_path='/mnt/c/Users/elefu/Documents/CERN/GitHub/WAKIS/Scripts/CST/'
#--- read the cst dictionary
with open(cst_path+'cst_out.txt', 'rb') as handle:
  cst_data = pickle.loads(handle.read())
  print('cst stored variables')
  print(cst_data.keys())

# Electric field
Ez_cst=cst_data.get('Ez')
t_cst = cst_data.get('t')
nz_cst=cst_data.get('nz')
nt_cst=cst_data.get('nt')
z_cst=cst_data.get('z')
# Charge distribution
charge_dist_cst=cst_data.get('charge_dist')
charge_dist_time=cst_data.get('charge_dist_time')
charge_dist_spectrum=cst_data.get('charge_dist_spectrum')
current=cst_data.get('current')
s_charge_dist=cst_data.get('s_charge_dist')
t_charge_dist=cst_data.get('t_charge_dist')
# Wake potential
Wake_potential_cst=cst_data.get('Wake_potential_cst')
WPx_cst=cst_data.get('WPx_cst')
WPy_cst=cst_data.get('WPy_cst')
WPx_cst=cst_data.get('WPx_cst')
WPy_cst=cst_data.get('WPy_cst')
WPx_dipolar_cst=cst_data.get('WPx_dipolar_cst')
WPy_dipolar_cst=cst_data.get('WPy_dipolar_cst')
WPx_quadrupolar_cst=cst_data.get('WPx_quadrupolar_cst')
WPy_quadrupolar_cst=cst_data.get('WPy_quadrupolar_cst')
s_cst=cst_data.get('s_cst')
# Impedance
Z_cst=cst_data.get('Z_cst')
Zx_cst=cst_data.get('Zx_cst')
Zy_cst=cst_data.get('Zy_cst')
freq_cst=cst_data.get('freq_cst')


#-------------------#
#     1D  Plots     #
#-------------------#

print('Cavity center is at z = '+str(z[len(z)//2])+'m')

#--- Plot electric Ez field on axis 
fig50 = plt.figure(50, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig50.gca()
ax.plot((np.array(t))*1.0e9, Ez_t[shapez//2, :], color='g', label='Ez Warpx')
ax.plot(np.array(t_cst)*1.0e9, Ez_cst[int(nz_cst/2), :], lw=0.8, color='black', ls='--',label='Ez CST')
ax.set(title='Electric field at cavity center',
        xlabel='t [ns]',
        ylabel='E [V/m]',         
        ylim=(np.min(Ez_cst[int(nz_cst/2), :])*1.1,np.max(Ez_cst[int(nz_cst/2), :])*1.1),
        xlim=(0,np.minimum(np.max(np.array(t)*1.0e9),np.max(t_cst*1.0e9)))
                )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

#--- Plot Electric field at cavity center 
fig50 = plt.figure(50, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig50.gca()
ax.plot((np.array(t)-9*dz/c)*1.0e9, Ez_t[shapez//2,:], color='g', label='Ez(0,0,z) Warpx')
ax.plot((np.array(t)-9*dz/c)*1.0e9, Ez_t1[shapez//2,:], color='seagreen', label='Ez(1,1,z) Warpx')
ax.plot((np.array(t)-9*dz/c)*1.0e9, Ez_t2[shapez//2,:], color='limegreen', label='Ez(2,2,z) Warpx')
ax.plot((np.array(t)-9*dz/c)*1.0e9, Ez_t3[shapez//2,:], color='springgreen', label='Ez(3,3,z) Warpx')
ax.plot(np.array(t_cst)*1.0e9, Ez_cst[int(nz_cst/2), :], lw=0.8, color='black', ls='--',label='Ez CST')
ax.set(title='Electric field at cavity center',
        xlabel='t [ns]',
        ylabel='$E [V/m]$',         
        ylim=(np.min(Ez_cst[int(nz_cst/2), :])*1.1,np.max(Ez_cst[int(nz_cst/2), :])*1.1),
        xlim=(0,np.minimum(np.max(np.array(t)*1.0e9),np.max(t_cst*1.0e9)))
                )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()


#--- Plot charge distribution
'''
fig60 = plt.figure(60, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig60.gca()
ax.plot(np.array(t)*1.0e9, charge_dist[len(z)//2, :]*c, color='r', label='$\lambda$(t) Warpx')
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
'''
#--- Plot charge distribution [normalized]
fig60 = plt.figure(60, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig60.gca()
factor=max(charge_dist_time)/max(charge_dist[len(z)//2, :]*c)
ax.plot(np.array(t)*1.0e9, charge_dist[len(z)//2, :]*c*factor, color='r', label='$\lambda$(t) Warpx')
ax.plot(np.array(t_charge_dist)*1.0e9+0.2, charge_dist_time, lw=0.8, color='black', ls='--',label='$\lambda$(t) CST') #correct with -0.2
ax.set(title='Charge distribution at cavity center (normalizing factor = '+str(round(factor,3))+')',
        xlabel='t [ns]',
        ylabel='$\lambda$(t) [C/s]',         
        ylim=(np.min(charge_dist_time)*1.1,np.max(charge_dist_time)*1.1),
        xlim=(0,np.minimum(np.max(np.array(t)*1.0e9),np.max(t_charge_dist*1.0e9)))
                )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()


fig70 = plt.figure(70, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig70.gca()
timestep=int((z[len(z)//2]/c+init_time+64*1e-3/c+3.1*sigmaz/c)/dt)+1
factor=max(charge_dist_time)/max(charge_dist[len(z)//2, :]*c)
ax.plot(z*1e3, charge_dist[:, timestep]*factor, color='r', label='$\lambda$(t) Warpx')
ax.plot(s_charge_dist*1e3, charge_dist_cst, lw=0.8, color='black', ls='--',label='$\lambda$(t) CST') #correct with -0.2
ax.set(title='Charge distribution at cavity center (normalizing factor = '+str(round(factor,3))+')',
        xlabel='z [mm]',
        ylabel='$\lambda$(z) [C/m]',         
        ylim=(np.min(charge_dist_cst)*1.1,np.max(charge_dist_cst)*1.1),
        xlim=(-64,64)
                )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

# Charge distribution integral along z
charge_integral=0
for n in range(nz):
    charge_integral+=charge_dist[n, timestep]*dz

print('Beam charge = '+str(round(charge_integral*1e9, 3))+ 'nC')
print('Difference with input charge: '+ str(round(1e-9/charge_integral, 4)))

# Charge distribution integral 3D
if flag_rho_3d:
    charge_integral=0
    rho=hf_rho.get(dataset_rho[timestep]) # [C/m3]
    charge_integral=np.sum(np.sum(np.sum(rho, axis=0), axis=0))*dx*dy*dz

    print('Beam charge = '+str(round(charge_integral*1e9, 3))+ 'nC')
    print('Difference with input charge: '+ str(round(1e-9/charge_integral, 4)))

    # Plot in 3d
    fig = plt.figure(80, figsize=(6,4), dpi=200, tight_layout=True)
    ax = fig.add_subplot(111, projection='3d')
 
    verts = [(z[i]*1e3, x[ixtest]*1e3, rho[3, 3, i]*dx*dy) for i in range(len(z))] + [(zmin*1e3,x[ixtest]*1e3,0),(zmax*1e3,x[ixtest]*1e3,0)]
    #ax.add_collection3d(Poly3DCollection([verts], color='r', alpha=0.7)) # Add a polygon instead of fill_between

    ax.plot(s_charge_dist*1e3, charge_dist_cst, x[ixtest]*1e3, zdir='y', lw=0.8, color='black', ls='--',label='$\lambda$(t) CST')
    ax.plot(z*1e3, rho[3, 3, :]*dx*dy, x[ixtest]*1e3, zdir='y', color='r', label='$\lambda$(t) WarpX')
    ax.plot(z*1e3, rho[0,0, :]*dx*dy, x[ixtest-3]*1e3, zdir='y', color='b')
    ax.plot(z*1e3, rho[-1, -1, :]*dx*dy, x[ixtest+3]*1e3, zdir='y', color='b')
    ax.plot(z*1e3, rho[1,1, :]*dx*dy, x[ixtest-2]*1e3, zdir='y', color='g')
    ax.plot(z*1e3, rho[-2, -2, :]*dx*dy, x[ixtest+2]*1e3, zdir='y', color='g')
    ax.plot(z*1e3, rho[2,2, :]*dx*dy, x[ixtest-1]*1e3, zdir='y', color='orange')
    ax.plot(z*1e3, rho[-3, -3, :]*dx*dy, x[ixtest+1]*1e3, zdir='y', color='orange')
    # Set plot properties

    ax.set_xlim3d(zmin*1e3,zmax*1e3)
    ax.set_ylim3d(xmin*1e3,xmax*1e3)
    ax.set_zlim3d(0,np.max(rho*dx*dy))

    #ax.grid(True, color='gray', linewidth=0.2)
    ax.set(title='3D charge distribution WarpX | total charge '+str(round(charge_integral*1e9, 3))+ ' nC', 
            xlabel='z [mm]',
            ylabel='x [mm]',
            zlabel='Charge distribution [C/m]'
            )
    plt.show()

#-------------------#
#     2D  Plots     #
#-------------------#


#--- loop with plot of the electric field Ez vs cst
'''
t_offset=69 #13
plt.ion()
nstep=0
for nstep in range(1000):
    if nstep % 1 == 0:
        #--- Plot Ez along z axis 
        fig500 = plt.figure(500, figsize=(6,4), dpi=200, tight_layout=True)
        ax=fig500.gca()
        ax.plot(np.array(z)*1.0e3, charge_dist[:,nstep+t_offset]/np.max(charge_dist)*10000, lw=1.2, color='r', label='$\lambda $') 
        ax.plot(z_Ez*1e3, Ez_t[:, nstep+t_offset], color='g', label='Ez(0,0) WarpX')
        ax.plot(z_Ez*1e3, Ez_t1[:, nstep+t_offset], color='seagreen', label='Ez(1,1) WarpX')
        ax.plot(z_Ez*1e3, Ez_t2[:, nstep+t_offset], color='limegreen', label='Ez(2,2) WarpX')
        ax.plot(z_Ez*1e3, Ez_t3[:, nstep+t_offset], color='springgreen', label='Ez(3,3) WarpX')
        ax.plot(z_cst*1e3, Ez_cst[:, nstep], lw=0.8, color='black', ls='--',label='Ez CST')
        ax.set(title='Electric field at time = '+str(round(t[nstep]*1e9,2))+' ns | timestep '+str(nstep),
                xlabel='z [mm]',
                ylabel='E [V/m]',         
                ylim=(np.min(Ez_cst)*1.1,np.max(Ez_cst)*1.1),
                xlim=(-64,+64),
                        )
        ax.legend(loc='best')
        ax.grid(True, color='gray', linewidth=0.2)
        fig500.canvas.draw()
        fig500.canvas.flush_events()
        fig500.clf()
plt.close()
'''