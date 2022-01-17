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
from scipy.constants import mu_0, pi, c, epsilon_0
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pickle
import h5py

out_folder='out/'

#------------------------------------#
#            1D variables            #
#------------------------------------#

#--- read the dictionary
with open(out_folder+'input_data_dict.txt', 'rb') as handle:
   input_data = pickle.loads(handle.read())
   print(input_data.keys())

#--- retrieve variables
x=input_data.get('x')
y=input_data.get('y')
z=input_data.get('z')
tot_nsteps=input_data.get('tot_nsteps')
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
t=[]

#------------------------------------#
#            diags files             #
#------------------------------------#

#create h5 files overwriting previous ones
hf_name='Ez.h5'
if os.path.exists(out_folder+hf_name):
    os.remove(out_folder+hf_name)

hf = h5py.File(out_folder+hf_name, 'w')

# Open the diag plot file
for i in np.linspace(0, tot_nsteps, tot_nsteps):
    filename = out_folder+ 'diags/warpx_diag' + str(int(i)).zfill(5)
    ds = yt.load(filename)
    data = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)

    t.append(ds.current_time.to_value())
    #Ex = data['boxlib','Ex'].to_ndarray()
    #Ey = data['boxlib','Ey'].to_ndarray()
    Ez = data['boxlib','Ez'].to_ndarray()  #shape(nx,ny,nz)
    #Bx = data['boxlib','Bx'].to_ndarray()
    #By = data['boxlib','By'].to_ndarray()

    # Save np.array into hdf5 file
    if i == 0:
        prefix='0'*5
        hf.create_dataset('Ez_'+prefix+str(int(i)), data=Ez)
    else:
        prefix='0'*(5-int(np.log10(i)))
        hf.create_dataset('Ez_'+prefix+str(int(i)), data=Ez)

#close the hdf5 file
hf.close()

#update the dictionary with time vector
input_data['t']=np.array(t)

with open(out_folder+'input_data_dict.txt', 'wb') as handle:
    pickle.dump(input_data, handle)

#------------------------------------#
#            3D variables            #
#------------------------------------#

#--- read the h5 file
hf = h5py.File(out_folder +'Ez.h5', 'r')
print('reading the h5 file '+ out_folder +'Ez.h5')
print('size of the file: '+str(round((os.path.getsize(out_folder+'Ez.h5')/10**9),2))+' Gb')
#get number of datasets
size_hf=0.0
dataset=[]
n_step=[]
for key in hf.keys():
    size_hf+=1
    dataset.append(key)
    n_step.append(int(key.split('_')[1]))

Ez_0=hf.get(dataset[0])
shapex=Ez_0.shape[0]  
shapey=Ez_0.shape[1] 
shapez=Ez_0.shape[2] 

print('Ez field is stored in matrices '+str(Ez_0.shape)+' in '+str(int(size_hf))+' datasets')

#...................#
#     2D  Plots     #
#...................#

'''
#--- loop with countours of electric field
plt.ion()
for n in tot_nsteps:
    if n % 1 == 0:
    Ez=hf.get(dataset[n])
    #--- Plot Ez - x cut 
    fig1 = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
    ax=fig1.gca()
    im=ax.imshow(Ez[int(shapex/2),:,:], vmin=-5.e4, vmax=5.e4, extent=[zmin*1e3, zmax*1e3, ymin*1e3, ymax*1e3], cmap='jet')
    ax.set(title='Warpx Ez field, t = ' + str(round(t*1e9,3)) + ' ns',
           xlabel='z    [mm]',
           ylabel='y    [mm]'
           )
    plt.colorbar(im, label = 'Ez    [V/m]')
    fig1.canvas.draw()
    fig1.canvas.flush_events()
    fig1.clf() 
plt.close()
''' 

#..................#
# Compare with CST #
#..................#
cst_path='/mnt/c/Users/elefu/Documents/CERN/GitHub/PyWake/Scripts/CST/'
#--- read the cst dictionary
with open(cst_path+'cst_out.txt', 'rb') as handle:
  cst_data = pickle.loads(handle.read())
  print('cst stored variables')
  print(cst_data.keys())

Ez_cst=cst_data.get('Ez')
t_cst = cst_data.get('t')
nz_cst=cst_data.get('nz')
nt_cst=cst_data.get('nt')
charge_dist_cst=cst_data.get('charge_dist')
s_charge_dist=cst_data.get('s_charge_dist')
distance=cst_data.get('distance')
Wake_potential_cst=cst_data.get('Wake_potential_cst')
Wake_potential_interfaces=cst_data.get('Wake_potential_interfaces')
Wake_potential_testbeams=cst_data.get('Wake_potential_testbeams')
WPx_cst=cst_data.get('WPx_cst')
WPy_cst=cst_data.get('WPy_cst')
s_cst=cst_data.get('s_cst')
Z_cst=cst_data.get('Z_cst')
Zx_cst=cst_data.get('Zx_cst')
Zy_cst=cst_data.get('Zy_cst')
freq_cst=cst_data.get('freq_cst')

#--- Plot Electric field at cavity center

fig50 = plt.figure(50, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig50.gca()
for n in range(tot_nsteps): 
    Ez=hf.get(dataset[n])
    ax.plot(np.array(t[n])*1.0e9, Ez[int(nx/2), int(ny/2), int(nz/2)], marker='.', color='g')
ax.plot(np.array(t[n])*1.0e9, Ez[int(nx/2), int(ny/2), int(nz/2)], marker='.', color='g', label='Ez Warpx')
ax.plot(np.array(t_cst)*1.0e9, Ez_cst[int(nz_cst/2), :], lw=0.8, color='black', ls='--',label='Ez CST')
ax.set(title='Electric field at cavity center',
        xlabel='t [ns]',
        ylabel='$E [V/m]$',         
        ylim=(np.min(Ez_cst[int(nz_cst/2), :])*1.1,np.max(Ez_cst[int(nz_cst/2), :])*1.1),
        xlim=(0,np.minimum(np.max(np.array(t[n])*1.0e9),np.max(t_cst*1.0e9)))
                )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()