'''
direct_wake_potential_CST.py

File for postprocessing CST simulations

--- Reads the input data dictionary with pickle
--- Reads the 3d data of the ez field from h5 file
--- Performs the direct integration of the longitudinal wake potential
--- Obtains the transverse wake potential through Panofsky Wenzel theorem
--- Performs the fourier trnasform to obtain the impedance
--- Plots the results

'''
print('---------------------')
print('|  Running PyWake   |')
print('---------------------')

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import scipy.constants as spc 
import scipy.interpolate as spi 
import pickle as pk
import h5py as h5py

unit = 1e-3 #mm to m
c=spc.c
beta=1.0 #TODO: obtain beta from Warp simulation

######################
#      Read data     #
######################
runs_path='/mnt/c/Users/elefu/Documents/CERN/PyWake/Scripts/CST/' 
out_folder=runs_path

#------------------------------------#
#            1D variables            #
#------------------------------------#

#--- read the dictionary
with open(out_folder+'cst_out.txt', 'rb') as handle:
   cst_data = pk.loads(handle.read())
   #print(cst_data.keys())
   
#---Input variables
x=cst_data.get('x')
y=cst_data.get('y')
z=cst_data.get('z')
nt=cst_data.get('nt')
init_time=cst_data.get('init_time')
nx=cst_data.get('nx')
ny=cst_data.get('ny')
nz=cst_data.get('nz')
w_cavity=cst_data.get('w_cavity')
h_cavity=cst_data.get('h_cavity')
L_cavity=cst_data.get('L_cavity')
w_pipe=cst_data.get('w_pipe')
h_pipe=cst_data.get('h_pipe')
L_pipe=cst_data.get('L_pipe')
sigmaz=cst_data.get('sigmaz')
#---Electric field
Ez_cst=cst_data.get('Ez')
t_cst = cst_data.get('t')
nz_cst=cst_data.get('nz')
nt_cst=cst_data.get('nt')
z_cst=cst_data.get('z')
#---Charge distribution
charge_dist_cst=cst_data.get('charge_dist')
charge_dist_time=cst_data.get('charge_dist_time')
charge_dist_spectrum=cst_data.get('charge_dist_spectrum')
current=cst_data.get('current')
s_charge_dist=cst_data.get('s_charge_dist')
t_charge_dist=cst_data.get('t_charge_dist')
#---Wake potential
Wake_potential_cst=cst_data.get('WP_cst')
Wake_potential_interfaces=cst_data.get('Wake_potential_interfaces')
Wake_potential_testbeams=cst_data.get('Wake_potential_testbeams')
WPx_cst=cst_data.get('WPx_cst')
WPy_cst=cst_data.get('WPy_cst')
WPx_dipolar_cst=cst_data.get('WPx_dipolar_cst')
WPy_dipolar_cst=cst_data.get('WPy_dipolar_cst')
WPx_quadrupolar_cst=cst_data.get('WPx_quadrupolar_cst')
WPy_quadrupolar_cst=cst_data.get('WPy_quadrupolar_cst')
s_cst=cst_data.get('s_cst')
#---Impedance
Z_cst=cst_data.get('Z_cst')
Zx_cst=cst_data.get('Zx_cst')
Zy_cst=cst_data.get('Zy_cst')
Zx_dipolar_cst=cst_data.get('Zx_dipolar_cst')
Zy_dipolar_cst=cst_data.get('Zy_dipolar_cst')
Zx_quadrupolar_cst=cst_data.get('Zx_quadrupolar_cst')
Zy_quadrupolar_cst=cst_data.get('Zy_quadrupolar_cst')
freq_cst=cst_data.get('freq_cst')

#------------------------------------#
#            3D variables            #
#------------------------------------#

# Read the Ez h5 file
h5_name='cst_Ez_quadrupolar_dh05.h5'
hf = h5py.File(out_folder+h5_name, 'r')
print('Reading the h5 file: '+ out_folder+h5_name)
print('---Size of the file: '+str(round((os.path.getsize(out_folder+h5_name)/10**9),2))+' Gb')
# get number of datasets
size_hf=0.0
dataset=[]
n_step=[]
for key in hf.keys():
    size_hf+=1
    dataset.append(key)
    n_step.append(int(key.split('_')[1]))
# get size of matrix
Ez_0=hf.get(dataset[0])
shapex=Ez_0.shape[0]  
shapey=Ez_0.shape[1] 
shapez=Ez_0.shape[2] 
print('---Ez field is stored in a matrix with shape '+str(Ez_0.shape)+' in '+str(int(size_hf))+' datasets')

# define time array
nt=len(dataset)
t=t_cst[0:nt]
dt=t[2]-t[1]

# define z array
zmin=-49.9313*unit #from txt
zmax=49.9313*unit
dz=(zmax-zmin)/(shapez-1)
z=np.arange(zmin, zmax+dz, dz)

# define x, y arrays
xmin=-0.854988*unit #from txt
xmax=0.854988*unit
dx=(xmax-xmin)/(shapex-1)
x=np.arange(xmin, xmax+dx, dx)

ymin=-0.854988*unit #from txt
ymax=0.854988*unit
dy=(ymax-ymin)/(shapey-1)
y=np.arange(ymin, ymax+dy, dy)

#--- read the dictionary
with open(out_folder+'field_data.txt', 'rb') as handle:
   Ez_data = pk.loads(handle.read())
   #print(cst_data.keys())

x=Ez_data.get('x')*unit
dx=x[2]-x[1]
y=Ez_data.get('y')*unit
dy=y[2]-y[1]
z=Ez_data.get('z')*unit
dz=z[2]-z[1]
t=Ez_data.get('t')

'''
#Define xtest, ytest, xsource, ysource
if h5_name == 'cst_Ez_dipolarx.h5':
    xsource, ysource = 3e-3, 3e-3
    xtest, ytest = 0, 0
elif h5_name == 'cst_Ez_quadrupolar.h5':
    xsource, ysource = 0, 0
    xtest, ytest = 3e-3, 3e-3
else: 
    xsource, ysource = 0, 0
    xtest, ytest = 0, 0
'''

######################
#   Wake potential   #
######################

t0 = time.time()

# set Wake_length, s
Wake_length=nt*dt*c - (zmax-zmin) - init_time*c
print('---Max simulated time = '+str(round(t[-1]*1.0e9,4))+' ns')
print('---Wakelength = '+str(Wake_length/unit)+' mm')
ns_neg=int(init_time/dt)        #obtains the length of the negative part of s
ns_pos=int(Wake_length/(dt*c))  #obtains the length of the positive part of s
s=np.linspace(-init_time*c, 0, ns_neg) #sets the values for negative s
s=np.append(s, np.linspace(0, Wake_length,  ns_pos))

# initialize Wp variables
Wake_potential=np.zeros_like(s)
t_s = np.zeros((nt, len(s)))

# interpolate fields so nz == nt
z_interp=np.linspace(zmin, zmax, nt)
dz_interp=z_interp[2]-z_interp[1]
Ez_interp=np.zeros((nt,nt))

# interpolate fields so xtest, ytest is in the grid
flag_interp_xy=False
xi=np.arange(min(x)+dx/2., max(x)-dx/2, dx)
yi=np.arange(min(y)+dy/2., max(y)-dy/2, dy)
Ezxy=np.zeros(len(z))

# initialize wake potential matrix
flag_fourth_order=False     #Default: False
flag_second_order=True      #Default: True
if flag_fourth_order:
    print('Using fourth order scheme for gradient')
    n_transverse_cells=2
    WP_3d=np.zeros((n_transverse_cells*2+1,n_transverse_cells*2+1,len(s)))
elif flag_second_order:
    print('Using second order scheme for gradient')
    n_transverse_cells=1
    WP_3d=np.zeros((n_transverse_cells*2+1,n_transverse_cells*2+1,len(s)))
else:    
    print('Using first order upwind scheme for gradient')
    n_transverse_cells=1    
    WP_3d=np.zeros((n_transverse_cells*2+1,n_transverse_cells*2+1,len(s)))

i0=n_transverse_cells
j0=n_transverse_cells

print('Calculating longitudinal wake potential...')
for i in range(-n_transverse_cells,n_transverse_cells+1,1): #field is stored around (xtest,ytest) selected by the user 
    for j in range(-n_transverse_cells,n_transverse_cells+1,1):

        n=0
        for n in range(nt):
            Ez=hf.get(dataset[n])
            if flag_interp_xy:
                for k in range(len(z)):
                    Ezxy[k]=spi.interpn((x,y,z), np.array(Ez), np.array([xi[shapex//2-1+i], yi[shapey//2-1+j], z[k]]))
                Ez_interp[:, n]=np.interp(z_interp, z, Ezxy[:])#Ezi[shapex//2-1+i,shapey//2-1+j,:])
            else:
                Ez_interp[:, n]=np.interp(z_interp, z, Ez[shapex//2-1+i,shapey//2-1+j,:])

        #-----------------------#
        #     Obtain W||(s)     #
        #-----------------------#

        # s loop -------------------------------------#                                                           
        n=0
        for n in range(len(s)):    

            #--------------------------------#
            # integral between zmin and zmax #
            #--------------------------------#

            #integral of (Ez(xtest, ytest, z, t=(s+z)/c))dz
            k=0
            for k in range(0, nt): 
                t_s[k,n]=(z_interp[k]+s[n])/c-zmin/c-t[0]+init_time

                if t_s[k,n]>0.0:
                    it=int(t_s[k,n]/dt)-1                                              #find index for t
                    Wake_potential[n]=Wake_potential[n]+(Ez_interp[k, it])*dz_interp   #compute integral

        q=(1e-9)*1e12                       # charge of the particle beam in pC
        Wake_potential=Wake_potential/q     # [V/pC]
        WP_3d[i0+i,j0+j,:]=Wake_potential 

WP=WP_3d[i0,j0,:]

#-----------------------#
#      Obtain W⊥(s)    #
#-----------------------#

# Initialize variables
n=0
k=0
i=0
j=0
ds=s[2]-s[1]
WPx=np.zeros_like(s)
WPy=np.zeros_like(s)

# Obtain the transverse wake potential through Panofsky Wenzel theorem
int_WP=np.zeros_like(WP_3d)
print('Calculating transverse wake potential...')
for n in range(len(s)):
    for i in range(-n_transverse_cells,n_transverse_cells+1,1):
        for j in range(-n_transverse_cells,n_transverse_cells+1,1):
            for k in range(n):
                # Perform the integral
                int_WP[i0+i,j0+j,n]=int_WP[i0+i,j0+j,n]+WP_3d[i0+i,j0+j,k]*ds 
    if flag_fourth_order:
    # Perform the gradient (fourth order scheme)
        WPx[n]= - (-int_WP[i0+2,j0,n]+8*int_WP[i0+1,j0,n]+ \
                                        -8*int_WP[i0-1,j0,n]+int_WP[i0-2,j0,n])/(12*dx)
        WPy[n]= - (-int_WP[i0,j0+2,n]+8*int_WP[i0,j0+1,n]+ \
                                        -8*int_WP[i0,j0-1,n]+int_WP[i0,j0-2,n])/(12*dy)
    if flag_second_order:
    # Perform the gradient (second order scheme)
        WPx[n]= - (int_WP[i0+1,j0,n]-int_WP[i0,j0,n])/(2*dx)
        WPy[n]= - (int_WP[i0,j0+1,n]-int_WP[i0,j0,n])/(2*dy)
    else:
    # Perform the gradient (first order scheme)
        WPx[n]= - (int_WP[i0+1,j0,n]-int_WP[i0,j0,n])/(dx)
        WPy[n]= - (int_WP[i0,j0+1,n]-int_WP[i0,j0,n])/(dy)    


#--------------------------------#
#      Obtain impedance Z||      #
#--------------------------------#
import solver_module as Wsol

#--- Obtain impedance Z with Fourier transform DFT
print('Obtaining longitudinal impedance...')
# Obtain charge distribution as a function of s, normalized
fmax=c/sigmaz/2.95
charge_dist_s=np.interp(s, s_charge_dist , charge_dist_cst/max(charge_dist_cst)) 
# Obtain the ffts and frequency bins
lambdaf, f=Wsol.FFT(charge_dist_s, ds/c, fmax=fmax, r=10.0)
WPf, f=Wsol.FFT(WP, ds/c, fmax=fmax, r=10.0)
# Obtain the impedance
Z = abs(- WPf / lambdaf) 


#--------------------------------#
#      Obtain impedance Z⊥       #
#--------------------------------#

#--- Obtain impedance Z with Fourier transform numpy.fft.fft
# to increase the resolution of fft, a longer wake length is needed
print('Obtaining transverse impedance...')
#---Zx⊥(w)
# Obtain the ffts and frequency bins
WPxf, f=Wsol.FFT(WPx, ds/c, fmax=fmax, r=10.0)
# Obtain the impedance
Zx = abs(- WPxf / lambdaf) 
#---Zy⊥(w)
# Obtain the ffts and frequency bins
WPyf, f=Wsol.FFT(WPy, ds/c, fmax=fmax, r=10.0)
# Obtain the impedance
Zy = abs(- WPyf / lambdaf) 

#--------------------------------#

#Calculate elapsed time
t1 = time.time()
totalt = t1-t0
print('Calculation terminated in %ds' %totalt)

# Save the data 

xsource, ysource = 3e-3, 3e-3
xtest, ytest = 0e-3, 0e-3

data = { 'WP' : WP, 
         's' : s,
         #'k_factor' : k_factor,
         'Z' : Z,
         'f' : f,
         'WPx' : WPx,
         'WPy' : WPy,
         'Zx' : Zx,
         'Zy' : Zy,
         'xsource' : xsource,
         'ysource' : ysource,
         'xtest' : xtest,
         'ytest' : ytest,
        }
# write the dictionary to a txt using pickle module
with open(out_folder + 'wake_solver.txt', 'wb') as handle:
    pk.dump(data, handle)


#--------------------------#
#   Comparison with CST    #
#--------------------------#
import plot_module as Wplt

cst_path='/mnt/c/Users/elefu/Documents/CERN/PyWake/Scripts/CST/'

Wplt.plot_PyWake(data=data, 
                cst_data=Wplt.read_CST_out(cst_path), 
                flag_compare_cst=True, 
                flag_normalize=True
                )

