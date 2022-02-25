'''
direct_wake_potential_1d.py

File for postprocessing WarpX simulations

--- Reads the input data disctionary with pickle module
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
import h5py as h5py

unit = 1e-3 #mm to m
c=sc.constants.c
beta=1.0 #TODO: obtain beta from Warp simulation

######################
#      Read data     #
######################
runs_path='/mnt/c/Users/elefu/Documents/CERN/GitHub/PyWake/Scripts/WarpX/runs/'
out_folder=runs_path+'out/'

#------------------------------------#
#            1D variables            #
#------------------------------------#

#--- read the dictionary
with open(out_folder+'input_data.txt', 'rb') as handle:
   input_data = pk.loads(handle.read())
   print(input_data.keys())

#--- retrieve variables
x=input_data.get('x')
y=input_data.get('y')
z=input_data.get('z')
nt=input_data.get('tot_nsteps')
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
# This needs to previosly run postproc_h5.py
Ez=input_data.get('Ez')
charge_dist=input_data.get('charge_dist')

# Extract Ez from h5 if not in the dictionary
if Ez is None:
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
    # Extract field on axis Ez (z,t) [V/m]
    Ez_t=[]
    for n in range(tot_nsteps):
        Ez=hf_Ez.get(dataset[n]) # [V/m]
        Ez_t.append(np.array(Ez[shapex//2, shapey//2,:])) # [V/m]

Ez_t=np.transpose(np.array(Ez_t))
# Extract charge_dist from h5 if not in the dictionary
if charge_dist is None:
    hf_rho = h5py.File(out_folder +'rho.h5', 'r')
    print('reading the h5 file '+ out_folder +'rho.h5')
    #get number of datasets
    dataset_rho=[]
    n_step_rho=[]
    for key in hf_rho.keys():
        dataset_rho.append(key)
        n_step_rho.append(int(key.split('_')[1]))
    # Extract charge distribution [C/m] lambda(z,t)
    charge_dist=[]
    for n in range(tot_nsteps):
        rho=hf_rho.get(dataset_rho[n]) # [C/m3]
        charge_dist.append(np.array(rho)*dx*dy) # [C/m]

    charge_dist=np.transpose(np.array(charge_dist)) # [C/m]

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

######################
#   Wake potential   #
######################

t0 = time.time()

# set up t, dt, 
t=np.array(t)#-9*dz/c
dt=t[2]-t[1]
dh=dx         #resolution in the transversal plane

# set Wake_length, s
Wake_length=nt*dt*c - (zmax-zmin) - init_time*c
print('Max simulated time = '+str(round(t[-1]*1.0e9,4))+' ns')
print('Wake_length = '+str(Wake_length*1e3)+' mm')
ns_neg=int(init_time/dt)        #obtains the length of the negative part of s
ns_pos=int(Wake_length/(dt*c))  #obtains the length of the positive part of s
s=np.linspace(-init_time*c, 0, ns_neg) #sets the values for negative s
s=np.append(s, np.linspace(0, Wake_length,  ns_pos))

# initialize Wp variables
Wake_potential=np.zeros_like(s)
Wake_potential_x=np.zeros_like(s)
Wake_potential_y=np.zeros_like(s)
integral = np.zeros(len(s))  #integral of ez between -l1, l2
t_s = np.zeros((nt, len(s)))

# interpolate fields so nz == nt
z_interp=np.linspace(zmin, zmax, nt)
dz_interp=z_interp[2]-z_interp[1]
Ez_interp=np.zeros((nt,nt))

n=0
for n in range(nt):
    if not np.any(Ez[:,n]): #if Ez=0 skip the interpolation
        pass
    else:
        Ez_interp[:, n]=np.interp(z_interp, z , Ez[:,n])

#-----------------------#
#      Obtain W||(s)    #
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
            it=int(t_s[k,n]/dt)                             #find index for t
            Wake_potential[n]=Wake_potential[n]+(Ez_interp[k, it])*dz_interp   #compute integral

q=(1e-9)*1e12                       # charge of the particle beam in pC
Wake_potential=Wake_potential/q     # [V/pC]

#--- plot wake potential 

fig1 = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig1.gca()
ax.plot(s*1.0e3, Wake_potential, lw=1.2, color='orange', label='W_//[0,0](s)')
ax.set(title='Longitudinal Wake potential',
        xlabel='s [mm]',
        ylabel='$W_{//}$ [V/pC]',
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

#--------------------------------#
#      Obtain k loss factor      #
#--------------------------------#

#initialize variables
k_factor=0.0
n=0
ds=s[2]-s[1]

'''
#obtain charge distribution with a gaussian profile
rho=np.transpose(np.array(rho_t)) #how to obtain rho(s) from warp as a function of s? TODO
#in the meantime, a gaussian is used...
charge_dist=(q*1e-12)*(1/(sigmaz*np.sqrt(2*np.pi)))*np.exp(-(0.5*(s-0.0)**2.)/(sigmaz)**2.)  #charge distribution [pC/m]
'''
timestep=int((z[nz//2]/c+init_time+3.1*sigmaz/c)/dt)+1 #timestep with beam at the center of the cavity
charge_dist_center=np.interp(s, z, charge_dist[:,timestep])
charge_dist_norm=charge_dist_center*4.728593798080774/(q*1e-12) #normalized charge distribution [-]

#perform the integral int{-inf,inf}(-lambda*Wake_potential*ds)
for n in range(len(s)): 
    k_factor=k_factor+charge_dist_norm[n]*Wake_potential[n]*ds

k_factor=-k_factor # [V/pC]
print('calculated k_factor = '+str(format(k_factor, '.3e')) + ' [V/pC]')


#--------------------------------#
#      Obtain impedance Z||      #
#--------------------------------#

#--- Obtain impedance Z with Fourier transform numpy.fft.fft

# to increase the resolution of fft, a longer wake length is needed
f_max=5.0*1e9
t_sample=int(1/(ds/c)/2/f_max) #obtains the time window to sample the time domain data
N_samples=int(len(s)/t_sample)
print('Performing FFT with '+str(N_samples)+' samples')
print('Frequency bin resolution '+str(round(1/(len(s)*ds/c)*1e-9,2))+ ' GHz')
print('Frequency range: 0 - '+str(round(f_max*1e-9,2)) +' GHz')

# Padding woth zeros to increase N samples = smoother FFT
charge_dist_padded=np.append(charge_dist_center, np.zeros(10000))
Wake_potential_padded=np.append(Wake_potential, np.zeros(10000))
charge_dist_fft=abs(np.fft.fft(charge_dist_padded[0:-1:t_sample]))
Wake_potential_fft=abs(np.fft.fft(Wake_potential_padded[0:-1:t_sample]))
Z_freq = np.fft.fftfreq(len(Wake_potential_padded[:-1:t_sample]), ds/c*t_sample)*1e-9 #GHz
Z = abs(- Wake_potential_fft / charge_dist_fft)

#--- Plot impedance

# Obtain the maximum frequency
ifreq_max=np.argmax(Z[0:len(Z)//2])
fig2 = plt.figure(2, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig2.gca()
ax.plot(Z_freq[ifreq_max], Z[ifreq_max], marker='o', markersize=4.0, color='cyan')
ax.annotate(str(round(Z_freq[ifreq_max],2))+ ' GHz', xy=(Z_freq[ifreq_max],Z[ifreq_max]), xytext=(-10,5), textcoords='offset points', color='grey') 
#arrowprops=dict(color='blue', shrink=1.0, headwidth=4, frac=1.0)
ax.plot(Z_freq[0:len(Z)//2], Z[0:len(Z)//2], lw=1, color='b', marker='s', markersize=2., label='Z// numpy FFT')

ax.set(title='Longitudinal impedance Z(w) magnitude',
        xlabel='f [GHz]',
        ylabel='Z [Ohm]',   
        ylim=(0.,np.max(Z)*1.2),
        xlim=(0.,np.max(Z_freq))      
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

# Transverse wake potential from Lorentz definition 
'''
#-----------------------#
#      Obtain W⊥(s)     #
#-----------------------#

# s loop -------------------------------------#                                                           
n=0
for n in range(len(s)):    

    #--------------------------------#
    #         W⊥(s) integral         #
    #--------------------------------#

    #Wx: integral of [(Ex(xtest, ytest, z, t=(s+z)/c))-beta*c*By(xtest, ytest, z, t=(s+z)/c)]dz
    #Wy: integral of [(Ey(xtest, ytest, z, t=(s+z)/c))-beta*c*Bx(xtest, ytest, z, t=(s+z)/c)]dz
    k=0
    for k in range(0, nt): 
        t_s[k,n]=(z_interp[k]+s[n])/c-zmin/c-t[0]+init_time

        if t_s[k,n]>0.0:
            it=int(t_s[k,n]/dt)                             #find index for t
            Wake_potential_x[n]=Wake_potential_x[n]+(Ex_interp[k, it]-beta*c*By_interp[k,it])*dz_interp   #compute integral
            Wake_potential_y[n]=Wake_potential_y[n]+(Ey_interp[k, it]-beta*c*Bx_interp[k,it])*dz_interp   #compute integral         

Wake_potential_x=Wake_potential_x/q     # [V/pC]
Wake_potential_y=Wake_potential_y/q     # [V/pC]

#--- plot transverse wake potential 

fig10 = plt.figure(10, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig10.gca()
ax.plot(s*1.0e3, Wake_potential_x, lw=1.2, color='g', label='Wx⊥[0,0](s)')
ax.plot(s*1.0e3, Wake_potential_y, lw=1.2, color='magenta', label='Wy⊥[0,0](s)')
ax.set(title='Transverse Wake potential W⊥(s)',
        xlabel='s [mm]',
        ylabel='$W_{//}$ [V/pC]',
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

#--------------------------------#
#      Obtain impedance Z⊥       #
#--------------------------------#

#--- Obtain impedance Z with Fourier transform numpy.fft.fft
# to increase the resolution of fft, a longer wake length is needed
# Padding woth zeros to increase N samples = smoother FFT

Wake_potential_x_padded=np.append(Wake_potential_x, np.zeros(10000))
Wake_potential_x_fft=abs(np.fft.fft(Wake_potential_x_padded[0:-1:t_sample]))
Z_x_freq = np.fft.fftfreq(len(Wake_potential_x_padded[:-1:t_sample]), ds/c*t_sample)*1e-9 #GHz
Z_x = abs(- Wake_potential_x_fft / charge_dist_fft)

Wake_potential_y_padded=np.append(Wake_potential_y, np.zeros(10000))
Wake_potential_y_fft=abs(np.fft.fft(Wake_potential_y_padded[0:-1:t_sample]))
Z_y_freq = np.fft.fftfreq(len(Wake_potential_y_padded[:-1:t_sample]), ds/c*t_sample)*1e-9 #GHz
Z_y = abs(- Wake_potential_y_fft / charge_dist_fft)

#--- Plot impedance

# Obtain the maximum frequency
ifreq_x_max=np.argmax(Z[0:len(Z_x)//2])
ifreq_y_max=np.argmax(Z[0:len(Z_y)//2])
fig20 = plt.figure(20, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig20.gca()
ax.plot(Z_x_freq[ifreq_x_max], Z_x[ifreq_x_max], marker='o', markersize=4.0, color='cyan')
ax.annotate(str(round(Z_x_freq[ifreq_x_max],2))+ ' GHz', xy=(Z_x_freq[ifreq_x_max],Z_x[ifreq_x_max]), xytext=(-10,5), textcoords='offset points', color='grey') 
ax.plot(Z_x_freq[0:len(Z_x)//2], Z_x[0:len(Z_x)//2], lw=1, color='g', marker='s', markersize=2., label='Zx⊥ numpy FFT')

ax.plot(Z_y_freq[ifreq_y_max], Z_x[ifreq_y_max], marker='o', markersize=4.0, color='cyan')
ax.annotate(str(round(Z_y_freq[ifreq_y_max],2))+ ' GHz', xy=(Z_y_freq[ifreq_y_max],Z_y[ifreq_y_max]), xytext=(-10,5), textcoords='offset points', color='grey') 
ax.plot(Z_y_freq[0:len(Z_y)//2], Z_y[0:len(Z_y)//2], lw=1, color='magenta', marker='s', markersize=2., label='Zy⊥ numpy FFT')

ax.set(title='Transverse impedance Z⊥(w) magnitude',
        xlabel='f [GHz]',
        ylabel='Z [Ohm]',   
        #ylim=(0.,np.max(Z_x)*1.2),
        #xlim=(0.,np.max(Z_x_freq))      
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

#--- Save the data 
data = { 'Wake_potential' : Wake_potential, 
         's' : s,
         'k_factor' : k_factor,
         'Impedance' : Z,
         'frequency' : Z_freq,
         'Wake_potential_x' : Wake_potential_x,
         'Wake_potential_y' : Wake_potential_y,
         'Impedance_x' : Z_x,
         'Impedance_y' : Z_y,
         'frequency_x' : Z_x_freq,
         'frequency_y' : Z_y_freq,

        }
# write the dictionary to a txt using pickle module
with open(out_folder + 'wake_solver.txt', 'wb') as handle:
  pk.dump(data, handle)
'''

#Calculate simulation time
t1 = time.time()
totalt = t1-t0
print('Calculation terminated in %ds' %totalt)


############################
#   Comparison with CST    #
############################

#--- read the cst dictionary
cst_path='/mnt/c/Users/elefu/Documents/CERN/GitHub/PyWake/Scripts/CST/'
with open(cst_path+'cst_out.txt', 'rb') as handle:
  cst_data = pk.loads(handle.read())
  print('cst stored variables')
  print(cst_data.keys())

charge_dist_cst=cst_data.get('charge_dist')
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

#--- Plot longitudinal WP comparison with CST

q=(1e-9)*1e12 # charge of the particle beam in pC
fig40 = plt.figure(40, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig40.gca()
ax.plot(s*1.0e3, Wake_potential, lw=1.2, color='orange', label='W_//(s) direct integration')
ax.plot(s_cst*1e3, Wake_potential_cst, lw=1.3, color='black', ls='--', label='W_//(s) CST')
ax.set(title='Longitudinal Wake potential',
        xlabel='s [mm]',
        ylabel='$W_{//}$ [V/pC]',
        xlim=(min(s*1.0e3), np.amin((np.max(s*1.0e3), np.max(s_cst*1.0e3))))
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

'''
#--- Plot transverse WP comparison with CST

q=(1e-9)*1e12 # charge of the particle beam in pC
fig4 = plt.figure(4, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig4.gca()
ax.plot(s*1.0e3, Wake_potential_x, lw=1.2, color='g', label='Wx⊥[0,0](s)')
ax.plot(s_cst*1.0e3, WPx_cst, lw=1.2, color='g', ls='--', label='Wx⊥[0,0](s) from CST')
ax.plot(s*1.0e3, Wake_potential_y, lw=1.2, color='magenta', label='Wy⊥[0,0](s)')
ax.plot(s_cst*1.0e3, WPy_cst, lw=1.2, color='magenta', ls='--', label='Wy⊥[0,0](s) from CST')
ax.set(title='Transverse Wake potential W⊥(s)',
        xlabel='s [mm]',
        ylabel='$W_{//}$ [V/pC]',
        xlim=(min(s*1.0e3), np.amin((np.max(s*1.0e3), np.max(s_cst*1.0e3)))),
        ylim=(-0.04, 0.07)
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()
'''

#--- Plot impedance comparison with CST [normalized]

# Plot comparison with CST [normalized]
norm=max(Z)/max(Z_cst) #diference between max in CST and in numpy.fft
ifreq_max=np.argmax(Z[0:len(Z)//2])
fig3 = plt.figure(3, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig3.gca()
ax.plot(Z_freq[ifreq_max], Z[ifreq_max]/norm, marker='o', markersize=4.0, color='cyan')
ax.annotate(str(round(Z_freq[ifreq_max],2))+ ' GHz', xy=(Z_freq[ifreq_max],Z[ifreq_max]/norm), xytext=(-20,5), textcoords='offset points', color='cyan') 
ax.plot(Z_freq[0:len(Z)//2], Z[0:len(Z)//2]/norm, lw=1, color='b', marker='s', markersize=2., label='numpy FFT')

ifreq_max=np.argmax(Z_cst)
ax.plot(freq_cst[ifreq_max]*1e-9, Z_cst[ifreq_max], marker='o', markersize=5.0, color='pink')
ax.annotate(str(round(freq_cst[ifreq_max]*1e-9,2))+ ' GHz', xy=(freq_cst[ifreq_max]*1e-9,Z_cst[ifreq_max]), xytext=(+20,5), textcoords='offset points', color='pink') 
ax.plot(freq_cst*1.0e-9, Z_cst, lw=1.2, color='red', marker='s', markersize=2., label='W// from CST')

ax.set(title='Longitudinal impedance Z(w) magnitude',
        xlabel='f [GHz]',
        ylabel='Z [Ohm]',   
        ylim=(0.,np.max(Z_cst)*1.2),
        xlim=(0.,np.max(freq_cst)*1e-9)      
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()
