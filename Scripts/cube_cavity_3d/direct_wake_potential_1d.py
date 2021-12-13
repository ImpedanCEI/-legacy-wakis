'''
direct_wake_potential_1d.py

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
import h5py as h5py

unit = 1e-3 #mm to m
c=sc.constants.c
beta=1.0 #TODO: obtain beta from Warp simulation

######################
#      Read data     #
######################

out_folder='out_10e7_refinedx2/'

#------------------------------------#
#            1D variables            #
#------------------------------------#

#--- read the dictionary
with open(out_folder+'out.txt', 'rb') as handle:
  data = pk.loads(handle.read())
  print('stored 1D variables')
  print(data.keys())

#--- retrieve 1D variables

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
L_cavity=data.get('L_cavity')
w_pipe=data.get('w_pipe')
h_pipe=data.get('h_pipe')
L_pipe=data.get('L_pipe')
t=data.get('t')
init_time=data.get('init_time')
nt=data.get('nt') #number of timesteps
nz=data.get('nz')
sigmaz=data.get('sigmaz')
xtest=data.get('xtest')
ytest=data.get('ytest')

#--- auxiliary variables
zmin=min(z)
zmax=max(z)
xmin=min(x)
xmax=max(x)
ymin=min(y)
ymax=max(y)

#reshape electric and magnetic fields
Ez=np.transpose(np.array(Ez_t))     #array to matrix (z,t)
Ex=np.transpose(np.array(Ex_t))
Ey=np.transpose(np.array(Ey_t))
Bx=np.transpose(np.array(Bx_t))
By=np.transpose(np.array(By_t))


######################
#   Wake potential   #
######################

t0 = time.time()

# set up z, t, dt, dz
z=np.array(z)
t=np.array(t)
dz=z[2]-z[1]
dt=t[2]-t[1]
dh=x[2]-x[1]    #resolution in the transversal plane

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
Ex_interp=np.zeros((nt,nt))
Ey_interp=np.zeros((nt,nt))
Bx_interp=np.zeros((nt,nt))
By_interp=np.zeros((nt,nt))

n=0
for n in range(nt):
    if not np.any(Ez[:,n]): #if Ez=0 skip the interpolation
        pass
    else:
        Ez_interp[:, n]=np.interp(z_interp, z , Ez[:,n])
        Ex_interp[:, n]=np.interp(z_interp, z , Ex[:,n])
        Ey_interp[:, n]=np.interp(z_interp, z , Ey[:,n])
        Bx_interp[:, n]=np.interp(z_interp, z , Bx[:,n])
        By_interp[:, n]=np.interp(z_interp, z , By[:,n])

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

#obtain charge distribution with a gaussian profile
rho=np.transpose(np.array(rho_t)) #how to obtain rho(s) from warp as a function of s? TODO
#in the meantime, a gaussian is used...
charge_dist=(q*1e-12)*(1/(sigmaz*np.sqrt(2*np.pi)))*np.exp(-(0.5*(s-0.0)**2.)/(sigmaz)**2.)  #charge distribution [pC/m]
charge_dist_norm=charge_dist/(q*1e-12) #normalized charge distribution [-]

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
charge_dist_padded=np.append(charge_dist, np.zeros(10000))
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
    #Wy: integral of [(Ex(xtest, ytest, z, t=(s+z)/c))-beta*c*By(xtest, ytest, z, t=(s+z)/c)]dz
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

#Calculate simulation time
t1 = time.time()
totalt = t1-t0
print('Calculation terminated in %ds' %totalt)

'''
############################
#   Comparison with CST    #
############################

#--- read the cst dictionary
with open('cst/cst_out.txt', 'rb') as handle:
  cst_data = pk.loads(handle.read())
  print('cst stored variables')
  print(cst_data.keys())

charge_dist_cst=cst_data.get('charge_dist')
distance=cst_data.get('distance')
Wake_potential_cst=cst_data.get('Wake_potential_cst')
s_cst=cst_data.get('s_cst')
Z_cst=cst_data.get('Z_cst')
freq_cst=cst_data.get('freq_cst')

#--- Plot comparison

q=(1e-9)*1e12 # charge of the particle beam in pC
fig4 = plt.figure(4, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig4.gca()
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
'''