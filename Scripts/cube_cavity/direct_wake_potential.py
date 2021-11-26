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
init_time=data.get('init_time')
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
dh=x[2]-x[1]	#resolution in the transversal plane

#--- set Wake_length, s
Wake_length=nt*dt*c - (zmax-zmin) - init_time*c
print('Max simulated time = '+str(round(t[-1]*1.0e9,4))+' ns')
print('Wake_length = '+str(Wake_length*1e3)+' mm')
ns_neg=int(init_time/dt)        #obtains the length of the negative part of s
ns_pos=int(Wake_length/(dt*c))  #obtains the length of the positive part of s
s=np.linspace(-init_time*c, 0, ns_neg) #sets the values for negative s
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
    #E - the correct integral is only obtained when integrating the field in the cavity only

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
ax.plot(s*1.0e3, Wake_potential, lw=1.2, color='orange', label='W_//(s)')
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

#--- DFT function definition like in CST [not working]

class Fourier:
    def __init__(self, dft, freqs):
        self.dft = dft
        self.freqs = freqs

def DFT(F, dt, N): 
        #function to obtain the DFT with 1000 samples
        #--F: function in time domain
        #--dt: time sampling width
        #--N: number of time samples

        #define frequency domain
        N_samples=1000  # same number as CST
        f_max = 5.0     # maximum freq in GHz
        freqs=np.linspace(-f_max,f_max,N_samples)*1e9 #frequency range [Hz]
        dft=np.zeros_like(freqs)*1j
        padding=1     #length of the padding with zero
        F=np.append(F,np.zeros(padding))
        print('Performing DFT with '+str(N_samples)+'samples')
        print('Frequency bin resolution'+str(round(1/(N*dt)*1e-9,3))+ 'GHz')
        print('Frequency range')

        for m in range(N_samples):
            for k in range(N+padding):
                dft[m]=dft[m]+F[k]*np.exp(-1j*k*dt*freqs[m]) 

        dft=dt/np.sqrt(np.pi)*dft #Magnitude in [Ohm]
        freqs=freqs*1e-9 #in [GHz]
        return Fourier(dft,freqs)        

#--- Obtain impedance Z
# charge_dist_fft=DFT(charge_dist, ds/c, len(s)) 
# Wake_potential_fft=DFT(Wake_potential, ds/c, len(s))
# Z = abs(- Wake_potential_fft.dft / charge_dist_fft.dft)/c
# Z_freq = Wake_potential_fft.freqs 
# ifreq_max=np.argmax(Z[0:len(Z)//2])+len(Z)//2 # obtains the largest value's index

#--- Obtain impedance Z considering only the positive part of s vector
# charge_dist_fft=DFT(charge_dist[ns_neg:], ds, len(s)-ns_neg) 
# Wake_potential_fft=DFT(Wake_potential[ns_neg:], ds, len(s)-ns_neg)
# Z = abs(- Wake_potential_fft.dft / charge_dist_fft.dft )/c
# Z_freq = Wake_potential_fft.freqs

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
ax.plot(Z_freq[0:len(Z)//2], Z[0:len(Z)//2], lw=1, color='b', marker='s', markersize=2., label='numpy FFT')

ax.set(title='Longitudinal impedance Z(w) magnitude',
        xlabel='f [GHz]',
        ylabel='Z [Ohm]',   
        ylim=(0.,np.max(Z)*1.2),
        xlim=(0.,np.max(Z_freq))      
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()
