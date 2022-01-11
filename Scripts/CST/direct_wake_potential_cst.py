'''
wake_potential_cst.py

File for postprocessing CST simulations

--- Reads the cst_out file from cst_to_dict.py to obtain Ez
--- Performs the direct integration to obtain wake potential
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

'''
#--- read the dictionary from Warp cube cavity dict
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
'''

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
charge_dist_cst=cst_data.get('charge_dist')
distance=cst_data.get('distance')
Wake_potential_cst=cst_data.get('Wake_potential_cst')
s_cst=cst_data.get('s_cst')
Z_cst=cst_data.get('Z_cst')
freq_cst=cst_data.get('freq_cst')
x=cst_data.get('x')
y=cst_data.get('y')
z=cst_data.get('z')
w_cavity=cst_data.get('w_cavity')
h_cavity=cst_data.get('h_cavity')
L_cavity=cst_data.get('L_cavity')
w_pipe=cst_data.get('w_pipe')
h_pipe=cst_data.get('h_pipe')
L_pipe=cst_data.get('L_pipe')
sigmaz=cst_data.get('sigmaz')
xtest=cst_data.get('xtest')
ytest=cst_data.get('ytest')
zmax=np.max(z)
zmin=np.min(z)

# Wake_potential_cst=np.array(Wake_potential_cst) # in V/pC
# s_cst=np.array(s_cst)  # in [m]


######################
# 	Wake potential   #
######################

#----------#
#  Set up  #
#----------#

#--- set up z, t, dt, dz
z=np.linspace(zmin,zmax,nz)
t=np.array(t)
dz=z[2]-z[1]
dt=t[2]-t[1]

init_time=5.332370636221942e-10 #time when the center of the bunch enters the cavity
dh=x[2]-x[1]	#resolution in the transversal plane

#--- set Wake_length, s
Wake_length=nt*dt*c - (zmax-zmin) - init_time*c
print('Max simulated time = '+str(round(t[-1]*1.0e9,4))+' ns')
print('Wake_length = '+str(Wake_length*1e3)+' mm')
ns_neg=int(init_time/dt)		#obtains the length of the negative part of s
ns_pos=int(Wake_length/(dt*c))	#obtains the length of the positive part of s
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
l1=(L_cavity/2.0)         #[m]
l2=(L_cavity/2.0)         #[m] 
iz_l1=int((-l1-z_interp[0])/dz_interp)
iz_l2=int((l2-z_interp[0])/dz_interp)

#--- initialize variables
Ez_dt=np.zeros((nt,nt))  #time derivative of Ez
Ez_dz=np.zeros((nt,nt))  #z spatial derivative of Ez
t_s = np.zeros((nt, len(s)))


#-------------------------#
#      Obtain W||(s)      #
#-------------------------#

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

# Plot comparison with CST [normalized]
norm=max(Z)/max(Z_cst) #diference between max in CST and in numpy.fft
ifreq_max=np.argmax(Z[0:len(Z)//2])
fig3 = plt.figure(3, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig3.gca()
ax.plot(Z_freq[ifreq_max], Z[ifreq_max]/norm, marker='o', markersize=4.0, color='cyan')
ax.annotate(str(round(Z_freq[ifreq_max],2))+ ' GHz', xy=(Z_freq[ifreq_max],Z[ifreq_max]/norm), xytext=(-20,5), textcoords='offset points', color='cyan') 
ax.plot(Z_freq[0:len(Z)//2], Z[0:len(Z)//2]/norm, lw=1, color='b', label='numpy FFT')

ifreq_max=np.argmax(Z_cst)
ax.plot(freq_cst[ifreq_max]*1e-9, Z_cst[ifreq_max], marker='o', markersize=5.0, color='pink')
ax.annotate(str(round(freq_cst[ifreq_max]*1e-9,2))+ ' GHz', xy=(freq_cst[ifreq_max]*1e-9,Z_cst[ifreq_max]), xytext=(+20,5), textcoords='offset points', color='pink') 
ax.plot(freq_cst*1.0e-9, Z_cst, lw=1.2, color='red', label='W// from CST')

ax.set(title='Longitudinal impedance Z(w) magnitude',
        xlabel='f [GHz]',
        ylabel='Z [Ohm]',   
        ylim=(0.,np.max(Z_cst)*1.2),
        xlim=(0.,np.max(freq_cst)*1e-9)      
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

#------------------------#
#      Obtain W⊥(s)      #
#------------------------#


############################
#   Comparison with CST    #
############################

#--- Plot comparison

q=(1e-9)*1e12 # charge of the particle beam in pC
fig4 = plt.figure(4, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig4.gca()
ax.plot(s*1.0e3, Wake_potential, lw=1.2, color='orange', label='W_//(s) direct integration')
ax.plot(s_cst*1e3, Wake_potential_cst, lw=1.3, color='black', ls='--', label='W_//(s) CST')
ax.set(title='Longitudinal Wake potential',
        xlabel='s [mm]',
        ylabel='$W_{//}$ [V/pC]',
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

'''
#--- Plot normalized comparison comparison

fig5 = plt.figure(5, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig5.gca()
ax.plot(s*1e3, Wake_potential/np.max(Wake_potential), lw=1.3, color='orange', label='$W_{//}(s)$ direct integration')
ax.plot(s_cst*1e3, Wake_potential_cst/np.max(Wake_potential_cst), lw=1.2, color='orange', ls='--', label='$W_{//}(s)$ from CST')
ax.set(title='Normalized longitudinal Wake potential',
        xlabel='s [mm]',
        ylabel='$W_{//}$ [V/pC]',
        ylim=(-1.5,1.5)
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()
'''

