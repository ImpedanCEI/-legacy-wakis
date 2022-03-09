'''
FFT_testing.py

Benchmark of the FFT method with CST files
'''
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import scipy.constants as sc  
from copy import copy
import pickle as pk   

import solver_module as Wsol

c=sc.c

#--- read the cst dictionary
with open('cst_out.txt', 'rb') as handle:
  cst_data = pk.loads(handle.read())

# charge_dist=cst_data.get('charge_dist_time')
# t=cst_data.get('t_charge_dist')
# t0=cst_data.get('init_time')
# s_charge_dist=t*c - t0*c

charge_dist=cst_data.get('charge_dist')
s_charge_dist=cst_data.get('s_charge_dist')

#'''
Wake_potential_cst=cst_data.get('WP_cst')
s_cst=cst_data.get('s_cst')
Z_cst=cst_data.get('Z_cst')
freq_cst=cst_data.get('freq_cst')
sigmaz=cst_data.get('sigmaz')
#'''
'''
Wake_potential_cst=cst_data.get('WPx_dipolar_cst')
s_cst=cst_data.get('s_cst_dipolar')
Z_cst=cst_data.get('Zx_dipolar_cst')
freq_cst=cst_data.get('freq_cst_dipolar')
'''
'''
Wake_potential_cst=cst_data.get('WPy_quadrupolar_cst')
s_cst=cst_data.get('s_cst_quadrupolar')
Z_cst=cst_data.get('Zy_quadrupolar_cst')
freq_cst=cst_data.get('freq_cst_quadrupolar')
'''

#--- Auxiliary variables
ds=s_cst[2]-s_cst[1]
s=np.arange(np.min(s_cst),np.max(s_cst),ds) #constant ds vector

#--- Obtain impedance Z with Fourier transform numpy.fft.fft
# MAKE A SYMMETRIC SIGNAL

# Interpolate charge distribution
# INTERPOLATE TO HAVE A CONSTANT ds. PLOT CST DS DISTRIBUTION
charge_dist_interp=np.interp(s, s_charge_dist, charge_dist/max(charge_dist))
Wake_potential_interp=np.interp(s, s_cst, Wake_potential_cst)

#lambdaf, f=Wsol.FFT(charge_dist_interp, ds/c, fmax=np.max(freq_cst), r=10.0)
#WPf, f=Wsol.FFT(Wake_potential_interp, ds/c, fmax=np.max(freq_cst), r=10.0)


lambdaf, f=Wsol.FFT(charge_dist_interp, ds/c, fmax=np.max(freq_cst), r=10.0)
WPf, f=Wsol.FFT(Wake_potential_interp, ds/c, fmax=np.max(freq_cst), r=10.0)

#lambdaf=np.exp(-4*np.pi**2*f*f*(sigmaz**2)/(2*c**2))

# Compute the impedance
Z = abs(- WPf / lambdaf) # * 2/(t_sample*ds/np.sqrt(np.pi)) #normalized according to CST wakesolver manual

# Plot Impedance and maximum frequency
fig = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig.gca()

# add CST fft result
ifreq_max_cst=np.argmax(Z_cst)
ax.plot(freq_cst[ifreq_max_cst]*1e-9, Z_cst[ifreq_max_cst], marker='o', markersize=4.0, color='black')
ax.annotate(str(round(freq_cst[ifreq_max_cst]*1e-9,2))+ ' GHz', xy=(freq_cst[ifreq_max_cst]*1e-9,Z_cst[ifreq_max_cst]), xytext=(1,1), textcoords='offset points', color='black') 
ax.plot(freq_cst*1.0e-9, Z_cst, lw=1.2, color='black', label='Z// from CST')

# add numpy.fft result (normalized)
factor=np.max(Z_cst)/np.max(Z)
ifreq_max=np.argmax(Z)
ax.plot(f[ifreq_max]*1e-9, Z[ifreq_max]*factor, marker='o', markersize=4.0, color='red')
ax.annotate(str(round(f[ifreq_max]*1e-9,2))+ ' GHz', xy=(f[ifreq_max]*1e-9,Z[ifreq_max]*factor), xytext=(1,1), textcoords='offset points', color='red') 
ax.plot(f*1.0e-9, Z*factor, lw=1.2, color='red', label='Z// from numpy FFT')

ax.set(title='Longitudinal impedance Z from CST magnitude' + '\n normalized by '+str(round(factor, 3)),
        xlabel='frequency [GHz]',
        ylabel='Z//(s) [Ohm]',         #ylim=(-8.0e4,8.0e4)
        xlim=(0.,np.max(freq_cst)*1e-9)
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()


# Plot charge dist and wake potential

fig = plt.figure(2, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig.gca()
ax.plot(s*1e3, charge_dist_interp, color='red', label='$\lambda$(s)')
ax.plot(s*1e3, Wake_potential_cst, color='orange', label='W||(s)')
ax.set(title='Wake potential and charge distribution',
        xlabel='s [mm]',
        ylabel='W||(s) [V/pC]',         #ylim=(-8.0e4,8.0e4)
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()


fig = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig.gca()
ax.plot(f, abs(lambdaf), color='red', label='$\lambda$(w)')
ax.plot(f, abs(WPf), color='orange', label='W||(w)')
ax.set(title='Wake potential and charge distribution',
        xlabel='s [mm]',
        ylabel='W||(s) [V/pC]',         #ylim=(-8.0e4,8.0e4)
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

#Obtain integrals (Parsevals identity)

int_t=np.sum(abs(charge_dist_interp)**2)
int_w=np.sum(2.0*abs(lambdaf)**2)/(len(f))
k_lambda=np.sqrt(int_t/int_w)
'''
int_t=np.sum(abs(Wake_potential_interp)**2)
int_w=np.sum(2.0*abs(WPf)**2)/(len(f))
k_WP=np.sqrt(int_t/int_w)
'''
t=cst_data.get('t')
dt=t[2]-t[1]
#VALUE IN 0 SHOULD BE THE SAME AS THE INTEGRAL FOR THE CHARGE DIST