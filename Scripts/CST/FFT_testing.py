'''
FFT_testing.py

Benchmark of the FFT method with CST files
'''
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import scipy as sc  
from copy import copy
import pickle as pk


def DFT(x):
    """
    Function to calculate the 
    discrete Fourier Transform 
    of a 1D real-valued signal x
    """

    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    
    X = np.dot(e, x)
    
    return X


class Fourier:
    def __init__(self, dft, freqs):
        self.dft = dft
        self.freqs = freqs

def CST_DFT(F, dt, N): 
        #function to obtain the DFT with 1000 samples
        #--F: function in time domain
        #--dt: time sampling width
        #--N: number of time samples

        #define frequency domain
        N_samples=1000*2  # number of samples in frequency domain
        f_max = 5.5*1e9     # maximum freq in GHz
        t_sample=1/(dt)/2/f_max+1 #obtains the time window to sample the time domain data

        #define frequency bins
        freq=np.linspace(-f_max,f_max,N_samples) # [Hz]
        #Add padding
        padding=int((N_samples*t_sample-N))    #length of the padding with zero
        F=np.append(F,np.zeros(padding))

        dft=np.zeros_like(freq)*1j
        for m in range(N_samples):
            for k in range(0,N+padding, int(t_sample)):
                dft[m]=dft[m]+F[k]*np.exp(-1j*k*dt*freq[m]) 

        dft=dt/np.sqrt(np.pi)*dft #Magnitude in [Ohm]
        freqs=freq*1e-9 #in [GHz]
        return Fourier(dft,freq)    

#--- read the cst dictionary
with open('cst_out.txt', 'rb') as handle:
  cst_data = pk.loads(handle.read())


charge_dist=cst_data.get('charge_dist')
s_charge_dist=cst_data.get('s_charge_dist')
Wake_potential_cst=cst_data.get('WP_cst')
s_cst=cst_data.get('s_cst')
Z_cst=cst_data.get('Z_cst')
freq_cst=cst_data.get('freq_cst')

#--- Auxiliary variables

ds=s_cst[2]-s_cst[1]
c=sc.constants.c
s=s_cst

#--- Obtain impedance Z with Fourier transform numpy.fft.fft

# to increase the resolution of fft, a longer wake length is needed
f_max=5.5*1e9
t_sample=1/(ds/c)/2/f_max+1 #obtains the time window to sample the time domain data
N_samples=1001 #int((len(s)+2*padding)/t_sample)
padding=int((N_samples*t_sample-len(s)))
print('Performing FFT with '+str(N_samples)+' samples')
#print('Frequency bin resolution '+str(round(1/(len(s)*ds/c)*1e-9,2))+ ' GHz')
#print('Frequency range: 0 - '+str(round(f_max*1e-9,2)) +' GHz')

# Interpolate charge distribution
charge_dist_interp=np.interp(s, s_charge_dist, charge_dist/np.max(charge_dist))
# Padding with zeros 
charge_dist_padded=np.append(np.append(np.zeros(padding),charge_dist_interp), np.zeros(padding))
Wake_potential_padded=np.append(np.append(np.zeros(padding), Wake_potential_cst), np.zeros(padding))
# Obtain the FFT
charge_dist_fft=np.fft.fft(charge_dist_padded[0:-1:int(t_sample)])
Wake_potential_fft=np.fft.fft(Wake_potential_padded[0:-1:int(t_sample)])
# Obtain the frequencies
freq = np.fft.fftfreq(len(Wake_potential_padded[:-1:int(t_sample)]), ds/c*int(t_sample)) #Hz
# Compute the impedance
Z = abs(- Wake_potential_fft / charge_dist_fft) * 2/(t_sample*ds/np.sqrt(np.pi)) #normalized according to CST wakesolver manual

print('Frequency bin resolution '+str(round(1/(len(s)*ds/c)*1e-9,2))+ ' GHz')
print('Frequency bin resolution after padding '+str(round((freq[2]-freq[1])*1e-6,2))+ ' MHz')
print('Frequency range: 0 - '+str(round(np.max(freq)*1e-9,2)) +' GHz')

#--- Obtain the frequency with DFT function
charge_dist_dft=DFT(charge_dist_padded[0:-1:int(t_sample)])
Wake_potential_dft=DFT(Wake_potential_padded[0:-1:int(t_sample)])
# Compute the impedance
Z_dft = abs(- Wake_potential_dft / charge_dist_dft) / len(charge_dist_dft)//2 #normalized according to DFT theory
freq_dft=np.arange(len(Z_dft))/(ds*t_sample)/len(Z_dft)

#--- Obtain the frequency with CST_DFT function
charge_dist_dft=CST_DFT(charge_dist_interp, ds/c, len(s) )
Wake_potential_dft=CST_DFT(Wake_potential_cst, ds/c, len(s))
# Compute the impedance
Z_dft = abs(- Wake_potential_dft.dft / charge_dist_dft.dft) / 2/(t_sample*ds/np.sqrt(np.pi))
freq_dft=Wake_potential_dft.freqs

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
ifreq_max=np.argmax(Z[0:len(Z)//2])
ax.plot(freq[ifreq_max]*1e-9, Z[ifreq_max]*factor, marker='o', markersize=4.0, color='red')
ax.annotate(str(round(freq[ifreq_max]*1e-9,2))+ ' GHz', xy=(freq[ifreq_max]*1e-9,Z[ifreq_max]*factor), xytext=(1,1), textcoords='offset points', color='red') 
ax.plot(freq[0:len(Z)//2]*1.0e-9, Z[0:len(Z)//2]*factor, lw=1.2, color='red', label='Z// from numpy FFT')

# add DFT result (normalized)
factor=np.max(Z_cst)/np.max(Z_dft)
ifreq_max=np.argmax(Z_dft[0:len(Z_dft)//2])
ax.plot(freq[ifreq_max]*1e-9, Z_dft[ifreq_max]*factor, marker='o', markersize=4.0, color='b')
ax.annotate(str(round(freq[ifreq_max]*1e-9,2))+ ' GHz', xy=(freq[ifreq_max]*1e-9,Z_dft[ifreq_max]*factor), xytext=(1,1), textcoords='offset points', color='b') 
ax.plot(freq_dft[0:len(Z_dft)//2]*1.0e-9, Z_dft[0:len(Z_dft)//2]*factor, lw=1.2, color='b', label='Z// from DFT')

ax.set(title='Longitudinal impedance Z from CST magnitude \n normalized by '+str(round(factor, 3)),
        xlabel='frequency [GHz]',
        ylabel='Z//(s) [Ohm]',         #ylim=(-8.0e4,8.0e4)
        xlim=(0.,np.max(freq_cst)*1e-9)
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

'''
# Plot charge dist and wake potential

fig = plt.figure(2, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig.gca()
ax.plot(s*1e3, charge_dist_interp*np.max(Wake_potential_cst), color='red', label='$\lambda$(s)')
ax.plot(s*1e3, Wake_potential_cst, color='orange', label='W||(s)')
ax.set(title='Wake potential and charge distribution',
        xlabel='s [mm]',
        ylabel='W||(s) [V/pC]',         #ylim=(-8.0e4,8.0e4)
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()
'''