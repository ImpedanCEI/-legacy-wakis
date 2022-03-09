'''
Auxiliary functions for PyWake results read/plotting:
- Longitudinal wake potential
- Longitudinal Impedance 
- Transverse wake potential
- Transverse Impedance
'''

import numpy as np
import scipy.fftpack as fftpack
import os 
import matplotlib.pyplot as plt
import scipy.constants as spc  
import pickle as pk
import h5py as h5py

def FFT(Xt, dt, fmax=None, r=2.0, flag_zeropadding=True):
    ''' 
    Calculate the FFT of a signal
    -Xt: time domain signal with a constant dt
    -dt: resolution in time domain [s]
    -fmax: fmax to analyse, defined by the sigmat of the bunch: fmax=1/(3*sigmat)
    -r: relative length of the zero padding
    '''
    if fmax is None:
        fmax=1/dt

    # Define FFT parameters
    N=len(Xt)    # Number of time domain samples
    T=N*dt       # Total time [s]
    fres=1/T     # Resolution in frequency [Hz]
    dts=1/(2.0*fmax)    # Time window [s]  

    #Sample the time signal
    t=np.arange(0, T, dt)       # Original time array
    ts=np.arange(0, T, dts)     # Sampled time array
    Xs=np.interp(ts,t,Xt)       # Sampled time domain signal
    Ns=N/(dts/dt)               # Number of FFT samples

    #Add zero padding
    if flag_zeropadding:
        pad=int(r*Ns)          # Adjust by changing the relative length r  
        Xpad=np.append(np.append(np.zeros(pad), Xs), np.zeros(pad))

        Xs=Xpad 

    #Perform FFT
    Xfft=np.fft.fft(Xs)                     #FFT of the full spectrum
    ffft=np.fft.fftfreq(len(Xfft), dts)     #frequencies of the full specrtum
    mask= ffft >= 0

    Xf=2.0*Xfft[mask]/Ns    # Positive FFT, normalized
    f=ffft[mask]            # Positive frequencies

    print('------------------------------------------------------')
    print('Performing FFT')
    print(' - fmax = ' + str(fmax*1e-9) + ' GHz')
    print(' - fres = ' + str(fres*1e-6) + ' MHz')
    print(' - N samples = ' + str(Ns) + ' GHz' + '\n')

    #Parsevals identity
    Et=np.sum(abs(Xs)**2.0)
    Ef=np.sum(abs(Xfft)**2.0)/len(ffft)
    K=np.sqrt(Et/Ef)

    print('Parseval identity check')
    print('Energy(time)/Energy(frequency) = '+ str(K)+' == 1.0')
    print('Energy(time)-Energy(frequency) = '+ str(round(Et-Ef, 3))+' == 0.0')
    print('------------------------------------------------------')

    Xf=K*Xf

    return Xf, f

def DFT(Xt, dt, fmax=None, Nf=1000):
    ''' 
    Calculate the DFT of a signal
    -Xt: time domain signal with a constant dt
    -dt: resolution in time domain [s]
    -Nf:number of samples in frequency domain
    -fmax: fmax to analyse, defined by the sigmat of the bunch: fmax=1/(3*sigmat)
    '''
    if fmax is None:
        fmax=1/dt

    # Define FFT parameters
    N=len(Xt)    # Number of time domain samples
    T=N*dt       # Total time [s]
    fres=1/T     # Resolution in frequency [Hz]
    dts=1/(2.0*fmax)    # Time window [s]  

    #Sample the time signal
    t=np.arange(0, T, dt)       # Original time array
    ts=np.arange(0, T, dts)     # Sampled time array
    Xs=np.interp(ts,t,Xt)       # Sampled time domain signal
    Ns=N/(dts/dt)               # Number of FFT samples

    #Perform FFT
    Xf=fftpack.rfft(Xs, Nf)              #FFT of the full spectrum
    f=fftpack.rfftfreq(len(Xf), dts)     #frequencies of the full specrtum

    print('------------------------------------------------------')
    print('Performing DFT')
    print(' - fmax = ' + str(fmax*1e-9) + ' GHz')
    print(' - fres = ' + str(fres*1e-6) + ' MHz')
    print(' - N samples = ' + str(Ns) + ' GHz' + '\n')
    
    #Parsevals identity
    Et=np.sum(abs(Xs)**2.0)
    Ef=(Xf[0]**2 + 2 * np.sum(Xf[1:]**2)) / len(f)
    K=np.sqrt(Et/Ef)
    
    print('Parseval identity check')
    print('Energy(time)/Energy(frequency) = '+ str(K)+' == 1.0')
    print('Energy(time)-Energy(frequency) = '+ str(round(Et-Ef, 3))+' == 0.0')
    print('------------------------------------------------------')

    mask=np.arange(Nf)%2.0 == 0.0  #Take the imaginary values of Xf
    Z=1j*np.zeros(len(Xf[mask]))
    Zf=np.zeros(len(Xf[mask]))

    if Nf%2.0 == 0.0:
        Re = ~mask
        Re[-1]=False
        Im = mask
        Im[0]=False

        Z[1:]=Xf[Re]+1j*Xf[Im]   #Reconstruct de complex array
        Z[0]=Xf[0]               #Take the DC value

        Zf[1:]=f[Im]
        Zf[0]=0.0

    else:
        Re = ~mask
        Im = mask
        Im[0]=False

        Z[1:]=Xf[Re]+1j*Xf[Im]   #Reconstruct de complex array
        Z[0]=Xf[0]               #Take the DC value

        Zf[1:]=f[Im]
        Zf[0]=0.0

    Xf=K*Z/(Ns/2)
    f=Zf

    return Xf, f

if __name__ == "__main__":
    
    N=500
    T=100
    w=2.0*np.pi/T
    t=np.linspace(0,T,N)
    dt=T/N

    Xt1=1.0*np.sin(5.0*w*t)
    Xt2=2.0*np.sin(10.0*w*t)
    Xt3=0.5*np.sin(20.0*w*t)

    Xt=Xt1+Xt2+Xt3

    # Plot time domain
    fig = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
    ax=fig.gca()
    ax.plot(t, Xt, marker='o', markersize=1.0, color='black', label='Xt1+Xt2+Xt3')
    ax.plot(t, Xt1, marker='o', markersize=1.0, color='blue', label='Xt1')
    ax.plot(t, Xt2, marker='o', markersize=1.0, color='red', label='Xt2')
    ax.plot(t, Xt3, marker='o', markersize=1.0, color='green', label='Xt3')

    ax.grid(True, color='gray', linewidth=0.2)
    ax.legend(loc='best')
    plt.show()

    Xf, f = FFT(Xt, dt, fmax=0.5, flag_zeropadding=True, r=3.0)
    Xdft, fdft = DFT(Xt, dt, fmax=0.5, Nf=1000)

    # Plot frequency domain
    fig = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
    ax=fig.gca()
    ax.plot(f, abs(Xf), marker='o', markersize=3.0, color='blue', label='FFT')
    ax.plot(fdft, abs(Xdft), marker='o', markersize=3.0, color='red', label='DFT')
    ax.grid(True, color='gray', linewidth=0.2)
    ax.legend(loc='best')
    plt.show()

