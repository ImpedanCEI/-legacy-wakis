'''
Analytical study to test WAKIS FFT routine 
Based on the formulas from C.Zannini presentation
link: https://indico.cern.ch/event/844487/contributions/3545750/attachments/1900674/3138204/ISC_wake_function_v2.pdf
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi, epsilon_0, e, m_p
from scipy.special import kn

import solver_module as Wsol

T=100*1e-9
N=10000
tau = np.linspace(-T/2, T/2, N) #[s]
dtau = abs(tau[2]-tau[1])

# Wake function formula
L = 1 #[m]
b = 0.03 #[m]
beta = 0.5

gamma = 1/np.sqrt(1-beta**2)
a = b/(c*beta*gamma)
Z_0 = 376.73 #[Ohm]

W = L*Z_0 / (4*pi*c**2*beta**3*gamma**4*np.sqrt((tau**2 + a**2)**3))  #[V/pC/m]

# Impedance formula

f = np.linspace(1e4, 1e10, 100000)
Z = 1j * Z_0*L*2*pi*f/(2*pi*b*c*beta**2*gamma**3) * kn(1,a*2*pi*f) #[Ohm/m]

# Impedance from FFT
 
Z_fft, f_fft = Wsol.FFT(W, dtau, flag_zeropadding=False)
Z_fft=Z_fft

# Impedance from numpy FFT

Xfft=np.fft.fft(W)                #FFT of the full spectrum
ffft=np.fft.fftfreq(len(Xfft), dtau)   #frequencies of the full specrtum
mask= ffft >= 0

Z_np=Xfft[mask]*T/N    # Positive FFT, normalized
f_np=ffft[mask]    # Positive frequencies

# Wake function from inverse FFT

fres=f_fft[2]-f_fft[1]
Zi= np.interp(f_np,f, Z)
Z_full = np.append(Zi, Zi[::-1])/2

#Xifft = np.fft.ifft(Z_full)
#W_ifft=np.append(Xifft[0], np.flip(Xifft[1:len(Z_full)//2])-Xifft[len(Z_full)//2+1:])
#t=np.linspace(0,1/fres,len(Z_full)//2) - 0.5/fres

W_ifft = abs(np.fft.ifft(Xfft))
t=np.linspace(0,1/fres,len(Xfft)) - 0.5/fres

# Plot 
fig = plt.figure(1, figsize=(8,5), dpi=150, tight_layout=True)
ax=fig.gca()

ax.plot(t*1e9, W_ifft*1e-3, lw=1.2, color='red', label='Wf(t) from numpy.ifft')
ax.plot(tau*1e9, W*1e-3, lw=1.2, color='black', ls = '--', label = 'Wf(t) from formula')

ax.set(title='Wake function W(t)',
        xlabel='time [ns]',
        ylabel='Wake function [V/pC/mm]',
        )
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

fig = plt.figure(2, figsize=(8,5), dpi=150, tight_layout=True)
ax=fig.gca()
#ax.plot(f, abs(Z), lw=1.2, color='black', ls = '--', label='Z(f) formula')
ax.plot(f_fft, abs(Z_fft), lw=1.2, color='blue', label='Z(f) from WAKIS FFT') #to match Z, it needs to be multiplied by the number of samples
ax.plot(f_np, abs(Z_np), lw=1.2, color='red', ls = '--', label='Z(f) from numpy.FFT')

ax.set(title='Impedance Z(f)',
        xlabel='f [Hz]',
        ylabel='Impedance [$\Omega$/m]',
        )
ax.set_xscale('log')
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()