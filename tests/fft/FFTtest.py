# test for CST WP to Impedance FFT

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c

def read_cst_1d(path, file):
    '''
    Read CST plot data saved in ASCII .txt format
    '''
    entry=file.split('.')[0]
    X = []
    Y = []

    i=0
    with open(path+file) as f:
        for line in f:
            i+=1
            columns = line.split()

            if i>1 and len(columns)>1:

                X.append(float(columns[0]))
                Y.append(float(columns[1]))

    X=np.array(X)
    Y=np.array(Y)

    return {'X':X , 'Y': Y}

#path to files
path = os.getcwd() + '/'

#bunch parameters
q = 1e-9
sigmaz = 18.737*1e-3

# read charge dist in distance lambda(z)
d1 = read_cst_1d(path,'lambda.txt')
lambdaz = d1['Y'] 
z = d1['X']*1e-3

# read wake potential WPz(s)
d2 = read_cst_1d(path, 'WP.txt')
WP = d2['Y']
s = d2['X']*1e-3

# interpolate charge dist to s
lambdas = np.interp(s, z, lambdaz/q)

# Set up the DFT computation
ds = s[2]-s[1]
fmax=1*c/sigmaz/3
N=int((c/ds)//fmax*1001) #to obtain a 1000 sample single-sided DFT

# Obtain DFTs
lambdafft = np.fft.fft(lambdas*c, n=N)
WPfft = np.fft.fft(WP*1e12, n=N)
ffft=np.fft.fftfreq(len(WPfft), ds/c)

# Mask invalid frequencies
mask  = np.logical_and(ffft >= 0 , ffft < fmax)
WPf = WPfft[mask]*ds
lambdaf = lambdafft[mask]*ds
f = ffft[mask]            # Positive frequencies

# Compute the impedance
Z = - WPf / lambdaf

# Plot impedance
fig = plt.figure(2, figsize=(8,5), dpi=150, tight_layout=True)
ax=fig.gca()
ax.plot(f*1e-9, abs(Z), lw=1.2, c='b', label = 'abs Z(w)')
ax.plot(f*1e-9, np.real(Z), lw=1.2, c='r', ls='--', label = 'Re Z(w)')
ax.plot(f*1e-9, np.imag(Z), lw=1.2, c='g', ls='--', label = 'Im Z(w)')

# Compare with CST
d = read_cst_1d(path,'Z.txt')
ax.plot(d['X'], d['Y'], lw=1, c='k', ls='--', label='abs Z(w) CST')

ax.set( title='Impedance Z(w)',
        xlabel='f [GHz]',
        ylabel='Z(w) [$\Omega$]',   
        xlim=(0.,np.max(f)*1e-9)    )
ax.legend(loc='upper left')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

#fig.savefig(path+'Zz.png', bbox_inches='tight')