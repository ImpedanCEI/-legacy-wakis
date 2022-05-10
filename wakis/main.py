'''
WAKIS main.py

Main routine of WAKIS. Obtains the Wake Potential and Impedance from 
pre-computed electromagnetic fields of structures with a passing beam

STEPS
-----
-Reads the input data dictionary with pickle
-Reads the 3d data of the Ez field from h5 file
-Performs the direct integration of the longitudinal Wake Potential
-Obtains the transverse Wake Potential through Panofsky Wenzel theorem
-Performs the fourier trnasform to obtain the Impedance
-Saves the data with pickle
-Plots the results

'''

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import argparse #[TODO]: pass the out folder as in-line command
import pickle as pk
import h5py as h5py
from scipy.constants import c

import solver_module as Wsol
import plot_module as Wplt

##############################################
# User global variables

OUT_FOLDER='out/'   #default: 'out/'

##############################################

#--------------------#
#      Read data     #
#--------------------#

# Set path
runs_path=os.getcwd() + '/runs/'
out_path=runs_path+OUT_FOLDER

# Read WarpX out dictionary
data = Wsol.read_WarpX(out_path=out_path)

# data checks (q, charge dist, units)
data = Wsol.check_data(data=data)

print('---------------------')
print('|   Running WAKIS   |')
print('---------------------')

t0 = time.time()

#-----------------------#
#     Obtain W||(s)     #
#-----------------------#

# Obtain the longitudinal wake potential from the Ez.h5 file
WP_3d, s = Wsol.calc_long_WP(data=data, out_path=out_path)
WP=WP_3d[1,1,:]

#-----------------------#
#      Obtain W⊥(s)     #
#-----------------------#

# Obtain the transverse wake potential from the longitudinal WP_3d
WPx, WPy = Wsol.calc_trans_WP(WP_3d, s, data=data)

#--------------------------------#
#      Obtain impedance Z||      #
#--------------------------------#

print('Obtaining longitudinal impedance...')

# Obtain charge distribution as a function of s, normalized
charge_dist = data.get('charge_dist')
q = data.get('q')
z0 = data.get('z0')
ds=s[2]-s[1]

timestep=np.argmax(charge_dist[len(z0)//2, :])         #max at cavity center
lambdas=np.interp(s, z0, charge_dist[:,timestep]/q)     #normed charge distribution

# Define maximum frequency
sigmaz=data.get('sigmaz')
fmax=1.01*c/sigmaz/3
# Obtain the ffts and frequency bins
lambdaf, f=Wsol.FFT(lambdas, ds/c, fmax=fmax, r=10.0)
WPf, f=Wsol.FFT(WP, ds/c, fmax=fmax, r=10.0)
# Obtain the impedance
Z = - WPf / lambdaf * 3400


#--------------------------------#
#      Obtain impedance Z⊥       #
#--------------------------------#

print('Obtaining transverse impedance...')

#---Zx⊥(w)

# Obtain the fft and frequency bins
WPxf, f=Wsol.FFT(WPx, ds/c, fmax=fmax, r=10.0)
# Obtain the impedance
Zx = - WPxf / lambdaf * 3400

#---Zy⊥(w)

# Obtain the fft and frequency bins
WPyf, f=Wsol.FFT(WPy, ds/c, fmax=fmax, r=10.0)
# Obtain the impedance
Zy = - WPyf / lambdaf * 3400


#-------------------------------------------------------

# Calculate elapsed time
t1 = time.time()
totalt = t1-t0
print('Calculation terminated in %ds' %totalt)

#-------------------#
#     Save data     #
#-------------------#

data = { 'WP' : WP, 
         's' : s,
         #'k_factor' : k_factor,
         'Z' : Z,
         'f' : f,
         'WPx' : WPx,
         'WPy' : WPy,
         'Zx' : Zx,
         'Zy' : Zy,
         'xsource' : data.get('xsource'),
         'ysource' : data.get('ysource'),
         'xtest' : data.get('xtest'),
         'ytest' : data.get('ytest'), 
         'sigmaz' : data.get('sigmaz'),
         'q' : q, 
         'lambda' : lambdas*q, #[C/m]
        }

with open(out_path + 'wake_solver.txt', 'wb') as handle:
    pk.dump(data, handle)

#-------------------#
#   Plot results    #
#-------------------#

Wplt.subplot_WAKIS(data=data)

#--------------------------#
#   Comparison with CST    #
#--------------------------#

'''
cst_path='/mnt/c/Users/elefu/Documents/CERN/WAKIS/Scripts/CST/'

Wplt.plot_WAKIS(data=data, 
                cst_data=Wplt.read_CST_out(cst_path), 
                flag_compare_cst=True, 
                flag_normalize=False
                )
'''