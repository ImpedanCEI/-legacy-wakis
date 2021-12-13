'''
cst_to_dict.py

File for postprocessing logfiles from cst

--- Reads 1 log file and plots the field and the frequency
--- Reads all log files and dumps the E(z,t) matrix into a dict
--- Saves the dict in a out file 'cst.txt' with pickle

'''

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import glob, os
import scipy as sc  
from scipy import constants
from copy import copy
import pickle as pk

c=constants.c

#--- read one Ez/ file
fname = 'Ez_050'

#Initialize variables
Ez=[]
t=[]
i=0 

with open('Ez/'+fname+'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Ez.append(float(columns[1]))
            t.append(float(columns[0]))

Ez=np.array(Ez) # in V/m
t=np.array(t)*1.0e-9   # in s

#close file
f.close()

#--- read all Ez files

Ez_t=np.zeros((len(glob.glob("Ez/*.txt")),len(t)))
k=0
i=0

for file in sorted(glob.glob("Ez/*.txt")):
    print('Scanning file '+ file)
    with open(file) as f:
        for line in f:
            i+=1
            columns = line.split()

            if i>1 and len(columns)>1:

                Ez_t[k,i-3]=(float(columns[1]))
    k+=1
    i=0
    #close file
    f.close()
    
print('Finished scanning files')

#--- read charge distribution file
fname = 'lambda'

#Initialize variables
charge_dist=[]
distance=[]
i=0

with open('lambda/'+fname+'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            charge_dist.append(float(columns[1]))
            distance.append(float(columns[0]))

charge_dist=np.array(charge_dist) # in C/m
distance=np.array(distance)*1.0e-3   # in m

#close file
f.close()

#--- read wake potential obtained from CST

Wake_potential_cst=[]
Wake_potential_interfaces=[]
Wake_potential_testbeams=[]
WPx=[]
WPy=[]
s_cst=[]
i=0
# Longitudinal wake potential
# fname='Wake_potential' #default direct WP
fname='WPz'
with open(fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Wake_potential_cst.append(float(columns[1]))
            s_cst.append(float(columns[0]))

Wake_potential_cst=np.array(Wake_potential_cst) # in V/pC
s_cst=np.array(s_cst)*1.0e-3  # in [m]

#close file
f.close()

# indirect WP with the indirect interfaces method
fname='indirect_interf_WP'
i=0
with open(fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Wake_potential_interfaces.append(float(columns[1]))

Wake_potential_interfaces=np.array(Wake_potential_interfaces) # in V/pC

#close file
f.close()

# indirect WP with the indirect testbeams method
fname='indirect_test_WP'
i=0
with open(fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Wake_potential_testbeams.append(float(columns[1]))

Wake_potential_testbeams=np.array(Wake_potential_testbeams) # in V/pC

#close file
f.close()


# Transverse wake potential
fname='WPx'
i=0
with open(fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            WPx.append(float(columns[1]))

WPx=np.array(WPx) # in V/pC

#close file
f.close()

fname='WPy'
i=0
with open(fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            WPy.append(float(columns[1]))

WPy=np.array(WPy) # in V/pC

#close file
f.close()


#--- read impedance obtained from CST

Z_cst=[]
Zx=[]
Zy=[]
freq_cst=[]

# Longitudinal impedance
# fname='Impedance'  #default direct impedance
fname='Zz'
i=0
with open(fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Z_cst.append(float(columns[1]))
            freq_cst.append(float(columns[0]))

Z_cst=np.array(Z_cst) # in [Ohm]
freq_cst=np.array(freq_cst)*1e9  # in [Hz]

#close file
f.close()

# Transverse impedance
fname='Zx'
i=0
with open(fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Zx.append(float(columns[1]))

Zx=np.array(Zx) # in V/pC

#close file
f.close()

fname='Zy'
i=0
with open(fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Zy.append(float(columns[1]))

Zy=np.array(Zy) # in V/pC

#close file
f.close()

#...................#
#     1D  Plots     #
#...................#

#--- 1 Ez file
# Plot electric field

fig1 = plt.figure(10, figsize=(6,4), dpi=200, tight_layout=True)
ax1=fig1.gca()
ax1.plot(t*1.0e9, Ez, lw=1.2, color='g', label='Ez CST')
ax1.set(title='Electric field at cavity center',
        xlabel='t [ns]',
        ylabel='$E [V/m]$',         #ylim=(-8.0e4,8.0e4)
        )
ax1.legend(loc='best')
ax1.grid(True, color='gray', linewidth=0.2)
plt.show()

# Plot frequency

freq=np.fft.fftfreq(len(t[300:]), d=(t[1]-t[0])*1.0e9)
Ez_fft=np.fft.fft(Ez[300:])
Amp=np.abs(Ez_fft)
Amp_max=np.argmax(Amp)

fig2 = plt.figure(10, figsize=(6,4), dpi=200, tight_layout=True)
ax2=fig2.gca()
ax2.plot(freq[Amp_max], Amp[Amp_max], marker='o', markersize=3.0, color='pink')
ax2.annotate(str(round(freq[Amp_max],2))+ ' GHz', xy=(freq[Amp_max],Amp[Amp_max]), xytext=(1,1), textcoords='offset points', color='grey') 
#arrowprops=dict(color='blue', shrink=1.0, headwidth=4, frac=1.0)
ax2.plot(freq, Amp, lw=1, color='r', label='fft CST')
#ax2.plot(freq, Amp.imag, lw=1.2, color='r', label='Imaginary')
ax2.set(title='Frequency of Electric field at cavity center',
        xlabel='f [GHz]',
        ylabel='Amplitude [dB]',   
        ylim=(0,np.max(Amp)*1.3),
        xlim=(0,np.max(freq))      
        )
ax2.legend(loc='best')
ax2.grid(True, color='gray', linewidth=0.2)
plt.show()

#--- Wake_potential and Impedance

# Plot wake potential and charge distribution

# Longitudinal wake potentials
fig2 = plt.figure(20, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig2.gca()
ax.plot(s_cst*1.0e3, Wake_potential_cst, lw=1.2, color='orange', label='Direct W// from CST')
ax.plot(s_cst*1.0e3, Wake_potential_interfaces, lw=1.2, color='magenta', label='Indirect interfaces W// from CST')
ax.plot(s_cst*1.0e3, Wake_potential_testbeams, lw=1.2, color='cyan', label='Indirect testbeams W// from CST')
ax.set(title='Longitudinal wake potential from CST',
        xlabel='s [mm]',
        ylabel='W//(s) [V/pC]',         #ylim=(-8.0e4,8.0e4)
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

# Transverse wake potentials
fig3 = plt.figure(30, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig3.gca()
ax.plot(s_cst*1.0e3, WPx, lw=1.2, color='green', label='Transverse W⊥,x from CST')
ax.plot(s_cst*1.0e3, WPx, lw=1.2, color='red', label='Transverse W⊥,y from CST')
ax.set(title='Transverse wake potential from CST',
        xlabel='s [mm]',
        ylabel='W⊥(s) [V/pC]',         #ylim=(-8.0e4,8.0e4)
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

# charge distribution
fig5 = plt.figure(50, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig5.gca()
ax.plot(distance*1.0e3, charge_dist, lw=1.2, color='r', label='$\lambda$ from CST')
ax.set(title='Charge distribution from CST',
        xlabel='distance [mm]',
        ylabel='$\lambda$(s) [C/m]',         #ylim=(-8.0e4,8.0e4)
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

# Plot impedance and maximum frequency
ifreq_max=np.argmax(Z_cst)
fig4 = plt.figure(40, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig4.gca()
ax.plot(freq_cst[ifreq_max]*1e-9, Z_cst[ifreq_max], marker='o', markersize=4.0, color='pink')
ax.annotate(str(round(freq_cst[ifreq_max]*1e-9,2))+ ' GHz', xy=(freq_cst[ifreq_max]*1e-9,Z_cst[ifreq_max]), xytext=(1,1), textcoords='offset points', color='red') 
ax.plot(freq_cst*1.0e-9, Z_cst, lw=1.2, color='red', label='Z// from CST')
ax.set(title='Longitudinal impedance Z from CST',
        xlabel='frequency [GHz]',
        ylabel='Z//(s) [Ohm]',         #ylim=(-8.0e4,8.0e4)
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

#------------------------------------#
#        Create cst_out file         #
#------------------------------------#

#--- declare aux variables from CST logfile info
unit=1e-3

# width of the rectangular beam pipe (x direction)
w_pipe = 15*unit
# height of the rectangular beam pipe (y direction)
h_pipe = 15*unit
# total length of the domain
L_pipe = 50*unit 
 
# width of the rectangular cavity (x direction)
w_cavity = 50*unit
# height of the rectangular beam pipe (y direction)
h_cavity = 50*unit
# length of each side of the beam pipe (z direction)
L_cavity = 30*unit 

# mesh cells per direction 
nx = 55                     
ny = 55
nz = k

# mesh bounds
xmin = -0.55*w_cavity
xmax = 0.55*w_cavity
ymin = -0.55*h_cavity
ymax = 0.55*h_cavity
zmin = -L_pipe
zmax = L_pipe

# beam sigma in time and longitudinal direction
sigmat= 1.000000e-09/16.     
sigmaz = sigmat*sc.constants.c 

#setup vectors
x=np.linspace(xmin,xmax,nx)
y=np.linspace(ymin,ymax,ny)
z=np.linspace(zmin,zmax,nz)

#define integration path
xtest=0.0
ytest=0.0

#--- save the matrix into a txt

data = { 'Ez' : Ez_t, #shape = (k, len(t))
         't' : t, #time [s]
         'nz' : nz, #mesh cells in z direction
         'nt' : len(t), #number of timesteps
         'charge_dist' : charge_dist, # [C/m]
         'distance' : distance, # [m]
         'Wake_potential_cst' : Wake_potential_cst, # [V/pC]
         'Wake_potential_interfaces' : Wake_potential_interfaces, 
         'Wake_potential_testbeams': Wake_potential_testbeams, 
         'WPx_cst' : WPx,
         'WPy_cst' : WPy,
         's_cst' : s_cst, # [m]
         'Z_cst' : Z_cst, # [Ohm]
         'Zx_cst' : Zx,
         'Zy_cst' : Zy,
         'freq_cst' : freq_cst, # [Hz]
         'x' : x,
         'y' : y,
         'z' : z,
         'w_cavity' : w_cavity,
         'h_cavity' : h_cavity,
         'L_cavity' : L_cavity,
         'w_pipe' : w_pipe,
         'h_pipe' : h_pipe,
         'L_pipe' : L_pipe,
         'sigmaz' : sigmaz,
         'xtest' : xtest,
         'ytest' : ytest
        }
# write the dictionary to a txt using pickle module
with open('cst_out.txt', 'wb') as handle:
    pk.dump(data, handle)

