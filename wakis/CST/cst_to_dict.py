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
from scipy.constants import c
import pickle as pk

c=constants.c
data_path='data/'
flag_show_plots = False

#--- read one Ez/ file
fname = 'Ez_050'

#Initialize variables
Ez=[]
t=[]
i=0 

with open('data/Ez/'+fname+'.txt') as f:
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

Ez_t=np.zeros((len(glob.glob("data/Ez/*.txt")),len(t)))
k=0
i=0

for file in sorted(glob.glob("data/Ez/*.txt")):
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


#--------------------------------#
#   Charge distribution files    #
#--------------------------------#   

charge_dist=[]
charge_dist_time=[]
charge_dist_spectrum=[]
current=[]
distance=[]
t_charge_dist=[]

# Charge distribution in distance (s)
fname = 'lambda'
i=0

with open('data/lambda/'+fname+'.txt') as f:
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

# Charge distribution in time
fname = 'charge_dist_time'
i=0

with open('data/lambda/'+fname+'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            charge_dist_time.append(float(columns[1]))
            t_charge_dist.append(float(columns[0]))

charge_dist_time=np.array(charge_dist_time) # in C
t_charge_dist=np.array(t_charge_dist)*1.0e-9   # in s
f.close()

# Charge distribution spectrum
fname = 'charge_dist_spectrum'
i=0

with open('data/lambda/'+fname+'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            charge_dist_spectrum.append(float(columns[1]))

charge_dist_spectrum=np.array(charge_dist_spectrum) # in C
f.close()

# Current
fname = 'current'
i=0

with open('data/lambda/'+fname+'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            current.append(float(columns[1]))

current=np.array(current) # J=rho*v
f.close()

#---------------------------#
#   Wake Potential files    #
#---------------------------#   

# Longitudinal wake potential [DIRECT method]
WP=[]
s_cst=[]

fname='WP'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            WP.append(float(columns[1]))
            s_cst.append(float(columns[0]))

WP=np.array(WP) # in V/pC
s_cst=np.array(s_cst)*1.0e-3  # in [m]

#close file
f.close()


# Longitudinal wake potential [INDIRECT method]  
Indirect_WP_interfaces=[]
Indirect_WP_testbeams=[]

fname='indirect_interf_WP'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Indirect_WP_interfaces.append(float(columns[1]))

Indirect_WP_interfaces=np.array(Indirect_WP_interfaces) # in V/pC
f.close()

fname='indirect_test_WP'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Indirect_WP_testbeams.append(float(columns[1]))

Indirect_WP_testbeams=np.array(Indirect_WP_testbeams) # in V/pC
f.close()


# Transverse wake potential [xytest==0] [xysource==0]
WPx=[]
WPy=[]

fname='WPx'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            WPx.append(float(columns[1]))

WPx=np.array(WPx) # in V/pC
f.close()

fname='WPy'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            WPy.append(float(columns[1]))

WPy=np.array(WPy) # in V/pC
f.close()

# Dipolar wake potential [xysource!=0]
WPx_dipolar=[]
WPy_dipolar=[]
WPx_dipolarX=[]
WPy_dipolarX=[]
WPx_dipolarY=[]
WPy_dipolarY=[]
s_cst_dipolar=[]

fname='WPx_dipolar'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            WPx_dipolar.append(float(columns[1]))
            s_cst_dipolar.append(float(columns[0]))

WPx_dipolar=np.array(WPx_dipolar) # in V/pC
s_cst_dipolar=np.array(s_cst_dipolar)*1.0e-3  # in [m]
f.close()

fname='WPy_dipolar'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            WPy_dipolar.append(float(columns[1]))

WPy_dipolar=np.array(WPy_dipolar) # in V/pC
f.close()

fname='WPx_dipolarX'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            WPx_dipolarX.append(float(columns[1]))

WPx_dipolarX=np.array(WPx_dipolarX) # in V/pC
f.close()

fname='WPy_dipolarX'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            WPy_dipolarX.append(float(columns[1]))

WPy_dipolarX=np.array(WPy_dipolarX) # in V/pC
f.close()

fname='WPx_dipolarY'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            WPx_dipolarY.append(float(columns[1]))

WPx_dipolarY=np.array(WPx_dipolarY) # in V/pC
f.close()

fname='WPy_dipolarY'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            WPy_dipolarY.append(float(columns[1]))

WPy_dipolarY=np.array(WPy_dipolarY) # in V/pC
f.close()


# Quadrupolar wake potential [xytest!=0]
WPx_quadrupolar=[]
WPy_quadrupolar=[]
WPx_quadrupolarX=[]
WPy_quadrupolarX=[]
WPx_quadrupolarY=[]
WPy_quadrupolarY=[]
s_cst_quadrupolar=[]

fname='WPx_quadrupolar'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            WPx_quadrupolar.append(float(columns[1]))
            s_cst_quadrupolar.append(float(columns[0]))

WPx_quadrupolar=np.array(WPx_quadrupolar) # in V/pC
s_cst_quadrupolar=np.array(s_cst_quadrupolar)*1.0e-3  # in [m]
f.close()

fname='WPy_quadrupolar'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            WPy_quadrupolar.append(float(columns[1]))

WPy_quadrupolar=np.array(WPy_quadrupolar) # in V/pC
f.close()

fname='WPx_quadrupolarX'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            WPx_quadrupolarX.append(float(columns[1]))

WPx_quadrupolarX=np.array(WPx_quadrupolarX) # in V/pC
f.close()

fname='WPy_quadrupolarX'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            WPy_quadrupolarX.append(float(columns[1]))

WPy_quadrupolarX=np.array(WPy_quadrupolarX) # in V/pC
f.close()

fname='WPx_quadrupolarY'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            WPx_quadrupolarY.append(float(columns[1]))

WPx_quadrupolarY=np.array(WPx_quadrupolarY) # in V/pC
f.close()

fname='WPy_quadrupolarY'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            WPy_quadrupolarY.append(float(columns[1]))

WPy_quadrupolarY=np.array(WPy_quadrupolarY) # in V/pC
f.close()

#---------------------------#
#      Impedance files      #
#---------------------------#  

# Longitudinal Impedance [DIRECT method]
Z=[]
freq_cst=[]
fname='Z'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Z.append(float(columns[1]))
            freq_cst.append(float(columns[0]))

Z=np.array(Z) # in [Ohm]
freq_cst=np.array(freq_cst)*1e9  # in [Hz]
f.close()

# Transverse Impedance [xytest=0]
Zx=[]
fname='Zx'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Zx.append(float(columns[1]))

Zx=np.array(Zx) # in V/pC
f.close()

Zy=[]
fname='Zy'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Zy.append(float(columns[1]))

Zy=np.array(Zy) # in V/pC
f.close()

# Transverse dipolar Impedance [xysource!=0]
Zx_dipolar=[]
Zy_dipolar=[]
Zx_dipolarX=[]
Zy_dipolarX=[]
Zx_dipolarY=[]
Zy_dipolarY=[]
freq_cst_dipolar=[]

fname='Zx_dipolar'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Zx_dipolar.append(float(columns[1]))

Zx_dipolar=np.array(Zx_dipolar) # in V/pC
f.close()

fname='Zy_dipolar'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Zy_dipolar.append(float(columns[1]))
            freq_cst_dipolar.append(float(columns[0]))

Zy_dipolar=np.array(Zy_dipolar) # in V/pC
freq_cst_dipolar=np.array(freq_cst_dipolar)*1e9  # in [Hz]
f.close()

fname='Zx_dipolarX'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Zx_dipolarX.append(float(columns[1]))

Zx_dipolarX=np.array(Zx_dipolarX) # in V/pC
f.close()

fname='Zy_dipolarX'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Zy_dipolarX.append(float(columns[1]))

Zy_dipolarX=np.array(Zy_dipolarX) # in V/pC
f.close()

fname='Zx_dipolarY'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Zx_dipolarY.append(float(columns[1]))

Zx_dipolarY=np.array(Zx_dipolarY) # in V/pC
f.close()

fname='Zy_dipolarY'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Zy_dipolarY.append(float(columns[1]))

Zy_dipolarY=np.array(Zy_dipolarY) # in V/pC
f.close()


# Transverse quadrupolar Impedance [xytest!=0]
Zx_quadrupolar=[]
Zy_quadrupolar=[]
Zx_quadrupolarX=[]
Zy_quadrupolarX=[]
Zx_quadrupolarY=[]
Zy_quadrupolarY=[]
freq_cst_quadrupolar=[]

fname='Zx_quadrupolar'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Zx_quadrupolar.append(float(columns[1]))
            freq_cst_quadrupolar.append(float(columns[0]))

Zx_quadrupolar=np.array(Zx_quadrupolar) # in V/pC
freq_cst_quadrupolar=np.array(freq_cst_quadrupolar)*1e9  # in [Hz]
f.close()

fname='Zy_quadrupolar'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Zy_quadrupolar.append(float(columns[1]))

Zy_quadrupolar=np.array(Zy_quadrupolar) # in V/pC
f.close()

fname='Zx_quadrupolarX'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Zx_quadrupolarX.append(float(columns[1]))

Zx_quadrupolarX=np.array(Zx_quadrupolarX) # in V/pC
f.close()

fname='Zy_quadrupolarX'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Zy_quadrupolarX.append(float(columns[1]))

Zy_quadrupolarX=np.array(Zy_quadrupolarX) # in V/pC
f.close()

fname='Zx_quadrupolarY'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Zx_quadrupolarY.append(float(columns[1]))

Zx_quadrupolarY=np.array(Zx_quadrupolarY) # in V/pC
f.close()

fname='Zy_quadrupolarY'
i=0
with open('data/'+fname +'.txt') as f:
    for line in f:
        i+=1
        columns = line.split()

        if i>1 and len(columns)>1:

            Zy_quadrupolarY.append(float(columns[1]))

Zy_quadrupolarY=np.array(Zy_quadrupolarY) # in V/pC
f.close()

#...................#
#     1D  Plots     #
#...................#

if flag_show_plots:
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

    # Longitudinal wake potential
    fig2 = plt.figure(20, figsize=(6,4), dpi=200, tight_layout=True)
    ax=fig2.gca()
    ax.plot(s_cst*1.0e3, WP, lw=1.2, color='orange', label='Direct W// from CST')
    # ax.plot(s_cst*1.0e3, Indirect_WP_interfaces, lw=1.2, color='magenta', label='Indirect interfaces W// from CST')
    # ax.plot(s_cst*1.0e3, Indirect_WP_testbeams, lw=1.2, color='cyan', label='Indirect testbeams W// from CST')
    ax.set(title='Longitudinal wake potential from CST',
            xlabel='s [mm]',
            ylabel='W//(s) [V/pC]',         #ylim=(-8.0e4,8.0e4)
            )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    # Transverse wake potential [xytest=0]
    fig3 = plt.figure(30, figsize=(6,4), dpi=200, tight_layout=True)
    ax=fig3.gca()
    ax.plot(s_cst*1.0e3, WPx, lw=1.2, color='green', label='Transverse W⊥,x from CST')
    ax.plot(s_cst*1.0e3, WPy, lw=1.2, color='red', label='Transverse W⊥,y from CST')
    ax.set(title='Transverse wake potential from CST',
            xlabel='s [mm]',
            ylabel='W⊥(s) [V/pC]',         #ylim=(-8.0e4,8.0e4)
            )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    # Transverse DIPOLAR wake potentials 
    fig3 = plt.figure(30, figsize=(6,4), dpi=200, tight_layout=True)
    ax=fig3.gca()
    ax.plot(s_cst*1.0e3, WPx_dipolar, lw=1.2, color='green', label='Transverse W⊥,x from CST')
    ax.plot(s_cst*1.0e3, WPy_dipolar, lw=1.2, color='red', label='Transverse W⊥,y from CST')
    ax.set(title='Transverse dipolar wake potential from CST | xytest=5', 
            xlabel='s [mm]',
            ylabel='W⊥(s) [V/pC]',         #ylim=(-8.0e4,8.0e4)
            )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    # Transverse QUADRUPOLAR wake potentials 
    fig3 = plt.figure(30, figsize=(6,4), dpi=200, tight_layout=True)
    ax=fig3.gca()
    ax.plot(s_cst*1.0e3, WPx_quadrupolar, lw=1.2, color='green', label='Transverse W⊥,x from CST')
    ax.plot(s_cst*1.0e3, WPy_quadrupolar, lw=1.2, color='red', label='Transverse W⊥,y from CST')
    ax.set(title='Transverse quadrupolar wake potential from CST | xysource=5',
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

    # Longitudinal Impedance and maximum frequency
    ifreq_max=np.argmax(Z)
    fig4 = plt.figure(40, figsize=(6,4), dpi=200, tight_layout=True)
    ax=fig4.gca()
    ax.plot(freq_cst[ifreq_max]*1e-9, Z[ifreq_max], marker='o', markersize=4.0, color='pink')
    ax.annotate(str(round(freq_cst[ifreq_max]*1e-9,2))+ ' GHz', xy=(freq_cst[ifreq_max]*1e-9,Z[ifreq_max]), xytext=(1,1), textcoords='offset points', color='red') 
    ax.plot(freq_cst*1.0e-9, Z, lw=1.2, color='red', label='Z// from CST')
    ax.set(title='Longitudinal impedance Z from CST',
            xlabel='frequency [GHz]',
            ylabel='Z//(s) [Ohm]',         #ylim=(-8.0e4,8.0e4)
            )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    # Transverse Impedance and maximum frequency [xytest=0]
    ifreq_max_x=np.argmax(Zx)
    ifreq_max_y=np.argmax(Zy)

    fig5 = plt.figure(50, figsize=(6,4), dpi=200, tight_layout=True)
    ax=fig5.gca()
    ax.plot(freq_cst[ifreq_max_x]*1e-9, Zx[ifreq_max_x], marker='o', markersize=4.0, color='red')
    ax.annotate(str(round(freq_cst[ifreq_max_x]*1e-9,2))+ ' GHz', xy=(freq_cst[ifreq_max_x]*1e-9,Zx[ifreq_max_x]), xytext=(1,1), textcoords='offset points', color='red') 
    ax.plot(freq_cst*1.0e-9, Zx, lw=1.2, color='red', label='Zx⊥ from CST')
    ax.plot(freq_cst[ifreq_max_y]*1e-9, Zy[ifreq_max_y], marker='o', markersize=4.0, color='g')
    ax.annotate(str(round(freq_cst[ifreq_max_y]*1e-9,2))+ ' GHz', xy=(freq_cst[ifreq_max_y]*1e-9,Zy[ifreq_max_y]), xytext=(1,1), textcoords='offset points', color='g') 
    ax.plot(freq_cst*1.0e-9, Zy, lw=1.2, color='g', label='Zy⊥ from CST')
    ax.set(title='Transverse impedance Z from CST',
            xlabel='frequency [GHz]',
            ylabel='Z⊥(w) [Ohm]',         #ylim=(-8.0e4,8.0e4)
            )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    # Transverse DIPOLAR Impedance and maximum frequency 
    ifreq_max_x=np.argmax(Zx_dipolar)
    ifreq_max_y=np.argmax(Zy_dipolar)

    fig6 = plt.figure(60, figsize=(6,4), dpi=200, tight_layout=True)
    ax=fig6.gca()
    ax.plot(freq_cst[ifreq_max_x]*1e-9, Zx_dipolar[ifreq_max_x], marker='o', markersize=4.0, color='red')
    ax.annotate(str(round(freq_cst[ifreq_max_x]*1e-9,2))+ ' GHz', xy=(freq_cst[ifreq_max_x]*1e-9,Zx_dipolar[ifreq_max_x]), xytext=(1,1), textcoords='offset points', color='red') 
    ax.plot(freq_cst*1.0e-9, Zx_dipolar, lw=1.2, color='red', label='Zx⊥ from CST')
    ax.plot(freq_cst[ifreq_max_y]*1e-9, Zy_dipolar[ifreq_max_y], marker='o', markersize=4.0, color='g')
    ax.annotate(str(round(freq_cst[ifreq_max_y]*1e-9,2))+ ' GHz', xy=(freq_cst[ifreq_max_y]*1e-9,Zy_dipolar[ifreq_max_y]), xytext=(1,1), textcoords='offset points', color='g') 
    ax.plot(freq_cst*1.0e-9, Zy_dipolar, lw=1.2, color='g', label='Zy⊥ from CST')
    ax.set(title='Transverse dipolar impedance Z from CST',
            xlabel='frequency [GHz]',
            ylabel='Z⊥(w) [Ohm]',         #ylim=(-8.0e4,8.0e4)
            )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    # Transverse QUADRUPOLAR Impedance and maximum frequency 
    ifreq_max_x=np.argmax(Zx_quadrupolar)
    ifreq_max_y=np.argmax(Zy_quadrupolar)

    fig6 = plt.figure(60, figsize=(6,4), dpi=200, tight_layout=True)
    ax=fig6.gca()
    ax.plot(freq_cst[ifreq_max_x]*1e-9, Zx_quadrupolar[ifreq_max_x], marker='o', markersize=4.0, color='red')
    ax.annotate(str(round(freq_cst[ifreq_max_x]*1e-9,2))+ ' GHz', xy=(freq_cst[ifreq_max_x]*1e-9,Zx_quadrupolar[ifreq_max_x]), xytext=(1,1), textcoords='offset points', color='red') 
    ax.plot(freq_cst*1.0e-9, Zx_quadrupolar, lw=1.2, color='red', label='Zx⊥ from CST')
    ax.plot(freq_cst[ifreq_max_y]*1e-9, Zy_quadrupolar[ifreq_max_y], marker='o', markersize=4.0, color='g')
    ax.annotate(str(round(freq_cst[ifreq_max_y]*1e-9,2))+ ' GHz', xy=(freq_cst[ifreq_max_y]*1e-9,Zy_quadrupolar[ifreq_max_y]), xytext=(1,1), textcoords='offset points', color='g') 
    ax.plot(freq_cst*1.0e-9, Zy_quadrupolar, lw=1.2, color='g', label='Zy⊥ from CST')
    ax.set(title='Transverse quadrupolar impedance Z from CST',
            xlabel='frequency [GHz]',
            ylabel='Z⊥(w) [Ohm]',         #ylim=(-8.0e4,8.0e4)
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
nz = 101  #k

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
init_time = -5.332370636221942e-10

#setup vectors
x=np.linspace(xmin,xmax,nx)
y=np.linspace(ymin,ymax,ny)
z=np.linspace(zmin,zmax,nz)

#--- save the matrix into a txt

data = { 'Ez' : Ez_t, #shape = (k, len(t))
         't' : t, #time [s]
         'init_time' : -init_time, # [s]
         'nz' : nz, #mesh cells in z direction
         'nt' : len(t), #number of timesteps
         'charge_dist' : charge_dist, # [C/m]
         'charge_dist_time' : charge_dist_time,
         'charge_dist_spectrum' : charge_dist_spectrum,
         'current' : current,
         's_charge_dist' : distance, #[m]
         't_charge_dist' : t_charge_dist,
         'WP_cst' : WP, # [V/pC]
         'Indirect_WP_interfaces' : Indirect_WP_interfaces, 
         'Indirect_WP_testbeams': Indirect_WP_testbeams, 
         'WPx_cst' : WPx,
         'WPy_cst' : WPy,
         'WPx_dipolar_cst' : WPx_dipolar,
         'WPy_dipolar_cst' : WPy_dipolar,
         'WPx_dipolarX_cst' : WPx_dipolarX,
         'WPy_dipolarX_cst' : WPy_dipolarX,
         'WPx_dipolarY_cst' : WPx_dipolarY,
         'WPy_dipolarY_cst' : WPy_dipolarY,
         'WPx_quadrupolar_cst' : WPx_quadrupolar,
         'WPy_quadrupolar_cst' : WPy_quadrupolar,
         'WPx_quadrupolarX_cst' : WPx_quadrupolarX,
         'WPy_quadrupolarX_cst' : WPy_quadrupolarX,
         'WPx_quadrupolarY_cst' : WPx_quadrupolarY,
         'WPy_quadrupolarY_cst' : WPy_quadrupolarY,
         's_cst' : s_cst, # [m]
         's_cst_dipolar' : s_cst_dipolar,
         's_cst_quadrupolar' : s_cst_quadrupolar,
         'Z_cst' : Z, # [Ohm]
         'Zx_cst' : Zx,
         'Zy_cst' : Zy,
         'Zx_dipolar_cst' : Zx_dipolar,
         'Zy_dipolar_cst' : Zy_dipolar,
         'Zx_dipolarX_cst' : Zx_dipolarX,
         'Zy_dipolarX_cst' : Zy_dipolarX,
         'Zx_dipolarY_cst' : Zx_dipolarY,
         'Zy_dipolarY_cst' : Zy_dipolarY,
         'Zx_quadrupolar_cst' : Zx_quadrupolar,
         'Zy_quadrupolar_cst' : Zy_quadrupolar,
         'Zx_quadrupolarX_cst' : Zx_quadrupolarX,
         'Zy_quadrupolarX_cst' : Zy_quadrupolarX,
         'Zx_quadrupolarY_cst' : Zx_quadrupolarY,
         'Zy_quadrupolarY_cst' : Zy_quadrupolarY,
         'freq_cst_dipolar' : freq_cst_dipolar,
         'freq_cst_quadrupolar' : freq_cst_quadrupolar,
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
        }
# write the dictionary to a txt using pickle module
with open('cst_out.txt', 'wb') as handle:
    pk.dump(data, handle)

