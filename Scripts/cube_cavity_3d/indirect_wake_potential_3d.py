'''
wake_potential.py

File for postprocessing warp simulations

--- Reads the out file with pickle module
--- Performs the indirect integration to obtain wake potential
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
import h5py as h5py

#to be able to use pyPIC the BIN path is needed
BIN = os.path.expanduser("/home/edelafue/PyCOMPLETE")
if BIN not in sys.path:
    sys.path.append(BIN)

import PyPIC.geom_impact_poly as poly 
import PyPIC.FiniteDifferences_Staircase_SquareGrid as PIC_FD

unit = 1e-3 #mm to m
c=sc.constants.c

######################
#      Read data     #
######################

out_folder='out/'

#------------------------------------#
#            1D variables            #
#------------------------------------#

#--- read the dictionary
with open(out_folder+'out.txt', 'rb') as handle:
  data = pk.loads(handle.read())
  print('stored 1D variables')
  print(data.keys())

#--- retrieve 1D variables

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
L_cavity=data.get('L_cavity')
w_pipe=data.get('w_pipe')
h_pipe=data.get('h_pipe')
L_pipe=data.get('L_pipe')
t=data.get('t')
init_time=data.get('init_time')
nt=data.get('nt') #number of timesteps
nz=data.get('nz')
sigmaz=data.get('sigmaz')
xtest=data.get('xtest')
ytest=data.get('ytest')

#--- auxiliary variables
zmin=min(z)
zmax=max(z)
xmin=min(x)
xmax=max(x)
ymin=min(y)
ymax=max(y)

#reshape electric field
Ez_1d=np.transpose(np.array(Ez_t))     #array to matrix (z,t)

#------------------------------------#
#            3D variables            #
#------------------------------------#

#--- read the h5 file
hf = h5py.File(out_folder+'Ez.h5', 'r')
print('reading the h5 file '+ out_folder +'Ez.h5')
print('size of the file: '+str(round((os.path.getsize(out_folder+'Ez.h5')/10**9),2))+' Gb')
#get number of datasets
size_hf=0.0
dataset=[]
n_step=[]
for key in hf.keys():
    size_hf+=1
    dataset.append(key)
    n_step.append(int(key.split('_')[1]))

Ez_0=hf.get(dataset[0])
shapex=Ez_0.shape[0]  
shapey=Ez_0.shape[1] 
shapez=Ez_0.shape[2] 

print('Ez field is stored in a matrix with shape '+str(Ez_0.shape)+' in '+str(int(size_hf))+' datasets')


######################
# 	Wake potential   #
######################

#---------------------------------------
# Set up the poisson solver from PyPIC #
#---------------------------------------

#--- set up z, t, dt, dz
z=np.linspace(min(z), max(z), nz+1)
#z=np.array(z)
t=np.array(t)
dz=z[2]-z[1]
dt=t[2]-t[1]
zmax=np.max(z)
zmin=np.min(z)

dh=x[2]-x[1]	#resolution in the transversal plane
init_time=5.332370636221942e-10 #time when the center of the bunch enters the cavity

#--- set Wake_length, s
Wake_length=nt*dt*c - (zmax-zmin) - init_time*c
print('Max simulated time = '+str(round(t[-1]*1.0e9,4))+' ns')
print('Wake_length = '+str(Wake_length*1e3)+' mm')
ns_neg=int(init_time/dt)        #obtains the length of the negative part of s
ns_pos=int(Wake_length/(dt*c))  #obtains the length of the positive part of s
s=np.linspace(-init_time*c, 0, ns_neg) #sets the values for negative s
s=np.append(s, np.linspace(0, Wake_length,  ns_pos))

#--- define the limits for the poisson and the integral
l1=(L_cavity/2.0)         #[m]
l2=(L_cavity/2.0)         #[m] 
b1=h_pipe/2.0             #[m]
b2=h_pipe/2.0             #[m]
# define the rectangle size (aperture = half the area)
w_rect = w_pipe/2.0
h_rect = h_pipe/2.0


#--- initialize variables for the indirect integration

phi_l1 = np.zeros((int((w_rect*2)/dh + 10 + 1), int((h_rect*2)/dh + 8 + 1), len(s)))  
phi_l2 = np.zeros((int((w_rect*2)/dh + 10 + 1), int((h_rect*2)/dh + 8 + 1), len(s)))  
integral = np.zeros(len(s))  #integral of ez between -l1, l2
t_s = np.zeros((nt, len(s)))
Wake_potential=np.zeros_like(s)

#--- interpolate Ez so nz == nt
z_interp=np.linspace(zmin, zmax, nt)
Ez_interp=np.zeros((nt,nt))
dz_interp=z_interp[2]-z_interp[1]
n=0
for n in range(nt):
    Ez_interp_l1[:, n]=np.interp(z_interp, z, Ez[shapex//2, shapey//2, n])

#--- obtain the derivatives
Ez_dt=np.zeros((nt,nt))  #time derivative of Ez
Ez_dz=np.zeros((nt,nt))  #z spatial derivative of Ez
# n=0
# k=0
# for n in range(nt-1):
#     for k in range(nt-1):
#         Ez_dz[k,n]= (Ez_interp[k+1, n] - Ez_interp[k, n])/dz_interp
#         Ez_dt[k,n]= (Ez_interp[k, n+1] - Ez_interp[k, n])/dt

# find indexes for l1, l2 
iz_l1=int((-l1-z_interp[0])/dz_interp)
iz_l2=int((l2-z_interp[0])/dz_interp)

# s loop -------------------------------------#                                                           

for n in range(len(s)-1):    

    #-----------------------#
    # first poisson z=(-l1) #
    #-----------------------#

    t_l1=(-l1+s[n])/c-zmin/c-t[0]+init_time

    #--- obtain the rectangle size for -l1 
    # define the rectangle size (aperture = half the area)
    w_rect = w_pipe/2.0
    h_rect = h_pipe/2.0

    # PyPIC function to declare the implicit function for the conductors (this acts as BCs)
    PyPIC_chamber = poly.polyg_cham_geom_object({'Vx' : np.array([w_rect, -w_rect, -w_rect, w_rect]),
                                                 'Vy': np.array([h_rect, h_rect, -h_rect, -h_rect]),
                                                 'x_sem_ellip_insc' : 0.99*w_rect, #important to add this
                                                 'y_sem_ellip_insc' : 0.99*h_rect})
    # solver object
    picFD = PIC_FD.FiniteDifferences_Staircase_SquareGrid(chamb = PyPIC_chamber, Dh = dh, sparse_solver = 'PyKLU')
    phi_l1[:, :, n] = np.zeros_like(picFD.rho) 

    if t_l1 > 0.0:
        # find t index for -l1 and s[n]
        it_l1=int(t_l1/dt)

        # define the left side of the laplacian (driving term rho = 1/c*dEz/dt-dEz/dz)
        # [TODO: use a higer order method for the derivatives]

        # store the 3D e field for this time and the previous time
        Ez_next=hf.get(dataset[it_l1+1])
        Ez=hf.get(dataset[it_l1])
        for i in range(shapex):
            for j in range(shapey):
                # obtain the derivatives for each x, y in the transverse plane
                Ez_dz[i,j]=(Ez[i, j, l1+1] - Ez[i, j, l1])/dz    #obtains the derivative around l1
                Ez_dt[i,j]=(Ez_next[i, j, l1] - Ez[i,j,l1])/dt   #obtains the derivative around t_l1
                # obtain rho for the transversal plane
                rho[i,j] = (Ez_dt[i,j]/c - Ez_dz[i,j])           #this rho is evaluated at z=-l1, t=(s-l1)/c

        # solve the laplacian and obtain phi(0,0)
        picFD.solve(rho = rho)      #the dimensions are selected by pyPIC solver
        phi_l1[:, :, n] = picFD.phi.copy()

    #-----------------------#
    # second poisson z=(l2) #
    #-----------------------#

    t_l2=(l2+s[n])/c-zmin/c-t[0]+init_time

    #--- obtain the rectangle size for l2 
    # define the rectangle size (aperture = half the area)
    w_rect = w_pipe/2.0
    h_rect = h_pipe/2.0

    # PyPIC function to declare the implicit function for the conductors (this acts as BCs)
    PyPIC_chamber = poly.polyg_cham_geom_object({'Vx' : np.array([w_rect, -w_rect, -w_rect, w_rect]),
                                                 'Vy': np.array([h_rect, h_rect, -h_rect, -h_rect]),
                                                 'x_sem_ellip_insc' : 0.99*w_rect, #important to add this
                                                 'y_sem_ellip_insc' : 0.99*h_rect})
    # solver object
    picFD = PIC_FD.FiniteDifferences_Staircase_SquareGrid(chamb = PyPIC_chamber, Dh = dh, sparse_solver = 'PyKLU')
    phi_l2[:, :, n] = np.zeros_like(picFD.rho) 

    if t_l2>0.0:

        # find t index for l2 and s[n]
        it_l2=int(t_l2/dt)
        # define the left side of the laplacian (driving term rho = 1/c*dEz/dt-dEz/dz)
        # [TODO: use a higer order method for the derivatives]

        # store the 3D e field for this time and the previous time
        Ez_next=hf.get(dataset[it_l2+1])
        Ez=hf.get(dataset[it_l2])
        for i in range(shapex):
            for j in range(shapey):
                # obtain the derivatives for each x, y in the transverse plane
                Ez_dz[i,j]=(Ez[i, j, l2+1] - Ez[i, j, l2])/dz    #obtains the derivative around l1
                Ez_dt[i,j]=(Ez_next[i, j, l2] - Ez[i,j,l2])/dt   #obtains the derivative around t_l1
                # obtain rho for the transversal plane
                rho[i,j] = (Ez_dt[i,j]/c - Ez_dz[i,j])          #this rho is evaluated at z=-l1, t=(s-l1)/c

        # solve the laplacian and obtain phi(xtest,ytest)
        picFD.solve(rho = rho)      #rho has to be a matrix
        phi_l2[:, :, n] = picFD.phi.copy()   #phi[n] size [pyPIC_nx,pyPIC_ny]


    #-----------------------------#
    # integral between -l1 and l2 #
    #-----------------------------#

    #integral of (Ez(xtest, ytest, z, t=(s+z)/c))dz
    k=0
    for k in range(iz_l1, iz_l2):
        t_s[k,n]=(z_interp[k]+s[n])/c-zmin/c-t[0]+init_time

        if t_s[k,n]>0.0:
            it=int(t_s[k,n]/dt)                             #find index for t
            integral[n]=integral[n]+(Ez_interp[k, it])*dz_interp   #compute integral


    #-----------------------#
    #      Obtain W(s)      #
    #-----------------------#

    # Define phi(x, y=b1) 
    # Phis indexes go from x(-w_rect,w_rect) and y(-h_rect,h_rect)
    # x direction is gridded with (w_rect*2)/dh + 10 ghost cells
    # y direction is gridded with (h_rect*2)/dh + 8 ghost cells
    # +info see: class PyPIC_Scatter_Gather(object):
    iy_b1=int((b1-picFD.bias_y)/dh)     
    iy_b2=int((b2-picFD.bias_y)/dh)
    ixtest_phi=int((xtest-picFD.bias_x)/dh)+1
    iytest_phi=int((ytest-picFD.bias_y)/dh)+1

    phi_b1=phi_l1[ixtest_phi, iy_b1, n] #phi(x=0, y=b1, s[n])
    phi_b2=phi_l2[ixtest_phi, iy_b1, n] #phi(x=0, y=b1, s[n])

    Wake_potential[n]=(phi_b1-phi_l1[ixtest_phi, iytest_phi, n])-integral[n]+(phi_l2[ixtest_phi, iytest_phi, n]-phi_b2)

q=(1e-9)*1e12                       # charge of the particle beam in pC
Wake_potential=-Wake_potential/q     # [V/pC]

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

#--- plot integral l1, l2

fig2 = plt.figure(2, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig2.gca()
ax.plot(s*1.0e3, integral/q, lw=1.2, color='r', label='W_//(s)')
ax.set(title='Longitudinal Wake potential - Integral(l1, l2)',
        xlabel='s [mm]',
        ylabel='$ W_{//} $ integral (l1, l2) [V/pC]',
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

#--- plot phi

fig3 = plt.figure(3, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig3.gca()
ax.plot(s*1.0e3, phi_l1[ixtest_phi, iytest_phi, :]/q, lw=1, color='r', label='Phi_l1(s)')
ax.plot(s*1.0e3, (phi_b1-phi_l1[ixtest_phi, iytest_phi, :])/q, lw=1, color='pink', label='Phi_b1-Phi_l1(s)', ls='--')
ax.plot(s*1.0e3, phi_l2[ixtest_phi, iytest_phi, :]/q, lw=1, color='b', label='Phi_l2(s)')
ax.plot(s*1.0e3, (phi_l2[ixtest_phi, iytest_phi, :]-phi_b2)/q, lw=1, color='cyan', label='Phi_l2(s)-Phi_b2', ls='--')
ax.set(title='Poisson solver results',
        xlabel='s [mm]',
        ylabel='$ Phi $ [V/pC]',
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


############################
#   Comparison with CST    #
############################

#--- read the cst dictionary
with open('cst/cst_out.txt', 'rb') as handle:
  cst_data = pk.loads(handle.read())
  print('cst stored variables')
  print(cst_data.keys())

charge_dist_cst=cst_data.get('charge_dist')
distance=cst_data.get('distance')
Wake_potential_cst=cst_data.get('Wake_potential_cst')
s_cst=cst_data.get('s_cst')
Z_cst=cst_data.get('Z_cst')
freq_cst=cst_data.get('freq_cst')

#--- Plot comparison

q=(1e-9)*1e12 # charge of the particle beam in pC
fig4 = plt.figure(4, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig4.gca()
ax.plot(s*1.0e3, Wake_potential, lw=1.2, color='orange', label='W_//(s) indirect integration')
ax.plot(s_cst*1e3, Wake_potential_cst, lw=1.3, color='black', ls='--', label='W_//(s) CST')
ax.set(title='Longitudinal Wake potential',
        xlabel='s [mm]',
        ylabel='$W_{//}$ [V/pC]',
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()


#--- Plot normalized comparison comparison

fig5 = plt.figure(5, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig5.gca()
ax.plot(s*1e3, Wake_potential/np.max(Wake_potential), lw=1.3, color='orange', label='$W_{//}(s)$ indirect integration')
ax.plot(s_cst*1e3, Wake_potential_cst/np.max(Wake_potential_cst), lw=1.2, color='black', ls='--', label='$W_{//}(s)$ from CST')
ax.set(title='Normalized longitudinal Wake potential',
        xlabel='s [mm]',
        ylabel='$W_{//}$ [V/pC]',
        ylim=(-1.5,1.5)
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

#--- Plot impedance comparison with CST [normalized]

norm=max(Z)/max(Z_cst) #diference between max in CST and in numpy.fft
ifreq_max=np.argmax(Z[0:len(Z)//2])
fig3 = plt.figure(3, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig3.gca()
ax.plot(Z_freq[ifreq_max], Z[ifreq_max]/norm, marker='o', markersize=4.0, color='cyan')
ax.annotate(str(round(Z_freq[ifreq_max],2))+ ' GHz', xy=(Z_freq[ifreq_max],Z[ifreq_max]/norm), xytext=(-10,5), textcoords='offset points', color='grey') 
ax.plot(Z_freq[0:len(Z)//2], Z[0:len(Z)//2]/norm, lw=1, color='b', label='numpy FFT')

ifreq_max=np.argmax(Z_cst)
ax.plot(freq_cst[ifreq_max]*1e-9, Z_cst[ifreq_max], marker='o', markersize=5.0, color='pink')
ax.annotate(str(round(freq_cst[ifreq_max]*1e-9,2))+ ' GHz', xy=(freq_cst[ifreq_max]*1e-9,Z_cst[ifreq_max]), xytext=(-10,2), textcoords='offset points', color='grey') 
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
