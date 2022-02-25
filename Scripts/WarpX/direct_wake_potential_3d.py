'''
direct_wake_potential_3d.py

File for postprocessing WarpX simulations

--- Reads the input data dictionary with picle
--- Reads the 3d data of the ez field from h5 file
--- Performs the direct integration of the longitudinal wake potential
--- Performs the fourier trnasform to obtain the impedance
--- Obtains the transverse wake potential through Panofsky Wenzel theorem
--- Plots the results

'''
print('---------------------')
print('|  Running PyWake   |')
print('---------------------')

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import scipy.constants as spc  
import pickle as pk
import h5py as h5py

unit = 1e-3 #mm to m
c=spc.c
beta=1.0 #TODO: obtain beta from Warp simulation

######################
#      Read data     #
######################
runs_path='/mnt/c/Users/elefu/Documents/CERN/GitHub/PyWake/Scripts/WarpX/' #runs/'
out_folder=runs_path+'out/'

#------------------------------------#
#            1D variables            #
#------------------------------------#

#--- read the dictionary
with open(out_folder+'input_data.txt', 'rb') as handle:
   input_data = pk.loads(handle.read())
   #print(input_data.keys())

#--- retrieve variables
x=input_data.get('x')
y=input_data.get('y')
z=input_data.get('z')
nt=input_data.get('tot_nsteps')
init_time=input_data.get('init_time')
nx=input_data.get('nx')
ny=input_data.get('ny')
nz=input_data.get('nz')
w_cavity=input_data.get('w_cavity')
h_cavity=input_data.get('h_cavity')
L_cavity=input_data.get('L_cavity')
w_pipe=input_data.get('w_pipe')
h_pipe=input_data.get('h_pipe')
L_pipe=input_data.get('L_pipe')
sigmaz=input_data.get('sigmaz')
xsource=input_data.get('xsource')
ysource=input_data.get('ysource')
xtest=input_data.get('xtest')
ytest=input_data.get('ytest')
# retrieve arrays
t=input_data.get('t')
#This needs to previously run postproc_h5.py
Ez_t=input_data.get('Ez')
charge_dist=input_data.get('charge_dist')

#--- auxiliary variables
zmin=min(z)
zmax=max(z)
xmin=min(x)
xmax=max(x)
ymin=min(y)
ymax=max(y)
dx=x[2]-x[1]
dy=y[2]-y[1]
dz=z[2]-z[1]
dt=t[2]-t[1]

# Extract charge_dist from h5 if not in the dictionary
if charge_dist is None:
    hf_rho = h5py.File(out_folder +'rho.h5', 'r')
    print('reading the h5 file '+ out_folder +'rho.h5')
    #get number of datasets
    dataset_rho=[]
    n_step_rho=[]
    for key in hf_rho.keys():
        dataset_rho.append(key)
        n_step_rho.append(int(key.split('_')[1]))
    # Extract charge distribution [C/m] lambda(z,t)
    charge_dist=[]
    for n in range(nt):
        rho=hf_rho.get(dataset_rho[n]) # [C/m3]
        charge_dist.append(np.array(rho)*dx*dy) # [C/m]

    charge_dist=np.transpose(np.array(charge_dist)) # [C/m]


#------------------------------------#
#            3D variables            #
#------------------------------------#

# Read the Ez h5 file
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
shapex=Ez_0.shape[0]-2  
shapey=Ez_0.shape[1]-2 
shapez=Ez_0.shape[2] 
z_Ez=z[nz//2-shapez//2:nz//2+shapez//2+1]
print('Ez field is stored in a matrix with shape '+str(Ez_0.shape)+' in '+str(int(size_hf))+' datasets')

# PML cells conrrection (WarpX)
flag_correct_pml = False 
PML_cells = nz//2 - int(L_pipe/dz) + 1 

######################
#   Wake potential   #
######################

t0 = time.time()

# set up t, dt, 
t=np.array(t)#-9*dz/c #time correction to match CST
dt=t[2]-t[1]
dh=dx         #resolution in the transversal plane

# set Wake_length, s
Wake_length=nt*dt*c - (zmax-zmin) - init_time*c
print('---Max simulated time = '+str(round(t[-1]*1.0e9,4))+' ns')
print('---Wake_length = '+str(Wake_length*1e3)+' mm')
ns_neg=int(init_time/dt)        #obtains the length of the negative part of s
ns_pos=int(Wake_length/(dt*c))  #obtains the length of the positive part of s
s=np.linspace(-init_time*c, 0, ns_neg) #sets the values for negative s
s=np.append(s, np.linspace(0, Wake_length,  ns_pos))

# initialize Wp variables
Wake_potential=np.zeros_like(s)
t_s = np.zeros((nt, len(s)))

# initialize interpolated Ez so nz == nt
z_interp=np.linspace(np.min(z_Ez), np.max(z_Ez), nt)
dz_interp=z_interp[2]-z_interp[1]
Ez_interp=np.zeros((nt,nt))

# initialize wake potential matrix
flag_fourth_order=False     #Default: False
flag_second_order=True      #Default: True
if flag_fourth_order:
    print('Using fourth order scheme for gradient')
    n_transverse_cells=2
    Wake_potential_3d=np.zeros((n_transverse_cells*2+1,n_transverse_cells*2+1,len(s)))
elif flag_second_order:
    print('Using second order scheme for gradient')
    n_transverse_cells=1
    Wake_potential_3d=np.zeros((n_transverse_cells*2+1,n_transverse_cells*2+1,len(s)))
else:    
    print('Using first order upwind scheme for gradient')
    n_transverse_cells=1    
    Wake_potential_3d=np.zeros((n_transverse_cells*2+1,n_transverse_cells*2+1,len(s)))

i0=n_transverse_cells
j0=n_transverse_cells

print('Calculating longitudinal wake potential...')
for i in range(-n_transverse_cells,n_transverse_cells+1,1): #field is stored around (xtest,ytest) selected by the user 
    for j in range(-n_transverse_cells,n_transverse_cells+1,1):

        n=0
        for n in range(nt-1):
            Ez=hf.get(dataset[n])
            if flag_correct_pml:
                Ez_interp[:, n]=np.interp(z_interp, z_Ez[PML_cells:-PML_cells] , Ez[i,j,PML_cells:-PML_cells])
            else:
                Ez_interp[:, n]=np.interp(z_interp, z_Ez, Ez[i+1,j+1,:])
        #-----------------------#
        #     Obtain W||(s)     #
        #-----------------------#

        # s loop -------------------------------------#                                                           
        n=0
        for n in range(len(s)):    

            #--------------------------------#
            # integral between zmin and zmax #
            #--------------------------------#

            #integral of (Ez(xtest, ytest, z, t=(s+z)/c))dz
            k=0
            for k in range(0, nt): 
                t_s[k,n]=(z_interp[k]+s[n])/c-zmin/c-t[0]+init_time

                if t_s[k,n]>0.0:
                    it=int(t_s[k,n]/dt)                                                #find index for t
                    Wake_potential[n]=Wake_potential[n]+(Ez_interp[k, it])*dz_interp   #compute integral

        q=(1e-9)*1e12                       # charge of the particle beam in pC
        Wake_potential=Wake_potential/q     # [V/pC]
        Wake_potential_3d[i0+i,j0+j,:]=Wake_potential #matrix[shapex,shapey,len(s)]

Long_wake_potential=Wake_potential_3d[i0,j0,:]

#-----------------------#
#      Obtain W⊥(s)    #
#-----------------------#

# Initialize variables
n=0
k=0
i=0
j=0
ds=s[2]-s[1]
Transverse_wake_potential_x=np.zeros_like(s)
Transverse_wake_potential_y=np.zeros_like(s)

#'''
# Obtain the transverse wake potential through Panofsky Wenzel theorem
int_wake_potential=np.zeros_like(Wake_potential_3d)
print('Calculating transverse wake potential...')
for n in range(len(s)):
    for i in range(-n_transverse_cells,n_transverse_cells+1,1):
        for j in range(-n_transverse_cells,n_transverse_cells+1,1):
            for k in range(n):
                # Perform the integral
                int_wake_potential[i0+i,j0+j,n]=int_wake_potential[i0+i,j0+j,n]+Wake_potential_3d[i0+i,j0+j,k]*ds 
    if flag_fourth_order:
    # Perform the gradient (fourth order scheme)
        Transverse_wake_potential_x[n]= - (-int_wake_potential[i0+2,j0,n]+8*int_wake_potential[i0+1,j0,n]+ \
                                        -8*int_wake_potential[i0-1,j0,n]+int_wake_potential[i0-2,j0,n])/(12*dx)
        Transverse_wake_potential_y[n]= - (-int_wake_potential[i0,j0+2,n]+8*int_wake_potential[i0,j0+1,n]+ \
                                        -8*int_wake_potential[i0,j0-1,n]+int_wake_potential[i0,j0-2,n])/(12*dy)
    if flag_second_order:
    # Perform the gradient (second order scheme)
        Transverse_wake_potential_x[n]= - (int_wake_potential[i0+1,j0,n]-int_wake_potential[i0-1,j0,n])/(2*dx)
        Transverse_wake_potential_y[n]= - (int_wake_potential[i0,j0+1,n]-int_wake_potential[i0,j0-1,n])/(2*dy)
    else:
    # Perform the gradient (first order scheme)
        Transverse_wake_potential_x[n]= - (int_wake_potential[i0+1,j0,n]-int_wake_potential[i0,j0,n])/(dx)
        Transverse_wake_potential_y[n]= - (int_wake_potential[i0,j0+1,n]-int_wake_potential[i0,j0,n])/(dy)    


#--------------------------------#
#      Obtain impedance Z||      #
#--------------------------------#

#--- Obtain impedance Z with Fourier transform numpy.fft.fft
# to increase the resolution of fft, a longer wake length is needed
print('Obtaining longitudinal impedance...')
f_max=5.5*1e9
t_sample=int(1/(ds/c)/2/f_max) #obtains the time window to sample the time domain data
N_samples=int(len(s)/t_sample)
print('Performing FFT with '+str(N_samples)+' samples')
print('Frequency bin resolution '+str(round(1/(len(s)*ds/c)*1e-9,2))+ ' GHz')
print('Frequency range: 0 - '+str(round(f_max*1e-9,2)) +' GHz')

# Obtain normalized charge distribution as a function of s: lambda(s)
timestep=int((z[nz//2]/c+init_time+3.1*sigmaz/c)/dt)+1 #charge distribution at cavity center
charge_dist_s=np.interp(s, z , charge_dist[:,timestep]/np.max(charge_dist[:,timestep])) # norm charge distribution as a function of s
# Padding with zeros to increase N samples = smoother FFT
charge_dist_padded=np.append(charge_dist_s, np.zeros(100000))
Wake_potential_padded=np.append(Long_wake_potential, np.zeros(100000))
# Obtain the ffts and frequency bins
charge_dist_fft=abs(np.fft.fft(charge_dist_padded[0:-1:t_sample]))
Wake_potential_fft=abs(np.fft.fft(Wake_potential_padded[0:-1:t_sample]))
Z_freq = np.fft.fftfreq(len(Wake_potential_padded[:-1:t_sample]), ds/c*t_sample)*1e-9 #GHz
# Obtain the impedance
Z = abs(- Wake_potential_fft / charge_dist_fft)

#--------------------------------#
#      Obtain impedance Z⊥       #
#--------------------------------#

#--- Obtain impedance Z with Fourier transform numpy.fft.fft
# to increase the resolution of fft, a longer wake length is needed

#---Zx⊥(s)
# Padding with zeros to increase N samples = smoother FFT
Wake_potential_x_padded=np.append(Transverse_wake_potential_x, np.zeros(100000))
# Obtain the ffts and frequency bins
Wake_potential_x_fft=abs(np.fft.fft(Wake_potential_x_padded[0:-1:t_sample]))
Z_x_freq = np.fft.fftfreq(len(Wake_potential_x_padded[:-1:t_sample]), ds/c*t_sample)*1e-9 #GHz
# Obtain the impedance
Z_x = abs(-1j* Wake_potential_x_fft / charge_dist_fft)

#---Zy⊥(s)
# Padding with zeros to increase N samples = smoother FFT
Wake_potential_y_padded=np.append(Transverse_wake_potential_y, np.zeros(100000))
# Obtain the ffts and frequency bins
Wake_potential_y_fft=abs(np.fft.fft(Wake_potential_y_padded[0:-1:t_sample]))
Z_y_freq = np.fft.fftfreq(len(Wake_potential_y_padded[:-1:t_sample]), ds/c*t_sample)*1e-9 #GHz
# Obtain the impedance
Z_y = abs(-1j* Wake_potential_y_fft / charge_dist_fft)


#Calculate elapsed time
t1 = time.time()
totalt = t1-t0
print('Calculation terminated in %ds' %totalt)

#-------------------#
#     1D  Plots     #
#-------------------#

# Plot longitudinal wake potential W||(s)

fig1 = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig1.gca()
ax.plot(s*1.0e3, Long_wake_potential, lw=1.2, color='orange', label='$W_{//}$[0,0](s)')
ax.set(title='Longitudinal Wake potential',
        xlabel='s [mm]',
        ylabel='$W_{//}$(s) [V/pC]',
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

# Plot transverse wake potential Wx⊥(s), Wy⊥(s)

fig10 = plt.figure(10, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig10.gca()
ax.plot(s*1.0e3, Transverse_wake_potential_x, lw=1.2, color='g', label='Wx⊥(s)')
ax.plot(s*1.0e3, Transverse_wake_potential_y, lw=1.2, color='m', label='Wy⊥(s)')
ax.set(title='Transverse Wake potential W⊥(s)',
        xlabel='s [mm]',
        ylabel='W⊥(s) [V/pC]',
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

# Plot longitudinal impedance Z||(w)
#--- obtain the maximum frequency
ifreq_max=np.argmax(Z[0:len(Z)//2])
fig2 = plt.figure(2, figsize=(6,4), dpi=200, tight_layout=True)
#--- plot Z||(s)
ax=fig2.gca()
ax.plot(Z_freq[ifreq_max], Z[ifreq_max], marker='o', markersize=4.0, color='blue')
ax.annotate(str(round(Z_freq[ifreq_max],2))+ ' GHz', xy=(Z_freq[ifreq_max],Z[ifreq_max]), xytext=(-10,5), textcoords='offset points', color='grey') 
#arrowprops=dict(color='blue', shrink=1.0, headwidth=4, frac=1.0)
ax.plot(Z_freq[0:len(Z)//2], Z[0:len(Z)//2], lw=1, color='b', marker='s', markersize=2., label='Z// numpy FFT')
ax.set(title='Longitudinal impedance Z(w) magnitude',
        xlabel='f [GHz]',
        ylabel='Z||(w) [$\Omega$]',   
        ylim=(0.,np.max(Z)*1.2),
        xlim=(0.,np.max(Z_freq))      
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

# Plot transverse impedance Zx⊥(w), Zy⊥(w)
#--- obtain the maximum frequency
ifreq_x_max=np.argmax(Z_x[0:len(Z_x)//2])
ifreq_y_max=np.argmax(Z_y[0:len(Z_y)//2])
#--- plot Zx⊥(s)
fig20 = plt.figure(20, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig20.gca()
ax.plot(Z_x_freq[ifreq_x_max], Z_x[ifreq_x_max], marker='o', markersize=4.0, color='green')
ax.annotate(str(round(Z_x_freq[ifreq_x_max],2))+ ' GHz', xy=(Z_x_freq[ifreq_x_max],Z_x[ifreq_x_max]), xytext=(-10,5), textcoords='offset points', color='grey') 
ax.plot(Z_x_freq[0:len(Z_x)//2], Z_x[0:len(Z_x)//2], lw=1, color='g', marker='s', markersize=2., label='Zx⊥ numpy FFT')
#--- plot Zy⊥(s)
ax.plot(Z_y_freq[ifreq_y_max], Z_y[ifreq_y_max], marker='o', markersize=4.0, color='magenta')
ax.annotate(str(round(Z_y_freq[ifreq_y_max],2))+ ' GHz', xy=(Z_y_freq[ifreq_y_max],Z_y[ifreq_y_max]), xytext=(-10,5), textcoords='offset points', color='grey') 
ax.plot(Z_y_freq[0:len(Z_y)//2], Z_y[0:len(Z_y)//2], lw=1, color='magenta', marker='s', markersize=2., label='Zy⊥ numpy FFT')
ax.set(title='Transverse impedance Z⊥(w) magnitude',
        xlabel='f [GHz]',
        ylabel='Z⊥(w) [$\Omega$]',   
        #ylim=(0.,np.max(Z_x)*1.2),
        #xlim=(0.,np.max(Z_x_freq))      
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

# Save the data 
data = { 'Longituginal wake potential' : Long_wake_potential, 
         's' : s,
         #'k_factor' : k_factor,
         'Longitudinal impedance' : Z,
         'frequency' : Z_freq,
         'Transverse wake potential x' : Transverse_wake_potential_x,
         'Transverse wake potential y' : Transverse_wake_potential_y,
         'Transverse impedance x' : Z_x,
         'Transverse impedance y' : Z_y,
         'frequency x' : Z_x_freq,
         'frequency y' : Z_y_freq,

        }
# write the dictionary to a txt using pickle module
with open(out_folder + 'wake_solver.txt', 'wb') as handle:
    pk.dump(data, handle)


############################
#   Comparison with CST    #
############################

#--- read the dictionary
cst_path='/mnt/c/Users/elefu/Documents/CERN/GitHub/PyWake/Scripts/CST/'
with open(cst_path+'cst_out.txt', 'rb') as handle:
   cst_data = pk.loads(handle.read())
   #print(cst_data.keys())

#---Input variables
x=cst_data.get('x')
y=cst_data.get('y')
z=cst_data.get('z')
nt=cst_data.get('nt')
init_time=cst_data.get('init_time')
nx=cst_data.get('nx')
ny=cst_data.get('ny')
nz=cst_data.get('nz')
w_cavity=cst_data.get('w_cavity')
h_cavity=cst_data.get('h_cavity')
L_cavity=cst_data.get('L_cavity')
w_pipe=cst_data.get('w_pipe')
h_pipe=cst_data.get('h_pipe')
L_pipe=cst_data.get('L_pipe')
sigmaz=cst_data.get('sigmaz')
#---Electric field
Ez_cst=cst_data.get('Ez')
t_cst = cst_data.get('t')
nz_cst=cst_data.get('nz')
nt_cst=cst_data.get('nt')
z_cst=cst_data.get('z')
#---Charge distribution
charge_dist_cst=cst_data.get('charge_dist')
charge_dist_time=cst_data.get('charge_dist_time')
charge_dist_spectrum=cst_data.get('charge_dist_spectrum')
current=cst_data.get('current')
s_charge_dist=cst_data.get('s_charge_dist')
t_charge_dist=cst_data.get('t_charge_dist')
#---Wake potential
Wake_potential_cst=cst_data.get('WP_cst')
Wake_potential_interfaces=cst_data.get('Wake_potential_interfaces')
Wake_potential_testbeams=cst_data.get('Wake_potential_testbeams')
WPx_cst=cst_data.get('WPx_cst')
WPy_cst=cst_data.get('WPy_cst')
WPx_dipolar_cst=cst_data.get('WPx_dipolar_cst')
WPy_dipolar_cst=cst_data.get('WPy_dipolar_cst')
WPx_quadrupolar_cst=cst_data.get('WPx_quadrupolar_cst')
WPy_quadrupolar_cst=cst_data.get('WPy_quadrupolar_cst')
s_cst=cst_data.get('s_cst')
s_cst_dipolar=cst_data.get('s_cst_dipolar')
#---Impedance
Z_cst=cst_data.get('Z_cst')
Zx_cst=cst_data.get('Zx_cst')
Zy_cst=cst_data.get('Zy_cst')
Zx_dipolar_cst=cst_data.get('Zx_dipolar_cst')
Zy_dipolar_cst=cst_data.get('Zy_dipolar_cst')
Zx_quadrupolar_cst=cst_data.get('Zx_quadrupolar_cst')
Zy_quadrupolar_cst=cst_data.get('Zy_quadrupolar_cst')
freq_cst=cst_data.get('freq_cst')
freq_cst_dipolar=cst_data.get('freq_cst_dipolar')

'''
# Plot interpolated electric field Ez on axis 
fig50 = plt.figure(50, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig50.gca()
ax.plot((np.array(t)-9*dz/c)*1.0e9, Ez_interp[nt//2, :], color='g', label='Ez Warpx (interpolated)')
ax.plot(np.array(t_cst)*1.0e9, Ez_cst[int(nz_cst/2), :], lw=0.8, color='black', ls='--',label='Ez CST')
ax.set(title='Electric field at cavity center',
        xlabel='t [ns]',
        ylabel='E [V/m]',         
        ylim=(np.min(Ez_cst[int(nz_cst/2), :])*1.1,np.max(Ez_cst[int(nz_cst/2), :])*1.1),
        #xlim=(0,np.minimum(np.max(np.array(t[n])*1.0e9),np.max(t_cst*1.0e9)))
                )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()
'''

# Plot longitudinal wake potential comparison with CST W||(s)

fig3 = plt.figure(3, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig3.gca()
ax.plot((s+9*dz)*1.0e3, Long_wake_potential, lw=1.2, color='orange', label='$W_{//}$(s) WarpX')
ax.plot(s_cst*1e3, Wake_potential_cst, lw=1.3, color='black', ls='--', label='$W_{//}$(s) CST')
ax.set(title='Longitudinal Wake potential $W_{//}$(s)',
        xlabel='s [mm]',
        ylabel='$W_{//}$(s) [V/pC]',
        xlim=(min(s*1.0e3), np.amin((np.max(s*1.0e3), np.max(s_cst*1.0e3))))
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

# Plot transverse wake potential Wx⊥(s), Wy⊥(s) comparison with CST

if xtest != 0.0:
    WPx_cst=WPx_quadrupolar_cst
    s_cst=s_cst_dipolar
if ytest != 0.0:
    WPy_cst=WPy_quadrupolar_cst
    s_cst=s_cst_dipolar
if xsource != 0.0:
    WPx_cst=WPx_dipolar_cst
    s_cst=s_cst_dipolar
if ysource != 0.0:
    WPy_cst=WPy_dipolar_cst
    s_cst=s_cst_dipolar

fig4 = plt.figure(4, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig4.gca()
ax.plot(s*1.0e3, Transverse_wake_potential_x, lw=1.2, color='g', label='Wx⊥(s)')
ax.plot(s_cst*1.0e3, WPx_cst, lw=1.2, color='g', ls='--', label='Wx⊥(s) from CST')
ax.plot(s*1.0e3, Transverse_wake_potential_y, lw=1.2, color='magenta', label='Wy⊥(s)')
ax.plot(s_cst*1.0e3, WPy_cst, lw=1.2, color='magenta', ls='--', label='Wy⊥(s) from CST')
ax.set(title='Transverse Wake potential W⊥(s) \n xsource, ysource = '+str(xsource*1e3)+' mm | xtest, ytest = '+str(xtest*1e3)+' mm',
        xlabel='s [mm]',
        ylabel='$W_{//}$ [V/pC]',
        xlim=(min(s*1.0e3), np.amin((np.max(s*1.0e3), np.max(s_cst*1.0e3)))),
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

# Plot longitudinal impedance Z||(w) comparison with CST [normalized]

#---normalizing factor between CST and in numpy.fft
norm=max(Z)/max(Z_cst) 
#--- obtain the maximum frequency for WarpX and plot
ifreq_max=np.argmax(Z[0:len(Z)//2])
fig5 = plt.figure(5, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig5.gca()
ax.plot(Z_freq[ifreq_max], Z[ifreq_max]/norm, marker='o', markersize=4.0, color='blue')
ax.annotate(str(round(Z_freq[ifreq_max],2))+ ' GHz', xy=(Z_freq[ifreq_max],Z[ifreq_max]/norm), xytext=(-20,5), textcoords='offset points', color='blue') 
ax.plot(Z_freq[0:len(Z)//2], Z[0:len(Z)//2]/norm, lw=1, color='b', marker='s', markersize=2., label='Z||(w) from WarpX')
#--- obtain the maximum frequency for CST and plot
ifreq_max=np.argmax(Z_cst)
ax.plot(freq_cst[ifreq_max]*1e-9, Z_cst[ifreq_max], marker='o', markersize=5.0, color='red')
ax.annotate(str(round(freq_cst[ifreq_max]*1e-9,2))+ ' GHz', xy=(freq_cst[ifreq_max]*1e-9,Z_cst[ifreq_max]), xytext=(+20,5), textcoords='offset points', color='red') 
ax.plot(freq_cst*1.0e-9, Z_cst, lw=1.2, color='red', marker='s', markersize=2., label='Z||(w) from CST')
#--- plot Z||(s)
ax.set(title='Longitudinal impedance Z||(w) [normalized by '+str(round(norm,3))+']',
        xlabel='f [GHz]',
        ylabel='Z||(w) [$\Omega$]',   
        ylim=(0.,np.max(Z_cst)*1.2),
        xlim=(0.,np.max(freq_cst)*1e-9)      
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

# Plot transverse impedance Zx⊥(w), Zy⊥(w) comparison with CST [normalized]
if xtest != 0.0:
    Zx_cst=Zx_quadrupolar_cst
    freq_cst=freq_cst_dipolar
if ytest != 0.0:
    Zy_cst=Zy_quadrupolar_cst
    freq_cst=freq_cst_dipolar
if xsource != 0.0:
    Zx_cst=Zx_dipolar_cst
    freq_cst=freq_cst_dipolar
if ysource != 0.0:
    Zy_cst=Zy_dipolar_cst
    freq_cst=freq_cst_dipolar

#---normalizing factor between CST and in numpy.fft
norm_x=max(Z_x)/max(Zx_cst) 
norm_y=max(Z_y)/max(Zy_cst) 
#--- obtain the maximum frequency
ifreq_x_max=np.argmax(Z_x[0:len(Z_x)//2])
ifreq_y_max=np.argmax(Z_y[0:len(Z_y)//2])
#--- plot Zx⊥(w)
fig6 = plt.figure(6, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig6.gca()
ax.plot(Z_x_freq[ifreq_x_max], Z_x[ifreq_x_max]/norm_x, marker='o', markersize=4.0, color='green')
ax.annotate(str(round(Z_x_freq[ifreq_x_max],2))+ ' GHz', xy=(Z_x_freq[ifreq_x_max],Z_x[ifreq_x_max]), xytext=(-10,5), textcoords='offset points', color='grey') 
ax.plot(Z_x_freq[0:len(Z_x)//2], Z_x[0:len(Z_x)//2]/norm_x, lw=1, color='g', marker='s', markersize=2., label='Zx⊥ from WarpX')
#--- obtain the maximum frequency for CST Zx⊥(w) and plot
ifreq_max=np.argmax(Zx_cst)
ax.plot(freq_cst[ifreq_max]*1e-9, Zx_cst[ifreq_max], marker='o', markersize=5.0, color='black')
ax.annotate(str(round(freq_cst[ifreq_max]*1e-9,2))+ ' GHz', xy=(freq_cst[ifreq_max]*1e-9,Zx_cst[ifreq_max]), xytext=(+20,5), textcoords='offset points', color='black') 
ax.plot(freq_cst*1.0e-9, Zx_cst, lw=1.2, ls='--', color='black', marker='s', markersize=2., label='Zx⊥(w) from CST')
#--- plot Zy⊥(w)
ax.plot(Z_y_freq[ifreq_y_max], Z_y[ifreq_y_max]/norm_y, marker='o', markersize=4.0, color='magenta')
ax.annotate(str(round(Z_y_freq[ifreq_y_max],2))+ ' GHz', xy=(Z_y_freq[ifreq_y_max],Z_y[ifreq_y_max]), xytext=(-10,5), textcoords='offset points', color='grey') 
ax.plot(Z_y_freq[0:len(Z_y)//2], Z_y[0:len(Z_y)//2]/norm_y, lw=1, color='magenta', marker='s', markersize=2., label='Zy⊥(w) from WarpX')
#--- obtain the maximum frequency for CST Zy⊥(w) and plot
ifreq_max=np.argmax(Zy_cst)
ax.plot(freq_cst[ifreq_max]*1e-9, Zy_cst[ifreq_max], marker='o', markersize=5.0, color='black')
ax.annotate(str(round(freq_cst[ifreq_max]*1e-9,2))+ ' GHz', xy=(freq_cst[ifreq_max]*1e-9,Zy_cst[ifreq_max]), xytext=(+20,5), textcoords='offset points', color='black') 
ax.plot(freq_cst*1.0e-9, Zy_cst, lw=1.2, ls='--', color='black', marker='s', markersize=2., label='Zy⊥(w) from CST')

ax.set(title='Transverse impedance Z⊥(w) [normalized by '+str(round(norm_x,3))+']',
        xlabel='f [GHz]',
        ylabel='Z⊥(w) [$\Omega$]',   
        #ylim=(0.,np.max(Z_x)*1.2),
        #xlim=(0.,np.max(Z_x_freq))      
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()