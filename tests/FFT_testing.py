'''
FFT_testing.py

Benchmark of the FFT method with CST files
'''
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from scipy.constants import c, mu_0, pi 
from scipy.special import iv
from copy import copy
import pickle as pk   

# UNIT=1e-3 #mm to m

# # Gaussian bunch 
# t0=0.53e-10         #injection time [s]
# WL=50*UNIT          #wakelength [m]
# sigmaz=1.0*UNIT     # [m]
# sigmat=sigmaz/c     # [s]
# q=1e-9              # [C]

# #--- define time
# N=10000
# tau=np.linspace(1e-12, WL/c, N)
# dt=tau[2]-tau[1]

# #--- define s
# s=np.linspace(-t0*c, WL, N)
# ds=s[2]-s[1]

# # bunch=np.exp(-((tau-2*t0)**2)/(2*(sigmat**2))) #*1/(sigmat*np.sqrt(2*pi))
# bunch=q*np.exp(-((s)**2)/(2*(sigmaz**2)))*1/(sigmaz*np.sqrt(2*pi))

# # Wakefunction

# F=1.0           #form factor
# L=300*UNIT      #length [m]
# b=1.0*UNIT      #radius [m]
# Zo=376.73       #vacuum impedance [Ohm]
# sigma_el=1.0e6   #electric conductivity [S/m]
# #s_wf=np.linspace(1.0e-5, WL, 1000)
# WF=np.zeros_like(s)

# #WF=-F*L/(4*pi*b)*np.sqrt(Zo/(pi*c*sigma_el))*(1/np.power(tau,3/2))
# mask = s > 0
# WF[mask]=F*L*c/(4*pi*b)*np.sqrt(Zo/(pi*sigma_el))*(1/np.power(abs(s[mask]),3/2))
# #WF[np.logical_not(mask)] = 0.0

# # Wakepotential

# #--- with convolution

# '''
# WFf, f = Wsol.DFT(WF, ds/c, fmax=fmax, Nf=1000)
# bunchf, f=Wsol.DFT(bunch, ds/c, fmax=fmax, Nf=1000)

# convf = WFf*bunchf
# WP = np.fft.ifft(Wf)
# '''

# #WP_conv=(1/(q*1e12))*np.convolve( bunch[mask] , WF[mask] )   #convolution of Wakefunction and bunch charge distribution [V/pC]
# WP_conv=(1/(q*1e12))*np.convolve( bunch , WF )
# s_conv=np.linspace(0, WL, len(WP_conv))
# WP_conv = np.interp(s, s_conv, WP_conv)


# #--- from theory
# x=s/(2*sigmaz)
# WPth=-c*L/(4*pi*b*np.power(sigmaz, 3/2))*np.sqrt(Zo/sigma_el)*np.power(abs(x), 3/2)*np.exp(-x**2)*(iv(-3/4, x*x)-iv(1/4, x*x)+np.sign(s)*(iv(3/4, x*x)-iv(-1/4, x*x)))
# WPth=WPth*1e-12 #[V/pC]

# # Impedance

# fmax=1/(3*sigmat)

# #--- with FFT
# bunchf, f2=Wsol.FFT(bunch/q, ds/c, fmax=2*fmax, flag_zeropadding=False)
# WPf, f=Wsol.FFT(WPth*1e12, ds/c, flag_zeropadding=False)

# bunchf = np.interp(f,f2,bunchf)

# #--- with DFT
# #WPf, f=Wsol.DFT(WPth, ds/c, fmax=fmax, Nf=1000)
# #bunchf, f=Wsol.DFT(bunch/q, ds/c, fmax=fmax, Nf=1000)
# #WPf_conv, f=Wsol.DFT(WP_conv, ds/c, fmax=fmax, Nf=1000)

# Z = - WPf / bunchf 
# Z_abs= abs(Z)
# Z_re=np.real(Z)
# Z_im=np.imag(Z)

# #Z_conv = - WPf_conv / bunchf 

# #--- from theory
# fth=np.linspace(0, fmax, 1000)
# Zth=(1-1j*np.sign(fth))*L/(2*pi*b)*np.sqrt(Zo*2*pi*fth/(2*c*sigma_el))

# Zth_abs=abs(Zth)
# Zth_re=np.real(Zth)
# Zth_im=np.imag(Zth)

# '''
# # Plot WP 
# fig = plt.figure(2, figsize=(6,4), dpi=200, tight_layout=True)
# ax=fig.gca()
# ax.plot(s, WP_conv, color='red', label='Wake potential from convolution [norm]')
# ax.plot(s, WPth, color='red', ls='--', label='Wake potential from theory')
# ax.plot(s, bunch/max(bunch)*max(abs(WPth)), color='orange', label='lambda(s)')
# ax.set(title='Resistive wall Wake potential W//(s)',
#         xlabel='s [m]',
#         ylabel='WP [V/pC]',         
#         )
# ax.legend(loc='best')
# ax.grid(True, color='gray', linewidth=0.2)
# plt.show()
# '''

# '''
# # Plot WF
# fig = plt.figure(2, figsize=(6,4), dpi=200, tight_layout=True)
# ax=fig.gca()
# ax.plot(s, WF, color='blue', ls='--', label='Wake function from theory')
# ax.plot(s, bunch, color='orange', label='lambda(s)')
# ax.set(title='Resistive wall Wake function',
#         xlabel='s [m]',
#         ylabel='WF [V]',         
#         )
# ax.legend(loc='best')
# ax.grid(True, color='gray', linewidth=0.2)
# plt.show()
# '''

# # Plot Z vs Zth
# factor=sum(Zth_abs)/len(Zth_abs)/(sum(Z_abs)/len(Z_abs))
# fig = plt.figure(2, figsize=(6,4), dpi=200, tight_layout=True)
# ax=fig.gca()
# #ax.plot(fth, Zth_abs, color='green', ls='--', label='|Z(f)| from theory')
# ax.plot(f, Z_abs, color='green', label='|Z(f)| from FFT')
# #ax.plot(f, abs(Z_conv), color='m', label='|Z(f)| from convolution')
# ax.plot(f, Z_re, color='red', label='Zre(f) from FFT')
# ax.plot(f, Z_im, color='blue', label='Zim(f) imaginary from FFT')
# ax.set(title='Resistive wall impedance Z//(f)', #' \n Zth / Z = ' + str(round(factor,3)),
#         xlabel='f [Hz]',
#         ylabel='Z [Ohm]',  
#         xlim=(0, 1/(3*sigmat)),      
#         )
# ax.legend(loc='best')
# ax.grid(True, color='gray', linewidth=0.2)
# plt.show()


# '''
# # Plot Z vs Zth Re and Im
# fig = plt.figure(2, figsize=(6,4), dpi=200, tight_layout=True)
# ax=fig.gca()
# ax.plot(f, Z_re, color='red', label='Zre(f) from FFT')
# ax.plot(f, Z_im, color='blue', label='Zim(f) imaginary from FFT')
# ax.plot(f, Zth_re, color='green', ls='--', label='Zre(f) from theory')
# ax.plot(f, Zth_im, color='m', ls='--', label='Zim(f) from theory')
# ax.set(title='Resistive wall impedance',
#         xlabel='f [Hz]',
#         ylabel='Z [Real / Imag]',         
#         )
# ax.legend(loc='best')
# ax.grid(True, color='gray', linewidth=0.2)
# plt.show()
# '''





















































# #--- read the cst dictionary
# with open('cst_out.txt', 'rb') as handle:
#   cst_data = pk.loads(handle.read())

# print(cst_data.keys())
# charge_dist_time=cst_data.get('charge_dist_time')
# t=cst_data.get('t_charge_dist')
# t0=cst_data.get('init_time')
# dt=t[2]-t[1]

# charge_dist=cst_data.get('charge_dist')
# s_charge_dist=cst_data.get('s_charge_dist')
# ds_charge_dist=s_charge_dist[2]-s_charge_dist[1]
# spectrum=cst_data.get('charge_dist_spectrum')
# df=0.0094650788*1e9
# f_spectrum=np.arange(0, 1001*df, df)
# q=1e-9 #[C]

# #'''
# Wake_potential=cst_data.get('WP_cst')
# s_cst=cst_data.get('s_cst')
# Z_cst=cst_data.get('Z_cst')
# freq_cst=cst_data.get('freq_cst')
# sigmaz=cst_data.get('sigmaz')
# #'''
# '''
# Wake_potential_cst=cst_data.get('WPx_dipolar_cst')
# s_cst=cst_data.get('s_cst_dipolar')
# Z_cst=cst_data.get('Zx_dipolar_cst')
# freq_cst=cst_data.get('freq_cst_dipolar')
# '''
# '''
# Wake_potential_cst=cst_data.get('WPy_quadrupolar_cst')
# s_cst=cst_data.get('s_cst_quadrupolar')
# Z_cst=cst_data.get('Zy_quadrupolar_cst')
# freq_cst=cst_data.get('freq_cst_quadrupolar')
# '''

# #--- Auxiliary variables
# ds=s_cst[2]-s_cst[1]
# s=np.arange(np.min(s_cst),np.max(s_cst),ds) #constant ds vector

# #--- Obtain impedance Z with Fourier transform numpy.fft.fft
# # MAKE A SYMMETRIC SIGNAL

# # Interpolate charge distribution
# # INTERPOLATE TO HAVE A CONSTANT ds. PLOT CST DS DISTRIBUTION
# charge_dist_interp=np.interp(s, s_charge_dist, charge_dist)
# Wake_potential_interp=np.interp(s, s_cst, Wake_potential)

# #lambdaf, f=Wsol.FFT(charge_dist_interp, ds/c, fmax=np.max(freq_cst), r=10.0)
# #WPf, f=Wsol.FFT(Wake_potential_interp, ds/c, fmax=np.max(freq_cst), r=10.0)
        

# #lambdaf, f2=Wsol.DFT(charge_dist/q, ds_charge_dist/c, fmax=max(freq_cst), Nf=2001)
# WPf, f=Wsol.DFT(Wake_potential*1e12, ds/c, fmax=max(freq_cst), Nf=2001)
# #WPf=WPf*sum(Wake_potential*1e12)*ds/c/np.sqrt(pi)

# #lambdaf=np.interp(f,f2,lambdaf)
# lambdaf=np.interp(f, f_spectrum, spectrum/q)*c


# # Compute the impedance
# Z = abs(- WPf / lambdaf) # * 2/(t_sample*ds/np.sqrt(pi)) #normalized according to CST wakesolver manual

# # Plot Impedance and maximum frequency
# fig = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
# ax=fig.gca()

# # add CST fft result
# ax.plot(freq_cst*1.0e-9, Z_cst, lw=1.2, color='black', label='Z// from CST')

# # add numpy.fft result (normalized)
# factor=np.max(Z_cst)/np.max(Z)
# ax.plot(f*1.0e-9, Z, lw=1.2, color='red', label='Z// from numpy FFT')

# ax.set(title='Longitudinal impedance Z from CST magnitude',
#         xlabel='frequency [GHz]',
#         ylabel='Z//(s) [Ohm]',         #ylim=(-8.0e4,8.0e4)
#         xlim=(0.,np.max(freq_cst)*1e-9)
#         )
# ax.legend(loc='best')
# ax.grid(True, color='gray', linewidth=0.2)
# plt.show()


# # Plot charge dist and wake potential
# '''
# fig = plt.figure(2, figsize=(6,4), dpi=200, tight_layout=True)
# ax=fig.gca()
# ax.plot(s*1e3, charge_dist_interp, color='red', label='$\lambda$(s)')
# ax.plot(s*1e3, Wake_potential_cst, color='orange', label='W||(s)')
# ax.set(title='Wake potential and charge distribution',
#         xlabel='s [mm]',
#         ylabel='W||(s) [V/pC]',         #ylim=(-8.0e4,8.0e4)
#         )
# ax.legend(loc='best')
# ax.grid(True, color='gray', linewidth=0.2)
# plt.show()
# '''

# fig = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
# ax=fig.gca()
# ax.plot(f/1e9, abs(lambdaf), color='red', label='$\lambda$(w)')
# ax.plot(f/1e9, abs(WPf), color='orange', label='W||(w)')
# ax.set(title='Wake potential and charge distribution',
#         xlabel='f [GHz]',
#         ylabel='W||(s) [V/pC]',         #ylim=(-8.0e4,8.0e4)
#         )
# ax.legend(loc='best')
# ax.grid(True, color='gray', linewidth=0.2)
# plt.show()


# #VALUE IN 0 SHOULD BE THE SAME AS THE INTEGRAL FOR THE CHARGE DIST


# spectrum_fft,f=Wsol.DFT(charge_dist_time, dt, fmax=max(f_spectrum), Nf=2001)
# spectrum_fft=spectrum_fft*sum(charge_dist_time)*dt/np.sqrt(pi)

# fig = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
# ax=fig.gca()
# ax.plot(f/1e9, abs(spectrum_fft), color='red', label='DFT')
# ax.plot(f_spectrum/1e9, spectrum, color='blue', label='CST')
# ax.set(title='Charge distribution spectrum',
#         xlabel='f [GHz]',
#         ylabel='Spectrum',         #ylim=(-8.0e4,8.0e4)
#         )
# ax.legend(loc='best')
# ax.grid(True, color='gray', linewidth=0.2)
# plt.show()












#--- read the cst dictionary
with open('cst_out.txt', 'rb') as handle:
  cst_data = pk.loads(handle.read())

bunch=cst_data.get('charge_dist')
bunch_time=cst_data.get('charge_dist_time')
spectrum=cst_data.get('charge_dist_spectrum')
current=cst_data.get('current')
s_bunch=cst_data.get('s_charge_dist')
ds_bunch=s_bunch[2]-s_bunch[1]
df=0.0094650788*1e9
f_spectrum=np.arange(0, 1001*df, df)
dt=0.0013634439*1e-9
t_current=np.arange(0,7.5371175*1e-9, dt)
q=1e-9 #[C]

WP=cst_data.get('WP_cst')
Z_cst=cst_data.get('Z_cst')
WP_dip=cst_data.get('WPx_dipolarX_cst')
Z_dip_cst=cst_data.get('Zx_dipolarX_cst')
WP_quad=cst_data.get('WPx_quadrupolarX_cst')
Z_quad_cst=cst_data.get('Zx_quadrupolarX_cst')

f_cst=cst_data.get('freq_cst_dipolar')
s_cst=cst_data.get('s_cst')
ds = s_cst[2]-s_cst[1]
df = f_cst[2]-f_cst[1]

bunch_i=np.interp(s_cst, s_bunch, bunch)

'''
lambdaf, f2=Wsol.DFT(bunch_i/q, ds/c, fmax=max(f_spectrum), Nf=2001)
WPf, f=Wsol.DFT(WP*1e12, ds/c, fmax=max(f_cst), Nf=2001)
WPf_dip, f=Wsol.DFT(WP_dip*1e12, ds/c, fmax=max(f_cst), Nf=2001)
WPf_quad, f=Wsol.DFT(WP_quad*1e12, ds/c, fmax=max(f_cst), Nf=2001)
'''
lambdafft = np.fft.fft(bunch_i/q*c, n=200000)
WPfft = np.fft.fft(WP*1e12, n=200000)
ffft=np.fft.fftfreq(len(WPfft), ds/c)

mask  = np.logical_and(ffft >= 0 , ffft < 5.5*1e9)
WPf = WPfft[mask]*ds
lambdaf = lambdafft[mask]*ds
f = ffft[mask]            # Positive frequencies

#lambdaf=np.interp(f,f2,lambdaf)*c/(2*pi)
lambdaf_cst=np.interp(f, f_spectrum, spectrum/q)*c

# Compute the impedance
Z = abs(- WPf / lambdaf)
#Z_dip = abs(1j* WPf_dip / lambdaf) 
#Z_quad = abs(1j* WPf_quad / lambdaf)


# Plot Impedance and maximum frequency
fig = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig.gca()

# add CST fft result

ax.plot(f*1.0e-9, Z, lw=1.2, color='red', label='Z// from numpy FFT')
ax.plot(f_cst*1.0e-9, Z_cst, lw=1.2, color='black', ls='--', label='Z// from CST')

#ax.plot(f_cst*1.0e-9, Z_cst/Z_dip_cst, lw=1.2, color='black', label='Z// from CST')
#ax.plot(f*1.0e-9, Z/Z_dip, lw=1.2, color='red', label='Z// from numpy FFT')
#ax.plot(s_cst*1e3, WP_quad, lw=1.2, color='blue', label='Z// from numpy FFT')

#ax.plot(f_cst*1.0e-9, abs(WPf_dip)/Z_dip_cst, lw=1.2, color='blue', label='Z// from numpy FFT')
#ax.plot(f*1.0e-9, abs(lambdaf_cst), lw=1.2, color='blue', label='Z// from numpy FFT')

ax.set(title='Longitudinal impedance Z from CST magnitude',
        xlabel='frequency [GHz]',
        ylabel='Z//(s) [Ohm]',         #ylim=(-8.0e4,8.0e4)
        #xlim=(0.,np.max(f_cst)*1e-9)
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()



