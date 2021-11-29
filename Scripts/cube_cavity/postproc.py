'''
File for postprocessing warp simulations

--- Reads the out file with pickle module
--- Plots the Electric field in the longitudinal direction
--- Obtains the frequency of the Electric field

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

#--- read the dictionary
with open('out_fixedfield/out.txt', 'rb') as handle:
  data = pk.loads(handle.read())
  print('stored variables')
  print(data.keys())

#--- retrieve the variables

Ez_t=data.get('Ez')
Ex_t=data.get('Ex')
Ey_t=data.get('Ey')
Bx_t=data.get('Bx')
By_t=data.get('By')
rho_t=data.get('rho')
x=data.get('x')
y=data.get('y')
z=data.get('z')
t=data.get('t')
nt=data.get('nt')
nz=data.get('nz')
xtest=data.get('xtest')
ytest=data.get('ytest')

###################
# 	Plots     #
###################

#--- Plot Electric field at cavity center
z=np.linspace(min(z), max(z), nz+1)
Ez=np.transpose(np.array(Ez_t))     #np.reshape(Ez_t, (nz+1,nt))      #array to matrix (z,t)
Ex=np.transpose(np.array(Ex_t))     #list to matrix (z,t)
Ey=np.transpose(np.array(Ey_t))     #list to matrix (z,t)
t=np.array(t)
E_abs=np.sqrt(Ez[int(nz/2), :]**2+Ex[int(nz/2), :]**2+Ey[int(nz/2), :]**2)

fig1 = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
ax1=fig1.gca()
ax1.plot(t*1.0e9, Ez[int(nz/2), :], lw=1.2, color='g', label='Ez warp')
#ax1.plot(np.array(t)*1.0e9, E_abs, lw=1.2, color='k', label='E_abs')
ax1.set(title='Electric field at cavity center',
        xlabel='t [ns]',
        ylabel='$E [V/m]$',         
        ylim=(np.min(Ez[int(nz/2), :])*1.1,np.max(Ez[int(nz/2), :])*1.1),
        xlim=(0,np.max(t*1.0e9))
		)
ax1.legend(loc='best')
ax1.grid(True, color='gray', linewidth=0.2)
plt.show()

#--- Plot frequency

freq=np.fft.fftfreq(len(t[0:-1:5]), d=(t[1]-t[0])*5.0e9)
Ez_fft=np.fft.fft(Ez[int(nz/2), 0:-1:5])
Amp=np.abs(Ez_fft)
Amp_max=np.argmax(Amp[:int(len(freq)/2-1)])

print("FFT parameters: Freq resolution = "+str())
fig2 = plt.figure(2, figsize=(6,4), dpi=200, tight_layout=True)
ax2=fig2.gca()
ax2.plot(freq[Amp_max], Amp[Amp_max], marker='o', markersize=4.0, color='cyan')
ax2.annotate(str(round(freq[Amp_max],2))+ ' GHz', xy=(freq[Amp_max],Amp[Amp_max]), xytext=(1,1), textcoords='offset points', color='grey') 
#arrowprops=dict(color='blue', shrink=1.0, headwidth=4, frac=1.0)
ax2.plot(freq[0:int(len(freq)/2)], Amp[0:int(len(freq)/2)], lw=1, color='b', label='fft')
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

#--- Plot charge density at cavity center

n=int(738)
rho=np.transpose(np.array(rho_t))      #array to matrix (z,t)

fig3 = plt.figure(3, figsize=(6,4), dpi=200, tight_layout=True)
ax3=fig3.gca()
ax3.plot(np.array(z)*1.0e3, rho[:, n], lw=1.2, color='r', label='Charge density')
ax3.set(title='Charge density in t='+str(round(t[n]*1.0e9,4))+' ns',
        xlabel='z [mm]',
        ylabel='$ rho $')
ax3.legend(loc='best')
ax3.grid(True, color='gray', linewidth=0.2)
plt.show()


#--- compare with cst
with open('cst/cst_out.txt', 'rb') as handle:
  data = pk.loads(handle.read())
  print('stored variables')
  print(data.keys())

#--- retrieve the variables

Ez_cst=data.get('Ez')
t_cst=data.get('t')
nz_cst=data.get('nz')
nt_cst=data.get('nt')


#--- Plot Electric field at cavity center


fig4 = plt.figure(4, figsize=(6,4), dpi=200, tight_layout=True)
ax4=fig4.gca()
ax4.plot(np.array(t)*1.0e9, Ez[int(nz/2), :], lw=0.8, color='b', label='Ez Warp')
ax4.plot(np.array(t_cst)*1.0e9, Ez_cst[int(nz_cst/2), :], lw=0.8, color='r', label='Ez CST')
ax4.set(title='Electric field at cavity center',
        xlabel='t [ns]',
        ylabel='$E [V/m]$',         
        ylim=(np.min(Ez[int(nz/2), :])*1.1,np.max(Ez[int(nz/2), :])*1.1),
        xlim=(0,np.max(t*1.0e9))
		)
ax4.legend(loc='best')
ax4.grid(True, color='gray', linewidth=0.2)
plt.show()


#--- Plot frequency

freq_cst=np.fft.fftfreq(len(t_cst[0:-1:2]), d=(t_cst[1]-t_cst[0])*2.0e9)
Ez_fft_cst=np.fft.fft(Ez_cst[int(nz_cst/2), 0:-1:2])
Amp_cst=np.abs(Ez_fft_cst)
Amp_cst_max=np.argmax(Amp_cst[:int(len(freq_cst)/2-1)])

fig5 = plt.figure(5, figsize=(6,4), dpi=200, tight_layout=True)
ax5=fig5.gca()

ax5.plot(freq[0:int(len(freq)/2)], Amp[0:int(len(freq)/2)], lw=1, color='b', label='fft Warp')
ax5.plot(freq[Amp_max], Amp[Amp_max], marker='o', markersize=4.0, color='cyan')
ax5.annotate(str(round(freq[Amp_max],2))+ ' GHz', xy=(freq[Amp_max],Amp[Amp_max]), xytext=(10,5), textcoords='offset points', color='blue')

ax5.plot(freq_cst[0:int(len(freq_cst)/2)], Amp_cst[0:int(len(freq_cst)/2)], lw=1, color='r', label='fft CST') 
ax5.plot(freq_cst[Amp_cst_max], Amp_cst[Amp_cst_max], marker='o', markersize=4.0, color='pink')
ax5.annotate(str(round(freq_cst[Amp_cst_max],2))+ ' GHz', xy=(freq_cst[Amp_cst_max],Amp_cst[Amp_cst_max]), xytext=(10,1), textcoords='offset points', color='red')

ax5.set(title='Frequency of Electric field at cavity center',
        xlabel='f [GHz]',
        ylabel='Amplitude [dB]',   
        #ylim=(0,np.max(Amp)*1.3),
        #xlim=(0,np.max(freq))      
        )
ax5.legend(loc='best')
ax5.grid(True, color='gray', linewidth=0.2)
ax5.grid(True, color='gray', linewidth=0.2)
plt.show()



#--- Plot *normalized* Electric field at cavity center

max_warp=np.max(abs(Ez[int(nz/2), :]))
max_cst=np.max(abs(Ez_cst[int(nz_cst/2), :]))

fig6 = plt.figure(6, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig6.gca()
ax.plot(np.array(t)*1.0e9, abs(Ez[int(nz/2), :])/max_warp, lw=0.8, color='b', label='Ez Warp')
ax.plot(np.array(t_cst)*1.0e9, abs(Ez_cst[int(nz_cst/2),:])/max_cst, lw=0.8, color='r', label='Ez CST')
ax.set(title='Electric field at cavity center',
        xlabel='t [ns]',
        ylabel='$E [V/m]$',         
        ylim=(-0.1,1.3),
        xlim=(0,np.max(t*1.0e9))
		)
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

''' 
#loop over time
for n in range(nt):
	fig2 = plt.figure(20, figsize=(6,4), dpi=200, tight_layout=True)
	ax2=fig2.gca()
	ax2.plot(np.array(z)*1.0e3, rho[:, n], lw=1.2, color='g', label='Ez from warp')
	ax2.set(title='Charge density in t='+str(t[n]),
	        xlabel='z [mm]',
	        ylabel='$ rho $')
	ax2.legend(loc='best')
	plt.show()
'''

