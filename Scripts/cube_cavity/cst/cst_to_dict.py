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
import pickle as pk


#--- read one file
fname = 'Ez_050'

#Initialize variables
Ez=[]
t=[]
i=0 

with open('cst_files/'+fname+'.txt') as f:
    for line in f:
        i+=1
        content = f.readline()
        columns = content.split()

        if i>1 and len(columns)>1:

            Ez.append(float(columns[1]))
            t.append(float(columns[0]))

Ez=np.array(Ez) # in V/m
t=np.array(t)*1.0e-9   # in s

#close file
f.close()

#--- Plot electric field

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

#--- Plot frequency

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

#--- read all files

Ez_t=np.zeros((len(glob.glob("cst_files/*.txt")),len(t)))
k=0
i=0

for file in sorted(glob.glob("cst_files/*.txt")):
    print('Scanning file '+ file)
    with open(file) as f:
        for line in f:
            i+=1
            content = f.readline()
            columns = content.split()

            if i>1 and len(columns)>1:

                Ez_t[k,i-2]=(float(columns[1]))
    k+=1
    i=0
    #close file
    f.close()
    
print('Finished scanning files')

#--- save the matrix into a txt

data = { 'Ez' : Ez_t, #len(k, len(t))
         't' : t, #time in [s]
         'nz' : k,
         'nt' : len(t)
        }
# write the dictionary to a txt using pickle module
with open('cst_out.txt', 'wb') as handle:
    pk.dump(data, handle)
