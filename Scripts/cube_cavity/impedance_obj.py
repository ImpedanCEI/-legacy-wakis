'''
    test script for different DFT options

'''


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
charge_dist_fft=abs(np.fft.fft(charge_dist[0:-1:t_sample]))
Wake_potential_fft=abs(np.fft.fft(Wake_potential[0:-1:t_sample]))
Z_freq = np.fft.fftfreq(len(s[:-1:t_sample]), ds/c*t_sample)*1e-9 #GHz
Z = abs(- Wake_potential_fft / charge_dist_fft)


#--- Plot impedance

# Obtain the maximum frequency
# Amp=np.abs(Z)
# Amp_max=np.argmax(Amp)
ifreq_max=np.argmax(Z[len(Z)//2:])+len(Z)//2 # obtains the largest value's index

# Plot with annotations
fig2 = plt.figure(2, figsize=(6,4), dpi=200, tight_layout=True)
ax=fig2.gca()
ax.plot(Z_freq[ifreq_max], Z[ifreq_max], marker='o', markersize=3.0, color='cyan')
ax.annotate(str(round(Z_freq[ifreq_max],2))+ ' GHz', xy=(Z_freq[ifreq_max],Z[ifreq_max]), xytext=(1,1), textcoords='offset points', color='grey') 
#arrowprops=dict(color='blue', shrink=1.0, headwidth=4, frac=1.0)
ax.plot(Z_freq, Z, lw=1, color='b', label='fft CST')
#ax2.plot(freq, Amp.imag, lw=1.2, color='r', label='Imaginary')
ax.set(title='Longitudinal impedance Z(w) magnitude',
        xlabel='f [GHz]',
        ylabel='Z [Ohm]',   
        ylim=(0.,np.max(Z)*1.2),
        xlim=(0.,np.max(Z_freq))      
        )
ax.legend(loc='best')
ax.grid(True, color='gray', linewidth=0.2)
plt.show()


#--- classical definition

class Fourier:
    def __init__(self, dft, freqs):
        self.dft = dft
        self.freqs = freqs

def DFT(F, dt): 
        #function to obtain the DFT with 1000 samples
        #--F: function in time domain
        #--dt: time sampling width

        #define frequency domain
        N_samples=1000      # same number as CST
        f_max = 1/dt        # maximum freq in GHz
        f_res = 1/(N_samples*dt)    # frequency bin resolution
        freqs=np.linspace(0,f_max,N_samples)*1e9 #frequency range [Hz]
        dft=np.zeros_like(freqs)*1j

        print('Performing DFT with '+str(N_samples)+' samples')
        print('Max Frequency bin resolution: '+str(round(f_res*1e-9,3))+ ' GHz')
        print('Max Frequency range: 0 to '+str(round(f_max*(1e-9)/2,2))+ ' GHz')

        #--- interpolate F so len(F) == N_samples
        t=np.linspace(0, 1, len(F))
        t_interp=np.linspace(0, 1, N_samples)
        F_interp=np.interp(t_interp, t, F)

        #--- Obtain DFT
        m=0
        for m in range(N_samples):
            for k in range(N_samples):
                dft[m]=dft[m]+F_interp[k]*np.exp(-1j*2*np.pi*m*k/N_samples)

        # Removing above Nyquist frequencies
        dft_nyq=dft[:N_samples//2] 
        freqs_nyq=freqs[:N_samples//2]*1e-9 #in [GHz]
        return Fourier(dft_nyq,freqs_nyq)  


charge_dist_fft=DFT(charge_dist, ds/c) 
Wake_potential_fft=DFT(Wake_potential, ds/c)
Z = abs(- Wake_potential_fft.dft / charge_dist_fft.dft )/c
Z_freq = Wake_potential_fft.freqs