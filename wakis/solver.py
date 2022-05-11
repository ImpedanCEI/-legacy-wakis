'''
-----------------------
| WAKIS solver module |
-----------------------
Functions for WAKIS solver

Functions: [TODO]
---------
- a
- b
- c

Requirements:
------------- 
pip install matplotlib, numpy, h5py, scipy

'''
import os 
import pickle as pk

import numpy as np
import scipy.fftpack as fftpack  
import h5py 
from scipy.constants import c, pi
from proc import read_WarpX, read_Ez

UNIT = 1e-3 #conversion to m
OUT_PATH = os.getcwd() + '/'

def check_input(data):
    '''
    Check if all the needed variables for the wake solver are defined.
    - Charge distribution (z.t) extraction from rho.h5 file [C/m]
    - Beam charge 'q': default 1e-9 [C]
    - Beam longitudinal sigma 'sigmaz': default 0.02 [m]
    - Unit conversion 'unit': default 1e-3 [m]
    '''
    if data.get('charge_dist') is None:
        hf_rho = h5py.File(out_path +'rho.h5', 'r')

        #get number of datasets
        dataset_rho=[]
        for key in hf_rho.keys():
            dataset_rho.append(key)

        # Extract charge distribution [C/m] lambda(z,t)
        charge_dist=[]
        x=data.get('x')
        y=data.get('y')
        nt=data.get('nt')

        dx=x[2]-x[1]
        dy=y[2]-y[1]
        for n in range(nt):
            rho=hf_rho.get(dataset_rho[n]) # [C/m3]
            charge_dist.append(np.array(rho)*dx*dy) # [C/m]

        charge_dist=np.transpose(np.array(charge_dist)) # [C/m]

        # Correct the maximum value so the integral along z = q
        timestep=np.argmax(charge_dist[nz//2, :])   #max at cavity center
        qz=np.sum(charge_dist[:,timestep])*dz       #charge along the z axis
        charge_dist = charge_dist[int(nz/2-L_pipe/dz):int(nz/2+L_pipe/dz+1), :]*bunch_charge/qz    #total charge in the z axis

        data['charge_dist']=charge_dist
        print('charge_dist checked')

    if data.get('q') is None:
        data['q']=1e-9
        print('q checked')

    if data.get('unit') is None:
        data['unit']=1e-3
        print('unit checked')

    return data


def read_WarpX(out_path):
    '''
    Read the input data of warpx stored in a dict with pickle
    '''
    if os.path.exists(out_path+'input_data.txt'):
        with open(out_path+'input_data.txt', 'rb') as handle:
            input_data = pk.loads(handle.read())

        return input_data
    else: 
        return None
        print('[! WARNING] No WarpX data found')

def read_Ez(out_path, filename='Ez.h5'):
    '''
    Read the Ez h5 file
    '''
    hf = h5py.File(out_path+filename, 'r')
    print('Reading the h5 file: '+ out_path+filename)
    print('--- Size of the file: '+str(round((os.path.getsize(out_path+filename)/10**9),2))+' Gb')

    # get number of datasets
    size_hf=0.0
    dataset=[]
    n_step=[]
    for key in hf.keys():
        size_hf+=1
        dataset.append(key)
        n_step.append(int(key.split('_')[1]))

    # get size of matrix
    Ez_0=hf.get(dataset[0])
    shapex=Ez_0.shape[0]  
    shapey=Ez_0.shape[1] 
    shapez=Ez_0.shape[2] 
    print('--- Ez field is stored in a matrix with shape '+str(Ez_0.shape)+' in '+str(int(size_hf))+' datasets')

    return hf, dataset


def calc_long_WP(data, out_path, filename='Ez.h5'):
    '''
    Obtains the Longitudinal Wake Potential 
    from the electric field Ez 
    through the Direct method

    -data=read_WarpX(OUT_PATH)
    '''

    # Read data
    hf, dataset = read_Ez(out_path, filename)

    t = data.get('t')               #simulated time [s]
    z = data.get('z')               #z axis values  [m]
    t_inj = data.get('init_time')   #injection time [s]
    q = data.get('q')               #beam charge [C]
    unit = data.get('unit')         #convserion from [m] to [xm]
    z0 = data.get('z0')             #full domain length (+pmls) [m]

    if z0 is None: z0 = z

    # Aux variables
    nt = len(t)
    dt = t[2] - t[1]

    nz = len(z)
    zmax = max(z)
    zmin = min(z)

    zi=np.linspace(zmin, zmax, nt)  #interpolated z
    dzi=zi[2]-zi[1]                 #interpolated z resolution

    # Set Wake_length, s
    Wake_length=nt*dt*c - (zmax-zmin) - t_inj*c

    ns_neg=int(t_inj/dt)            #obtains the length of the negative part of s
    ns_pos=int(Wake_length/(dt*c))  #obtains the length of the positive part of s
    s=np.linspace(-t_inj*c, 0, ns_neg) #sets the values for negative s
    s=np.append(s, np.linspace(0, Wake_length,  ns_pos))

    print('[! INFO] Max simulated time = '+str(round(t[-1]*1.0e9,4))+' ns')
    print('[! INFO] Wakelength = '+str(Wake_length/unit)+' mm')

    # Initialize variables
    Ezi = np.zeros((nt,nt))     #interpolated Ez field
    ts = np.zeros((nt, len(s))) #result of (z+s)/c for each z, s

    WP = np.zeros_like(s)
    WP_3d = np.zeros((3,3,len(s)))

    i0=1    #center of the array in x
    j0=1    #center of the array in y

    print('[PROGRESS] Calculating longitudinal wake potential...')
    for i in range(-i0,i0+1,1):  
        for j in range(-j0,j0+1,1):

            # Interpolate Ez field
            n=0
            for n in range(nt):
                Ez=hf.get(dataset[n])
                Ezi[:, n]=np.interp(zi, z, Ez[i0+i,j0+j,:])

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
                    ts[k,n]=(zi[k]+s[n])/c-zmin/c-t[0]+t_inj

                    if ts[k,n]>0.0:
                        it=int(ts[k,n]/dt)-1                                              #find index for t
                        WP[n]=WP[n]+(Ezi[k, it])*dzi   #compute integral

            WP=WP/(q*1e12)     # [V/pC]

            WP_3d[i0+i,j0+j,:]=WP 

    return WP_3d, s


def calc_trans_WP(WP_3d, s, data):
    '''
    Obtains the transverse wake potetential 
    through Panofsky-Wenzel theorem using the 
    longitudinal wake potential calculation 
    '''
    # Obtain x, y 
    x=data.get('x')
    y=data.get('y')
    dx=x[2]-x[1]
    dy=y[2]-y[1]

    # Initialize variables
    i0 = 1 
    j0 = 1
    ds = s[2]-s[1]
    WPx = np.zeros_like(s)
    WPy = np.zeros_like(s)
    int_WP = np.zeros_like(WP_3d)

    # Obtain the transverse wake potential 
    print('[PROGRESS] Calculating transverse wake potential...')
    for n in range(len(s)):
        for i in range(-i0,i0+1,1):
            for j in range(-j0,j0+1,1):
                # Perform the integral
                int_WP[i0+i,j0+j,n]=np.sum(WP_3d[i0+i,j0+j,0:n])*ds 

        # Perform the gradient (second order scheme)
        WPx[n] = - (int_WP[i0+1,j0,n]-int_WP[i0-1,j0,n])/(2*dx)
        WPy[n] = - (int_WP[i0,j0+1,n]-int_WP[i0,j0-1,n])/(2*dy)

    return WPx, WPy

def FFT(Xt, dt, fmax=None, r=2.0, flag_zeropadding=True):
    ''' 
    Calculate the FFT of a signal
    -Xt: time domain signal with a constant dt
    -dt: resolution in time domain [s]
    -fmax: fmax to analyse, defined by the sigmat of the bunch: fmax=1/(3*sigmat)
    -r: relative length of the zero padding
    '''

    # Define FFT parameters
    N=len(Xt)    # Number of time domain samples
    T=N*dt       # Total time [s]
    fres=1/T     # Resolution in frequency [Hz]
    if fmax is None:
        fmax=1/dt
        dts=dt    
    else:
        dts=1/(2.0*fmax)    # Time window [s]  

    #Sample the time signal
    t=np.linspace(0, T, N)      # Original time array
    ts=np.arange(0, T, dts)     # Sampled time array
    Xs=np.interp(ts,t,Xt)       # Sampled time domain signal
    Ns=N/(dts/dt)               # Number of FFT samples

    #Add zero padding
    if flag_zeropadding:
        pad=int(r*Ns)          # Adjust by changing the relative length r  
        Xpad=np.append(np.append(np.zeros(pad), Xs), np.zeros(pad))

        Xs=Xpad 

    #Perform FFT
    Xfft=np.fft.fft(Xs)                     #FFT of the full spectrum
    ffft=np.fft.fftfreq(len(Xfft), dts)     #frequencies of the full specrtum
    mask= ffft >= 0

    Xf=2.0*Xfft[mask]/Ns    # Positive FFT, normalized
    f=ffft[mask]            # Positive frequencies

    print('------------------------------------------------------')
    print('Performing FFT')
    print(' - fmax = ' + str(fmax*1e-9) + ' GHz')
    print(' - fres = ' + str(fres*1e-6) + ' MHz')
    print(' - N samples = ' + str(Ns) + '\n')

    #Parsevals identity
    Et=np.sum(abs(Xs)**2.0)
    Ef=np.sum(abs(Xfft)**2.0)/len(ffft)
    K=np.sqrt(Et/Ef)

    print('Parseval identity check')
    print('Energy(time)/Energy(frequency) = '+ str(K)+' == 1.0')
    print('Energy(time)-Energy(frequency) = '+ str(round(Et-Ef, 3))+' == 0.0')
    print('------------------------------------------------------')

    Xf=K*Xf

    return Xf, f


def DFT(Xt, dt, fmax=None, Nf=1000):
    ''' 
    Calculate the DFT of a signal
    -Xt: time domain signal with a constant dt
    -dt: resolution in time domain [s]
    -Nf:number of samples in frequency domain
    -fmax: fmax to analyse, defined by the sigmat of the bunch: fmax=1/(3*sigmat)
    '''
    
    # Define FFT parameters
    N=len(Xt)    # Number of time domain samples
    T=N*dt       # Total time [s]
    fres=1/T     # Resolution in frequency [Hz]
    if fmax is None:
        fmax=1/dt
        dts=dt    
    else:
        dts=1/(2.0*fmax)    # Time window [s]   

    #Sample the time signal
    t=np.arange(0, T, dt)       # Original time array
    ts=np.arange(0, T, dts)     # Sampled time array
    Xs=np.interp(ts,t,Xt)       # Sampled time domain signal
    Ns=N/(dts/dt)               # Number of FFT samples

    #Perform FFT
    Xf=fftpack.rfft(Xs, Nf)              #FFT of the full spectrum
    f=fftpack.rfftfreq(len(Xf), dts)     #frequencies of the full specrtum

    print('------------------------------------------------------')
    print('Performing DFT')
    print(' - fmax = ' + str(fmax*1e-9) + ' GHz')
    print(' - fres = ' + str(fres*1e-6) + ' MHz')
    print(' - N samples = ' + str(Ns) + ' GHz' + '\n')
    
    #Parsevals identity
    Et=np.sum(abs(Xs)**2.0)
    Ef=(Xf[0]**2 + 2 * np.sum(Xf[1:]**2)) / len(f)
    K=np.sqrt(Et/Ef)
    
    print('Parseval identity check')
    print('Energy(time)/Energy(frequency) = '+ str(K)+' == 1.0')
    print('Energy(time)-Energy(frequency) = '+ str(round(Et-Ef, 3))+' == 0.0')
    print('------------------------------------------------------')

    mask=np.arange(Nf)%2.0 == 0.0  #Take the imaginary values of Xf
    Z=1j*np.zeros(len(Xf[mask]))
    Zf=np.zeros(len(Xf[mask]))

    if Nf%2.0 == 0.0:
        Re = ~mask
        Re[-1]=False
        Im = mask
        Im[0]=False

        Z[1:]=Xf[Re]+1j*Xf[Im]   #Reconstruct de complex array
        Z[0]=Xf[0]               #Take the DC value

        Zf[1:]=f[Im]
        Zf[0]=0.0

    else:
        Re = ~mask
        Im = mask
        Im[0]=False

        Z[1:]=Xf[Re]+1j*Xf[Im]   #Reconstruct de complex array
        Z[0]=Xf[0]               #Take the DC value

        Zf[1:]=f[Im]
        Zf[0]=0.0

    Xf=K*Z/(Ns/2)
    f=Zf

    return Xf, f

def test_FT():
    #Proof of FFT / DFT algorithm performance with sines

    N=500
    T=100
    w=2.0*np.pi/T
    t=np.linspace(0,T,N)
    dt=T/N

    Xt1=1.0*np.sin(5.0*w*t)
    Xt2=2.0*np.sin(10.0*w*t)
    Xt3=0.5*np.sin(20.0*w*t)

    Xt=Xt1+Xt2+Xt3

    # Plot time domain
    fig = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
    ax=fig.gca()
    ax.plot(t, Xt, marker='o', markersize=1.0, color='black', label='Xt1+Xt2+Xt3')
    ax.plot(t, Xt1, marker='o', markersize=1.0, color='blue', label='Xt1')
    ax.plot(t, Xt2, marker='o', markersize=1.0, color='red', label='Xt2')
    ax.plot(t, Xt3, marker='o', markersize=1.0, color='green', label='Xt3')

    ax.grid(True, color='gray', linewidth=0.2)
    ax.legend(loc='best')
    plt.show()

    Xf, f = FFT(Xt, dt, fmax=0.5, flag_zeropadding=True, r=3.0)
    Xdft, fdft = DFT(Xt, dt, fmax=0.5, Nf=1000)

    # Plot frequency domain
    fig = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
    ax=fig.gca()
    ax.plot(f, abs(Xf), marker='o', markersize=3.0, color='blue', label='FFT')
    ax.plot(fdft, abs(Xdft), marker='o', markersize=3.0, color='red', label='DFT')
    ax.grid(True, color='gray', linewidth=0.2)
    ax.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    
    test_FT()
    

