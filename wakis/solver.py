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
import h5py 
from scipy.constants import c, pi

from proc import read_WAKIS_in, read_WAKIS_out, read_Ez

UNIT = 1e-3 #conversion to m
cwd = os.getcwd() + '/'

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


def calc_long_WP(data, path=cwd, hf_name='Ez.h5'):
    '''
    Obtains the longitudinal Wake Potential from the electric field Ez 
    stored in 'hf_name.h5' using the direct method

    Parameters:
    -----------
    -data: wakis.in data. Can be obtained using data=read_Wakis_in(cwd)
    -path=cwd [default]. Path to the h5 file with the Ez field
    -hf_name='Ez.h5' [default]. Name of the h5 file with the Ez field   

    Returns:
    -------- 
    -WP_3d : Longitudinal wake potential matrix WP_3d(x,y,s) [V/pC]
    -s : distance from bunch head to test position [m]

    '''

    # Read data
    hf, dataset = read_Ez(path, hf_name)

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

    print('[PROGRESS] Calculating longitudinal wake potential WP...')
    for i in range(-i0,i0+1,1):  
        for j in range(-j0,j0+1,1):

            # Interpolate Ez field
            n=0
            for n in range(nt):
                Ez=hf.get(dataset[n])
                Ezi[:, n]=np.interp(zi, z, Ez[Ez.shape[0]//2+i,Ez.shape[1]//2+j,:])

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
    Obtains the transverse Wake Potetential through Panofsky-Wenzel theorem from the
    pre-computed longitudinal wake potential in 3d

    Parameters:
    -----------
    - output of calc_long_WP(data):
        -WP_3d : Longitudinal wake potential matrix WP_3d(x,y,s) [V/pC]
        -s : distance from bunch head to test position [m]
    - data: wakis.in input data
    
    Returns:
    --------
    -WPx : Horizontal transverse wake potential WPx(s) [V/pC]
    -WPy : Vertical transverse wake potential WPy(s) [V/pC]

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
    print('[PROGRESS] Calculating transverse wake potential WPx, WPy...')
    for n in range(len(s)):
        for i in range(-i0,i0+1,1):
            for j in range(-j0,j0+1,1):
                # Perform the integral
                int_WP[i0+i,j0+j,n]=np.sum(WP_3d[i0+i,j0+j,0:n])*ds 

        # Perform the gradient (second order scheme)
        WPx[n] = - (int_WP[i0+1,j0,n]-int_WP[i0-1,j0,n])/(2*dx)
        WPy[n] = - (int_WP[i0,j0+1,n]-int_WP[i0,j0-1,n])/(2*dy)

    return WPx, WPy

def calc_long_Z(WP, s, data):
    '''
    Obtain impedance Z with single-sided DFT using 1000 samples

    Parameters:
    -----------
    - output of calc_long_WP(data):
        -WP : Longitudinal wake potential matrix WP(s) [V/pC]
        -s : distance from bunch head to test position [m]
    - data: wakis.in input data
    
    Returns:
    --------
    - Z: longitudinal impedance Z(w) [Ohm]
    - f: DFT frequencies [Hz]


    '''
    print('[PROGRESS] Obtaining longitudinal impedance Z...')

    #Check input
    if WP.ndim > 1:
        WP = WP[1,1,:]

    # Retrieve variables
    sigmaz=data.get('sigmaz')
    q=data.get('q')
    charge_dist= data.get('charge_dist')
    z=data.get('z')

    # Obtain charge distribution as a function of s, normalized
    lambdas = np.interp(s, z, charge_dist/q)

    # Set up the DFT computation
    ds = s[2]-s[1]
    fmax=1*c/sigmaz/3
    N=int((c/ds)//fmax*1001) #to obtain a 1000 sample single-sided DFT

    # Obtain DFTs
    lambdafft = np.fft.fft(lambdas*c, n=N)
    WPfft = np.fft.fft(WP*1e12, n=N)
    ffft=np.fft.fftfreq(len(WPfft), ds/c)

    # Mask invalid frequencies
    mask  = np.logical_and(ffft >= 0 , ffft < fmax)
    WPf = WPfft[mask]*ds
    lambdaf = lambdafft[mask]*ds
    f = ffft[mask]            # Positive frequencies

    # Compute the impedance
    Z = - WPf / lambdaf

    return Z, f

def calc_trans_Z(WPx, WPy, s, data):

    print('[PROGRESS] Obtaining transverse impedance Zx, Zy...')
    # Retrieve variables
    sigmaz=data.get('sigmaz')
    q=data.get('q')
    charge_dist= data.get('charge_dist')
    z=data.get('z')

    # Obtain charge distribution as a function of s, normalized
    lambdas = np.interp(s, z, charge_dist/q)

    # Set up the DFT computation
    ds = s[2]-s[1]
    fmax=1*c/sigmaz/3
    N=int((c/ds)//fmax*1001) #to obtain a 1000 sample single-sided DFT

    # Obtain DFTs

    # Normalized charge distribution λ(w) 
    lambdafft = np.fft.fft(lambdas*c, n=N)
    mask  = np.logical_and(ffft >= 0 , ffft < fmax)
    lambdaf = lambdafft[mask]*ds

    # Horizontal impedance Zx⊥(w)
    WPxfft = np.fft.fft(WPx*1e12, n=N)
    WPxf = WPxfft[mask]*ds

    Zx = 1j * WPxf / lambdaf

    # Vertical impedance Zy⊥(w)
    WPyfft = np.fft.fft(WPy*1e12, n=N)
    WPyf = WPyfft[mask]*ds

    Zy = 1j * WPyf / lambdaf

    return Zx, Zy


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
    

