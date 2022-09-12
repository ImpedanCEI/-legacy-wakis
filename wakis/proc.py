'''
------------------------
| WAKIS process module |
------------------------
Functions to pre-process Wakis input and post-process the output

Functions:
----------
Pre-processing functions:
- read_WAKIS_in
- read_Ez
- preproc_WarpX
- preproc_CST
- preproc_Ez
- preproc_rho
- preproc_lambda
- check_input

Post-processing functions:
- read_WAKIS_out
- animate_Ez
- contour_Ez
- plot_charge_dist
- plot_long_WP
- plot_long_Z
- plot_trans_WP
- plot_trans_Z
- plot_WAKIS
- subplot_WAKIS

Requirements:
------------- 
pip install matplotlib, numpy, h5py, scipy

'''

import os 
import pickle as pk

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5py

# Default working directory path for outputs
cwd = os.getcwd() + '/'

# Global parameters for plotting
plt.rcParams.update({'font.size': 12})
UNIT = 1e-3 #x-axis expressed in [mm]

#-----------------------------#
#   Pre-processing functions  #
#-----------------------------#

def read_WAKIS_in(path=cwd):
    '''
    Read the input data from file 'wakis.in' file 

    Parameters:
    -----------
    - path: path to the wakis.in input file. 
    '''

    if os.path.exists(path+'wakis.in'):
        with open(path+'wakis.in','rb') as handle:
            data = pk.loads(handle.read())
    else: 
        print('[! WARNING] wakis.in file not found')
        data=None

    return data

def read_Ez(out_path=cwd, filename='Ez.h5'):
    '''
    Read the Ez h5 file

    Parameters:
    -----------
    - out_path = cwd [default] path to the Ez.h5 file. The default is the current working directory 
    - filename = 'Ez.h5' [default]. Specify the name of the Ez file
    '''

    hf = h5py.File(out_path+filename, 'r')
    print('[PROGRESS] Reading the h5 file: '+ out_path+filename + ' ...')
    print('[INFO] Size of the file: '+str(round((os.path.getsize(out_path+filename)/10**9),2))+' Gb')

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

    print('[INFO] Ez field is stored in a matrix with shape '+str(Ez_0.shape)+' in '+str(int(size_hf))+' datasets')

    return hf, dataset

def preproc_WarpX(warpx_path, path=cwd):
    '''
    Pre-process WaprX simulation output warpx.out

    Parameters:
    -----------
    - warpx_path: path to the warpx output file. 
    - path: path to the wakis.in file

    Outputs:
    --------
    - wakis.in file ready to use by the solver module

    '''

    if os.path.exists(warpx_path+'warpx.out'):
        with open(warpx_path+'warpx.out', 'rb') as handle:
            data = pk.loads(handle.read())
    else: 
        print('[! WARNING] warpx.out file not found')
        data=None

    # Check input data
    data=check_input(data)

    # Check charge distribution
    if data.get('charge_dist') is None:
        data['charge_dist'] = preproc_rho(warpx_path)

    # Remove previous .in files
    if os.path.exists(path+'wakis.in'):
        os.remove(path+'wakis.in')

    # Save dictionary with pickle
    with open(path+'wakis.in', 'wb') as fp:
        pk.dump(data, fp)

    print('[! OUT] wakis.in file succesfully generated') 


def preproc_CST(cst_path, hf_name='Ez.h5', path=cwd, **kwargs):
    '''
    Pre-process the CST 3D field monitor output 

    Parameters:
    -----------
    - data_path: specify the path where the Ez and charge distribution data is.
                 [!] The folder containing the 3d Ez data should be named '3d'
                 [!] The file containing the charge distribution data should be named 'lambda.txt' 
    - **kwargs: input the rest of the required data if not stored already in cst.out
        - 'q': Beam charge, default 1e-9 [C]
        - 'sigmaz': Beam longitudinal sigma, default 0.02 [m]
        - 'unit': Unit conversion, default 1e-3 [m]
        - 'xsource', 'ysource': Beam position in transverse plane, default 0.0 [m]
        - 'xtest', 'ytest': Integration path position in transverse plane, default 0.0 [m]

    Default Parameters:
    -------------------
    - hf_name = 'Ez.h5' [default] specify the name of the output hdf5 file
    - out_path = cwd [default] path to the Ez.h5 file. The default is the current working directory 

    Outputs:
    --------
    - wakis.in file ready to use by the solver module

    '''

    # Save kwargs in cst.out
    if bool(kwargs):
        with open(cst_path+'cst.out', 'wb') as fp:
            pk.dump(kwargs, fp)

    # Pre-process 3d Ez field data
    preproc_Ez(cst_path=cst_path,
                      n_transverse_cells=n_transverse_cells, 
                      n_longitudinal_cells=n_longitudinal_cells, 
                      hf_name=hf_name, 
                      out_path=path
                      )

    # Charge distribution vs distance (s)
    preproc_lambda(cst_path=cst_path)

    # Read cst.out
    with open(cst_path+'cst.out') as handle:
        data = pk.loads(handle.read())

    # Check input data
    data=check_input(data)

    # Remove previous .in files
    if os.path.exists(path+'wakis.in'):
        os.remove(path+'wakis.in')

    # Generate wakis.in file
    with open(path+'wakis.in', 'wb') as fp:
        pk.dump(data, fp)

    print('[! OUT] wakis.in file succesfully generated')
    

def preproc_Ez(cst_path, hf_name='Ez.h5', out_path=cwd):
    '''
    Pre-process the CST 3D field monitor output 

    Parameters:
    -----------
    - cst_path: specify the path where the 3d Ez data folder is.
                [!] The folder containing the 3d Ez data should be named '3d'

    Default Parameters:
    -------------------
    - hf_name = 'Ez.h5' [default] specify the name of the output hdf5 file
    - out_path = cwd [default] path to the Ez.h5 file. The default is the current working directory 

    Outputs:
    --------
    - updated cst.out file
    - hdf5 file 'hf_name'.h5 with the Ez(x,y,z) matrix for every timestep
    '''

    data_path = cst_path + '3d/'

    # Rename files with E-02, E-03
    for file in glob.glob(data_path +'*E-02.txt'): 
        file=file.split(data_path)
        title=file[1].split('_')
        num=title[1].split('E')
        num[0]=float(num[0])/100

        ntitle=title[0]+'_'+str(num[0])+'.txt'
        os.rename(data_path+file[1], data_path+ntitle)

    for file in glob.glob(data_path +'*E-03.txt'): 
        file=file.split(data_path)
        title=file[1].split('_')
        num=title[1].split('E')
        num[0]=float(num[0])/1000

        ntitle=title[0]+'_'+str(num[0])+'.txt'
        os.rename(data_path+file[1], data_path+ntitle)

    for file in glob.glob(data_path +'*_0.txt'): 
        file=file.split(data_path)
        title=file[1].split('_')
        num=title[1].split('.')
        num[0]=float(num[0])

        ntitle=title[0]+'_'+str(num[0])+'.txt'
        os.rename(data_path+file[1], data_path+ntitle)

    fnames = sorted(glob.glob(data_path+'*.txt'))

    #Get the number of longitudinal and transverse cells used for Ez
    i=0
    with open(fnames[0]) as f:
        lines=f.readlines()
        n_rows = len(lines)-3 #n of rows minus the header
        x1=lines[3].split()[0]

        while True:
            i+=1
            x2=lines[i+3].split()[0]
            if x1==x2:
                break

    n_transverse_cells=i
    n_longitudinal_cells=int(n_rows/(n_transverse_cells**2))

    # Create h5 file 
    if os.path.exists(out_path+hf_name):
        os.remove(out_path+hf_name)

    hf_Ez = h5py.File(out_path+hf_name, 'w')

    # Initialize variables
    Ez=np.zeros((n_transverse_cells, n_transverse_cells, n_longitudinal_cells))
    x=np.zeros((n_transverse_cells))
    y=np.zeros((n_transverse_cells))
    z=np.zeros((n_longitudinal_cells))
    t=[]

    nsteps, i, j, k = 0, 0, 0, 0
    skip=-4 #number of rows to skip
    rows=skip 

    # Start scan
    for file in fnames:
        print('[PROGRESS] Scanning file '+ file + '...')
        title=file.split(data_path)
        title2=title[1].split('_')
        num=title2[1].split('.txt')
        t.append(float(num[0])*1e-9)

        with open(file) as f:
            for line in f:
                rows+=1
                columns = line.split()

                if rows>=0 and len(columns)>1:
                    k=int(rows/n_transverse_cells**2)
                    j=int(rows/n_transverse_cells-n_transverse_cells*k)
                    i=int(rows-j*n_transverse_cells-k*n_transverse_cells**2) 
                    
                    Ez[i,j,k]=float(columns[5])
                    x[i]=float(columns[0])
                    y[j]=float(columns[1])
                    z[k]=float(columns[2])

        if nsteps == 0:
            prefix='0'*5
            hf_Ez.create_dataset('Ez_'+prefix+str(nsteps), data=Ez)
        else:
            prefix='0'*(5-int(np.log10(nsteps)))
            hf_Ez.create_dataset('Ez_'+prefix+str(nsteps), data=Ez)

        i, j, k = 0, 0, 0          
        rows=skip
        nsteps+=1

        #close file
        f.close()

    hf_Ez.close()    


    print('[PROGRESS] Finished scanning files')
    print('[INFO] Ez field is stored in a matrix with shape '+str(Ez.shape)+' in '+str(int(nsteps))+' datasets')
    print('[! OUT] hdf5 file'+hf_name+'succesfully generated')
    
    with open(cst_path+'cst.out', 'rb') as handle:
        data = pk.loads(handle.read())

    unit=data.get('unit')

    #Save x,y,z,t in dictionary
    data['x'] = x*unit,
    data['y'] = y*unit, 
    data['z'] = z*unit,
    data['t'] = np.array(t)

    # Update cst.out with pickle
    with open(cst_path+'cst.out', 'wb') as fp:
        pk.dump(data, fp)

    print('[! OUT] cst.out file updated with field data')
    

def preproc_rho(path):
    '''
    Obtain charge distribution in [C/m] from the rho.h5 file

    '''
    with open(path+'warpx.out', 'rb') as handle:
            data = pk.loads(handle.read())

    hf_rho = h5py.File(path +'rho.h5', 'r')
    print("[PROGRESS] Processing rho.h5 file")

    #get number of datasets
    dataset_rho=[]
    for key in hf_rho.keys():
        dataset_rho.append(key)

    # Extract charge distribution [C/m] lambda(z,t)
    bunch=[]
    x=data.get('x')
    y=data.get('y')
    z=data.get('z')
    nt=data.get('nt')
    q=data.get('q')

    dx=x[2]-x[1]
    dy=y[2]-y[1]
    dz=z[2]-z[1]

    for n in range(nt):
        rho=hf_rho.get(dataset_rho[n]) # [C/m3]
        bunch.append(np.array(rho)*dx*dy) # [C/m]

    bunch=np.transpose(np.array(bunch)) # [C/m]
    nz=bunch.shape[0]
    
    # Correct the maximum value so the integral along z = q
    timestep=np.argmax(bunch[nz//2, :])   #max at cavity center
    qz=np.sum(bunch[:,timestep])*dz       #charge along the z axis
    charge_dist = bunch[:,timestep]*q/qz  #total charge in the z axis

    # Add to dict
    data['charge_dist'] = charge_dist

    # Update warpx.out with pickle
    with open(path+'warpx.out', 'wb') as fp:
        pk.dump(data, fp)

    print('[! OUT] warpx.out file updated with charge distribution data')
    
    return charge_dist

def preproc_lambda(cst_path):
    '''
    Obtain charge distribution vs z from lambda.txt file
    '''
    # Charge distribution vs distance (s)
    charge_dist=[]
    fname = 'lambda'
    i=0

    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname+'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    charge_dist.append(float(columns[1]))
                    distance.append(float(columns[0]))

        #close file
        f.close()
        charge_dist_s = np.array(charge_dist)
        s = np.array(distance)*1.0e-3
    else: 
        print("[! WARNING] file for charge distribution not found")

    with open(cst_path+'cst.out') as handle:
        data = pk.loads(handle.read())

    # Update dictionary with charge distribution vs z
    z = data.get('z')           
    charge_dist = np.interp(z, s, charge_dist_s) # in C/m
    data['charge_dist']=charge_dist
    data['s_charge_dist']=s

    # Update cst.out with pickle
    with open(cst_path+'cst.out', 'wb') as fp:
        pk.dump(data, fp)

    print('[! OUT] cst.out file updated with charge distribution data')
    
def check_input(data):
    '''
    Check if all the needed variables for the wake solver are defined.
    - Beam charge 'q': default 1e-9 [C]
    - Beam longitudinal sigma 'sigmaz': default 0.02 [m]
    - Unit conversion 'unit': default 1e-3 [m]
    - Beam position in transverse plane 'xsource', 'ysource': default 0.0 [m]
    - Integration path position in transverse plane 'xtest', 'ytest': default 0.0 [m]
    '''
    if data.get('q') is None:
        data['q']=1e-9
        print("[! WARNING] beam charge 'q' not defined, using default value 1e-9 [C]")

    if data.get('unit') is None:
        data['unit']=1e-3
        print("[! WARNING] unit conversion 'unit' not defined, using default value 1e-3 [m/mm]")

    if data.get('sigmaz') is None:
        data['sigmaz']=0.02
        print("[! WARNING] beam longitudinal sigma 'sigmaz' not defined, using default value 0.02 [m]")

    if data.get('xsource') is None:
        data['xsource']=0.0
        print("[! WARNING] beam center x position 'xsource' not defined, using default value 0.0 [m]")

    if data.get('ysource') is None:
        data['ysource']=0.0
        print("[! WARNING] beam center x position 'ysource' not defined, using default value 0.0 [m]")

    if data.get('xtest') is None:
        data['xtest']=0.0
        print("[! WARNING] integration path x position 'xtest' not defined, using default value 0.0 [m]")

    if data.get('ytest') is None:
        data['ytest']=0.0
        print("[! WARNING] integration path x position 'ytest' not defined, using default value 0.0 [m]")

    return data

#------------------------------#
#   Post-processing functions  #
#------------------------------#

def read_WAKIS_out(path=cwd):
    '''
    Read the output file 'wakis.out' generated by WAKIS 

    Parameters:
    -----------
    - out_path: path to the wakis output file. 
    '''
    if os.path.exists(path+'wakis.out'):
        with open('wakis.out', 'rb') as handle:
            data = pk.loads(handle.read())
    else: 
        print('[! WARNING] wakis.out file not found')
        data=None 

    return data 

def animate_Ez(path, filename='Ez.h5', flag_charge_dist=True, flag_transverse_field=False):
    '''
    Creates an animated plot showing the Ez field along the z axis for every timestep

    Parameters:
    -----------
    -flag_charge_dist=True [def]: plots the passing beam charge distribution 
    -flag_compare_cst=True : add the comparison with CST field in cst dict
    -flag_transverse_field=True : add the Ez field in adjacent transverse cells Ez1(0+dx, 0+dy, z), Ez2(0+2dx, 0+2dy, z)
    '''

    # Read data
    hf, dataset = read_Ez(path, filename)
    data =  read_WAKIS_in(path)

    t = data.get('t')               #simulated time [s]
    z = data.get('z')               #z axis values  [m]
    charge_dist = data.get('charge_dist')
    z0 = data.get('z0')             #full domain length (+pmls) [m]

    # Extract field on axis Ez (z,t)
    Ez0=[]
    for n in range(len(dataset)):
        Ez=hf.get(dataset[n]) # [V/m]
        Ez0.append(np.array(Ez[Ez.shape[0]//2, Ez.shape[1]//2,:])) # [V/m]

    Ez0=np.transpose(np.array(Ez0))

    if flag_transverse_field:
        Ez1=[]
        Ez2=[]
        for n in range(len(dataset)):
            Ez=hf.get(dataset[n]) # [V/m]
            Ez1.append(np.array(Ez[Ez.shape[0]//2+1, Ez.shape[1]//2+1,:])) # 1st adjacent cell Ez [V/m]
            Ez2.append(np.array(Ez[Ez.shape[0]//2+2, Ez.shape[1]//2+2,:])) # 2nd adjacent cell Ez [V/m]

        Ez1=np.transpose(np.array(Ez1))
        Ez2=np.transpose(np.array(Ez2))

    plt.ion()
    n=0
    for n in range(10,1000):
        if n % 1 == 0:
            #--- Plot Ez along z axis 
            fig = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
            ax=fig.gca()
            ax.plot(np.array(z0)/UNIT, charge_dist[:,n]/np.max(charge_dist)*np.max(Ez0)*0.4, lw=1.3, color='r', label='$\lambda $') 
            ax.plot(z/UNIT, Ez0[:, n], color='g', label='Ez(0,0,z) WarpX')

            if flag_transverse_field:
                ax.plot(z/UNIT, Ez1[:, n], color='seagreen', label='Ez(0+dx, 0+dy, z) WarpX')
                ax.plot(z/UNIT, Ez2[:, n], color='limegreen', label='Ez(0+2dx, 0+2dy, z) WarpX')

            ax.set(title='Electric field at time = '+str(round(t[n]*1e9,2))+' ns | timestep '+str(n),
                    xlabel='z [mm]',
                    ylabel='E [V/m]',         
                    ylim=(-np.max(Ez0)*1.1,np.max(Ez0)*1.1),
                    xlim=(min(z)/UNIT,max(z)/UNIT),
                            )
            ax.legend(loc='best')
            ax.grid(True, color='gray', linewidth=0.2)
            fig.canvas.draw()
            fig.canvas.flush_events()
            fig.clf()
    plt.close()

def contour_Ez(out_path, filename='Ez.h5', vmin=-1.0e5, vmax=1.0e5):
    '''
    Creates an animated contour of the Ez field in the Y-Z plane at x=0

    Parameters:
    -----------
    -vmin=-1.0e5 [def]: minimum value of the colorbar
    -vmax=+1.0e5 [def]: maximum value of the colorbar
    '''

    # Read data
    hf, dataset = read_Ez(out_path, filename)
    data =  read_WarpX(out_path)

    t = data.get('t')       #simulated time [s]
    z = data.get('z')       #z axis masked values [m]
    y = data.get('y0')      #y axis domain values [m]

    # Check for 3d field
    Ez=hf.get(dataset[0])
    if Ez.shape[1] < 8:
        raise Exception("[! WARNING] Ez field not stored in 3D, contour will not render")

    else:
        plt.ion()
        for n in range(len(dataset)):
            Ez=hf.get(dataset[n])
            fig = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
            ax=fig.gca()
            im=ax.imshow(Ez[int(Ez.shape[0]/2),:,:], vmin=vmin, vmax=vmax, extent=[min(z)/UNIT, max(z)/UNIT, min(y)/UNIT, max(y)/UNIT], cmap='jet')
            ax.set(title='WarpX Ez field, t = ' + str(round(t[n]*1e9,3)) + ' ns',
                   xlabel='z [mm]',
                   ylabel='y [mm]'
                   )
            plt.colorbar(im, label = 'Ez [V/m]')
            fig1.canvas.draw()
            fig1.canvas.flush_events()
            fig1.clf() 

        plt.close()


def plot_charge_dist(data):
    '''
    Plots the charge distribution λ(s) 

    Parameters:
    -----------
    - data = read_WAKIS_out(out_path)
    '''

    # Obtain WAKIS variables
    s = data.get('s')
    q = data.get('q')
    lambdas = data.get('lambda') #[C/m]

    # Plot charge distribution λ(s) & comparison with CST 
    fig = plt.figure(1, figsize=(8,5), dpi=150, tight_layout=True)
    ax=fig.gca()
    ax.plot(s/UNIT, lambdas, lw=1.2, color='red', label='$\lambda$(s)')
    ax.set(title='Charge distribution $\lambda$(s)',
            xlabel='s [mm]',
            ylabel='$\lambda$(s) [C/m]',
            xlim=(min(s/UNIT), np.max(s/UNIT))
            )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    return fig

def plot_long_WP(data):
    '''
    Plots the longitudinal wake potential W||(s) 

    Parameters:
    -----------
    - data = read_WAKIS_out(path)
    '''

    # Obtain WAKIS variables
    WP=data.get('WP')
    s=data.get('s')

    # Plot longitudinal wake potential W||(s) & comparison with CST 
    fig = plt.figure(1, figsize=(8,5), dpi=150, tight_layout=True)
    ax=fig.gca()
    ax.plot(s/UNIT, WP, lw=1.2, color='orange', label='$W_{||}$(s)')
    ax.set(title='Longitudinal Wake potential $W_{||}$(s)',
            xlabel='s [mm]',
            ylabel='$W_{||}$(s) [V/pC]',
            xlim=(min(s/UNIT),np.max(s/UNIT)),
            ylim=(min(WP)*1.2, max(WP)*1.2)
            )
    ax.legend(loc='lower right')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    return fig

def plot_long_Z(data, 
                flag_plot_Real=False, flag_plot_Imag=False, flag_plot_Abs=True):
    '''
    Plots the longitudinal impedance Z||(w)

    Parameters:
    -----------
    - data = read_WAKIS_out(path)
    - flag_plot_Real = False [default]
    - flag_plot_Imag = False [default]
    - flag_plot_Abs = True [default]
    '''

    # Obtain wakis variables
    Z=data.get('Z')
    f=data.get('f')

    if np.iscomplex(Z[1]):
        ReZ=np.real(Z)
        ImZ=np.imag(Z)
        Z=abs(Z)

    # Plot longitudinal impedance Z||(w) comparison with CST 
    fig = plt.figure(2, figsize=(8,5), dpi=150, tight_layout=True)
    ax=fig.gca()

    if flag_plot_Real:
        ax.plot(f*1e-9, ReZ, lw=1, color='r', marker='v', markersize=2., label='Real Z||(w)')

    if flag_plot_Imag:
        ax.plot(f*1e-9, ImZ, lw=1, color='g', marker='s', markersize=2., label='Imag Z||(w)')

    if flag_plot_Abs:
        # obtain the maximum frequency and plot Z||(s)
        ifmax=np.argmax(Z)
        ax.plot(f[ifmax]*1e-9, Z[ifmax], marker='o', markersize=4.0, color='blue')
        ax.annotate(str(round(f[ifmax]*1e-9,2))+ ' GHz', xy=(f[ifmax]*1e-9,Z[ifmax]), xytext=(-20,5), textcoords='offset points', color='blue') 
        ax.plot(f*1e-9, Z, lw=1, color='b', marker='s', markersize=2., label='Z||(w) magnitude')
    
    ax.set( title='Longitudinal impedance Z||(w)',
            xlabel='f [GHz]',
            ylabel='Z||(w) [$\Omega$]',   
            xlim=(0.,np.max(f)*1e-9)      
            )
    ax.legend(loc='upper left')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    return fig


def plot_trans_WP(data):
    '''
    Plots the transverse wake potential Wx⊥(s), Wy⊥(s) 

    Parameters:
    -----------
    - data = read_WAKIS_out(path)
    '''
    # Obtain wakis variables
    WPx=data.get('WPx')
    WPy=data.get('WPy')
    s=data.get('s')
    # Obtain the offset of the source beam and test beam
    xsource=data.get('xsource')
    ysource=data.get('ysource')
    xtest=data.get('xtest')
    ytest=data.get('ytest') 

    # Plot transverse wake potential Wx⊥(s), Wy⊥(s) & comparison with CST
    fig = plt.figure(3, figsize=(8,5), dpi=150, tight_layout=True)
    ax=fig.gca()
    ax.plot(s/UNIT, WPx, lw=1.2, color='g', label='Wx⊥(s)')
    ax.plot(s/UNIT, WPy, lw=1.2, color='magenta', label='Wy⊥(s)')
    ax.set(title='Transverse Wake potential W⊥(s) \n (x,y) source = ('+str(round(xsource/UNIT,1))+','+str(round(ysource/UNIT,1))+') mm | test = ('+str(round(xtest/UNIT,1))+','+str(round(ytest/UNIT,1))+') mm',
            xlabel='s [mm]',
            ylabel='$W_{⊥}$ [V/pC]',
            xlim=(np.min(s/UNIT), np.max(s/UNIT)),
            )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    return fig

def plot_trans_Z(data, 
                 flag_plot_Real=False, flag_plot_Imag=False, flag_plot_Abs=True):
    '''
    Plots the transverse impedance Zx⊥(w), Zy⊥(w) 

    Parameters:
    -----------
    - data = read_WAKIS_out(path)
    - flag_plot_Real = False [default]
    - flag_plot_Imag = False [default]
    - flag_plot_Abs = True [default]
    '''

    # Obtain wakis variables
    Zx=data.get('Zx')
    Zy=data.get('Zy')
    f=data.get('f')

    if np.iscomplex(Zx[1]):
        ReZx=np.real(Zx)
        ImZx=np.imag(Zx)
        Zx=abs(Zx)

    if np.iscomplex(Zy[1]):
        ReZy=np.real(Zy)
        ImZy=np.imag(Zy)
        Zy=abs(Zy)

    # Obtain the offset of the source beam and test beam
    xsource=data.get('xsource')
    ysource=data.get('ysource')
    xtest=data.get('xtest')
    ytest=data.get('ytest') 

    # Plot the transverse impedance Zx⊥(w), Zy⊥(w) 
    fig = plt.figure(4, figsize=(8,5), dpi=150, tight_layout=True)
    ax=fig.gca()

    #--- plot Zx⊥(w)
    if flag_plot_Real:
        ax.plot(f*1e-9, ReZx, lw=1, color='b', marker='v', markersize=2., label='Real Zx⊥(w)')

    if flag_plot_Imag:
        ax.plot(f*1e-9, ImZx, lw=1, color='c', marker='s', markersize=2., label='Imag Zx⊥(w)')

    if flag_plot_Abs:
        # obtain the maximum frequency and plot
        ifmax=np.argmax(Zx)
        ax.plot(f[ifmax]*1e-9, Zx[ifmax], marker='o', markersize=4.0, color='green')
        ax.annotate(str(round(f[ifxmax]*1e-9,2))+ ' GHz', xy=(f[ifxmax]*1e-9,Zx[ifxmax]), xytext=(-50,-5), textcoords='offset points', color='green') 
        ax.plot(f*1e-9, Zx, lw=1, color='g', marker='s', markersize=2., label='Zx⊥(w)')

    #--- plot Zy⊥(w)
    if flag_plot_Real:
        ax.plot(f*1e-9, ReZy, lw=1, color='r', marker='v', markersize=2., label='Real Zy⊥(w)')

    if flag_plot_Imag:
        ax.plot(f*1e-9, ImZy, lw=1, color='m', marker='s', markersize=2., label='Imag Zy⊥(w)')

    if flag_plot_Abs:
        # obtain the maximum frequency and plot
        ifymax=np.argmax(Zy)
        ax.plot(f[ifymax]*1e-9, Zy[ifymax], marker='o', markersize=4.0, color='magenta')
        ax.annotate(str(round(f[ifymax]*1e-9,2))+ ' GHz', xy=(f[ifymax]*1e-9, Zy[ifymax]), xytext=(-50,-5), textcoords='offset points', color='magenta') 
        ax.plot(f*1e-9, Zy, lw=1, color='magenta', marker='s', markersize=2., label='Zy⊥(w)')

    ax.set(title='Transverse impedance Z⊥(w) \n (x,y) source = ('+str(round(xsource/UNIT,1))+','+str(round(ysource/UNIT,1))+') mm | test = ('+str(round(xtest/UNIT,1))+','+str(round(ytest/UNIT,1))+') mm',
            xlabel='f [GHz]',
            ylabel='Z⊥(w) [$\Omega$]',   
            xlim=(0.,np.max(f)*1e-9)      
            )

    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    return fig

def plot_WAKIS(data, flag_charge_dist=False,
               flag_plot_Real=False, flag_plot_Imag=False, flag_plot_Abs=True):
    '''
    Plot results of WAKIS wake solver in different figures

    Parameters
    ---------- 
    - data: [default] data=read_WAKIS_out(path). Dictionary containing the wake solver output
    - flag_charge_dist: [default] False. Plots the charge distribution as a function of s 
    - flag_plot_Real = False [default]
    - flag_plot_Imag = False [default]
    - flag_plot_Abs = True [default]

    Returns
    -------
    - fig 1-4: if flag_charge_dist=False 
    or
    - fig 1-5: if flag_charge_dist=True
    
    fig1 = plot_long_WP
    fig2 = plot_long_Z
    fig3 = plot_trans_WP
    fig4 = plot_trans_Z
    fig5 = plot_charge_dist

    '''
    fig1 = plot_long_WP(data=data)
    fig2 = plot_long_Z(data=data, flag_plot_Real=flag_plot_Real, flag_plot_Imag=flag_plot_Imag, flag_plot_Abs=flag_plot_Abs)
    fig3 = plot_trans_WP(data=data)
    fig4 = plot_trans_Z(data=data, flag_plot_Real=flag_plot_Real, flag_plot_Imag=flag_plot_Imag, flag_plot_Abs=flag_plot_Abs)

    if flag_charge_dist:
        fig5 = plot_charge_dist(data=data, cst_data=cst_data, flag_compare_cst=flag_compare_cst)
        return fig1, fig2, fig3, fig4, fig5
    else: 
        return fig1, fig2, fig3, fig4 

def subplot_WAKIS(data, flag_charge_dist=True, flag_plot_Real=False, flag_plot_Imag=False, flag_plot_Abs=True):
    '''
    Plot results of WAKIS wake solver in the same figure

    Parameters
    ---------- 
    - data = [default] read_WAKIS_out(path). Dictionary containing the wake solver output
    - flag_charge_dist = [default] True : Plots (normalized) charge distribution on top of the wake potential
    - flag_plot_Real = [default] False : Adds the real part of the impedance Z to the plot
    - flag_plot_Imag = [default] False : Adds the imaginary part of the impedance Z to the plot
    - flag_plot_Abs  = [default] True : Adds the magnitude of the impedance Z to the plot

    Returns
    -------
    - fig: figure object

    '''  
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.set_size_inches(16, 10)

    # Add title
    xsource=data.get('xsource')
    ysource=data.get('ysource')
    xtest=data.get('xtest')
    ytest=data.get('ytest') 

    plt.text(x=0.5, y=0.96, s="WAKIS wake solver result", fontsize='x-large', fontweight='bold', ha="center", transform=fig.transFigure)
    plt.text(x=0.5, y=0.93, s= '(x,y) source = ('+str(round(xsource/UNIT,1))+','+str(round(ysource/UNIT,1))+') mm | test = ('+str(round(xtest/UNIT,1))+','+str(round(ytest/UNIT,1))+') mm', fontsize='large', ha="center", transform=fig.transFigure)

    # Longitudinal WP
    WP=data.get('WP')
    s=data.get('s')

    ax1.plot(s/UNIT, WP, lw=1.2, color='orange', label='$W_{||}$(s)')

    if flag_charge_dist:
        lambdas = data.get('lambda')
        ax1.plot(s/UNIT, lambdas*max(WP)/max(lambdas), lw=1, color='red', label='$\lambda$(s) [norm]')

    ax1.set(title='Longitudinal Wake potential $W_{||}$(s)',
            xlabel='s [mm]',
            ylabel='$W_{||}$(s) [V/pC]',
            )
    ax1.legend(loc='best')
    ax1.grid(True, color='gray', linewidth=0.2)

    # Longitudinal Z
    Z=data.get('Z')
    f=data.get('f')

    if np.iscomplex(Z[1]):
        ReZ=np.real(Z)
        ImZ=np.imag(Z)
        Z=abs(Z)

    if flag_plot_Real:
        ax2.plot(f*1e-9, ReZ, lw=1, ls='--', color='r', marker='s', markersize=2., label='Real Z||(w)')

    if flag_plot_Imag:
        ax2.plot(f*1e-9, ImZ, lw=1, ls='--', color='g', marker='s', markersize=2., label='Imag Z||(w)')

    if flag_plot_Abs:
        ifmax=np.argmax(Z)
        ax2.plot(f[ifmax]*1e-9, Z[ifmax], marker='o', markersize=4.0, color='blue')
        ax2.annotate(str(round(f[ifmax]*1e-9,2))+ ' GHz', xy=(f[ifmax]*1e-9,Z[ifmax]), xytext=(-20,5), textcoords='offset points', color='blue') 
        ax2.plot(f*1e-9, Z, lw=1, color='b', marker='s', markersize=2., label='Z||(w)')

    ax2.set(title='Longitudinal impedance Z||(w)',
            xlabel='f [GHz]',
            ylabel='Z||(w) [$\Omega$]',   
            #ylim=(0.,np.max(Z)*1.2),
            xlim=(0.,np.max(f)*1e-9)      
            )
    ax2.legend(loc='best')
    ax2.grid(True, color='gray', linewidth=0.2)

    # Transverse WP    
    WPx=data.get('WPx')
    WPy=data.get('WPy')

    ax3.plot(s/UNIT, WPx, lw=1.2, color='g', label='Wx⊥(s)')
    ax3.plot(s/UNIT, WPy, lw=1.2, color='magenta', label='Wy⊥(s)')
    ax3.set(title='Transverse Wake potential W⊥(s)',
            xlabel='s [mm]',
            ylabel='$W_{⊥}$ [V/pC]',
            )
    ax3.legend(loc='best')
    ax3.grid(True, color='gray', linewidth=0.2)

    # Transverse Z
    Zx=data.get('Zx')
    Zy=data.get('Zy')

    if np.iscomplex(Zx[1]):
        ReZx=np.real(Zx)
        ImZx=np.imag(Zx)
        Zx=abs(Zx)

    if np.iscomplex(Zy[1]):
        ReZy=np.real(Zy)
        ImZy=np.imag(Zy)
        Zy=abs(Zy)

    #--- plot Zx⊥(w)
    if flag_plot_Real:
        ax4.plot(f*1e-9, ReZx, lw=1, color='b', marker='v', markersize=2., label='Real Zx⊥(w)')

    if flag_plot_Imag:
        ax4.plot(f*1e-9, ImZx, lw=1, color='c', marker='s', markersize=2., label='Imag Zx⊥(w)')

    if flag_plot_Abs:
        # obtain the maximum frequency and plot
        ifxmax=np.argmax(Zx)
        ax4.plot(f[ifxmax]*1e-9, Zx[ifxmax], marker='o', markersize=4.0, color='green')
        ax4.annotate(str(round(f[ifxmax]*1e-9,2))+ ' GHz', xy=(f[ifxmax]*1e-9,Zx[ifxmax]), xytext=(-50,-5), textcoords='offset points', color='green') 
        ax4.plot(f*1e-9, Zx, lw=1, color='g', marker='s', markersize=2., label='Zx⊥(w)')

    #--- plot Zy⊥(w)
    if flag_plot_Real:
        ax4.plot(f*1e-9, ReZy, lw=1, color='r', marker='v', markersize=2., label='Real Zy⊥(w)')

    if flag_plot_Imag:
        ax4.plot(f*1e-9, ImZy, lw=1, color='m', marker='s', markersize=2., label='Imag Zy⊥(w)')

    if flag_plot_Abs:
        # obtain the maximum frequency and plot
        ifymax=np.argmax(Zy)
        ax4.plot(f[ifymax]*1e-9, Zy[ifymax], marker='o', markersize=4.0, color='magenta')
        ax4.annotate(str(round(f[ifymax]*1e-9,2))+ ' GHz', xy=(f[ifymax]*1e-9, Zy[ifymax]), xytext=(-50,-5), textcoords='offset points', color='magenta') 
        ax4.plot(f*1e-9, Zy, lw=1, color='magenta', marker='s', markersize=2., label='Zy⊥(w)')


    ax4.set(title='Transverse impedance Z⊥(w)',
            xlabel='f [GHz]',
            ylabel='Z⊥(w) [$\Omega$]',   
            #ylim=(0.,np.maximum(max(Zx)*1.2, max(Zy)*1.2)),
            xlim=(0.,np.max(f)*1e-9)      
            )
    ax4.legend(loc='best')
    ax4.grid(True, color='gray', linewidth=0.2)

    plt.show()

    return fig

if __name__ == "__main__":
    
    out_path=os.getcwd()+'/'+'runs/out/'
    flag_animate_Ez=True
    flag_individual_figs=False

    if flag_animate_Ez:

        #Plot Ez field animation
        animate_Ez(out_path, 
                   flag_charge_dist=True, 
                   flag_compare_cst=False, 
                   flag_transverse_field=False)
    try:
        #Plot Ez contour in the YZ plane
        contour_Ez(out_path, vmin=-1.0e5, vmax=1.0e5)
    except: print("Ez field not stored in 3D, contour will not render -> check simulation script")

    
    if os.path.exists(out_path+'wake_solver.txt'):

        # Read WAKIS results
        data=read_WAKIS_out(out_path)
    
        # Plot results
        fig = subplot_WAKIS(data=data, 
                            flag_charge_dist=True,
                            flag_plot_Real=True, 
                            flag_plot_Imag=True,
                            flag_plot_Abs=False
                            )

        # Save figure
        fig.savefig(out_path+'subplot_WAKIS.png',  bbox_inches='tight')
    

    if flag_individual_figs:

        # Plot in individual figures
        figs = plot_WAKIS(data=data, 
                    cst_data=read_CST(cst_path), 
                    flag_compare_cst=True, 
                    flag_normalize=False,
                    flag_charge_dist=True,
                    flag_plot_Real=True, 
                    flag_plot_Imag=True,
                    flag_plot_Abs=False
                    )

        figs[0].savefig(out_path+'longWP.png', bbox_inches='tight')
        figs[1].savefig(out_path+'longZ.png',  bbox_inches='tight')
        figs[2].savefig(out_path+'transWP.png',  bbox_inches='tight')
        figs[3].savefig(out_path+'transZ.png',  bbox_inches='tight')

        if len(figs) > 4:
            figs[4].savefig(out_path+'charge_dist')
        
