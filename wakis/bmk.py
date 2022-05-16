'''
-------------------
| WAKIS benchmark |
-------------------
Submodule to benchmark Wakis output with CST Wake solver[1]

[1] https://space.mit.edu/RADIO/CST_online/mergedProjects/3D/special_overview/special_beams_wakefield_solver_overview.htm

Functions: [TODO]
----------
- a
- b
- c
'''

import glob, os 
import pickle as pk

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5py

from .proc import read_WAKIS_out, read_Ez
from .solver import run_WAKIS

# Global parameters for plotting
cwd = os.getcwd() + '/'
plt.rcParams.update({'font.size': 12})
UNIT = 1e-3 

def run_benchmark(cst_path, wakis_path=cwd):
    '''
    Runs the benchmark vs CST

    Inputs:
    -------
    -cst.bmk file. If it does not exist in path, it is generated
    -wakis.out file. If it does not exist in path, WAKIS is run
    '''
    if not os.exists(cst_path+'cst.bmk'):
        generate_bmk(cst_path)

    if not os.exists(wakis_path+'wakis.out'):
        run_WAKIS(wakis_path)

    # Plot in individual figures
    figs = bmk_WAKIS(data=read_WAKIS_out(wakis_path), 
                bmk=read_bmk(cst_path), 
                flag_compare_cst=True, 
                flag_charge_dist=False,
                flag_plot_Real=True, 
                flag_plot_Imag=True,
                flag_plot_Abs=False
                )

    figs[0].savefig(out_path+'longWP_bmk.png', bbox_inches='tight')
    figs[1].savefig(out_path+'longZ_bmk.png',  bbox_inches='tight')
    figs[2].savefig(out_path+'transWP_bmk.png',  bbox_inches='tight')
    figs[3].savefig(out_path+'transZ_bmk.png',  bbox_inches='tight')

    if len(figs) > 4:
        figs[4].savefig(out_path+'charge_dist_bmk.png', bbox_inches='tight')


def read_bmk(cst_path):
    '''
    Read the benchmark data from file 'cst.bmk'

    Parameters:
    -----------
    - cst_path: path to the 'cst.bmk' file 
    '''

    if os.path.exists(cst_path+'cst.bmk'):
        with open(cst_path+'cst.bmk') as handle:
            bmk = js.load(handle)
    else: 
        try:
            generate_bmk(cst_path)
            with open(path+'cst.bmk') as handle:
                bmk = js.load(handle)
        except:
            bmk = None 
            print('[! WARNING] CST benchmark data not found')

    return bmk

def bmk_Ez(out_path, cst_path, filename='Ez.h5', flag_charge_dist=True, flag_transverse_field=False):
    '''
    Creates an animated plot showing the Ez field along the z axis for every timestep

    Parameters:
    -----------
    -flag_charge_dist=True [def]: plots the passing beam charge distribution 
    -flag_compare_cst=True : add the comparison with CST field in cst dict
    -flag_transverse_field=False [def] : add the Ez field in adjacent transverse cells Ez1(0+dx, 0+dy, z), Ez2(0+2dx, 0+2dy, z)
    '''

    # Read data
    hf, dataset = read_Ez(out_path, filename)
    data =  read_WarpX(out_path)

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

            try:
                bmk=read_bmk(cst_path)
                z_cst=bmk.get('z_cst')
                Ez_cst=bmk.get('Ez_cst')
                ax.plot(z_cst/UNIT, Ez_cst[:, n], lw=1.0, color='black', ls='--',label='Ez(0,0,z) CST')
            except: 
                print('[! WARNING] CST Ez data is not well stored and will not be plotted')

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


def bmk_charge_dist(data, bmk):
    '''
    Plots the charge distribution λ(s) and compares it with CST wake solver results

    Parameters:
    -----------
    - data = read_WAKIS_in(out_path)
    - bmk = read_bmk(cst_path)
    '''

    # Obtain WAKIS variables
    s = data.get('s')
    q = data.get('q')
    lambdas = data.get('lambda') #[C/m]

    # Obtain CST variables
    lambda_cst=bmk.get('charge_dist') #[C/m]
    s_cst=bmk.get('s_charge_dist')

    # Plot charge distribution λ(s) & comparison with CST 
    fig = plt.figure(1, figsize=(8,5), dpi=150, tight_layout=True)
    ax=fig.gca()
    ax.plot(s/UNIT, lambdas, lw=1.2, color='red', label='$\lambda$(s)')
    ax.plot(s_cst/UNIT, lambda_cst, lw=1, color='red', ls='--', label='$\lambda$(s) CST')
    ax.set(title='Charge distribution $\lambda$(s)',
            xlabel='s [mm]',
            ylabel='$\lambda$(s) [C/m]',
            xlim=(min(s/UNIT), np.max(s/UNIT))
            )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    return fig

def bmk_long_WP(data, bmk):
    '''
    Plots the longitudinal wake potential W||(s) and compares it with CST wake solver results

    Parameters:
    -----------
    - data = read_WAKIS(out_path)
    - bmk=read_bmk(cst_path)
    '''

    # Obtain WAKIS variables
    WP=data.get('WP')
    s=data.get('s')

    # Obtain CST variables
    WP_cst=bmk.get('WP_cst')
    s_cst=bmk.get('s_cst')

    # Plot longitudinal wake potential W||(s) & comparison with CST 
    fig = plt.figure(1, figsize=(8,5), dpi=150, tight_layout=True)
    ax=fig.gca()
    ax.plot(s/UNIT, WP, lw=1.2, color='orange', label='$W_{||}$(s) from WAKIS')
    ax.plot(s_cst/UNIT, WP_cst, lw=1.2, color='black', ls='--', label='$W_{||}$(s) CST')
    ax.set(title='Longitudinal Wake potential $W_{||}$(s)',
            xlabel='s [mm]',
            ylabel='$W_{||}$(s) [V/pC]',
            xlim=(min(s/UNIT), np.amin((np.max(s/UNIT), np.max(s_cst/UNIT)))),
            ylim=(min(WP)*1.2, max(WP)*1.2)
            )
    ax.legend(loc='lower right')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    return fig

def bmk_long_Z(data, bmk, 
                flag_normalize=False, flag_plot_Real=False, flag_plot_Imag=False, flag_plot_Abs=True):
    '''
    Plots the longitudinal impedance Z||(w) and compares it with CST wake solver results

    Parameters:
    -----------
    - data = read_WAKIS(out_path)
    - bmk=read_bmk(cst_path)
    - flag_normalize = False [default]
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

    # Obtain CST variables
    Z_cst=bmk.get('Z_cst')
    freq_cst=bmk.get('freq_cst')
    ReZ_cst=bmk.get('ReZ')
    ImZ_cst=bmk.get('ImZ')

    # Plot longitudinal impedance Z||(w) comparison with CST 
    
    if flag_normalize:
        #---normalizing factor between CST and Wakis
        norm=max(Z_cst)/max(Z) 
        title='Longitudinal impedance Z||(w) \n [normalized by '+str(round(norm,3))+']'
    else:
        norm=1.0
        title='Longitudinal impedance Z||(w)'

    fig = plt.figure(2, figsize=(8,5), dpi=150, tight_layout=True)
    ax=fig.gca()

    if flag_plot_Real:
        try:
            ax.plot(freq_cst*1e-9, ReZ_cst, lw=1, ls='--', color='r', label='Real Z||(w) from CST')
        except: print('Real Z is not stores in cst dict')
        ax.plot(f*1e-9, ReZ, lw=1, color='r', marker='v', markersize=2., label='Real Z||(w) from WAKIS')

    if flag_plot_Imag:
        try:
            ax.plot(freq_cst*1e-9, ImZ_cst, lw=1, ls='--', color='g', label='Imag Z||(w) from CST')
        except: print('Imag Z is not stored in cst dict')
        ax.plot(f*1e-9, ImZ, lw=1, color='g', marker='s', markersize=2., label='Imag Z||(w) from WAKIS')

    if flag_plot_Abs:
        #--- obtain the maximum frequency for CST and plot
        ifmax=np.argmax(Z_cst)
        ax.plot(freq_cst[ifmax]*1e-9, Z_cst[ifmax], marker='o', markersize=5.0, color='red')
        ax.annotate(str(round(freq_cst[ifmax]*1e-9,2))+ ' GHz', xy=(freq_cst[ifmax]*1e-9,Z_cst[ifmax]), xytext=(+10,0), textcoords='offset points', color='red') 
        ax.plot(freq_cst*1.0e-9, Z_cst, lw=1, color='red', marker='s', markersize=1., label='Z||(w) from CST')

        #--- obtain the maximum frequency and plot Z||(s)
        ifmax=np.argmax(Z)
        ax.plot(f[ifmax]*1e-9, Z[ifmax], marker='o', markersize=4.0, color='blue')
        ax.annotate(str(round(f[ifmax]*1e-9,2))+ ' GHz', xy=(f[ifmax]*1e-9,Z[ifmax]), xytext=(-20,5), textcoords='offset points', color='blue') 
        ax.plot(f*1e-9, Z, lw=1, color='b', marker='s', markersize=2., label='Z||(w) from WAKIS')
    
    ax.set( title=title,
            xlabel='f [GHz]',
            ylabel='Z||(w) [$\Omega$]',   
            xlim=(0.,np.max(f)*1e-9)      
            )
    ax.legend(loc='upper left')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    return fig


def bmk_trans_WP(data, bmk):
    '''
    Plots the transverse wake potential Wx⊥(s), Wy⊥(s) and compares it with CST wake solver results

    Parameters:
    -----------
    - data = read_WAKIS(out_path)
    - bmk = read_bmk(cst_path)
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

    # Obtain CST variables
    WPx_cst=bmk.get('WPx_cst')
    WPy_cst=bmk.get('WPy_cst')
    s_cst=bmk.get('s_cst')

    #-- Quadrupolar cases
    if xtest != 0.0 and ytest == 0.0:
        WPx_cst=bmk.get('WPx_quadrupolarX_cst')
        WPy_cst=bmk.get('WPy_quadrupolarX_cst')
        s_cst=bmk.get('s_cst_quadrupolar')
    if xtest == 0.0 and ytest != 0.0:
        WPx_cst=bmk.get('WPx_quadrupolarY_cst')
        WPy_cst=bmk.get('WPy_quadrupolarY_cst')
        s_cst=bmk.get('s_cst_quadrupolar')
    if xtest != 0.0 and ytest != 0.0:
        WPx_cst=bmk.get('WPx_quadrupolar_cst')
        WPy_cst=bmk.get('WPy_quadrupolar_cst')
        s_cst=bmk.get('s_cst_quadrupolar')

    #-- Dipolar cases
    if xsource != 0.0 and ysource == 0.0:
        WPx_cst=bmk.get('WPx_dipolarX_cst')
        WPy_cst=bmk.get('WPy_dipolarX_cst')
        s_cst=bmk.get('s_cst_dipolar')
    if xsource == 0.0 and ysource != 0.0:
        WPx_cst=bmk.get('WPx_dipolarY_cst')
        WPy_cst=bmk.get('WPy_dipolarY_cst')
        s_cst=bmk.get('s_cst_dipolar')
    if xsource != 0.0 and ysource != 0.0:
        WPx_cst=bmk.get('WPx_dipolar_cst')
        WPy_cst=bmk.get('WPy_dipolar_cst')
        s_cst=bmk.get('s_cst_dipolar')

    # Plot transverse wake potential Wx⊥(s), Wy⊥(s) & comparison with CST
    fig = plt.figure(3, figsize=(8,5), dpi=150, tight_layout=True)
    ax=fig.gca()
    # - Wx⊥(s)
    ax.plot(s_cst/UNIT, WPx_cst, lw=1, color='g', ls='--', label='Wx⊥(s) from CST')
    ax.plot(s/UNIT, WPx, lw=1.2, color='g', label='Wx⊥(s) from WAKIS')
    # - Wy⊥(s)
    ax.plot(s_cst/UNIT, WPy_cst, lw=1, color='magenta', ls='--', label='Wy⊥(s) from CST')
    ax.plot(s/UNIT, WPy, lw=1.2, color='magenta', label='Wy⊥ from WAKIS(s)')

    ax.set(title='Transverse Wake potential W⊥(s) \n xsource, ysource = '+str(xsource/UNIT)+' mm | xtest, ytest = '+str(xtest/UNIT)+' mm',
            xlabel='s [mm]',
            ylabel='$W_{⊥}$ [V/pC]',
            xlim=(min(s/UNIT), np.max(s/UNIT)),
            )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    return fig

def bmk_trans_Z(data, bmk, flag_normalize=False):
    '''
    Plots the transverse Impedance Zx⊥(w), Zy⊥(w) and compares it with CST wake solver results

    Parameters:
    -----------
    - data = read_WAKIS(out_path)
    - bmk=read_bmk(cst_path)
    - flag_normalize = False [default]
    '''

    # Obtain wakis variables
    Zx=data.get('Zx')
    Zy=data.get('Zy')
    freqx=data.get('f')
    freqy=data.get('f')

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

    # Obtain CST variables
    Zx_cst=bmk.get('Zx_cst')
    Zy_cst=bmk.get('Zy_cst')
    freq_cst=bmk.get('freq_cst')

    #-- Quadrupolar cases
    if xtest != 0.0 and ytest == 0.0:
        Zx_cst=bmk.get('Zx_quadrupolarX_cst')
        Zy_cst=bmk.get('Zy_quadrupolarX_cst')
        freq_cst=bmk.get('freq_cst_quadrupolar')
    if xtest == 0.0 and ytest != 0.0:
        Zx_cst=bmk.get('Zx_quadrupolarY_cst')
        Zy_cst=bmk.get('Zy_quadrupolarY_cst')
        freq_cst=bmk.get('freq_cst_quadrupolar')
    if xtest != 0.0 and ytest != 0.0:
        Zx_cst=bmk.get('Zx_quadrupolar_cst')
        Zy_cst=bmk.get('Zy_quadrupolar_cst')
        freq_cst=bmk.get('freq_cst_quadrupolar')

    #-- Dipolar cases
    if xsource != 0.0 and ysource == 0.0:
        Zx_cst=bmk.get('Zx_dipolarX_cst')
        Zy_cst=bmk.get('Zy_dipolarX_cst')
        freq_cst=bmk.get('freq_cst_dipolar')
    if xsource == 0.0 and ysource != 0.0:
        Zx_cst=bmk.get('Zx_dipolarY_cst')
        Zy_cst=bmk.get('Zy_dipolarY_cst')
        freq_cst=bmk.get('freq_cst_dipolar')
    if xsource != 0.0 and ysource != 0.0:
        Zx_cst=bmk.get('Zx_dipolar_cst')
        Zy_cst=bmk.get('Zy_dipolar_cst')
        freq_cst=bmk.get('freq_cst_dipolar')

    #--- normalizing factor between CST and in wakis results
    if flag_normalize:
        norm_x=max(Zx_cst)/max(Zx) 
        norm_y=max(Zy_cst)/max(Zy) 
        title='Transverse impedance Z⊥(w) [normalized by '+str(round(norm_x,3))+']'
    else:
        norm_x=1.0
        norm_y=1.0
        title='Transverse impedance Z⊥(w)'

    #--- obtain the maximum frequency
    ifxmax=np.argmax(Zx)
    ifymax=np.argmax(Zy)

    fig = plt.figure(4, figsize=(8,5), dpi=150, tight_layout=True)
    ax=fig.gca()

    #--- plot Zx⊥(w)
    ifmax=np.argmax(Zx_cst)
    ax.plot(freq_cst[ifmax]*1e-9, Zx_cst[ifmax], marker='o', markersize=5.0, color='black')
    ax.annotate(str(round(freq_cst[ifmax]*1e-9,2))+ ' GHz', xy=(freq_cst[ifmax]*1e-9,Zx_cst[ifmax]), xytext=(+5,-5), textcoords='offset points', color='black') 
    ax.plot(freq_cst*1.0e-9, Zx_cst, lw=1, ls='--', color='black', label='Zx⊥(w) from CST')

    ax.plot(freqx[ifxmax]*1e-9, Zx[ifxmax]*norm_x, marker='o', markersize=4.0, color='green')
    ax.annotate(str(round(freqx[ifxmax]*1e-9,2))+ ' GHz', xy=(freqx[ifxmax]*1e-9,Zx[ifxmax]), xytext=(-50,-5), textcoords='offset points', color='green') 
    ax.plot(freqx*1e-9, Zx*norm_x, lw=1, color='g', marker='s', markersize=2., label='Zx⊥(w) from WAKIS')

    #--- plot Zy⊥(w)
    ifmax=np.argmax(Zy_cst)
    ax.plot(freq_cst[ifmax]*1e-9, Zy_cst[ifmax], marker='o', markersize=5.0, color='black')
    ax.annotate(str(round(freq_cst[ifmax]*1e-9,2))+ ' GHz', xy=(freq_cst[ifmax]*1e-9,Zy_cst[ifmax]), xytext=(+5,-5), textcoords='offset points', color='black') 
    ax.plot(freq_cst*1.0e-9, Zy_cst, lw=1, ls='--', color='black', label='Zy⊥(w) from CST')

    ax.plot(freqy[ifymax]*1e-9, Zy[ifymax]*norm_y, marker='o', markersize=4.0, color='magenta')
    ax.annotate(str(round(freqy[ifymax]*1e-9,2))+ ' GHz', xy=(freqy[ifymax]*1e-9,Zy[ifymax]), xytext=(-50,-5), textcoords='offset points', color='magenta') 
    ax.plot(freqy*1e-9, Zy*norm_y, lw=1, color='magenta', marker='s', markersize=2., label='Zy⊥(w) from WAKIS')
    ax.set(title=title,
            xlabel='f [GHz]',
            ylabel='Z⊥(w) [$\Omega$]',   
            #ylim=(0.,np.max(Zx_cst)*1.2),
            xlim=(0.,np.max(freqx)*1e-9)      
            )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    return fig

def bmk_charge_dist(data, bmk):
    '''
    Plots the charge distribution λ(s) and compares it with CST wake solver results

    Parameters:
    -----------
    - data = read_WAKIS(out_path)
    - bmk = read_bmk(cst_path)
    - flag_compare_cst = False [default]
    '''

    # Obtain WAKIS variables
    s = data.get('s')
    q = data.get('q')
    lambdas = data.get('lambda') #[C/m]

    # Obtain CST variables
    lambda_cst=bmk.get('charge_dist') #[C/m]
    s_cst=bmk.get('s_charge_dist')

    # Plot charge distribution λ(s) & comparison with CST 
    fig = plt.figure(1, figsize=(8,5), dpi=150, tight_layout=True)
    ax=fig.gca()
    ax.plot(s/UNIT, lambdas, lw=1.2, color='red', label='$\lambda$(s)')
    ax.plot(s_cst/UNIT, lambda_cst, lw=1, color='red', ls='--', label='$\lambda$(s) CST')
    ax.set(title='Charge distribution $\lambda$(s)',
            xlabel='s [mm]',
            ylabel='$\lambda$(s) [C/m]',
            xlim=(min(s/UNIT), np.max(s/UNIT))
            )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    return fig

def bmk_WAKIS(data, bmk, flag_normalize=False, flag_charge_dist=False,
            flag_plot_Real=False, flag_plot_Imag=False, flag_plot_Abs=True):
    '''
    Plot results of WAKIS wake solver in different figures with the CST comparison

    Parameters
    ---------- 
    - data: [default] data=read_WAKIS(out_path). Dictionary containing the wake solver output
    - bmk: [default] bmk=read_bmk(cst_path). Dictionary containing the CST benchmark variables
    - flag_compare_cst: [default] True. Enables comparison with CST data 
    - flag_normalize: [default] False. Normalizes the shunt impedance to CST value
    - flag_charge_dist: [default] False. Plots the charge distribution as a function of s 

    Returns
    -------
    - fig 1-4: if flag_charge_dist=False 
    or
    - fig 1-5: if flag_charge_dist=True
    
    fig1 = bmk_long_WP
    fig2 = bmk_long_Z
    fig3 = bmk_trans_WP
    fig4 = bmk_trans_Z
    fig5 = bmk_charge_dist

    '''
    fig1 = bmk_long_WP(data=data, bmk=bmk)
    fig2 = bmk_long_Z(data=data, bmk=bmk, flag_normalize=flag_normalize, flag_plot_Real=flag_plot_Real, flag_plot_Imag=flag_plot_Imag, flag_plot_Abs=flag_plot_Abs)
    fig3 = bmk_trans_WP(data=data, bmk=bmk)
    fig4 = bmk_trans_Z(data=data, bmk=bmk, flag_normalize=flag_normalize)

    if flag_charge_dist:
        fig5 = bmk_charge_dist(data=data, bmk=bmk)
        return fig1, fig2, fig3, fig4, fig5
    else: 
        return fig1, fig2, fig3, fig4 


def generate_bmk(cst_path):
    '''
    Stores CST wake solver output data in a dictionary to perform the benchmark

    Parameters
    ---------- 
    - cst_path: path to the folder containing the benchmark files in .txt format

    Filenames supported
    -------------------
    - Charge distribution: 'lambda'
    - Longitudinal wake potential: 'WP', 'indirect_interf_WP', 'indirect_test_WP'
    - Transverse wake potential: 'WPx', 'WPy', 'WPx_dipolar', 'WPy_dipolar', 'WPx_dipolarX', 'WPy_dipolarX', 'WPx_dipolarY', 'WPy_dipolarY',
                                 'WPx_quadrupolar', 'WPy_quadrupolar', 'WPx_quadrupolarX', 'WPy_quadrupolarX', 'WPx_quadrupolarY', 'WPy_quadrupolarY'
    - Longitudinal impedance: 'Z', 'ReZ', 'ImZ'
    - Transverse impedance: 'Zx', 'Zy', 'Zx_dipolar', 'Zy_dipolar', 'Zx_dipolarX', 'Zy_dipolarX', 'Zx_dipolarY', 'Zy_dipolarY',
                            'Zx_quadrupolar', 'Zy_quadrupolar', 'Zx_quadrupolarX', 'Zy_quadrupolarX', 'Zx_quadrupolarY', 'Zy_quadrupolarY'
    Output
    ------
    - 'cst.bmk' file containing the data in a dictionary. 
        To retrieve it use data=read_bmk(cst_path)
        To access the dictionary keys use data.keys()
    '''
    data = {}

    #--------------------------------#
    #    Electric field Ez files     #
    #--------------------------------# 

    #[TODO]


    #--------------------------------#
    #   charge distribution files    #
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

    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname+'.txt') as f:
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

        #save in dict
        data['charge_dist_cst']=charge_dist
        data['s_charge_dist']=distance

    #---------------------------#
    #   Wake Potential files    #
    #---------------------------#   

    # Longitudinal wake potential [DIRECT method]
    WP=[]
    s_cst=[]

    fname='WP'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
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

        #save in dict
        data['WP_cst']=WP
        data['s_cst']=s_cst


    # Longitudinal wake potential [INDIRECT method]  
    Indirect_WP_interfaces=[]
    Indirect_WP_testbeams=[]

    fname='indirect_interf_WP'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    Indirect_WP_interfaces.append(float(columns[1]))

        Indirect_WP_interfaces=np.array(Indirect_WP_interfaces) # in V/pC
        f.close()
        data['Indirect_WP_interfaces']=Indirect_WP_interfaces

    fname='indirect_test_WP'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    Indirect_WP_testbeams.append(float(columns[1]))

        Indirect_WP_testbeams=np.array(Indirect_WP_testbeams) # in V/pC
        f.close()
        data['Indirect_WP_testbeams']=Indirect_WP_testbeams

    # Transverse wake potential [xytest==0] [xysource==0]
    WPx=[]
    WPy=[]

    fname='WPx'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    WPx.append(float(columns[1]))

        WPx=np.array(WPx) # in V/pC
        data['WPx_cst']=WPx
        f.close()

    fname='WPy'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    WPy.append(float(columns[1]))

        WPy=np.array(WPy) # in V/pC
        data['WPy_cst']=WPy
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
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    WPx_dipolar.append(float(columns[1]))
                    s_cst_dipolar.append(float(columns[0]))

        WPx_dipolar=np.array(WPx_dipolar) # in V/pC
        s_cst_dipolar=np.array(s_cst_dipolar)*1.0e-3  # in [m]
        data['WPx_dipolar_cst']=WPx_dipolar
        data['s_cst_dipolar']=s_cst_dipolar
        f.close()

    fname='WPy_dipolar'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    WPy_dipolar.append(float(columns[1]))

        WPy_dipolar=np.array(WPy_dipolar) # in V/pC
        data['WPy_dipolar_cst']=WPy_dipolar
        f.close()

    fname='WPx_dipolarX'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    WPx_dipolarX.append(float(columns[1]))

        WPx_dipolarX=np.array(WPx_dipolarX) # in V/pC
        data['WPx_dipolarX_cst']=WPx_dipolarX
        f.close()

    fname='WPy_dipolarX'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    WPy_dipolarX.append(float(columns[1]))

        WPy_dipolarX=np.array(WPy_dipolarX) # in V/pC
        data['WPy_dipolarX_cst']=WPy_dipolarX
        f.close()

    fname='WPx_dipolarY'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    WPx_dipolarY.append(float(columns[1]))

        WPx_dipolarY=np.array(WPx_dipolarY) # in V/pC
        data['WPx_dipolarY_cst']=WPx_dipolarY
        f.close()

    fname='WPy_dipolarY'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    WPy_dipolarY.append(float(columns[1]))

        WPy_dipolarY=np.array(WPy_dipolarY) # in V/pC
        data['WPy_dipolarY_cst']=WPy_dipolarY
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
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    WPx_quadrupolar.append(float(columns[1]))
                    s_cst_quadrupolar.append(float(columns[0]))

        WPx_quadrupolar=np.array(WPx_quadrupolar) # in V/pC
        s_cst_quadrupolar=np.array(s_cst_quadrupolar)*1.0e-3  # in [m]
        data['WPx_quadrupolar_cst']=WPx_quadrupolar
        data['s_cst_quadrupolar']=s_cst_quadrupolar
        f.close()

    fname='WPy_quadrupolar'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    WPy_quadrupolar.append(float(columns[1]))

        WPy_quadrupolar=np.array(WPy_quadrupolar) # in V/pC
        data['WPy_quadrupolar_cst']=WPy_quadrupolar
        f.close()

    fname='WPx_quadrupolarX'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    WPx_quadrupolarX.append(float(columns[1]))

        WPx_quadrupolarX=np.array(WPx_quadrupolarX) # in V/pC
        data['WPx_quadrupolarX_cst']=WPx_quadrupolarX
        f.close()

    fname='WPy_quadrupolarX'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    WPy_quadrupolarX.append(float(columns[1]))

        WPy_quadrupolarX=np.array(WPy_quadrupolarX) # in V/pC
        data['WPy_quadrupolarX_cst']=WPy_quadrupolarX
        f.close()

    fname='WPx_quadrupolarY'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    WPx_quadrupolarY.append(float(columns[1]))

        WPx_quadrupolarY=np.array(WPx_quadrupolarY) # in V/pC
        data['WPx_quadrupolarY_cst']=WPx_quadrupolarY
        f.close()

    fname='WPy_quadrupolarY'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    WPy_quadrupolarY.append(float(columns[1]))

        WPy_quadrupolarY=np.array(WPy_quadrupolarY) # in V/pC
        data['WPy_quadrupolarY_cst']=WPy_quadrupolarY
        f.close()

    #---------------------------#
    #      Impedance files      #
    #---------------------------#  

    # Longitudinal Impedance [DIRECT method]
    Z=[]
    ReZ=[]
    ImZ=[]
    freq_cst=[]

    fname='Z'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    Z.append(float(columns[1]))
                    freq_cst.append(float(columns[0]))

        Z=np.array(Z) # in [Ohm]
        freq_cst=np.array(freq_cst)*1e9  # in [Hz]
        data['Z_cst']=Z
        data['freq_cst']=freq_cst
        f.close()

    fname='ReZ'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    ReZ.append(float(columns[1]))

        data['ReZ']=np.array(ReZ) # in [Ohm]
        f.close()

    fname='ImZ'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    ImZ.append(float(columns[1]))

        data['ImZ']=np.array(ImZ) # in [Ohm]
        f.close()

    # Transverse Impedance [xytest=0]
    Zx=[]
    fname='Zx'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    Zx.append(float(columns[1]))

        Zx=np.array(Zx) # in V/pC
        data['Zx_cst']=Zx
        f.close()

    Zy=[]
    fname='Zy'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    Zy.append(float(columns[1]))

        Zy=np.array(Zy) # in V/pC
        data['Zy_cst']=Zy
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
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()

                if i>1 and len(columns)>1:

                    Zx_dipolar.append(float(columns[1]))

        Zx_dipolar=np.array(Zx_dipolar) # in V/pC
        data['Zx_cst']=Zx
        f.close()

    fname='Zy_dipolar'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()

                if i>1 and len(columns)>1:

                    Zy_dipolar.append(float(columns[1]))
                    freq_cst_dipolar.append(float(columns[0]))

        Zy_dipolar=np.array(Zy_dipolar) # in V/pC
        data['Zy_dipolar_cst']=Zy_dipolar
        f.close()

    fname='Zx_dipolarX'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):    
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()

                if i>1 and len(columns)>1:

                    Zx_dipolarX.append(float(columns[1]))

        Zx_dipolarX=np.array(Zx_dipolarX) # in V/pC
        data['Zx_dipolarX_cst']=Zx_dipolarX
        f.close()

    fname='Zy_dipolarX'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()

                if i>1 and len(columns)>1:

                    Zy_dipolarX.append(float(columns[1]))

        Zy_dipolarX=np.array(Zy_dipolarX) # in V/pC
        data['Zy_dipolarX_cst']=Zy_dipolarX
        f.close()

    fname='Zx_dipolarY'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()

                if i>1 and len(columns)>1:

                    Zx_dipolarY.append(float(columns[1]))

        Zx_dipolarY=np.array(Zx_dipolarY) # in V/pC
        data['Zx_dipolarY_cst']=Zx_dipolarY
        f.close()

    fname='Zy_dipolarY'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()

                if i>1 and len(columns)>1:

                    Zy_dipolarY.append(float(columns[1]))

        Zy_dipolarY=np.array(Zy_dipolarY) # in V/pC
        data['Zy_dipolarY_cst']=Zy_dipolarY
        f.close()


    # Transverse quadrupolar Impedance [xytest!=0]
    Zx_quadrupolar=[]
    Zy_quadrupolar=[]
    Zx_quadrupolarX=[]
    Zy_quadrupolarX=[]
    Zx_quadrupolarY=[]
    Zy_quadrupolarY=[]

    fname='Zx_quadrupolar'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    Zx_quadrupolar.append(float(columns[1]))

        Zx_quadrupolar=np.array(Zx_quadrupolar) # in V/pC
        data['Zx_quadrupolar_cst']=Zx_quadrupolar
        f.close()

    fname='Zy_quadrupolar'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()

                if i>1 and len(columns)>1:

                    Zy_quadrupolar.append(float(columns[1]))

        Zy_quadrupolar=np.array(Zy_quadrupolar) # in V/pC
        data['Zy_quadrupolar_cst']=Zy_quadrupolar
        f.close()

    fname='Zx_quadrupolarX'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    Zx_quadrupolarX.append(float(columns[1]))

        Zx_quadrupolarX=np.array(Zx_quadrupolarX) # in V/pC
        data['Zx_quadrupolarX_cst']=Zx_quadrupolarX
        f.close()

    fname='Zy_quadrupolarX'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    Zy_quadrupolarX.append(float(columns[1]))

        Zy_quadrupolarX=np.array(Zy_quadrupolarX) # in V/pC
        data['Zy_quadrupolarX_cst']=Zy_quadrupolarX
        f.close()

    fname='Zx_quadrupolarY'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    Zx_quadrupolarY.append(float(columns[1]))

        Zx_quadrupolarY=np.array(Zx_quadrupolarY) # in V/pC
        data['Zx_quadrupolarY_cst']=Zx_quadrupolarY
        f.close()

    fname='Zy_quadrupolarY'
    i=0
    if os.path.exists(cst_path+fname+'.txt'):
        with open(cst_path+fname +'.txt') as f:
            for line in f:
                i+=1
                columns = line.split()
                if i>1 and len(columns)>1:
                    Zy_quadrupolarY.append(float(columns[1]))

        Zy_quadrupolarY=np.array(Zy_quadrupolarY) # in V/pC
        data['Zy_quadrupolarY_cst']=Zy_quadrupolarY
        f.close()

    # write the dictionary to a txt using json
    with open(cst_path+'cst.bmk', 'w') as fp:
        js.dump(data, fp,  indent=4)

    print('[! OUT] "cst.bmk" file succesfully created')


if __name__ == "__main__":

    cst_path=cwd
    run_benchmark(cst_path, wakis_path=cwd)
