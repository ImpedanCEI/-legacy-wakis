'''
Auxiliary functions for PyWake results read/plotting:
- Longitudinal wake potential
- Longitudinal Impedance 
- Transverse wake potential
- Transverse Impedance
'''

import numpy as np
import os 
import matplotlib.pyplot as plt
import scipy.constants as spc  
import pickle as pk
import h5py as h5py

UNIT = 1e-3 #conversion to m
CST_PATH = '/mnt/c/Users/elefu/Documents/CERN/PyWake/Scripts/CST/' 
OUT_PATH = os.getcwd() +'/'

def read_WarpX_out(out_path=OUT_PATH):
    #--- read the dictionary
    with open(out_path+'input_data.txt', 'rb') as handle:
        input_data = pk.loads(handle.read())
    return input_data

def read_CST_out(cst_out_path=CST_PATH):
    with open(cst_out_path+'cst_out.txt', 'rb') as handle:
        cst_data = pk.loads(handle.read())
    return cst_data

def read_PyWake_out(out_path=OUT_PATH):
    with open(out_path+'wake_solver.txt', 'rb') as handle:
        PyWake_data = pk.loads(handle.read())
    return PyWake_data 

def plot_long_WP(data, cst_data=read_CST_out(CST_PATH), flag_compare_cst=True):
    # Obtain PyWake variables
    WP=data.get('Longitudinal wake potential')
    s=data.get('s')

    # Obtain CST variables
    WP_cst=cst_data.get('WP_cst')
    s_cst=cst_data.get('s_cst')

    #Plot longitudinal wake potential W||(s) & comparison with CST 
    fig = plt.figure(1, figsize=(6,4), dpi=200, tight_layout=True)
    ax=fig.gca()
    ax.plot(s*1.0e3, WP, lw=1.2, color='orange', label='$W_{||}$(0,0)(s)')
    if flag_compare_cst:
        ax.plot(s_cst*1e3, Wake_potential_cst, lw=1.3, color='black', ls='--', label='$W_{//}$(s) CST')
    ax.set(title='Longitudinal Wake potential $W_{||}$(s)',
            xlabel='s [mm]',
            ylabel='$W_{||}$(s) [V/pC]',
            xlim=(min(s*1.0e3), np.amin((np.max(s*1.0e3), np.max(s_cst*1.0e3))))
            )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

def plot_long_Z(data, cst_data=read_CST_out(CST_PATH), flag_compare_cst=False, flag_normalize=True):
    # Obtain PyWake variables
    Z=data.get('Longitudinal impedance')
    freq=data.get('frequency')

    # Obtain CST variables
    Z_cst=cst_data.get('Z_cst')
    freq_cst=cst_data.get('freq_cst')

    # Plot longitudinal impedance Z||(w) comparison with CST [normalized]
    #---normalizing factor between CST and in numpy.fft
    if flag_normalize:
        norm=max(Z)/max(Z_cst) 
        title='Longitudinal impedance Z||(w) \n [normalized by '+str(round(norm,3))+']'
    else:
        norm=1.0
        title='Longitudinal impedance Z||(w)'

    #--- obtain the maximum frequency for WarpX and plot
    ifreq_max=np.argmax(Z[0:len(Z)//2])
    fig = plt.figure(2, figsize=(6,4), dpi=200, tight_layout=True)
    ax=fig.gca()
    ax.plot(freq[ifreq_max], Z[ifreq_max]/norm, marker='o', markersize=4.0, color='blue')
    ax.annotate(str(round(freq[ifreq_max],2))+ ' GHz', xy=(freq[ifreq_max],Z[ifreq_max]/norm), xytext=(-20,5), textcoords='offset points', color='blue') 
    ax.plot(freq[0:len(Z)//2], Z[0:len(Z)//2]/norm, lw=1, color='b', marker='s', markersize=2., label='Z||(w) from WarpX')
    #--- obtain the maximum frequency for CST and plot
    if flag_compare_cst:
        ifreq_max=np.argmax(Z_cst)
        ax.plot(freq_cst[ifreq_max]*1e-9, Z_cst[ifreq_max], marker='o', markersize=5.0, color='red')
        ax.annotate(str(round(freq_cst[ifreq_max]*1e-9,2))+ ' GHz', xy=(freq_cst[ifreq_max]*1e-9,Z_cst[ifreq_max]), xytext=(+20,5), textcoords='offset points', color='red') 
        ax.plot(freq_cst*1.0e-9, Z_cst, lw=1.2, color='red', marker='s', markersize=2., label='Z||(w) from CST')
    #--- plot Z||(s)
    ax.set(title=title,
            xlabel='f [GHz]',
            ylabel='Z||(w) [$\Omega$]',   
            ylim=(0.,np.max(Z_cst)*1.2),
            xlim=(0.,np.max(freq_cst)*1e-9)      
            )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

def plot_trans_WP(data, cst_data=read_CST_out(CST_PATH), flag_compare_cst=True):
    # Obtain PyWake variables
    WPx=data.get('Transverse wake potential x')
    WPy=data.get('Transverse wake potential y')
    s=data.get('s')
    # Obtain the offset of the source beam and test beam
    xsource=data.get('xsource')
    ysource=data.get('ysource')
    xtest=data.get('xtest')
    ytest=data.get('ytest') 

    # Obtain CST variables
    WPx_cst=cst_data.get('WPx_cst')
    WPy_cst=cst_data.get('WPy_cst')
    s_cst=cst_data.get('s_cst')

    #-- Quadrupolar cases
    if xtest != 0.0 and ytest == 0.0:
        WPx_cst=cst_data.get('WPx_quadrupolarX_cst')
        WPy_cst=cst_data.get('WPy_quadrupolarX_cst')
        s_cst=cst_data.get('s_cst_quadrupolar')
    if xtest == 0.0 and ytest != 0.0:
        WPx_cst=cst_data.get('WPx_quadrupolarY_cst')
        WPy_cst=cst_data.get('WPy_quadrupolarY_cst')
        s_cst=cst_data.get('s_cst_quadrupolar')
    if xtest != 0.0 and ytest != 0.0:
        WPx_cst=cst_data.get('WPx_quadrupolar_cst')
        WPy_cst=cst_data.get('WPy_quadrupolar_cst')
        s_cst=cst_data.get('s_cst_quadrupolar')

    #-- Dipolar cases
    if xsource != 0.0 and ysource == 0.0:
        WPx_cst=cst_data.get('WPx_dipolarX_cst')
        WPy_cst=cst_data.get('WPy_dipolarX_cst')
        s_cst=cst_data.get('s_cst_dipolar')
    if xsource == 0.0 and ysource != 0.0:
        WPx_cst=cst_data.get('WPx_dipolarY_cst')
        WPy_cst=cst_data.get('WPy_dipolarY_cst')
        s_cst=cst_data.get('s_cst_dipolar')
    if xsource != 0.0 and ysource != 0.0:
        WPx_cst=cst_data.get('WPx_dipolar_cst')
        WPy_cst=cst_data.get('WPy_dipolar_cst')
        s_cst=cst_data.get('s_cst_dipolar')

    # Plot transverse wake potential Wx⊥(s), Wy⊥(s) & comparison with CST
    fig = plt.figure(3, figsize=(6,4), dpi=200, tight_layout=True)
    ax=fig.gca()
    ax.plot(s*1.0e3, Transverse_wake_potential_x, lw=1.2, color='g', label='Wx⊥(s)')
    ax.plot(s_cst*1.0e3, WPx_cst, lw=1.2, color='g', ls='--', label='Wx⊥(s) from CST')
    ax.plot(s*1.0e3, Transverse_wake_potential_y, lw=1.2, color='magenta', label='Wy⊥(s)')
    ax.plot(s_cst*1.0e3, WPy_cst, lw=1.2, color='magenta', ls='--', label='Wy⊥(s) from CST')
    ax.set(title='Transverse Wake potential W⊥(s) \n xsource, ysource = '+str(xsource*1e3)+' mm | xtest, ytest = '+str(xtest*1e3)+' mm',
            xlabel='s [mm]',
            ylabel='$W_{⊥}$ [V/pC]',
            xlim=(min(s*1.0e3), np.amin((np.max(s*1.0e3), np.max(s_cst*1.0e3)))),
            )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

def plot_trans_Z(data, cst_data=read_CST_out(CST_PATH), flag_compare_cst=True, flag_normalize=True):
    # Obtain PyWake variables
    Zx=data.get('Transverse impedance x')
    Zy=data.get('Transverse impedance y')
    freqx=data.get('frequency x')
    freqy=data.get('frequency y')

    # Obtain the offset of the source beam and test beam
    xsource=data.get('xsource')
    ysource=data.get('ysource')
    xtest=data.get('xtest')
    ytest=data.get('ytest') 

    # Obtain CST variables
    Zx_cst=cst_data.get('Zx_cst')
    Zy_cst=cst_data.get('Zy_cst')
    freq_cst=cst_data.get('freq_cst')

    #-- Quadrupolar cases
    if xtest != 0.0 and ytest == 0.0:
        Zx_cst=cst_data.get('Zx_quadrupolarX_cst')
        Zy_cst=cst_data.get('Zy_quadrupolarX_cst')
        freq_cst=cst_data.get('freq_cst_quadrupolar')
    if xtest == 0.0 and ytest != 0.0:
        Zx_cst=cst_data.get('Zx_quadrupolarY_cst')
        Zy_cst=cst_data.get('Zy_quadrupolarY_cst')
        freq_cst=cst_data.get('freq_cst_quadrupolar')
    if xtest != 0.0 and ytest != 0.0:
        Zx_cst=cst_data.get('Zx_quadrupolar_cst')
        Zy_cst=cst_data.get('Zy_quadrupolar_cst')
        freq_cst=cst_data.get('freq_cst_quadrupolar')

    #-- Dipolar cases
    if xsource != 0.0 and ysource == 0.0:
        Zx_cst=cst_data.get('Zx_dipolarX_cst')
        Zy_cst=cst_data.get('Zy_dipolarX_cst')
        freq_cst=cst_data.get('freq_cst_dipolar')
    if xsource == 0.0 and ysource != 0.0:
        Zx_cst=cst_data.get('Zx_dipolarY_cst')
        Zy_cst=cst_data.get('Zy_dipolarY_cst')
        freq_cst=cst_data.get('freq_cst_dipolar')
    if xsource != 0.0 and ysource != 0.0:
        Zx_cst=cst_data.get('Zx_dipolar_cst')
        Zy_cst=cst_data.get('Zy_dipolar_cst')
        freq_cst=cst_data.get('freq_cst_dipolar')

    #---normalizing factor between CST and in PyWake results
    if flag_normalize:
        norm_x=max(Zx)/max(Zx_cst) 
        norm_y=max(Zy)/max(Zy_cst) 
        title='Transverse impedance Z⊥(w) [normalized by '+str(round(norm_x,3))+']'
    else:
        norm_x=1.0
        norm_y=1.0
        title='Transverse impedance Z⊥(w)'

    #--- obtain the maximum frequency
    ifreq_x_max=np.argmax(Zx[0:len(Zx)//2])
    ifreq_y_max=np.argmax(Zy[0:len(Zy)//2])
    #--- plot Zx⊥(w)
    fig = plt.figure(4, figsize=(6,4), dpi=200, tight_layout=True)
    ax=fig.gca()
    ax.plot(freqx[ifreq_x_max], Zx[ifreq_x_max]/norm_x, marker='o', markersize=4.0, color='green')
    ax.annotate(str(round(freqx[ifreq_x_max],2))+ ' GHz', xy=(freqx[ifreq_x_max],Zx[ifreq_x_max]), xytext=(-10,5), textcoords='offset points', color='grey') 
    ax.plot(freqx[0:len(Zx)//2], Zx[0:len(Zx)//2]/norm_x, lw=1, color='g', marker='s', markersize=2., label='Zx⊥ from WarpX')
    #--- obtain the maximum frequency for CST Zx⊥(w) and plot
    if flag_compare_cst:
        ifreq_max=np.argmax(Zx_cst)
        ax.plot(freq_cst[ifreq_max]*1e-9, Zx_cst[ifreq_max], marker='o', markersize=5.0, color='black')
        ax.annotate(str(round(freq_cst[ifreq_max]*1e-9,2))+ ' GHz', xy=(freq_cst[ifreq_max]*1e-9,Zx_cst[ifreq_max]), xytext=(+20,5), textcoords='offset points', color='black') 
        ax.plot(freq_cst*1.0e-9, Zx_cst, lw=1.2, ls='--', color='black', marker='s', markersize=2., label='Zx⊥(w) from CST')
    #--- plot Zy⊥(w)
    ax.plot(freqy[ifreq_y_max], Zy[ifreq_y_max]/norm_y, marker='o', markersize=4.0, color='magenta')
    ax.annotate(str(round(freqy[ifreq_y_max],2))+ ' GHz', xy=(freqy[ifreq_y_max],Zy[ifreq_y_max]), xytext=(-10,5), textcoords='offset points', color='grey') 
    ax.plot(freqy[0:len(Zy)//2], Zy[0:len(Zy)//2]/norm_y, lw=1, color='magenta', marker='s', markersize=2., label='Zy⊥(w) from WarpX')
    #--- obtain the maximum frequency for CST Zy⊥(w) and plot
    if flag_compare_cst:
        ifreq_max=np.argmax(Zy_cst)
        ax.plot(freq_cst[ifreq_max]*1e-9, Zy_cst[ifreq_max], marker='o', markersize=5.0, color='black')
        ax.annotate(str(round(freq_cst[ifreq_max]*1e-9,2))+ ' GHz', xy=(freq_cst[ifreq_max]*1e-9,Zy_cst[ifreq_max]), xytext=(+20,5), textcoords='offset points', color='black') 
        ax.plot(freq_cst*1.0e-9, Zy_cst, lw=1.2, ls='--', color='black', marker='s', markersize=2., label='Zy⊥(w) from CST')

    ax.set(title=title,
            xlabel='f [GHz]',
            ylabel='Z⊥(w) [$\Omega$]',   
            #ylim=(0.,np.max(Zx)*1.2),
            #xlim=(0.,np.max(freqx))      
            )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

def plot_PyWake(data=read_PyWake_out(OUT_PATH), cst_data=read_CST_out(CST_PATH), flag_compare_cst=True, flag_normalize=True):
    # Plot results
    plot_long_WP(data, cst_data=cst_data, flag_compare_cst=flag_compare_cst)
    plot_long_Z(data, cst_data=cst_data, flag_compare_cst=flag_compare_cst, flag_normalize=flag_normalize)
    plot_trans_WP(data, cst_data=cst_data, flag_compare_cst=flag_compare_cst)
    plot_trans_Z(data, cst_data=cst_data, flag_compare_cst=flag_compare_cst, flag_normalize=flag_normalize)


if __name__ == "__main__":
    
    plot_PyWake(data=read_PyWake_out(OUT_PATH))