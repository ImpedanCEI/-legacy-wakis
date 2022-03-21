'''
Auxiliary functions for WAKIS results plotting:
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

# Global variables
UNIT = 1e-3 #conversion to m
CST_PATH = '/mnt/c/Users/elefu/Documents/CERN/WAKIS/Scripts/CST/' 
OUT_PATH = os.getcwd() + '/' +'runs/out/'

# Plot global parameters
plt.rcParams.update({'font.size': 12})

def read_WarpX_out(out_path=OUT_PATH):
    with open(out_path+'input_data.txt', 'rb') as handle:
        input_data = pk.loads(handle.read())
    return input_data

def read_CST_out(cst_out_path=CST_PATH):
    if os.path.exists(cst_out_path+'cst_out.txt'):
        with open(cst_out_path+'cst_out.txt', 'rb') as handle:
            cst_data = pk.loads(handle.read())
    else: cst_data = None 

    return cst_data

def read_WAKIS_out(out_path=OUT_PATH):
    if os.path.exists(out_path+'wake_solver.txt'):

        with open(out_path+'wake_solver.txt', 'rb') as handle:
            wakis_data = pk.loads(handle.read())
    else: wakis_data=None 

    return wakis_data 

def plot_charge_dist(data=read_WAKIS_out(out_path=OUT_PATH), cst_data=read_CST_out(CST_PATH), flag_compare_cst=True):
    # Obtain WAKIS variables
    s = data.get('s')
    q = data.get('q')
    lambdas = data.get('lambda')*q #[C/m]

    # Obtain CST variables
    lambda_cst=cst_data.get('charge_dist') #[C/m]
    s_cst=cst_data.get('s_charge_dist')

    # Plot charge distribution λ(s) & comparison with CST 
    fig = plt.figure(1, figsize=(8,5), dpi=150, tight_layout=True)
    ax=fig.gca()
    ax.plot(s*1.0e3, lambdas, lw=1.2, color='red', label='$\lambda$(s)')
    if flag_compare_cst:
        ax.plot(s_cst*1.0e3, lambda_cst, lw=1, color='red', ls='--', label='$\lambda$(s) CST')
    ax.set(title='Charge distribution $\lambda$(s)',
            xlabel='s [mm]',
            ylabel='$\lambda$(s) [C/m]',
            xlim=(min(s*1.0e3), np.max(s*1.0e3))
            )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    return fig

def plot_long_WP(data=read_WAKIS_out(out_path=OUT_PATH), cst_data=read_CST_out(CST_PATH), flag_compare_cst=True):
    # Obtain WAKIS variables
    WP=data.get('WP')
    s=data.get('s')

    # Obtain CST variables
    WP_cst=cst_data.get('WP_cst')
    s_cst=cst_data.get('s_cst')

    # Plot longitudinal wake potential W||(s) & comparison with CST 
    fig = plt.figure(1, figsize=(8,5), dpi=150, tight_layout=True)
    ax=fig.gca()
    ax.plot(s*1.0e3, WP, lw=1.2, color='orange', label='$W_{||}$(s)')
    if flag_compare_cst:
        ax.plot(s_cst*1e3, WP_cst, lw=1.2, color='black', ls='--', label='$W_{//}$(s) CST')
    ax.set(title='Longitudinal Wake potential $W_{||}$(s)',
            xlabel='s [mm]',
            ylabel='$W_{||}$(s) [V/pC]',
            xlim=(min(s*1.0e3), np.amin((np.max(s*1.0e3), np.max(s_cst*1.0e3))))
            )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    return fig

def plot_long_Z(data=read_WAKIS_out(out_path=OUT_PATH), cst_data=read_CST_out(CST_PATH), flag_compare_cst=False, flag_normalize=True):
    # Obtain wakis variables
    Z=data.get('Z')
    freq=data.get('f')

    # Obtain CST variables
    Z_cst=cst_data.get('Z_cst')
    freq_cst=cst_data.get('freq_cst')

    # Plot longitudinal impedance Z||(w) comparison with CST [normalized]
    #---normalizing factor between CST and in numpy.fft
    if flag_normalize:
        norm=max(Z_cst)/max(Z) 
        title='Longitudinal impedance Z||(w) \n [normalized by '+str(round(norm,3))+']'
    else:
        norm=1.0
        title='Longitudinal impedance Z||(w)'

    fig = plt.figure(2, figsize=(8,5), dpi=150, tight_layout=True)
    ax=fig.gca()
    #--- obtain the maximum frequency for CST and plot
    if flag_compare_cst:
        ifmax=np.argmax(Z_cst)
        ax.plot(freq_cst[ifmax]*1e-9, Z_cst[ifmax], marker='o', markersize=5.0, color='red')
        ax.annotate(str(round(freq_cst[ifmax]*1e-9,2))+ ' GHz', xy=(freq_cst[ifmax]*1e-9,Z_cst[ifmax]), xytext=(+10,0), textcoords='offset points', color='red') 
        ax.plot(freq_cst*1.0e-9, Z_cst, lw=1, color='red', marker='s', markersize=1., label='Z||(w) from CST')
    #--- obtain the maximum frequency and plot Z||(s)
    ifmax=np.argmax(Z)
    ax.plot(freq[ifmax]*1e-9, Z[ifmax]*norm, marker='o', markersize=4.0, color='blue')
    ax.annotate(str(round(freq[ifmax]*1e-9,2))+ ' GHz', xy=(freq[ifmax]*1e-9,Z[ifmax]*norm), xytext=(-50,0), textcoords='offset points', color='blue') 
    ax.plot(freq*1e-9, Z*norm, lw=1, color='b', marker='s', markersize=2., label='Z||(w)')
    ax.set( title=title,
            xlabel='f [GHz]',
            ylabel='Z||(w) [$\Omega$]',   
            ylim=(0.,np.max(Z)*1.2),
            xlim=(0.,np.max(freq)*1e-9)      
            )
    ax.legend(loc='upper left')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    return fig


def plot_trans_WP(data=read_WAKIS_out(out_path=OUT_PATH), cst_data=read_CST_out(CST_PATH), flag_compare_cst=True):
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
    fig = plt.figure(3, figsize=(8,5), dpi=150, tight_layout=True)
    ax=fig.gca()
    if flag_compare_cst:
        ax.plot(s_cst*1.0e3, WPx_cst, lw=1, color='g', ls='--', label='Wx⊥(s) from CST')
    ax.plot(s*1.0e3, WPx, lw=1.2, color='g', label='Wx⊥(s)')
    if flag_compare_cst:
        ax.plot(s_cst*1.0e3, WPy_cst, lw=1, color='magenta', ls='--', label='Wy⊥(s) from CST')
    ax.plot(s*1.0e3, WPy, lw=1.2, color='magenta', label='Wy⊥(s)')
    ax.set(title='Transverse Wake potential W⊥(s) \n xsource, ysource = '+str(xsource*1e3)+' mm | xtest, ytest = '+str(xtest*1e3)+' mm',
            xlabel='s [mm]',
            ylabel='$W_{⊥}$ [V/pC]',
            xlim=(min(s*1.0e3), np.max(s*1.0e3)),
            )
    ax.legend(loc='best')
    ax.grid(True, color='gray', linewidth=0.2)
    plt.show()

    return fig


def plot_trans_Z(data=read_WAKIS_out(out_path=OUT_PATH), cst_data=read_CST_out(CST_PATH), flag_compare_cst=True, flag_normalize=True):
    # Obtain wakis variables
    Zx=data.get('Zx')
    Zy=data.get('Zy')
    freqx=data.get('f')
    freqy=data.get('f')

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
    if flag_compare_cst:
        ifmax=np.argmax(Zx_cst)
        ax.plot(freq_cst[ifmax]*1e-9, Zx_cst[ifmax], marker='o', markersize=5.0, color='black')
        ax.annotate(str(round(freq_cst[ifmax]*1e-9,2))+ ' GHz', xy=(freq_cst[ifmax]*1e-9,Zx_cst[ifmax]), xytext=(+5,-5), textcoords='offset points', color='black') 
        ax.plot(freq_cst*1.0e-9, Zx_cst, lw=1, ls='--', color='black', label='Zx⊥(w) from CST')

    ax.plot(freqx[ifxmax]*1e-9, Zx[ifxmax]*norm_x, marker='o', markersize=4.0, color='green')
    ax.annotate(str(round(freqx[ifxmax]*1e-9,2))+ ' GHz', xy=(freqx[ifxmax]*1e-9,Zx[ifxmax]), xytext=(-50,-5), textcoords='offset points', color='green') 
    ax.plot(freqx*1e-9, Zx*norm_x, lw=1, color='g', marker='s', markersize=2., label='Zx⊥(w)')

    #--- plot Zy⊥(w)
    if flag_compare_cst:
        ifmax=np.argmax(Zy_cst)
        ax.plot(freq_cst[ifmax]*1e-9, Zy_cst[ifmax], marker='o', markersize=5.0, color='black')
        ax.annotate(str(round(freq_cst[ifmax]*1e-9,2))+ ' GHz', xy=(freq_cst[ifmax]*1e-9,Zy_cst[ifmax]), xytext=(+5,-5), textcoords='offset points', color='black') 
        ax.plot(freq_cst*1.0e-9, Zy_cst, lw=1, ls='--', color='black', label='Zy⊥(w) from CST')

    ax.plot(freqy[ifymax]*1e-9, Zy[ifymax]*norm_y, marker='o', markersize=4.0, color='magenta')
    ax.annotate(str(round(freqy[ifymax]*1e-9,2))+ ' GHz', xy=(freqy[ifymax]*1e-9,Zy[ifymax]), xytext=(-50,-5), textcoords='offset points', color='magenta') 
    ax.plot(freqy*1e-9, Zy*norm_y, lw=1, color='magenta', marker='s', markersize=2., label='Zy⊥(w)')
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

def plot_WAKIS(data=read_WAKIS_out(OUT_PATH), cst_data=read_CST_out(CST_PATH), flag_compare_cst=True, flag_normalize=False, flag_charge_dist=False):
    '''
    Plot results of WAKIS wake solver in different figures

    Parameters
    ---------- 
    - data: [default] data=read_WAKIS_out(OUT_PATH). Dictionary containing the wake solver output
    - cst_data: [default] cst_data=read_CST_out(CST_PATH). Dictionary containing the CST benchmark variables
    - flag_compare_cst: [default] True. Enables comparison with CST data 
    - flag_normalize: [default] False. Normalizes the shunt impedance to CST value
    - flag_charge_dist: [default] False. Plots the charge distribution as a function of s 

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
    fig1 = plot_long_WP(data=data, cst_data=cst_data, flag_compare_cst=flag_compare_cst)
    fig2 = plot_long_Z(data=data, cst_data=cst_data, flag_compare_cst=flag_compare_cst, flag_normalize=flag_normalize)
    fig3 = plot_trans_WP(data=data, cst_data=cst_data, flag_compare_cst=flag_compare_cst)
    fig4 = plot_trans_Z(data=data, cst_data=cst_data, flag_compare_cst=flag_compare_cst, flag_normalize=flag_normalize)

    if flag_charge_dist:
        fig5 = plot_charge_dist(data=data, cst_data=cst_data, flag_compare_cst=flag_compare_cst)
        return fig1, fig2, fig3, fig4, fig5
    else: 
        return fig1, fig2, fig3, fig4 

def subplot_WAKIS(data=read_WAKIS_out(OUT_PATH), flag_charge_dist=False):
    '''
    Plot results of WAKIS wake solver in the same figure

    Parameters
    ---------- 
    - data: [default] data=read_WAKIS_out(OUT_PATH). Dictionary containing the wake solver output
    - flag_charge_dist: [default] False. Plots the charge distribution as a function of s on top of the wake potential

    Returns
    -------
    - fig: figure object

    '''  
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.set_size_inches(16, 9)

    # Add title
    xsource=data.get('xsource')
    ysource=data.get('ysource')
    xtest=data.get('xtest')
    ytest=data.get('ytest') 

    plt.text(x=0.5, y=0.96, s="WAKIS wake solver result", fontsize='x-large', fontweight='bold', ha="center", transform=fig.transFigure)
    plt.text(x=0.5, y=0.93, s= 'xsource = '+str(xsource*1e3)+' mm , ysource = '+str(ysource*1e3)+'mm | xtest = '+str(xtest*1e3)+' mm , ytest = '+str(ytest*1e3)+'mm', fontsize='large', ha="center", transform=fig.transFigure)

    # Longitudinal WP
    WP=data.get('WP')
    s=data.get('s')

    ax1.plot(s*1.0e3, WP, lw=1.2, color='orange', label='$W_{||}$(s)')

    if flag_charge_dist:
        lambdas = data.get('lambda')
        ax1.plot(s*1.0e3, lambdas*max(WP)/max(lambdas), lw=1, color='red', label='$\lambda$(s) [norm]')

    ax1.set(title='Longitudinal Wake potential $W_{||}$(s)',
            xlabel='s [mm]',
            ylabel='$W_{||}$(s) [V/pC]',
            )
    ax1.legend(loc='best')
    ax1.grid(True, color='gray', linewidth=0.2)

    # Longitudinal Z
    Z=data.get('Z')
    f=data.get('f')
    ifmax=np.argmax(Z)

    ax2.plot(f[ifmax]*1e-9, Z[ifmax], marker='o', markersize=4.0, color='blue')
    ax2.annotate(str(round(f[ifmax]*1e-9,2))+ ' GHz', xy=(f[ifmax]*1e-9,Z[ifmax]), xytext=(-20,5), textcoords='offset points', color='blue') 
    ax2.plot(f*1e-9, Z, lw=1, color='b', marker='s', markersize=2., label='Z||(w)')
    ax2.set(title='Longitudinal impedance Z||(w)',
            xlabel='f [GHz]',
            ylabel='Z||(w) [$\Omega$]',   
            ylim=(0.,np.max(Z)*1.2),
            xlim=(0.,np.max(f)*1e-9)      
            )
    ax2.legend(loc='best')
    ax2.grid(True, color='gray', linewidth=0.2)

    # Transverse WP    
    WPx=data.get('WPx')
    WPy=data.get('WPy')

    ax3.plot(s*1.0e3, WPx, lw=1.2, color='g', label='Wx⊥(s)')
    ax3.plot(s*1.0e3, WPy, lw=1.2, color='magenta', label='Wy⊥(s)')
    ax3.set(title='Transverse Wake potential W⊥(s)',
            xlabel='s [mm]',
            ylabel='$W_{⊥}$ [V/pC]',
            )
    ax3.legend(loc='best')
    ax3.grid(True, color='gray', linewidth=0.2)

    # Transverse Z
    Zx=data.get('Zx')
    Zy=data.get('Zy')
    ifxmax=np.argmax(Zx)
    ifymax=np.argmax(Zy)

    #--- plot Zx⊥(w)
    ax4=fig.gca()
    ax4.plot(f[ifxmax]*1e-9, Zx[ifxmax] , marker='o', markersize=4.0, color='green')
    ax4.annotate(str(round(f[ifxmax]*1e-9,2))+ ' GHz', xy=(f[ifxmax]*1e-9,Zx[ifxmax]), xytext=(-10,5), textcoords='offset points', color='g') 
    ax4.plot(f*1e-9, Zx , lw=1, color='g', marker='s', markersize=2., label='Zx⊥(w)')
    #--- plot Zy⊥(w)
    ax4.plot(f[ifymax]*1e-9, Zy[ifymax] , marker='o', markersize=4.0, color='magenta')
    ax4.annotate(str(round(f[ifymax]*1e-9,2))+ ' GHz', xy=(f[ifymax]*1e-9,Zy[ifymax]), xytext=(-10,5), textcoords='offset points', color='m') 
    ax4.plot(f*1e-9, Zy , lw=1, color='magenta', marker='s', markersize=2., label='Zy⊥(w)')
    ax4.set(title='Transverse impedance Z⊥(w)',
            xlabel='f [GHz]',
            ylabel='Z⊥(w) [$\Omega$]',   
            ylim=(0.,np.maximum(max(Zx)*1.2, max(Zy)*1.2)),
            xlim=(0.,np.max(f)*1e-9)      
            )
    ax4.legend(loc='best')
    ax4.grid(True, color='gray', linewidth=0.2)

    plt.show()

    return fig

if __name__ == "__main__":
    
    out_path=os.getcwd()+'/'+'runs/out_cub_cav_default/'

    # Read WAKIS results
    data=read_WAKIS_out(out_path)
    
    # Plot results
    fig = subplot_WAKIS(data=data, flag_charge_dist=True)

    figs = plot_WAKIS(data=data, 
                cst_data=read_CST_out(), 
                flag_compare_cst=True, 
                flag_normalize=False,
                flag_charge_dist=True
                )

    # Save figures
    fig.savefig(out_path+'subplot_WAKIS.png',  bbox_inches='tight')

    figs[0].savefig(out_path+'longWP.png', bbox_inches='tight')
    figs[1].savefig(out_path+'longZ.png',  bbox_inches='tight')
    figs[2].savefig(out_path+'transWP.png',  bbox_inches='tight')
    figs[3].savefig(out_path+'transZ.png',  bbox_inches='tight')

    if len(figs) > 4:
        figs[4].savefig(out_path+'charge_dist')