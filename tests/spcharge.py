'''
Analytic study of impedance due to space charge
based on the formulas in chapter 2 of XSuite physics manual
url: https://raw.githubusercontent.com/xsuite/xsuite/main/docs/physics_manual/physics_man.pdf
'''
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.constants import c, pi, epsilon_0, e, m_p

#to be able to use pyPIC the BIN path is needed
BIN = os.path.expanduser("/home/edelafue/PyCOMPLETE")
if BIN not in sys.path:
    sys.path.append(BIN)

import PyPIC.geom_impact_poly as poly 
import PyPIC.FiniteDifferences_Staircase_SquareGrid as PIC_FD

import solver_module as Wsol
unit = 1e-3

#domain

w=10 * unit
h=10 * unit
L=100 * unit

dh=0.1*unit

x = np.arange(-w/2 - 5*dh, w/2 + 4*dh, dh)
y = np.arange(-h/2 - 4*dh,h/2 + 3*dh,dh)
z = np.arange(-L/2,L/2,dh)

t = np.linspace(0,2e-9, len(z))

#bunch
N = 2e11 #bunch population

beta = 0.5
gamma = 1/np.sqrt(1-beta**2)
s = z - beta*c*t
ds = abs(s[2]-s[1])

sigma_z = 20*unit
lambda_0 = N/(sigma_z*np.sqrt(2*pi))*np.exp(-1/2*(s/sigma_z)**2)

X, Y = np.meshgrid(y,x)
sigma_x = 0.1*unit
sigma_y = 0.1*unit
rho_x = 1/(sigma_x*np.sqrt(2*pi))*np.exp(-1/2*(X/sigma_x)**2)
rho_y = 1/(sigma_y*np.sqrt(2*pi))*np.exp(-1/2*(Y/sigma_y)**2)
rho_xy = rho_x*rho_y


# solve 2D poisson

# get indexes for x and y for the picFD solver 
# x direction is gridded with w/dh + 10 ghost cells
# y direction is gridded with h/dh + 8 ghost cells

rho = -1/epsilon_0*rho_xy

# PyPIC function to declare the implicit function for the conductors (this acts as BCs)
PyPIC_chamber = poly.polyg_cham_geom_object({'Vx' : np.array([w/2, -w/2, -w/2, w/2]),
                                             'Vy': np.array([h/2, h/2, -h/2, -h/2]),
                                             'x_sem_ellip_insc' : 0.99*w/2, #important to add this
                                             'y_sem_ellip_insc' : 0.99*h/2})
# solver object
picFD = PIC_FD.FiniteDifferences_Staircase_SquareGrid(chamb = PyPIC_chamber, Dh = dh, sparse_solver = 'PyKLU')
picFD.solve(rho = rho)      #the dimensions are selected by pyPIC solver

phi_xy = picFD.phi.copy()
phi_z = e*lambda_0

dphi_z = - e*N/(sigma_z*np.sqrt(2*pi))*(s/sigma_z**2)*np.exp(-1/2*(s/sigma_z)**2)

#wake potential
WP = -(L*(1-beta)/(m_p*gamma*beta**2*c**2))*dphi_z

#impedance

# Define maximum frequency
fmax=1.01*c/sigma_z/3
# Obtain the ffts and frequency bins
lambdaf, f=Wsol.FFT(lambda_0, ds/c, fmax=fmax, r=10.0)
WPf, f=Wsol.FFT(WP, ds/c, fmax=fmax, r=10.0)
# Obtain the impedance
Z = abs(- WPf / lambdaf) * 3400


# Plot 
fig = plt.figure(1, figsize=(8,5), dpi=150, tight_layout=True)
ax=fig.gca()
ax.plot(s*1.0e3, WP, lw=1.2, color='red', label='$W_{||}$(s)')

ax.set(title='Longitudinal Wake potential $W_{||}$(s)',
        xlabel='s [mm]',
        ylabel='$W_{||}$(s) [V/pC]',
        )
ax.grid(True, color='gray', linewidth=0.2)
plt.show()

fig = plt.figure(2, figsize=(8,5), dpi=150, tight_layout=True)
ax=fig.gca()
ax.plot(f[2:-2]/1.0e9, Z[2:-2], lw=1.2, color='blue', label='$Z_{||}$(f)')

ax.set(title='Longitudinal Wake potential $Z_{||}$(f)',
        xlabel='f [GHz]',
        ylabel='$Z_{||}$(s) [$\Omega$]',
        )
ax.grid(True, color='gray', linewidth=0.2)
plt.show()