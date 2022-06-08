'''
cubcavity.py
------------
Script to perform EM simulations of a pillbox cavity with WarpX 
- Define the simulation length based on the Wake length
- Define the test and source positions of the beam
- Define the geometru with implicit function object

WarpX
- Performs the simulation with WarpX-PIC solver
- Saves the electric field Ez and charge distribution in Ez.h5 file
- Saves the 3d charge distribution in rho.h5 file 
- Generates warpx.out file with simultation 1d data

Wakis
- Run wakis solver to obtain wake potential and impedance
- Saves the results in wakis.out file
- Generates the plots in wakis.png

'''

import numpy as np
from scipy.constants import c, e

from pywarpx import picmi
import wakis

#========================#
# User defined variables #
#========================#

# simulation parameters
CFL = 1.0               #Courant-Friedrichs-Levy criterion for stability
NUM_PROC = 1            #number of mpi processors wanted to use
UNIT = 1e-3             #conversion factor from input to [m]
Wake_length=10*UNIT    #Wake potential length in s [m]

# flags
flag_logfile = False        #generates a .log file with the simulation info
flag_mask_pml = False       #removes the pml cells from the E field data
flag_stl_geom = False       #takes geom info from .stl file
flag_plot_geom = True       #only for implicit function option

# beam parameters
q=1e-9                      #Total beam charge [C]
sigmat = 1.000000e-09/16.   #[s]
sigmaz = sigmat*c           #[m]

# beam source center 
# ---[longitudinal impedance: beam center in 0,0]
# ---[dipolar impedance: beam center in a,0 or 0,a or a,a]
xsource = 0.0*UNIT
ysource = 0.0*UNIT 

# beam test center 
# ---[longitudinal impedance: test axis in 0,0]
# ---[quadrupolar impedance: test axis in a,0 or 0,a or a,a]
xtest = 0.0*UNIT   
ytest = 0.0*UNIT

#=====================#
# Define the geometry #
#=====================#

# stl file option
if flag_stl_geom:

    # Initialize WarpX EB object
    embedded_boundary = picmi.EmbeddedBoundary(stl_file = 'cubcavity.stl')

    # Define domain limits
    # maximum width (x-dir)
    W = 50*UNIT
    # maximum heigth (y-dir)
    H = 50*UNIT
    # maximum length (z-dir)
    L = 100*UNIT

    # Define mesh resolution in x, y, z
    dh = 1.0*UNIT


# implicit function option
else:

    # width of the rectangular beam pipe (x direction)
    w_pipe = 15*UNIT
    # height of the rectangular beam pipe (y direction)
    h_pipe = 15*UNIT
    # length of the pipe (bigger than the cav to resemble infinite length)
    L_pipe = 100*UNIT

    # width of the rectangular beam pipe (x direction)
    w_cav = 50*UNIT
    # height of the rectangular beam pipe (y direction)
    h_cav = 50*UNIT
    # length of the pipe (bigger than the domain to resemble infinite length)
    L_cav = 30*UNIT

    # Define domain limits
    # maximum width (x-dir)
    W = w_cav
    # maximum heigth (y-dir)
    H = h_cav
    # maximum length (z-dir)
    L = L_pipe

    # Define the implicit function for the boundary conditions
    # If beam traverses z direction, vacuum should reach the end of the domain (-L/2, L/2)
    embedded_boundary = picmi.EmbeddedBoundary(
        implicit_function="w=w_pipe+(w_cav-w_pipe)*(z<L_cav/2)*(z>-L_cav/2); h=h_pipe+(h_cav-h_pipe)*(z<L_cav/2)*(z>-L_cav/2); max(max(x-w/2,-w/2-x),max(y-h/2,-h/2-y))",
        w_cav=w_cav, 
        h_cav=h_cav, 
        L_cav=L_cav, 
        w_pipe=w_pipe, 
        h_pipe=h_pipe, 
    )

    # Define mesh resolution in x, y, z
    dh = 1.0*UNIT

    if flag_plot_geom:
        wakis.triang_implicit(fun=wakis.eval_implicit, BC=embedded_boundary, bbox=(-L/2,L/2))


#======================#
# Run WarpX simulation #
#======================#

# execute the file
wakis.execfile('../warpx.py')


#==================#
# Run Wakis solver #
#==================#

# Run wakis
wakis.run_WAKIS()