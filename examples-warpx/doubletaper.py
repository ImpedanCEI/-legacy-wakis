'''
doubletaper.py

Script to perform simulations of a in-out taper in WarpX 
to obtain the wake potential and impedance

--- Define the simulation length based on the Wake length
--- Define the test and source positions of the beam
--- Performs the simulation with WarpX-PIC solver
--- Stores the electric field Ez and charge distribution in .h5 file
--- Stores the geometry and simulation input in a dictionary with pickle 

'''
import os, sys
import sys
import json as js

import numpy as np
from scipy.constants import c, e
from pywarpx import picmi

import wakis

#-----------------------------------------------------------------------

##################################
# User defined variables
##################################

# output path

# simulation parameters
CFL = 1.0               #Courant-Friedrichs-Levy criterion for stability
NUM_PROC = 1            #number of mpi processors wanted to use
UNIT = 1e-3             #conversion factor from input to [m]
Wake_length=100*UNIT    #Wake potential length in s [m]

# flags
flag_plot_geom = True
flag_logfile = False
flag_mask_pml = False

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

##################################
# Define the geometry
##################################

# width of the taper (x-dir)
a = 40*UNIT

# intial height of the taper (y-dir)
b = 30*UNIT
# final length of the taper (y-dir)
target=15*UNIT

# length of the straight part (z-dir)
L1 = 10*UNIT
# length of the inclined part (z-dir)
L2 = 25*UNIT
# length of the target part (z-dir)
L3 = 10*UNIT

# Define domain limits

# maximum width (x-dir)
W = a
# maximum heigth (y-dir)
H = np.maximum(b, target)
# maximum length (z-dir)
L =(L1 + L2 + L3)*2.0


# Define the implicit function for the boundary conditions
# If beam traverses z direction, vacuum should reach the ends of the domain
embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function="w=a; h=c*(z<(-Z+L3))+c*(z>(Z-L3))+b*(z>-L1)*(z<L1)+((b-c)/L2*(z+L1)+b)*(z<-L1)*(z>(-Z+L3))+((c-b)/L2*(z-L1)+b)*(z>L1)*(z<(Z-L3)); max(max(x-w/2,-w/2-x),max(y-h/2,-h/2-y))",
    a=a, 
    b=b, 
    c=target, 
    Z=L/2.0,
    L1=L1, 
    L2=L2, 
    L3=L3
)

# Define mesh resolution in x, y, z
dh = 1.0*UNIT

if flag_plot_geom:
    wakis.triang_implicit(fun=wakis.eval_implicit, BC=embedded_boundary, bbox=(-L/2,L/2))


#-----------------------------------------------------------------------

##################################
# Run WarpX simulation
##################################

# execute the file
wakis.execfile('warpx.py')


##################################
# Run Wakis solver
##################################

# Run wakis

