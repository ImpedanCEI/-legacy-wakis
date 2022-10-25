'''
wakis.py
===

Example script to obtain wake potential and impedance
using wakis module

'''
import wakis

#---------------------------#
#       User variables      #
#---------------------------#

# define EM solver: 'warpx' or 'cst'
case = 'cst'  

# abs path to input data (default is cwd)
# path = 'example/path/to/input/data/'

# set unit conversion 
unit_m = 1e-3  #default: mm
unit_t = 1e-9  #default: ns
unit_f = 1e9   #default: GHz

# Beam parameters
# input_file = 'filename.ext'     #filename containing the beam information (optional)
q = 1e-9                          #beam charge in [C]
sigmaz = 15                		  #beam longitudinal sigma [m]
xsource, ysource = 0e-3, 0e-3     #beam center offset [m]
xtest, ytest = 0e-3, 0e-3         #integration path offset [m]

# Field data
flag_preproc = True      #enable data pre-processing
# Ez_fname = 'Ez.h5'	 #name of filename (optional)
Ez_folder = '3d/'		 #relative path to folder containing multiple Ez files (optional)

# Output options
flag_save = True 
flag_plot = True
out_fname = 'wakis.json'  #define name and format of output data

#---------------------------#

# Initialize inputs
user = wakis.User(case = case, unit_m = unit_m, unit_t = unit_t, unit_f = unit_f)
beam = wakis.Beam(q = q, sigmaz = sigmaz, 
				  xource = xsource, ysource = ysource, 
				  xtest = xtest, ytest = ytest)
field = wakis.Field(preproc = flag_preproc, Ez_folder = Ez_folder)

# Get data object
data = wakis.Input().get_input(User = user, Beam = beam, Field = field) 

# Run solver
data = wakis,solve(data)

# Plot
if flag_plot:
	fig, ax = wakis.plot(data)

# Save
if flag_save:
	wakis.save(data, fname = out_fname)
