## Cube cavity

In this folder, are contained the scripts for wake potential and impedance calculation of a generic squared pillbox cavity.

**- cube_cavity.py:**  
---Input for the Warp simulation of the squared cavity for running locally. 
---Stores the results in an out file as a dict using pickle module

**- cube_cavity_mpi.py:**  
---Input for the Warp simulation of the squared cavity for running on parallel. 
---Stores the results in an out file as a dict using pickle module 

**- cst_to_dict.py:** File for postprocessing logfiles from cst
--- Reads 1 log file and plots the field and the frequency
--- Reads all log files and dumps the E(z,t) matrix into a dict
--- Reads the Wake potential and Impedance results from CST into a dict
--- Saves the dict in a out file with pickle

**- postproc.py:** File for postprocessing warp simulations
--- Reads the out file with pickle module
--- Plots the Electric field in the longitudinal direction and obtains the frequency of the Electric field
--- Plots the charge density for every timestep
--- Compares the results with CST (if cst out file from 'cst_to_dict.py' is available)

**- direct_wakepotential.py:**  Wake solver for Warp simulations
--- Reads the out file with pickle module
--- Performs the direct integration to obtain wake potential
--- Performs the fourier transform to obtain the impedance
--- Compares the results with CST wake solver computation

**--- direct_wakepotential_cst.py:**  Wake solver for CST simulations
--- Reads the CST out file with pickle module
--- Performs the direct integration to obtain wake potential
--- Performs the fourier transform to obtain the impedance
--- Compares the results with CST wake solver computation

**--- indirect_wakepotential.py:**  Wake solver for Warp simulations pre-computed fields
--- Reads the out file with pickle module
--- Performs the indirect integration to obtain wake potential using the [Gdfidl method](https://accelconf.web.cern.ch/e06/PAPERS/WEPCH110.PDF)
--- Performs the fourier transform to obtain the impedance
--- Compares the results with CST wake solver computation
Package needed: 'pyPIC' for the Poisson solver. Available [here](https://github.com/PyCOMPLETE/PyPIC)

**--- indirect_wakepotential_cst.py:**  Wake solver for CST simulations pre-computed fields
--- Reads the CST out file with pickle module
--- Performs the indirect integration to obtain wake potential using the [Gdfidl method](https://accelconf.web.cern.ch/e06/PAPERS/WEPCH110.PDF)
--- Performs the fourier transform to obtain the impedance
--- Compares the results with CST wake solver computation
Package needed: 'pyPIC' for the Poisson solver. Available [here](https://github.com/PyCOMPLETE/PyPIC)
