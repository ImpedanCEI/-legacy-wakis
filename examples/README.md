:file_folder: examples/
===

Each folder contains the script to perform simulations of a certain geometry in WarpX 

How to use:
---
1. Define the path and simulation parameters
2. Define beam parameters
3. Define beam center 
4. Define integration path
5. Define geometry with STL file
6. Define mesh parameters

Run with:

 ```
 ipython 
 run warpx.py
 ```

Output
---
*Ez.h5*: Ez 3d matrix for every timestep in HDF5 file format
*warpx.inp*: Stores the geometry and simulation input in a dictionary with pickle 

Requirements
---
`numpy`, `scipy`, `stl`, `h5py`, `pywarpx`