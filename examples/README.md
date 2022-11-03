:file_folder: examples/
===
The examples are meant to show the user how to use the Wakis tool together with the EM solvers supported: [WarpX](https://github.com/ECP-WarpX/WarpX) and [CST Studio](https://www.3ds.com/es/productos-y-servicios/simulia/productos/cst-studio-suite/).


Warpx simulations
---

Each folder contains the script to perform simulations of a certain geometry with WarpX in the file `[GEOMETRY].py`

**How to use:**

Inside the python script:

1. Define the path and simulation parameters
2. Define beam parameters
3. Define beam center 
4. Define integration path
5. Define geometry with STL file
6. Define mesh parameters

**Run with:**

 ```
 ipython 
 run warpx.py
 ```

**Output**

- **Ez.h5**: Ez 3d matrix for every timestep in HDF5 file format
- **warpx.inp**: Stores the geometry and simulation input in a dictionary with pickle 

**Requirements**
- WarpX installed with python bindings. For WarpX installation, check the [step-by-step guide](https://wakis.readthedocs.io/en/latest/installation.html#warpx-installation)
- Python modules:
`numpy`, `scipy`, `stl`, `h5py`


CST simulations
---
Open the `.cst` file in CST STUDIO Suite 2020, check the solver is set to Wakefield and click on run simulation. Results of Wake Potential and Impedance are available on the left tree in the 1d results section.

To export the field, set up a field monitor, set the frequency to time and time resolution near to the timestep size. To save some space use the subvolume option and calculate the field only 1 or 2 cells around the integration path *xtes*, *ytest*. In `Post-Processing>Result Templates`, select `Export 3d`, define the subvolume equal to the one used in the field monitor and click on `Set Frq/Time` menu to `sweep time`. Leave the `File option` to ASCII. Press `OK` to save, then click on `Evaluate`. 

Wait for 20-30 mins until the export is finished. 

To post-process the field data with wakis, copy the `3d/` folder stored in `CST/model_name/Export/` into your working directory. Wakis will automatically post-process all the files in the `3d/` folder into a single `Ez.h5` file and store other input data in cst.inp.  


Wakis use 
---
Wakis uses the pre-computed electromagnetic field to obtain the Wake Potential and Impedance of the device under study.

### a) Procedural script

How to use:

Once the WarpX simulation is finished or the CST field data is exported, inside `main.py`:

1. Set the case to postprocess: 'warpx' or 'cst'
2. Fill in user variables if needed
3. Check the path. needs to be set to the folder containing the Ez.h5 file or the 3d/ folder with the Electric field data

Then run with:
```
ipython 
run main.py
```

**Requirements**
- Electric field data in 3d from supported EM solvers
- Python modules:
`numpy`, `scipy`, `stl`, `h5py`, `matplotlib`
