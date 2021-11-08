# Run parallel Warp
## Setting up a python enviroment + openmpi
tutorial modified from: https://github.com/PyCOMPLETE/PyECLOUD/wiki/Setup-python-(including-mpi4py)-without-admin-rights 
The following guide shows how to get a complete python installation without administrator rights:

**Step 1:** We move to the folder where we want to place our installation:

```
/afs/cern.ch/work/e/edelafue/workspace_mpi
```

**Step 2:** We download, compile and install python 3.7 from
https://www.python.org/downloads/release/python-3712/ Python 3.7 is neededsince 3.8 has some issues with warp.

```
wget https://www.python.org/ftp/python/3.7.12/Python-3.7.12.tar.xz
mkdir python_src
mv Python-3.7.12.tar.xz python_src/
cd python_src
tar xvf Python-3.7.12.tar.xz 
cd Python-3.7.12
mkdir /afs/cern.ch/work/e/edelafue/sim_workspace_mpi/python37
./configure --prefix=/afs/cern.ch/work/e/edelafue/sim_workspace_mpi/python37
make
make install
```

**Step 3:** We install virtualenv

```
 cd ..
 curl -LO https://pypi.python.org/packages/source/v/virtualenv/virtualenv-15.0.0.tar.gz
 tar -zxvf virtualenv-15.0.0.tar.gz
 cd virtualenv-15.0.0
 /afs/cern.ch/work/e/edelafue/sim_workspace_mpi/python37/bin/python3 setup.py install
 cd ..
```

**Step 4:** We create our virtual environment

```
cd /afs/cern.ch/work/e/edelafue/sim_workspace_mpi
mkdir virtualenvs
cd virtualenvs
/afs/cern.ch/work/e/edelafue/sim_workspace_mpi/python37/bin/virtualenv py3.7 --python=/afs/cern.ch/work/e/edelafue/sim_workspace_mpi/python37/bin/python3
```

**Step 5:** We activate our brand new virtual environment

```
cd py3.7/bin
source ./activate
```

If we type: `which python` we get:
`/afs/cern.ch/work/e/edelafue/sim_workspace_mpi/virtualenvs/py3.7/bin/python`

**Step 6:** We use pip to install the python modules that we need

```
 pip install numpy scipy cython h5py ipython
 pip install matplotlib sympy linalg pandas seaborn
```

**Step 7** (optional): Installing mpi4py

We do not have an MPI installation we need to get one (for CERN users: skip this for CNAF cluster). We need the version 4.1.1 for python 3.7:

```
cd /afs/cern.ch/work/g/giadarol/sim_workspace_mpi_py3/python_src
wget https://www.open-mpi.org/software/ompi/v4.1/downloads/openmpi-4.1.1.tar.bz2
tar jxf openmpi-4.1.1.tar.bz2
cd openmpi-4.1.1
./configure --prefix=/afs/cern.ch/work/g/giadarol/sim_workspace_mpi_py3/openmpi
 make all install
```

We set the environment variable for the MPI compiler:

```
export MPICC=/afs/cern.ch/work/e/edelafue/sim_workspace_mpi/openmpi/bin/mpicc
```

At this point be careful not to have other MPI paths in your environment (for example set in your .bashrc) pip will use the compiler pointed by your MPICC variable to compile mpi4py:

```
pip install mpi4py
```

Important: Python jobs using mpi4py must be run with the mpiexec corresponding to the MPI compiler that has been used. In our case:

```
/afs/cern.ch/work/g/giadarol/sim_workspace_mpi/openmpi/bin/mpiexec
```

Of course we can add the folder to our PATH or create a shortcut.


## Instructions to install paralell Warp 

In the previously created envirmonment, activate it and install the Python packages needed for Warp (and more)

```
pip install numpy scipy Forthon picmistandard matplotlib ipython
```

Off the top of my head these are the fundamental ones but there could be more.. 

Then Install Warp. 

```
git clone https://bitbucket.org/berkeleylab/warp.git
cd warp/pywarp90
make install3
```
Note that ```gfortran``` (Fortran compiler) is needed. You can install it with ```sudo apt install gfortran```

Test your installation

Open an interactive python session and test the installation by running the following commands:

```
ipython       		  #this opens the python session
from warp import *        #this imports warp in the Python session
```
Then install parallel Warp. You will have to change the path inside warp/pywarp90/Makefile.Forthon3.pympi to give the path to openmpi
```
cd warp/pywarp90
vim Makefile.Forthon3.pympi
FARGS = --farg "-I/afs/cern.ch/work/e/edelafue/sim_workspace_mpi/openmpi/lib/" 
```
Then we can compile with 
```
make pinstall3
```
Test the parallel instalation by creating a `test.py` file inside `/pywarp90` with only `import warp` in it. To do so you can type
```
touch test.py
echo "import warp" >> test.py
```
or open it and edit it with `vim test.py`

Then run the test.py on parallel by doing:
```
/afs/cern.ch/work/e/edelafue/sim_workspace_mpi/openmpi/bin/mpiexec -np 2 python test.py -p 1 1 2
```
where `-np x` stands for the number of processors and `-p` defines the number of regions in which the domain will be splitted. We preffer to split only in the z direction, with the x in`-p 0 0 x` equal the number of processors declared in np.

