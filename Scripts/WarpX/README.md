## Instructions on how to install WarpX
This README cointains a summary of the tutorial in https://warpx.readthedocs.io/en/latest/install/cmake.html#compile on how to intall WarpX for developers.

### Step 1: WarpX source code
Before you start, you will need a copy of the WarpX source code:
```
git clone https://github.com/ECP-WarpX/WarpX.git $HOME/src/warpx
cd $HOME/src/warpx
```

### Step 2: Install dependencies
Summary of the content in https://warpx.readthedocs.io/en/latest/install/dependencies.html#install-dependencies
WarpX depends on the following popular third party software. 

* a mature C++17 compiler, e.g., GCC 7, Clang 6, NVCC 11.0, MSVC 19.15 or newer
* CMake 3.18.0+
* Git 2.18+
* AMReX: we automatically download and compile a copy of AMReX
* PICSAR: we automatically download and compile a copy of PICSAR

These dependencies and optional ones like CUDA or OpenMP can be installed via [Spack](https://spack.readthedocs.io/en/latest/getting_started.html#installation) or [Homebrew](https://brew.sh/)

**For Spack do:**
Install Spack: 
```
 git clone -c feature.manyFiles=true https://github.com/spack/spack.git
 . spack/share/spack/setup-env.sh
```
To install the depenedencies then do:
```
spack env create warpx-dev
spack env activate warpx-dev

spack add adios2        # for openPMD
spack add blaspp        # for PSATD in RZ
spack add ccache
spack add cmake
spack add fftw          # for PSATD
spack add hdf5          # for openPMD
spack add lapackpp      # for PSATD in RZ
spack add mpi
spack add openpmd-api   # for openPMD
spack add pkgconfig     # for fftw

spack install
```

**For Homebrew do:**
Install Homebrew: 
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
To install the dependencies then do: 
```
brew update
brew tap openpmd/openpmd
brew install adios2      # for openPMD
brew install ccache
brew install cmake
brew install fftw        # for PSATD
brew install git
brew install hdf5-mpi    # for openPMD
brew install libomp
brew install pkg-config  # for fftw
brew install open-mpi
brew install openblas    # for PSATD in RZ
brew install openpmd-api # for openPMD
```

### Step 3: Compile WarpX
From the base of the WarpX source directory, execute this to compile WarpX 

´´´
cmake -S . -B build -DWarpX_DIMS=3 -DWarpX_LIB=ON -DWarpX_EB=ON
cmake --build build -j 4
´´´
That’s all! WarpX binaries are now in build/bin/. 

### Step 4: Python bindings
Build and install pywarpx from the root of the WarpX source tree:
```
PYWARPX_LIB_DIR=$PWD/build/lib python3 -m pip wheel .
python3 -m pip install pywarpx-*whl
```
To avoid mistakes, the whole path to the python/python3 executable can be given
