.. _installation-page:

============
Installation
============

We assume that you have a recent python installation (python 3.8+). It this is not the case you can make one following the dedicated section on :ref:`how to get a miniconda installation<miniconda>`.


.. code-block:: bash
	cd ~
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	bash Miniconda3-latest-Linux-x86_64.sh
	source miniconda3/bin/activate
	pip install numpy scipy matplotlib ipython h5py


.. contents:: Table of Contents
    :depth: 3

Wakis installation
==================

The Wakis package can be installed using pip:

.. code-block:: bash

    pip install wakis


WarpX installation
==================

This page contains a summary of the tutorial in
https://warpx.readthedocs.io/en/latest/install/cmake.html#compile_ on how
to install WarpX for developers.

Step 1: WarpX source code
~~~~~~~~~~~~~~~~~~~~~~~~~

Before you start, you will need a copy of the WarpX source code:

.. code-block:: bash

   git clone https://github.com/ECP-WarpX/WarpX.git $HOME/src/warpx
   cd $HOME/src/warpx

Step 2: Install dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Summary of the content in
https://warpx.readthedocs.io/en/latest/install/dependencies.html#install-dependencies
WarpX depends on the following popular third party software.

-  a mature C++17 compiler, e.g., GCC 7, Clang 6, NVCC 11.0, MSVC 19.15
   or newer
-  CMake 3.18.0+
-  Git 2.18+
-  AMReX: we automatically download and compile a copy of AMReX
-  PICSAR: we automatically download and compile a copy of PICSAR

These dependencies and optional ones like CUDA or OpenMP can be
installed via `Spack<https://spack.readthedocs.io/en/latest/getting_started.html#installation>`_ or `Homebrew<https://brew.sh/>`_

**Spack**

If not installed already, install Spack by:

.. code-block:: bash

    git clone -c feature.manyFiles=true https://github.com/spack/spack.git
    . spack/share/spack/setup-env.sh

To install the depenedencies then do:

.. code-block:: bash

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

**Homebrew**

The preferred method to install dependencies is spack. Howeverm Homebrew is a good alternative in case you encounter some issues with Spack's installation. 

If not installed already, install Homebrew by:

.. code-block:: bash

   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

To install the dependencies then do:

.. code-block:: bash

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


Step 3: Compile WarpX
~~~~~~~~~~~~~~~~~~~~~

From the base of the WarpX source directory, execute this to compile
WarpX

.. code-block:: bash

   cmake -S . -B build -DWarpX_DIMS=3 -DWarpX_LIB=ON -DWarpX_EB=ON
   cmake --build build -j 4

That’s all! WarpX binaries are now in build/bin/.

Step 4: Python bindings
~~~~~~~~~~~~~~~~~~~~~~~

Build and install pywarpx from the root of the WarpX source tree:

.. code-block:: bash

   PYWARPX_LIB_DIR=$PWD/build/lib python3 -m pip wheel .
   python3 -m pip install pywarpx-*whl

To avoid mistakes, the whole path to the python/python3 executable can
be given.

The end! Now the python libraries of pywarpx are installed in the python
version used for the installation.


Developer installation
======================

If you need to develop Wakis, you can clone the package from GitHub and install it with pip in editable mode:

.. code-block:: bash

    git clone https://github.com/ImpedanCEI/WAKIS
    pip install -e wakis


Optional dependencies
=====================

For visualizing the embedded boundaries geometry defined with the implicit function option, Wakis relies on *scikit-image* package v>= 0.17. You can installing using pip:

.. code-block:: bash

	pip install scikit-image

.. _miniconda:

Install Miniconda
=================

If you don't have a miniconda installation, you can quickly get one ready for xsuite installation with the following steps.

On Linux
--------

.. code-block:: bash

    cd ~
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    source miniconda3/bin/activate
    pip install numpy scipy matplotlib h5py ipython pytest

On MacOS
--------

.. code-block:: bash

    cd ~
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh > miniconda_inst.sh
    bash miniconda_inst.sh
    source miniconda3/bin/activate
    conda install clang_osx-64
    pip install numpy scipy matplotlib h5py ipython pytest
