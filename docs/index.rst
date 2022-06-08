.. Wakis documentation master file, created by
   sphinx-quickstart on Fri Jun  3 14:29:48 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=====
Wakis
=====

Wakis is a postprocessing package that computes the Wake Potential and Impedance from pre-computed electromagnetic fields. These magnitudes are important to quantify the losses of accelerator structures with a passing beam.

Wakis has been coupled with `WarpX <https://warpx.readthedocs.io/en/latest/>`_, an open-source electromagnetic Particle-In-Cell code, in order to perform time-domain simulations in 3D. WarpX supports many features including Embedded Boundaries definition through '.stl' files, Perfectly-Matched-Layers (PML). It is based on `AMReX <https://amrex-codes.github.io/amrex/docs_html/>`_ allowing mesh refinement, MPI parallel computing and GPU acceleration. 

The source code is available in the `WAKIS GitHub repository <https://github.com/ImpedanCEI/WAKIS>`_.

Table of contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usersguide
   physicsguide
   modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
