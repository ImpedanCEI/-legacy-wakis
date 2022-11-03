:file_folder: wakis/
===

*Procedural version of Wakis* 

* **main.py**: contains all the source code of wakis tool to compute wake potential and impedance from pre-computed fields. Currently EM solver input supported: 
	- [CST Studio](https://www.3ds.com/es/productos-y-servicios/simulia/productos/cst-studio-suite/) (_commercial_)
	- [WarpX](https://github.com/ECP-WarpX/WarpX) (_open-source_)
* **helpers.py**: helper functions that support the pre- and post- processing of the data for main.py
* **warpx.py**: script to run warpx simulations importing geometry from an `.stl` file. See :file_folder: examples/ for specific geometries


