# SF-py

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
 
 Python scripts to compute Structure Function. This code computes the structure function using the vectorization approach, as shown in https://joss.theoj.org/papers/10.21105/joss.02185. 


 All the relevant files are contained in the following directories: 

* ``./2d/`` - contains the scripts to compute second and third order structure functions for two-dimensional input data

* ``./3d/`` - contains the scripts to compute second and third order structure functions for three-dimensional input data

 

## Installing SF

``SF`` relies on a few prerequisites which are Python, h5py, Numba, Cupy [For gpu]. 



## Running SF

Please set the following things before you run the code. 

*   
    #input field <br>
    #change the names of the input file and datasets according to your convenience <br>
    Vx = hdf5_reader("U.V1r.h5", "U.V1r") <br>
    Vz = hdf5_reader("U.V3r.h5", "U.V3r") <br>

Change the input file name and the names of the datasets. The similar thing is applicable for 3d also. 



 # 
