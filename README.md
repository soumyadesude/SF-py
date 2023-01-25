# SF-py

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
 
 Python scripts to compute Structure Function. This code computes the structure function using the vectorization approach, as shown in https://joss.theoj.org/papers/10.21105/joss.02185. 


 All the relevant files are contained in the following directories: 

* ``./2d/`` - contains the scripts to compute second and third order structure functions for two-dimensional input data

* ``./3d/`` - contains the scripts to compute second and third order structure functions for three-dimensional input data

 

## Installing SF-py

``SF`` relies on a few prerequisites which are Python, h5py, Numba, Cupy [For gpu]. 



## Running SF-py

Please set the following things before you run the code. 

*   
    #input field <br>
    #change the names of the input file and datasets according to your convenience <br>
    Vx = hdf5_reader("U.V1r.h5", "U.V1r") <br>
    Vz = hdf5_reader("U.V3r.h5", "U.V3r") <br>

Change the input file name and the names of the datasets. The above is the example of 2D input data. The similar thing is applicable for 3d also. 

* 
    ##input ## <br>
    L = 2\*np.pi #Length of the domain <br>

Change the domain length according to your need. 


``SF-py`` can be executed by the following way at the respective folders (2d/3d).

Example: for 2nd order structure function:

    python SF_2nd_order.py


## License

``SF-py`` is an open-source package made available under the New BSD License.


## Contributions and bug reports

Contributions to this project are very welcome.
If you wish to contribute, please create a branch with a [pull request](https://github.com/soumyadesude/SF-py/pulls) and the proposed changes can be discussed there.

If you find a bug, please open a new [issue](https://github.com/soumyadesude/SF-py/issues) on the GitHub repository to report the bug.
Please provide sufficient information for the bug to be reproduced.



 # 
