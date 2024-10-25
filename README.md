# SF-py

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10060564.svg)](https://doi.org/10.5281/zenodo.10060564)
 
This branch is currently under development for MHD. Only 'str_function.py' is available in the 2D folder.

 Python scripts to compute Structure Function. These codes computes the structure function using the vectorization approach, as shown in https://joss.theoj.org/papers/10.21105/joss.02185. This code is compatible with both CPU and GPU.  


 All the relevant files are contained in the following directories: 

* ``2d/SF_2nd_order.py`` - contains the scripts to compute second order structure functions for two-dimensional input data

* ``2d/SF_3rd_order.py`` - contains the scripts to compute third order structure functions for two-dimensional input data

* ``3d/SF_2nd_order.py`` - contains the scripts to compute second order structure functions for three-dimensional input data

* ``3d/SF_3rd_order.py`` - contains the scripts to compute third order structure functions for three-dimensional input data

 

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


* 
    ##device ## <br>
    device = "cpu" #by default code will run on cpu <br>

Change the device to "gpu" if you need.


* 
    ##device id ## <br>
    device_id = 0 #by default code will run on gpu0 <br>

Change the device_id accordingly to use a particular gpu on which you want to run the code.


``SF-py`` can be executed by the following way at the respective folders (2d/3d).

Example: for 2nd order structure function:

    python SF_2nd_order.py


## Output of the SF-py

``SF-py`` outputs "str_function.h5" file, which contains the following datasets: 


* ``For 2nd order`` - ``S``: This dataset corresponds to $\langle(|\delta u|)^2\rangle$.    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;``S_u_r``: This dataset corresponds to $\langle(|\delta u_{\parallel}|)^2\rangle$.   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;``S_u_x``: This dataset corresponds to $\langle(|\delta u_{x}|)^2\rangle$.   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;``S_u_z``: This dataset corresponds to $\langle(|\delta u_{z}|)^2\rangle$.   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;``S_u_y``: This dataset corresponds to $\langle(|\delta u_{y}|)^2\rangle$. This is available in the output only for 3d data.  


* ``For 3rd order`` - ``S``: This dataset corresponds to $\langle(|\delta u|)^2\delta u_{\parallel}\rangle$.    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;``S_u_r``: This dataset corresponds to $\langle(|\delta u_{\parallel}|)^3\rangle$.   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;``S_u_x``: This dataset corresponds to $\langle(|\delta u|)^2\delta u_{x}\rangle$.   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;``S_u_z``: This dataset corresponds to $\langle(|\delta u|)^2\delta u_{z}\rangle$.   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;``S_u_y``: This dataset corresponds to $\langle(|\delta u|)^2\delta u_{y}\rangle$. This is available in the output only for 3d data.





## License

``SF-py`` is an open-source package made available under the New BSD License.


## Contributions and bug reports

Contributions to this project are very welcome.
If you wish to contribute, please create a branch with a [pull request](https://github.com/soumyadesude/SF-py/pulls) and the proposed changes can be discussed there.

If you find a bug, please open a new [issue](https://github.com/soumyadesude/SF-py/issues) on the GitHub repository to report the bug.
Please provide sufficient information for the bug to be reproduced.



 # 
