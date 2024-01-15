# SF-py
# 
# Copyright (C) 2022, Soumyadeep Chatterjee
#
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#     3. Neither the name of the copyright holder nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# #########################################################
#  
# \file SF_2nd_order.py
#
# \brief 
# \author Soumyadeep Chatterjee
# \copyright New BSD License
#
# #########################################################


##### Please note we assume domain size 2*pi while binning. Please change line no 81 if you have different domain size. 

# Importing necessary libraries
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

# Setting plot style
mpl.style.use('classic')
plt.rcParams['xtick.major.size'] = 0.7 * 4.0
plt.rcParams['xtick.major.width'] = 2 * 0.5
plt.rcParams['xtick.minor.size'] = 0.7 * 2.5
plt.rcParams['xtick.minor.width'] = 2 * 0.5
plt.rcParams['ytick.major.size'] = 0.7 * 4.0
plt.rcParams['ytick.major.width'] = 2 * 0.5
plt.rcParams['ytick.minor.size'] = 0.7 * 2.5
plt.rcParams['ytick.minor.width'] = 2 * 0.5

# Font settings
A = 2.3 * 9.3
font = {'family': 'serif', 'weight': 'normal', 'size': A}
plt.rc('font', **font)

def hdf5_reader(filename, dataset):
    # Function to read HDF5 file
    file_read = h5py.File(filename, 'r')
    dataset_read = file_read["/" + dataset]
    V = dataset_read[:, :]
    return V

# Input field
# Change the names of the input file and datasets according to your convenience
SF3 = hdf5_reader("str_function.h5", "S_upll")

Nx, Nz = SF3.shape

Nr = int(np.ceil(np.sqrt((Nz // 2) ** 2 + (Nz // 2) ** 2)))

r = np.zeros([Nr])
for i in range(len(r)):
    r[i] = np.pi * np.sqrt(2) * i / (2 * len(r))

 
### binning
SF3_r = np.zeros([Nr])
counter = np.zeros([Nr])

for x in range(Nx // 2):
    for z in range(Nz // 2):
        l = int(np.ceil(np.sqrt(x ** 2 + z ** 2)))
        SF3_r[l] = SF3_r[l] + SF3[x, z]
        counter[l] = counter[l] + 1

SF3_r = SF3_r / counter

# Plotting
fig, axes = plt.subplots(figsize=(8, 6))
axes.plot(r[1:len(r)], SF3_r[1:len(r)] / r[1:len(r)], color='r', lw=1.5, label=r"$S_3^{u}(l)/l$")
axes.set_xlabel('$l$')
axes.set_ylabel('$S_3(l)/l$')
axes.set_yscale("symlog", linthresh=1e-4)
axes.set_xscale("log")

# Saving and displaying the plot
fig.tight_layout()
plt.savefig("SF_velocity_r2D.png", dpi=600)
plt.show()
