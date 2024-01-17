
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



import numpy as np
import h5py
import time

device = "cpu" #default device 

device_id = 0 #only applicable if gpu is in use

import numba as nb  #comment this line if device is not cpu

if device == "gpu":
   import cupy as cp

    
# read the data file ###############

# input field
#change the names of the input file and datasets according to your convenience
data = np.loadtxt("time_60.dat")

Vx = data[:, 1]

if device == "gpu":
    # select GPU device
    dev1 = cp.cuda.Device(device_id) #change the device number based on availability of GPUs.

    dev1.use()
############################


############ Calculate domain params ####

## input ###
L = 2*np.pi #Length of the domain 

q = 2 #order of the structure function

Nx = np.shape(Vx)[0]  #no of grid points in each directon

dx = L/Nx

#############################

# define cpu arrays
S_array_cpu = np.zeros([Nx])

# define gpu arrays
if device == "gpu":
    S_array = cp.asarray(S_array_cpu)
    
#############################
#comment this function if device is not cpu

@nb.jit(nopython=True, parallel=True) 
def str_function_cpu(Vx,l_cap_x, l_cap_z, S_array_cpu, q):

       
    N = len(l_cap_x) 

    for m in range(N):
         
        u1, u2 = Vx[0:Nx-Ix[m], 0:Nz-Iz[m]], Vx[Ix[m]:Nx, Iz[m]:Nz]

        del_u = np.abs(u2[:, :] - u1[:, :])
        
        S_array_cpu[Ix[m], Iz[m]] = np.mean(del_u[:, :]**q) # S = < (del u)^2> 

        print (m, Ix[m]*dx, Iz[m]*dz, S_array_cpu[Ix[m], Iz[m]])        

    
    return 

def str_function_gpu(Vx,l_cap_x, l_cap_z, S_array, q):

       
    N = len(l_cap_x) 

    Vx = cp.asarray(Vx) # copy the data on cpu
    print ("GPU copy done")

    cp._core.set_routine_accelerators(['cub', 'cutensor'])
    
    for m in range(N):
         
        u1, u2 = Vx[0:Nx-Ix[m], 0:Nz-Iz[m]], Vx[Ix[m]:Nx, Iz[m]:Nz]

        del_u = cp.abs(u2[:, :] - u1[:, :])   
        
        S_array[Ix[m], Iz[m]] = cp.mean(del_u[:, :]**q) # S = < (del u)^2>


        print (m, Ix[m]*dx, Iz[m]*dz, S_array_cpu[Ix[m], Iz[m]])        

    
    return 


## pre-process
l, Ix, Iz = [], [], []

count = 0

t_pre_process_start = time.time()

for ix in range(Nx):
    for iz in range(Nz):
        
        l_temp = np.sqrt((ix)**2+(iz)**2)
        #if (l_temp*dx > 0.4) and (l_temp*dx < 1.2): ## Upper and lower limit of the length scales for str function ##
        
        l.append(l_temp)

        Ix.append(ix)
        Iz.append(iz)
        count += 1
        
    print (ix, iz)


t_pre_process_stop = time.time()

print("preprocess loop = ", t_pre_process_stop - t_pre_process_start)

print("Total count", count)

l, Ix, Iz = np.asarray(l), np.asarray(Ix), np.asarray(Iz)

l_cap_x, l_cap_z = ((Ix[:])/l[:]), ((Iz[:])/l[:])

l_cap_x[0], l_cap_z[0] = 0, 0



## compute str_function
if device == "gpu":

    t_str_func_start = time.time()

    str_function_gpu(Vx,l_cap_x, l_cap_z, S_array_cpu, q)

    t_str_func_end = time.time()

    print("str func compute time = ", t_str_func_end-t_str_func_start)


else: 

    t_str_func_start = time.time()

    str_function_cpu(Vx,l_cap_x, l_cap_z, S_array, q)

    t_str_func_end = time.time()

    print("str func compute time = ", t_str_func_end-t_str_func_start)



if device == "gpu":
    S_array_cpu = cp.asnumpy(S_array)


## save file
hf = h5py.File("str_function.h5", 'w')
hf.create_dataset("S", data=S_array_cpu)
hf.create_dataset("l", data=l)

hf.close()


