import numpy as np
import h5py
import time

device = "cpu" #default device

import numba as nb  #comment this line if device is not cpu

if device == "gpu":
   import cupy as cp

    
# read the data file ###############
def hdf5_reader(filename,dataset):
    file_read = h5py.File(filename, 'r')
    dataset_read = file_read["/"+dataset]
    V = dataset_read[:,:]

    return V


# input field
#change the names of the input file and datasets according to your convenience
Vx = hdf5_reader("U.V1r.h5", "U.V1r")
Vz = hdf5_reader("U.V3r.h5", "U.V3r")


if device == "gpu":
    # select GPU device
    dev1 = cp.cuda.Device(1) #change the device number based on availability of GPUs.

    dev1.use()
############################


############ Calculate domain params ####

L = 2*np.pi #Length of the domain 

Nx, Nz = np.shape(Vx)[0], np.shape(Vx)[1]  #no of grid points in each directon

dx = L/Nx
dz = L/Nz

#############################

# define cpu arrays
S_upll_array_cpu = np.zeros([Nx//2, Nz//2])
S_u_r_array_cpu = np.zeros([Nx//2, Nz//2])

S_ux_array_cpu = np.zeros([Nx//2, Nz//2])
S_uz_array_cpu = np.zeros([Nx//2, Nz//2])

# define gpu arrays
if device == "gpu":
    S_upll_array = cp.asarray(S_upll_array_cpu)
    S_u_r_array = cp.asarray(S_u_r_array_cpu)


    S_ux_array = cp.asarray(S_ux_array_cpu)
    S_uz_array = cp.asarray(S_uz_array_cpu)


#############################
#comment this function if device is not cpu

@nb.jit(nopython=True, parallel=True)
def str_function_cpu(Vx, Vz, Ix, Iz, l_cap_x, l_cap_z, S_upll_array_cpu, S_u_r_array_cpu, S_ux_array_cpu, S_uz_array_cpu):

       
    N = len(l_cap_x) 

    for m in range(N):
         
        u1, u2 = Vx[0:Nx-Ix[m], 0:Nz-Iz[m]], Vx[Ix[m]:Nx, Iz[m]:Nz]
                
        w1, w2 = Vz[0:Nx-Ix[m], 0:Nz-Iz[m]], Vz[Ix[m]:Nx, Iz[m]:Nz]


        del_u, del_w  = u2[:, :] - u1[:, :], w2[:, :] - w1[:, :]

        diff_magnitude_sqr = (del_u)**2 + (del_w)**2     
        
        S_upll_array_cpu[Ix[m], Iz[m]] = np.mean(diff_magnitude_sqr[:, :]*(del_u[:, :]*l_cap_x[m] + del_w[:, :]*l_cap_z[m])) # S = < (del u)^2 (del upll)>

        S_u_r_array_cpu[Ix[m], Iz[m]] = np.mean((del_u[:, :]*l_cap_x[m] + del_w[:, :]*l_cap_z[m])**3) # S = < (del upll)^3>

        

        S_ux_array_cpu[Ix[m], Iz[m]] = np.mean(diff_magnitude_sqr[:, :]*(del_u[:, :]*l_cap_x[m])) # S = < (del u)^2 (del ux)>

        S_uz_array_cpu[Ix[m], Iz[m]] = np.mean(diff_magnitude_sqr[:, :]*(del_w[:, :]*l_cap_z[m])) # S = < (del u)^2 (del uz)>


        print (m, Ix[m]*dx, Iz[m]*dz)        

    
    return 

def str_function_gpu(Vx, Vz, Ix, Iz, l_cap_x, l_cap_z, S_upll_array, S_u_r_array, S_ux_array, S_uz_array):

       
    N = len(l_cap_x) 

    Vx, Vz = cp.asarray(Vx), cp.asarray(Vz) # copy the data on cpu
    print ("GPU copy done")

    cp._core.set_routine_accelerators(['cub', 'cutensor'])
    
    for m in range(N):
         
        u1, u2 = Vx[0:Nx-Ix[m], 0:Nz-Iz[m]], Vx[Ix[m]:Nx, Iz[m]:Nz]
                
        w1, w2 = Vz[0:Nx-Ix[m], 0:Nz-Iz[m]], Vz[Ix[m]:Nx, Iz[m]:Nz]


        del_u, del_w  = u2[:, :] - u1[:, :], w2[:, :] - w1[:, :]

        diff_magnitude_sqr = (del_u)**2 + (del_w)**2     
        
        S_upll_array[Ix[m], Iz[m]] = cp.mean(diff_magnitude_sqr[:, :]*(del_u[:, :]*l_cap_x[m] + del_w[:, :]*l_cap_z[m])) # S = < (del u)^2 (del upll)>

        S_u_r_array[Ix[m], Iz[m]] = cp.mean((del_u[:, :]*l_cap_x[m] + del_w[:, :]*l_cap_z[m])**3) # S = < (del upll)^3>

        

        S_ux_array[Ix[m], Iz[m]] = cp.mean(diff_magnitude_sqr[:, :]*(del_u[:, :]*l_cap_x[m])) # S = < (del u)^2 (del ux)>

        S_uz_array[Ix[m], Iz[m]] = cp.mean(diff_magnitude_sqr[:, :]*(del_w[:, :]*l_cap_z[m])) # S = < (del u)^2 (del uz)>


        print (m, Ix[m]*dx, Iz[m]*dz)        

    
    return 


## pre-process
l, Ix, Iz = [], [], []

count = 0

t_pre_process_start = time.time()

for ix in range(Nx//2):
    for iz in range(Nz//2):
        
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

    str_function_gpu(Vx, Vz, Ix, Iz, l_cap_x, l_cap_z, S_upll_array, S_u_r_array, S_ux_array, S_uz_array)

    t_str_func_end = time.time()

    print("str func compute time = ", t_str_func_end-t_str_func_start)


else: 

    t_str_func_start = time.time()

    str_function_cpu(Vx, Vz, Ix, Iz, l_cap_x, l_cap_z, S_upll_array, S_u_r_array, S_ux_array, S_uz_array)

    t_str_func_end = time.time()

    print("str func compute time = ", t_str_func_end-t_str_func_start)



if device == "gpu":
    S_upll_array_cpu = cp.asnumpy(S_upll_array)
    S_u_r_array_cpu = cp.asnumpy(S_u_r_array)

    S_ux_array_cpu = cp.asnumpy(S_ux_array)
    S_uz_array_cpu = cp.asnumpy(S_uz_array)



## save file
hf = h5py.File("str_function.h5", 'w')
hf.create_dataset("S_upll", data=S_upll_array_cpu)
hf.create_dataset("S_u_r", data=S_u_r_array_cpu)
hf.create_dataset("S_ux", data=S_ux_array_cpu)
hf.create_dataset("S_uz", data=S_uz_array_cpu)

hf.close()


