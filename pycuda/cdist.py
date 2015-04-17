# -*- coding: utf-8 -*-

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
import numpy.linalg as la
from pycuda.compiler import SourceModule
import  matplotlib.pyplot as plt
from pycuda import driver, compiler, gpuarray,tools
import scipy.spatial.distance as dist
#import profile
import pycuda.driver as drv

import time


N=5120
M=4
a_cpu=np.random.random((N,M))
b_cpu=np.random.random((N,M))


kernel_code_template="""
            #include<math.h>
            __global__ void gpu_cdist(float *res, float *a, float *b)
            {   
                int i;
                int ix = threadIdx.x+blockDim.x*blockIdx.x;
                int iy = blockIdx.y * blockDim.y + threadIdx.y;
                //int iz = blockIdx.z * blockDim.y + threadIdx.z;
                int nx = %(N)s;
                int ny = %(M)s;

                for(i = 0; i < ny; i++)
                {
                  res[nx*ix+iy]+=pow((a[ny*ix+i] - b[ny*iy+i]),2);
//                res[%(N)s*ix+iy]=sqrt(pow((a[%(M)s*ix]-b[%(M)s*iy]),2) + pow((a[%(M)s*ix+1]-b[%(M)s*iy+1]),2));
                }
                res[nx*ix+iy]=sqrt(res[nx*ix+iy]);
             

//                res[nx*ix+iy]+=iz;//+=pow((a[ny*ix+iz]-b[ny*iy+iz]),2);
            }"""


    
    
    
def gpu_cdist(a_cpu,b_cpu):
    N=np.shape(a_cpu)[0]
    M=np.shape(a_cpu)[1]
        
    block=(32,8,1)
        
        
    kernel_code=kernel_code_template % {
            'M': M,
            'N': N
    }
        
    mod=SourceModule(kernel_code)
    gpu_cdist_=mod.get_function("gpu_cdist")
    
    start=drv.Event()
    end=drv.Event()
    start.record()
    # transfer host (CPU) memory to device (GPU) memory 
    a_gpu = gpuarray.to_gpu(a_cpu.astype(np.float32)) 
    b_gpu = gpuarray.to_gpu(b_cpu.astype(np.float32))
    c_gpu = gpuarray.zeros((N,N), np.float32)# set zero for testing
        
        
    gpu_cdist_(c_gpu,a_gpu,b_gpu,grid=(N/block[0],N/block[1],1),block=block)
    end.record()
    end.synchronize()
        
    gpu_time=start.time_till(end)*1e-3
    print "gpu time: ", gpu_time
        
        
    start_c=time.clock()
    diff=c_gpu.get()-dist.cdist(a_cpu,b_cpu)
    end_c=time.clock()
    cpu_time=end_c-start_c
    print "cpu time: " , cpu_time
    print "speed up: ", cpu_time/gpu_time
    print "============="
    #assert (np.all(abs(diff)) < 1e-6)
    return diff#c_gpu.get()


#for i in range(10):
r=gpu_cdist(a_cpu,b_cpu)
assert (np.all(abs(r) < 1e-6))


