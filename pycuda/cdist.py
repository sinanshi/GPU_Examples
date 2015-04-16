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


mod = SourceModule("""
        #include<math.h>
__global__ void gpu_cdist(float *res, float *a, float *b)
{
          int ix = threadIdx.x+blockDim.x*blockIdx.x;
          int iy = blockIdx.y * blockDim.y + threadIdx.y;
        res[5*ix+iy]=sqrt(pow((a[2*ix]-b[2*iy]),2) + pow((a[2*ix+1]-b[2*iy+1]),2));
}
""")


N=5

a_cpu=np.indices((N,2))[1]
b_cpu=np.indices((N,2))[0]


# transfer host (CPU) memory to device (GPU) memory 
a_gpu = gpuarray.to_gpu(a_cpu.astype(np.float32)) 
b_gpu = gpuarray.to_gpu(b_cpu.astype(np.float32))

# create empty gpu array for the result (C = A * B)
c_gpu = gpuarray.zeros((N,N), np.float32)

gpu_cdist=mod.get_function("gpu_cdist")


gpu_cdist(c_gpu,a_gpu,b_gpu,block=(N,N,1))


diff=c_gpu.get()-dist.cdist(a_cpu,b_cpu)

print c_gpu

print diff

assert (np.any(abs(diff)) > 1e-6)


