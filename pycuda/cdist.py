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
mod = SourceModule("""
        #include<math.h>
__global__ void gpu_cdist(float *res, float *a, float *b)
{
          int ix = threadIdx.x+blockDim.x*blockIdx.x;
          int iy = blockIdx.y * blockDim.y + threadIdx.y;
          res[5120*2*ix+iy]=sqrt(pow((a[2*ix]-b[2*iy]),2) + pow((a[2*ix+1]-b[2*iy+1]),2));
}
""")


N=5120*2

a_cpu=np.indices((N,2))[1]
b_cpu=np.indices((N,2))[0]



start=drv.Event()
end=drv.Event()

start.record()
# transfer host (CPU) memory to device (GPU) memory 
a_gpu = gpuarray.to_gpu(a_cpu.astype(np.float32)) 
b_gpu = gpuarray.to_gpu(b_cpu.astype(np.float32))

# create empty gpu array for the result (C = A * B)
c_gpu = gpuarray.zeros((N,N), np.float32)
gpu_cdist=mod.get_function("gpu_cdist")
gpu_cdist(c_gpu,a_gpu,b_gpu,grid=(N/256,N/2,1),block=(256,2,1))
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



assert (np.all(abs(diff)) < 1e-6)


