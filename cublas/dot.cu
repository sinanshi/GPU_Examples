/*
   call cublas<S>dot function and compare CPU and GPU time
   Sinan SHI
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
//#include <helper_cuda.h>
#define N 100000000

int cpu_dot(const float *x, const float *y, float *z){
    int i;
    z[0]=0;
    for(i=0;i<N;i++){
        z[0]=z[0]+x[i]*y[i];
    }
    return(0);
}



int check(cublasStatus_t status){
    if (status != CUBLAS_STATUS_SUCCESS)
    {
    
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    
    }
    return(0);
}
    

/*Main function*/
int main(int argc, char **argv)
{



    clock_t start,diff;
    cublasHandle_t handle;
    cublasStatus_t status;
    float *x;
    float *y;
    float *z,*zg;
    float *d_x, *d_y, *d_z;
    int i;

    status = cublasCreate(&handle);
    x=(float *)malloc(N * sizeof(x[0]));
    y=(float *)malloc(N * sizeof(y[0]));
    z=(float *)malloc(sizeof(float));
    zg=(float *)malloc(  sizeof(float));


    cudaMalloc((void **)&d_x, N * sizeof(float));
    cudaMalloc((void **)&d_y, N * sizeof(float));
    cudaMalloc((void **)&d_z, sizeof(float));


    /*initialise*/
    for(i=0;i<N;i++){
        x[i]=0.001;
        y[i]=0.001;
        zg[0]=-9999;
        z[0]=-9999;
    }

    
    /*CPU dot product*/
    start=clock();
    cpu_dot(x,y,z);
    diff=clock()-start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken (CPU) %d seconds %d milliseconds\n", msec/1000, msec%1000);
    


    /*GPU dot product*/
    start=clock();
    cublasSetVector(N,sizeof(float),x,1,d_x,1);
    cublasSetVector(N,sizeof(float),y,1,d_y,1);
    check(cublasSdot(handle,N,d_x,1,d_y,1,zg));
    diff=clock()-start;
    msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken (GPU) %d seconds %d milliseconds\n", msec/1000, msec%1000);
    

    /*check result*/
    printf("%f - %f\n",zg[0],z[0]);





    free(x);
    free(y);
    free(z);
    free(zg);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
}
