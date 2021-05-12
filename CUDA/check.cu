#include <cuda_runtime.h>
#include <stdio.h>


#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if(error != CudaSuccess) \
    { \
        printf("Error:%s:%d,", __FILE__, __LINE__);\
        printf("Code:%d, reason: %s \n", error, cudaGetErrorString(error));\
        exit(1);\
    }\
}

__global__ void chechIndex(void)
{
    printf("threadIdx:(%d, %d, %d) blockIdx:(%d,%d,%d) blockdim:(%d,%d,%d)"
            "gridDim:(%d,%d,%d)\n", threadIdx.x, threadIdx.y, threadIdx.z,
            blockIdx.x,blockIdx.y,blockIdx.z, blockDim.x,blockDim.y,blockDim.z,
        gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char **argv)
{
    int nElem = 6;
    dim3 block(3);
    dim3 grid((nElem + block.x - 1) / block.x);
    printf("grid.x %d  gird.y  %d  grid.z %d \n", grid.x, grid.y, grid.z);
    printf("block.x %d  block.y  %d  block.z %d \n", block.x, block.y, block.z);

    chechIndex<<<grid, block>>>();
    cudaDeviceReset();
    return 0;

    // kernel_function <<<4, 8>>>(argument list);
}

