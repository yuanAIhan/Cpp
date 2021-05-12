#include <stdlib.h>
#include <string.h>
#include <time.h>

void sumArrayOnHost(float *A, float *B, float *C, const int N)
{
    for(int idx = 0; idx < N; idx ++) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArrayOnGpu(float *A, float *B, float *C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned int) time(&t));
    for(int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

int main()
{
    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);
    int  flag = 1;
    if(flag)
    {
        float *h_a, *h_b, *h_c;
        h_a = (float*)malloc(nBytes);
        h_b = (float*)malloc(nBytes);
        h_c = (float*)malloc(nBytes);
        initialData(h_a, nElem);
        initialData(h_b, nElem);
        sumArrayOnHost(h_a, h_b, h_c, nElem);
        free(h_a);
        free(h_b);
        free(h_c);
    }
    else 
    {
        float *h_a, *h_b, *h_c;
        h_a = (float*)malloc(nBytes);
        h_b = (float*)malloc(nBytes);
        h_c = (float*)malloc(nBytes);
        initialData(h_a, nElem);
        initialData(h_b, nElem);

        float *d_a, *d_b, *d_c;
        cudaMalloc((float**)&d_a, nBytes);
        cudaMalloc((float**)&d_b, nBytes);
        cudaMalloc((float**)&d_c, nBytes);

        cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
    return 0;
}