#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
void cpu(int *a, int N)
{
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
    }
}
__global__ void gpu(int *a, int N)
{
    int threadi = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = threadi; i < N; i += stride)
    {
        a[i] *= 2;
    }
}

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA runtime error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}
int main()
{
    const int N = 1000;
    size_t size = N * sizeof(int);
    int *a;

    cudaError_t err;
    err = cudaMallocManaged(&a, size);
    if (err != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    cpu(a, N);

    size_t threads = 256;
    size_t blocks = 1;
    gpu<<<blocks, -1>>>(a, N);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error gpu: %s\n", cudaGetErrorString(err));
    }

    checkCuda(cudaDeviceSynchronize());

    cudaFree(a);
}