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

    int id;
    cudaGetDevice(&id);

    cpu(a, N); // 初始在cpu上

    cudaMemPrefetchAsync(a, size, id); // 异步预取到gpu上，为后续的gpu()操作做准备，需要传入gpu的设备id

    size_t threads = 256;
    size_t blocks = 1;

    gpu<<<blocks, threads>>>(a, N);

    cudaMemPrefetchAsync(a, size, cudaCpuDeviceId); // 异步预取到cpu上，为的是后续的checkCuda函数，checkCuda函数是在cpu上进行，这时需传入cpu的设备id，直接传cudaCpuDeviceId参数即可，不需要函数获取id
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error gpu: %s\n", cudaGetErrorString(err));
    }

    checkCuda(cudaDeviceSynchronize());

    cudaFree(a);
}