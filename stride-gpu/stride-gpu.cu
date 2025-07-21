// gridDim.x表示当前网格有几个block
// blockIdx.x表示当前第几个线程块，
// blockDim.x表示当前线程块内的线程数量
// threadIdx.x表示当前是第几个线程
#include <stdio.h>
#include <stdlib.h>

void cpu(int *a, int N)
{
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
    }
}
__global__ void gpu(int *a, int N)
{
    int threadi = blockIdx.x * blockDim.x + threadIdx.x; // 初始位置
    int stride = gridDim.x * blockDim.x;                 // 跨步大小
    for (int i = threadi; i < N; i += stride)
    {
        a[i] *= 2;
    }
}

bool check_cpu(int *a, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (a[i] != i)
            return false;
    }
    return true;
}
bool check_gpu(int *a, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (a[i] != i * 2)
            return false;
    }
    return true;
}
int main()
{
    const int N = 1000;
    size_t size = N * sizeof(int);
    int *a;
    cudaMallocManaged(&a, size); // 统一分配内存，可以被gpu使用也可以被cpu使用
    cpu(a, N);

    check_cpu(a, N) ? printf("cpu ok\n") : printf("cpu error\n");

    size_t threads = 256;
    size_t blocks = 1;
    gpu<<<blocks, threads>>>(a, N);
    cudaDeviceSynchronize();

    check_gpu(a, N) ? printf("gpu ok") : printf("gpu error");

    cudaFree(a);
}