#include <stdio.h>
#define N 64

__global__ void gpu(int *a, int *b, int *c_gpu)
{
    int r = blockDim.x * blockIdx.x + threadIdx.x;
    int c = blockDim.y * blockIdx.y + threadIdx.y;
    if (r < N && c < N)
    {
        c_gpu[r * N + c] = a[r * N + c] + b[r * N + c];
    }
}

void cpu(int *a, int *b, int *c_cpu)
{
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++)
        {
            c_cpu[r * N + c] = a[r * N + c] + b[r * N + c];
        }
}

bool check(int *c_cpu, int *c_gpu)
{
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++)
        {
            if (c_cpu[r * N + c] != c_gpu[r * N + c])
                return false;
        }
    return true;
}
int main()
{
    int *a_cpu, *b_cpu, *a_gpu, *b_gpu, *c_cpu, *c_gpu, *c_gpu_cpu;
    size_t size = N * N * sizeof(int);

    cudaMallocHost(&a_cpu, size);
    cudaMallocHost(&b_cpu, size);
    cudaMalloc(&a_gpu, size);
    cudaMalloc(&b_gpu, size);
    cudaMallocHost(&c_cpu, size);
    cudaMallocHost(&c_gpu_cpu, size);
    cudaMalloc(&c_gpu, size);

    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++)
        {
            a_cpu[r * N + c] = r;
            b_cpu[r * N + c] = c;
            c_cpu[r * N + c] = 0;
            // c_gpu[r * N + c] = 0;
            c_gpu_cpu[r * N + c] = 0;
        }

    cpu(a_cpu, b_cpu, c_cpu);

    dim3 threads(16, 16, 1);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y, 1);

    cudaMemcpy(a_gpu, a_cpu, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(b_gpu, b_cpu, size, cudaMemcpyDeviceToHost);

    gpu<<<blocks, threads>>>(a_gpu, b_gpu, c_gpu);
    cudaDeviceSynchronize();

    cudaMemcpy(c_gpu_cpu, c_gpu, size, cudaMemcpyDeviceToHost); // 把cudamalloc() 申请的数据从c_gpu 拷贝到c_gpu_cpu, 因为cudamalloc申请的内存只能gpu使用，cpu无法访问指针

    check(c_cpu, c_gpu_cpu) ? printf("ok") : printf("error");

    cudaFreeHost(a_cpu);
    cudaFreeHost(b_cpu);
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFreeHost(c_cpu);
    cudaFreeHost(c_gpu_cpu);
    cudaFree(c_gpu);
}