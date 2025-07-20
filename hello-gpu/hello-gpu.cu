#include <stdio.h>
void cpu()
{
    printf("hello cpu!\n");
}
__global__ void gpu()
{
    printf("hello gpu!");
}
int main()
{
    cpu();
    gpu<<<1, 1>>>();         // 1， 1分别为 gpu的线程块个数与线程个数
    cudaDeviceSynchronize(); // 同步：cpu等待gpu的代码运行完成
}
// nvcc -arch=sm_89 -o hello-gpu hello-gpu.cu -run 跑起来hello gpu!
