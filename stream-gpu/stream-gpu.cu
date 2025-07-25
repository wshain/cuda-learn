const int N = 1 << 20;
__global__ void kernel(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x)
    {
        x[i] = sqrt(pow(3.14159, i));
    }
}
int main()
{
    const int num_streams = 8;
    cudaStream_t streams[num_streams];
    float *data[num_streams];

    for (int i = 0; i < num_streams; i++)
    {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&data[i], N * sizeof(float));
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N);
        kernel<<<1, 1>>>(0, 0); // 默认流会阻塞
        // 注意这里不要加cudaDeviceSynchronize();加了每步循环cpu都会阻塞等待gpu完成，看不到流并行
    }
    cudaDeviceReset();
    return 0;
}