//   .->MY
//  |
//  \/MX
#include <stdio.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define MX 2048
#define MY 2048

// native 版矩阵转置
__global__ void transpose(float *odata, float *idata)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    if (x >= MX || y >= MY)
        return;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[x * width + (y + j)] = idata[(y + j) * width + x];
}

// 共享内存替换，带宽基本没有提升，因为要跨步拷贝
__global__ void shared_memory_transpose(float *odata, float *idata)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // 存储块冲突优化

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    if (x >= MX || y >= MY)
        return;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
    __syncthreads();
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[x * width + y + j] = tile[threadIdx.y + j][threadIdx.x];
}

// 共享内存转置
__global__ void shared_memory_tr_transpose(float *odata, float *idata)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    if (x >= MX || y >= MY)
        return;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
    __syncthreads();
    // 块位置交换,内部直接复制
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j]; // shared-mem转置
}

bool check(float *h_odata, float *res)
{
    for (int r = 0; r < MX; r++)
        for (int c = 0; c < MY; c++)
        {
            if (h_odata[r * MY + c] != res[r * MY + c])
                return false;
        }
    return true;
}
int main()
{
    size_t size = MX * MY * sizeof(float);
    float *h_idata, *h_odata, *d_idata, *d_odata, *res;
    cudaMallocHost(&h_idata, size);
    cudaMallocHost(&h_odata, size);
    cudaMallocHost(&res, size);
    cudaMalloc(&d_idata, size);
    cudaMalloc(&d_odata, size);

    dim3 threads(TILE_DIM, BLOCK_ROWS, 1);
    // 上取整
    dim3 blocks((MX + TILE_DIM - 1) / TILE_DIM, (MY + TILE_DIM - 1) / TILE_DIM, 1);

    for (int r = 0; r < MX; r++)
        for (int c = 0; c < MY; c++)
        {
            h_idata[r * MY + c] = r * MY + c;
            res[r * MY + c] = c * MX + r;
        }

    cudaEvent_t startEvent, stopEvent;
    float ms;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice);

    cudaEventRecord(startEvent, 0);

    shared_memory_tr_transpose<<<blocks, threads>>>(d_odata, d_idata);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);

    printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");
    printf("%25s", "native transpose");
    // 1GB = 1e3MB = 1e6KB = 1e9B学到了
    printf("%20.2f\n", 2 * MX * MY * sizeof(float) * 1e-9 / (ms / 1000));
    cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    check(h_odata, res) ? printf("ok\n") : printf("error\n");

    cudaFreeHost(h_idata);
    cudaFreeHost(h_odata);
    cudaFreeHost(res);
    cudaFree(d_idata);
    cudaFree(d_odata);
}