#include <stdio.h>

int main()
{
    int id;
    cudaGetDevice(&id);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, id);
    printf("device id: %d \n sms: %d \n capability major: %d \n capability minor: %d \n warp size: %d \n", id, props.multiProcessorCount, props.major, props.minor, props.warpSize);
}

// device id 设备id
// sms 处理器的个数
// （算力）架构：capabilities major整数范围 capabilities minor小数范围
// warp size ：warp大小
