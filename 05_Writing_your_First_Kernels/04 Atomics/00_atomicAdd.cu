#include <cuda_runtime.h>
#include <stdio.h>

#define NUM_THREADS 1000   // 每个 block 中的线程数
#define NUM_BLOCKS 1000    // block 的数量

// 非原子操作的核函数（结果可能不正确）
__global__ void incrementCounterNonAtomic(int* counter) {
    // 未加锁读取旧值
    int old = *counter;
    int new_value = old + 1;
    // 未解锁直接写入新值
    *counter = new_value;
}

// 使用原子操作的核函数（结果正确）
__global__ void incrementCounterAtomic(int* counter) {
    // 原子加操作，确保并发安全
    int a = atomicAdd(counter, 1);
}

int main() {
    int h_counterNonAtomic = 0;  // 主机上的非原子计数器初始值
    int h_counterAtomic = 0;     // 主机上的原子计数器初始值
    int *d_counterNonAtomic, *d_counterAtomic;

    // 分配设备内存
    cudaMalloc((void**)&d_counterNonAtomic, sizeof(int));
    cudaMalloc((void**)&d_counterAtomic, sizeof(int));

    // 将初始计数器值从主机复制到设备
    cudaMemcpy(d_counterNonAtomic, &h_counterNonAtomic, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counterAtomic, &h_counterAtomic, sizeof(int), cudaMemcpyHostToDevice);

    // 启动核函数
    incrementCounterNonAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterNonAtomic);
    incrementCounterAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterAtomic);

    // 将结果从设备复制回主机
    cudaMemcpy(&h_counterNonAtomic, d_counterNonAtomic, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_counterAtomic, d_counterAtomic, sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("非原子操作计数器值: %d\n", h_counterNonAtomic);
    printf("原子操作计数器值: %d\n", h_counterAtomic);

    // 释放设备内存
    cudaFree(d_counterNonAtomic);
    cudaFree(d_counterAtomic);

    return 0;
}
