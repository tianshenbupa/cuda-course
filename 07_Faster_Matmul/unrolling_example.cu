/*
==========================================================
CUDA 向量加法性能对比实验：带循环展开 vs 不带循环展开
----------------------------------------------------------
本程序用于评估 `#pragma unroll` 指令对 GPU kernel 性能的影响。

主要内容：
1. 定义两个 CUDA kernel：
   - `vectorAddNoUnroll`: 普通版本，不使用循环展开。
   - `vectorAddUnroll`: 使用 `#pragma unroll` 展开循环。

2. 使用统一输入数组（a=1.0，b=2.0），执行 LOOP_COUNT 次累加操作。

3. 通过 CUDA Events 计时，分别比较展开与未展开版本的执行时间。

4. 运行多轮热身和基准测试，并输出平均耗时。

5. 验证计算正确性。

适用目的：
- 学习循环展开对计算密集型 kernel 的影响。
- 熟悉 GPU kernel 性能评估方法（使用 CUDA Events）。
==========================================================
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10000000               // 元素总数
#define THREADS_PER_BLOCK 256    // 每个线程块中的线程数
#define LOOP_COUNT 100           // 每个线程重复执行的累加次数
#define WARMUP_RUNS 5            // 热身运行次数（不计入时间）
#define BENCH_RUNS 10            // 计时运行次数

// ===========================
// 不使用循环展开的 kernel
// ===========================
__global__ void vectorAddNoUnroll(float *a, float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        float sum = 0;
        for (int j = 0; j < LOOP_COUNT; j++) {
            sum += a[tid] + b[tid];
        }
        c[tid] = sum;
    }
}

// ===========================
// 使用 #pragma unroll 的 kernel
// ===========================
__global__ void vectorAddUnroll(float *a, float *b, float *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        float sum = 0;
        #pragma unroll
        for (int j = 0; j < LOOP_COUNT; j++) {
            sum += a[tid] + b[tid];
        }
        c[tid] = sum;
    }
}

// ===========================
// 验证计算结果是否正确
// ===========================
bool verifyResults(float *c, int n) {
    float expected = (1.0f + 2.0f) * LOOP_COUNT;
    for (int i = 0; i < n; i++) {
        if (abs(c[i] - expected) > 1e-5) {
            return false;
        }
    }
    return true;
}

// ===========================
// 启动 kernel 并计时（使用 CUDA Events）
// ===========================
float runKernel(void (*kernel)(float*, float*, float*, int), float *d_a, float *d_b, float *d_c, int n) {
    int numBlocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEvent_t start, stop;
    float milliseconds;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    kernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

// ===========================
// 主函数
// ===========================
int main() {
    float *a, *b, *c;           // 主机内存
    float *d_a, *d_b, *d_c;     // 设备内存
    size_t size = N * sizeof(float);

    // 分配主机内存
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);

    // 初始化主机数组：a=1.0，b=2.0
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // 分配设备内存
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 拷贝输入数据到设备
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // 热身运行（避免首次运行的启动开销干扰计时）
    for (int i = 0; i < WARMUP_RUNS; i++) {
        runKernel(vectorAddNoUnroll, d_a, d_b, d_c, N);
        runKernel(vectorAddUnroll, d_a, d_b, d_c, N);
    }

    // 正式 benchmark：统计两种 kernel 平均运行时间
    float totalTimeNoUnroll = 0, totalTimeUnroll = 0;
    for (int i = 0; i < BENCH_RUNS; i++) {
        totalTimeNoUnroll += runKernel(vectorAddNoUnroll, d_a, d_b, d_c, N);
        totalTimeUnroll += runKernel(vectorAddUnroll, d_a, d_b, d_c, N);
    }

    // 计算平均时间
    float avgTimeNoUnroll = totalTimeNoUnroll / BENCH_RUNS;
    float avgTimeUnroll = totalTimeUnroll / BENCH_RUNS;

    // 输出结果
    printf("不使用循环展开的平均执行时间: %f ms\n", avgTimeNoUnroll);
    printf("使用循环展开的平均执行时间:   %f ms\n", avgTimeUnroll);

    // 拷贝输出数据到主机并验证
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    if (verifyResults(c, N)) {
        printf("计算结果正确 ✅\n");
    } else {
        printf("计算结果错误 ❌\n");
    }

    // 释放内存
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
