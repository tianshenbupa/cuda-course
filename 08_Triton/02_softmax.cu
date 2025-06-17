/*
===========================================================
CPU / GPU Softmax 对比实验（含 CUDA Kernel）
-----------------------------------------------------------
功能概要：
1. 在 CPU 上用数值稳定版 softmax 处理一行向量；
2. 在 GPU 上为一个批次 (B 行，每行 N 元素) 计算 softmax；
3. 将 GPU 结果拷回主机，与 CPU 结果（仅首行）做差异比较；
4. 输出 CPU 与 GPU 结果的最大绝对误差。

⚠️ 代码演示以易读为主，GPU kernel 采用「逐行重复计算」实现，
算法正确但**不具备高性能**——每个线程都会为所在行重复扫描 N 次。
如需优化，可改用共享内存归约或 Triton / Warp‑wide Primitives。

编译示例：
    nvcc -o softmax_cmp softmax_cmp.cu
===========================================================
*/

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdlib.h>

/* ------------------ CUDA Kernel：逐行 softmax ------------------
   输入  : input  (B×N)
   输出  : output (B×N)
   每个线程块负责一行内的一段元素 (tid)，但本示例中
   为简化起见，每个线程都对整行执行求 max / 求和操作。*/
__global__ void softmax_cuda(float* input,   // ➜ 输入张量
                             float* output,  // ➜ 输出张量
                             int B,          // ➜ 批次行数
                             int N) {        // ➜ 每行长度
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 行内列索引
    int bid = blockIdx.y;                            // 行号 (batch id)

    if (tid < N && bid < B) {
        int offset = bid * N;            // 该行在一维数组中的起始位置

        /* 1️⃣ 计算该行最大值 (数值稳定) */
        float max_val = input[offset];   // 先取行首为初值
        for (int i = 1; i < N; i++) {
            max_val = fmaxf(max_val, input[offset + i]);
        }

        /* 2️⃣ 计算 e^(x_i - max) 的总和 */
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += expf(input[offset + i] - max_val);
        }

        /* 3️⃣ 写回 softmax 结果 */
        for (int i = 0; i < N; i++) {
            output[offset + i] = expf(input[offset + i] - max_val) / sum;
        }
    }
}

/* ------------------ CPU 版本 softmax（单行） ------------------ */
void softmax_cpu(float *x, int N) {
    float max_val = x[0];
    for (int i = 1; i < N; i++) {                  // 找最大值
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {                  // 计算 e^(x-max)
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < N; i++) x[i] /= sum;       // 归一化
}

int main() {
    const int B = 32;     // 批次大小 (行数)
    const int N = 1024;   // 每行元素个数

    /* --------------- 主机端内存分配与初始化 --------------- */
    float *x_cpu = (float*)malloc(B * N * sizeof(float)); // 原始数据
    float *x_gpu = (float*)malloc(B * N * sizeof(float)); // GPU 结果

    for (int i = 0; i < B * N; i++) {
        x_cpu[i] = (float)rand() / RAND_MAX;  // 0~1 随机数
    }

    /* --------------- 设备端内存分配 --------------- */
    float *d_input, *d_output;
    cudaMalloc(&d_input,  B * N * sizeof(float));
    cudaMalloc(&d_output, B * N * sizeof(float));

    /* --------------- 数据拷贝：Host ➜ Device --------------- */
    cudaMemcpy(d_input, x_cpu, B * N * sizeof(float), cudaMemcpyHostToDevice);

    /* --------------- 启动 CUDA Kernel --------------- */
    int threadsPerBlock = 256;                         // 每块 256 线程
    int blocksPerGrid_x = (N + threadsPerBlock - 1) / threadsPerBlock;
    dim3 gridDim(blocksPerGrid_x, B);                  // (列块数, 行数)
    softmax_cuda<<<gridDim, threadsPerBlock>>>(d_input, d_output, B, N);
    cudaDeviceSynchronize();

    /* --------------- 结果拷回：Device ➜ Host --------------- */
    cudaMemcpy(x_gpu, d_output, B * N * sizeof(float), cudaMemcpyDeviceToHost);

    /* --------------- CPU 上计算首行 softmax 作对照 --------------- */
    softmax_cpu(x_cpu, N);

    /* --------------- 误差比较（首行） --------------- */
    float max_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(x_cpu[i] - x_gpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("CPU 与 GPU 结果最大绝对误差 (首行): %e\n", max_diff);

    /* --------------- 资源释放 --------------- */
    free(x_cpu);
    free(x_gpu);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
