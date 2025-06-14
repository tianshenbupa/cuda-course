// ============================================================================
//  文件名：cublasXt_simple_sgemm_chinese.cu
//  作者：ChatGPT
//  功能：演示如何使用 **cuBLAS‑Xt** 在单块 GPU 上完成单精度 (FP32) GEMM，
//        并与 CPU 朴素实现结果进行对比。
//        ➤ 默认矩阵尺寸为 256×256 × 256×256（1024/4），完全满足 cuBLAS‑Xt
//          对 m、n、k 必须为 4 的倍数的要求。
//  亮点：
//  1. 仅依赖 cublasXtSgemm，接口与经典 cublasSgemm 基本一致；
//  2. 提供 `CHECK_CUBLAS` 宏统一错误处理；
//  3. 随机初始化矩阵并逐元素验证 GPU / CPU 结果；
//  4. 代码结构极简，便于初学者理解。
//  编译示例：
//      nvcc -std=c++17 -lcublas -lcublasXt cublasXt_simple_sgemm_chinese.cu -o xt_sgemm
// ============================================================================

#include <cublasXt.h>        // cuBLAS‑Xt 头文件
#include <cublas_v2.h>       // 仅用于 error code 定义
#include <cuda_runtime.h>    // CUDA Runtime API
#include <iostream>
#include <cstdlib>           // rand / srand
#include <ctime>             // time
#include <cmath>             // fabsf

// ----------------------------
// (1) 矩阵维度（确保都是 4 的倍数）
// ----------------------------
const int M = 1024 / 4;  // 256
const int N = 1024 / 4;  // 256
const int K = 1024 / 4;  // 256

// ----------------------------
// (2) cuBLAS‑Xt 错误检查宏
// ----------------------------
#define CHECK_CUBLAS(call)                                                        \
    do {                                                                          \
        cublasStatus_t err = (call);                                              \
        if (err != CUBLAS_STATUS_SUCCESS) {                                       \
            std::cerr << "cuBLAS error in " << #call << " at line " << __LINE__  \
                      << ", status = " << err << std::endl;                       \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

int main() {
    // ---------------------------------------------------------------------
    // 1. 初始化随机种子（便于生成随机矩阵）
    // ---------------------------------------------------------------------
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // ---------------------------------------------------------------------
    // 2. 在主机端分配并初始化 A、B、C
    // ---------------------------------------------------------------------
    float *A_host = new float[M * K];   // A: (M × K)
    float *B_host = new float[K * N];   // B: (K × N)
    float *C_cpu  = new float[M * N];   // CPU 结果
    float *C_gpu  = new float[M * N];   // GPU 结果 (cuBLAS‑Xt)

    // 填充随机数（0 ~ 1）
    for (int i = 0; i < M * K; ++i) A_host[i] = static_cast<float>(std::rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) B_host[i] = static_cast<float>(std::rand()) / RAND_MAX;

    // ---------------------------------------------------------------------
    // 3. 朴素 CPU GEMM：C_cpu = A_host × B_host
    // ---------------------------------------------------------------------
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A_host[i * K + k] * B_host[k * N + j];
            C_cpu[i * N + j] = sum * alpha + beta * 0.0f; // 此处 beta=0，直接赋值
        }
    }

    // ---------------------------------------------------------------------
    // 4. 创建 cuBLAS‑Xt 句柄并选择设备（此例仅 1 块 GPU，device 0）
    // ---------------------------------------------------------------------
    cublasXtHandle_t xtHandle;
    CHECK_CUBLAS(cublasXtCreate(&xtHandle));

    int devices[1] = {0}; // 使用第 0 号 GPU
    CHECK_CUBLAS(cublasXtDeviceSelect(xtHandle, 1, devices));

    // ---------------------------------------------------------------------
    // 5. 调用 cuBLAS‑Xt SGEMM（列主序约定：C = alpha × (B × A) + beta × C）
    //    注意：cublasXtSgemm 的参数顺序与 cublasSgemm 类似，但直接传递主机指针；
    // ---------------------------------------------------------------------
    CHECK_CUBLAS(cublasXtSgemm(
        xtHandle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N,   // 第 1 维（列主序中的行）
        M,   // 第 2 维（列主序中的列）
        K,
        &alpha,
        B_host, N,   // B (N × K)
        A_host, K,   // A (K × M)
        &beta,
        C_gpu, N));  // C (N × M)

    // ---------------------------------------------------------------------
    // 6. 结果验证：逐元素比较 CPU / GPU
    // ---------------------------------------------------------------------
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = fabsf(C_cpu[i] - C_gpu[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "Max abs diff between CPU and cuBLAS‑Xt results = "
              << max_diff << std::endl;

    // ---------------------------------------------------------------------
    // 7. 资源释放
    // ---------------------------------------------------------------------
    CHECK_CUBLAS(cublasXtDestroy(xtHandle));

    delete[] A_host;
    delete[] B_host;
    delete[] C_cpu;
    delete[] C_gpu;

    return 0;
}
