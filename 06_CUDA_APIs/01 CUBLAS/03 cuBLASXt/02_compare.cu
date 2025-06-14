// ============================================================================
//  文件名：cublas_vs_cublasXt_benchmark.cu
//  作者：ChatGPT
//  功能：基准测试 **cuBLAS (cublasSgemm)** 与 **cuBLAS‑Xt (cublasXtSgemm)**
//        在超大规模单精度 (FP32) GEMM（16384×16384）上的平均运行时间，
//        并逐元素比较两者计算结果的相对误差。
//  亮点：
//  1. 使用 `std::chrono` 进行多次计时，统计平均耗时；
//  2. 直接向 `cublasXtSgemm` 传递主机指针，演示其自动分配 / 拷贝特性；
//  3. 自定义误差比较函数，输出首个不匹配元素的信息；
//  4. 关键步骤均配有中文注释，适合快速上手 cuBLAS‑Xt 性能测试；
//  5. 默认矩阵维度 16384×16384，约占用 4 GB 显存，可根据 GPU 调整 M/N/K。
//
//  编译示例：
//      nvcc -std=c++17 -lcublas -lcublasXt cublas_vs_cublasXt_benchmark.cu -o gemm_bench
// ============================================================================

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasXt.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>   // std::max
#include <cmath>       // std::abs
#include <cstdlib>     // rand / srand
#include <ctime>       // time

// ----------------------------
// (1) 简易错误检查宏
// ----------------------------
#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)               \
                      << ", line " << __LINE__ << std::endl;                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

#define CHECK_CUBLAS(call)                                                        \
    do {                                                                          \
        cublasStatus_t status = (call);                                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                    \
            std::cerr << "cuBLAS error: " << status                              \
                      << ", line " << __LINE__ << std::endl;                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// ----------------------------
// (2) 初始化矩阵为 0~1 随机数
// ----------------------------
void initMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
}

// ----------------------------
// (3) 结果比较函数：相对误差 > tol 则认为不匹配
// ----------------------------
bool compareResults(const float* ref, const float* test, int size, float tol) {
    for (int i = 0; i < size; ++i) {
        float diff = std::abs(ref[i] - test[i]);
        float denom = std::max(std::abs(ref[i]), std::abs(test[i]));
        if (denom == 0.0f) denom = 1.0f; // 避免除 0
        if (diff / denom > tol) {
            std::cout << "Mismatch @ index " << i
                      << " | ref=" << ref[i] << ", test=" << test[i]
                      << ", rel_err=" << diff / denom << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // ---------------------------------------------------------------------
    // 0. 设置随机种子 & 定义 GEMM 尺寸 (可按需调小)
    // ---------------------------------------------------------------------
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    const int M = 16384; // A: (M×K)  C: (M×N)
    const int N = 16384; // B: (K×N)  C: (M×N)
    const int K = 16384;

    std::cout << "GEMM size: " << M << "×" << K << "  *  " << K << "×" << N << std::endl;

    size_t bytes_A = static_cast<size_t>(M) * K * sizeof(float);
    size_t bytes_B = static_cast<size_t>(K) * N * sizeof(float);
    size_t bytes_C = static_cast<size_t>(M) * N * sizeof(float);

    // ---------------------------------------------------------------------
    // 1. 主机端分配 / 初始化
    // ---------------------------------------------------------------------
    float* h_A          = static_cast<float*>(malloc(bytes_A));
    float* h_B          = static_cast<float*>(malloc(bytes_B));
    float* h_C_cublas   = static_cast<float*>(malloc(bytes_C));
    float* h_C_cublasXt = static_cast<float*>(malloc(bytes_C));

    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    const int num_runs = 5;          // benchmark 次数
    std::vector<double> t_cublas(num_runs), t_xt(num_runs);

    const float alpha = 1.0f, beta = 0.0f;

    // =====================================================================
    // 2. cuBLAS (单 GPU) benchmark
    // =====================================================================
    {
        cublasHandle_t handle;           CHECK_CUBLAS(cublasCreate(&handle));

        // 2.1 设备内存
        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
        CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
        CHECK_CUDA(cudaMalloc(&d_C, bytes_C));

        CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

        // 2.2 warm‑up
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 d_B, N,
                                 d_A, K,
                                 &beta,
                                 d_C, N));
        CHECK_CUDA(cudaDeviceSynchronize());

        // 2.3 benchmark (多次)
        for (int i = 0; i < num_runs; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K,
                                     &alpha,
                                     d_B, N,
                                     d_A, K,
                                     &beta,
                                     d_C, N));
            CHECK_CUDA(cudaDeviceSynchronize());
            auto t1 = std::chrono::high_resolution_clock::now();
            t_cublas[i] = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "cuBLAS   run " << i + 1 << ": " << t_cublas[i] << " s" << std::endl;
        }

        CHECK_CUDA(cudaMemcpy(h_C_cublas, d_C, bytes_C, cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        CHECK_CUBLAS(cublasDestroy(handle));
    }

    // =====================================================================
    // 3. cuBLAS‑Xt benchmark（仍只用 1 卡，展示 API 调用）
    // =====================================================================
    {
        cublasXtHandle_t xtHandle;       CHECK_CUBLAS(cublasXtCreate(&xtHandle));
        int devices[1] = {0};           // 使用 device 0
        CHECK_CUBLAS(cublasXtDeviceSelect(xtHandle, 1, devices));

        // warm‑up
        CHECK_CUBLAS(cublasXtSgemm(xtHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   N, M, K,
                                   &alpha,
                                   h_B, N,
                                   h_A, K,
                                   &beta,
                                   h_C_cublasXt, N));

        // benchmark
        for (int i = 0; i < num_runs; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            CHECK_CUBLAS(cublasXtSgemm(xtHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                       N, M, K,
                                       &alpha,
                                       h_B, N,
                                       h_A, K,
                                       &beta,
                                       h_C_cublasXt, N));
            auto t1 = std::chrono::high_resolution_clock::now();
            t_xt[i] = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "cuBLAS‑Xt run " << i + 1 << ": " << t_xt[i] << " s" << std::endl;
        }

        CHECK_CUBLAS(cublasXtDestroy(xtHandle));
    }

    // =====================================================================
    // 4. 统计平均时间并输出
    // =====================================================================
    double avg_cublas   = std::accumulate(t_cublas.begin(), t_cublas.end(), 0.0) / num_runs;
    double avg_cublasXt = std::accumulate(t_xt.begin(),    t_xt.end(),    0.0) / num_runs;

    std::cout << "\n=== Average Times ===" << std::endl;
    std::cout << "cuBLAS   : " << avg_cublas   << " s" << std::endl;
    std::cout << "cuBLAS‑Xt: " << avg_cublasXt << " s" << std::endl;

    // =====================================================================
    // 5. 结果验证
    // =====================================================================
    const float tol = 1e-4f; // 相对误差容忍度
    bool match = compareResults(h_C_cublas, h_C_cublasXt, M * N, tol);
    std::cout << (match ? "Results match within tolerance." : "Results do NOT match!") << std::endl;

    // ---------------------------------------------------------------------
    // 6. 释放主机内存
    // ---------------------------------------------------------------------
    free(h_A); free(h_B); free(h_C_cublas); free(h_C_cublasXt);

    return 0;
}
