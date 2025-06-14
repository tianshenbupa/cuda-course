// ============================================================================
//  文件名：gemm_benchmark_with_chinese_comments.cu
//  作者：ChatGPT
//  功能：在同一程序中基准测试 4 种 GPU 矩阵乘实现，并与朴素 CUDA Kernel 对比
//        1)  cuBLAS   ‑ FP32 (cublasSgemm)
//        2)  cuBLASLt ‑ FP32 (cublasLtMatmul + CUDA_R_32F)
//        3)  cuBLAS   ‑ FP16 (cublasHgemm)
//        4)  cuBLASLt ‑ FP16 (cublasLtMatmul + CUDA_R_16F)
//        同时给出 Max 绝对误差验证，并测量平均运行时间 (ms)。
//  说明：
//  * 使用列主序 (column‑major) 参数接口，遵循 cuBLAS 约定。
//  * 提供基于 CUDA Event 的计时函数，以及 "warmup + 重复" 方式计算平均时间。
//  * 支持大矩阵 (4096×1024 × 1024×4096) 的性能验证，可自行修改 M/K/N。
//  * 演示 GPU half <‑> float 转换与误差评估流程。
//  环境：CUDA 11+，支持 Tensor Core 的 NVIDIA GPU。
// ============================================================================

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <functional>
#include <random>
#include <numeric>

// ----------------------------
// (1) 错误检查宏
// ----------------------------
#define CHECK_CUDA(call)                                    \
    do {                                                    \
        cudaError_t status = (call);                        \
        if (status != cudaSuccess) {                        \
            std::cerr << "CUDA error at line " << __LINE__  \
                      << ": " << cudaGetErrorString(status) \
                      << std::endl;                         \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

#define CHECK_CUBLAS(call)                                   \
    do {                                                    \
        cublasStatus_t status = (call);                     \
        if (status != CUBLAS_STATUS_SUCCESS) {              \
            std::cerr << "cuBLAS error at line " << __LINE__ \
                      << ": " << status << std::endl;        \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

// ----------------------------
// (2) 矩阵尺寸 (可修改)
// ----------------------------
const int M = 4096;   // A 行数 / C 行数
const int K = 1024;   // A 列数 / B 行数
const int N = 4096;   // B 列数 / C 列数

// ============================================================================
//  朴素 CUDA Kernel：每线程计算 C 的一个元素 (行 × 列)
// ============================================================================
__global__ void naiveMatrixMultiply(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ----------------------------
// (3) 随机初始化矩阵 (‑0.5 ~ 0.5)
// ----------------------------
void initializeMatrix(std::vector<float>& matrix, int rows, int cols) {
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
    for (int i = 0; i < rows * cols; ++i) matrix[i] = dis(gen);
}

// ----------------------------
// (4) 结果校验：逐元素绝对误差 < tolerance
// ----------------------------
bool verifyResults(const std::vector<float>& expected,
                   const std::vector<float>& actual,
                   float tolerance = 1e-2f) {
    if (expected.size() != actual.size()) return false;
    for (size_t i = 0; i < expected.size(); ++i) {
        float abs_err = fabsf(expected[i] - actual[i]);
        if (abs_err > tolerance) {
            std::cout << "Mismatch @ " << i << ": expected "
                      << expected[i] << ", got " << actual[i]
                      << ", abs_err=" << abs_err << std::endl;
            return false;
        }
    }
    return true;
}

// ----------------------------
// (5) 使用 CUDA Events 计时的工具函数
// ----------------------------
float time_kernel(const std::function<void()>& kernel_func) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    kernel_func();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms;
}

// 重复多次求平均，含 warmup
float benchmark_kernel(const std::function<void()>& kernel_func,
                       int warmup_runs, int benchmark_runs) {
    for (int i = 0; i < warmup_runs; ++i) kernel_func();

    std::vector<float> times;
    times.reserve(benchmark_runs);
    for (int i = 0; i < benchmark_runs; ++i)
        times.push_back(time_kernel(kernel_func));

    return std::accumulate(times.begin(), times.end(), 0.0f) / benchmark_runs;
}

// ============================================================================
//  主程序入口
// ============================================================================
int main() {
    std::cout << "Matrix GEMM size: " << M << "×" << K << "  *  "
              << K << "×" << N << std::endl;

    // ------------------------
    // (1) 主机端内存准备
    // ------------------------
    std::vector<float> h_A(M * K), h_B(K * N);
    std::vector<float> h_C_naive(M * N);
    std::vector<float> h_C_cublas_fp32(M * N), h_C_cublasLt_fp32(M * N);
    std::vector<float> h_C_cublas_fp16(M * N), h_C_cublasLt_fp16(M * N);

    initializeMatrix(h_A, M, K);
    initializeMatrix(h_B, K, N);

    // ------------------------
    // (2) 设备端内存 (FP32 / FP16)
    // ------------------------
    float *d_A, *d_B, *d_C;
    half  *d_A_h, *d_B_h, *d_C_h;
    CHECK_CUDA(cudaMalloc(&d_A,  M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B,  K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C,  M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_A_h, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_h, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C_h, M * N * sizeof(half)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    // 主机 FP16 缓冲区
    std::vector<half> h_A_h(M * K), h_B_h(K * N);
    for (int i = 0; i < M * K; ++i) h_A_h[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_h[i] = __float2half(h_B[i]);
    CHECK_CUDA(cudaMemcpy(d_A_h, h_A_h.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_h, h_B_h.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));

    // ------------------------
    // (3) 创建 cuBLAS / cuBLASLt 句柄
    // ------------------------
    cublasHandle_t   cublas_handle;  CHECK_CUBLAS(cublasCreate(&cublas_handle));
    cublasLtHandle_t cublasLt_handle;CHECK_CUBLAS(cublasLtCreate(&cublasLt_handle));

    const float alpha = 1.0f, beta = 0.0f;
    const half  alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);

    const int warmup_runs = 3;
    const int bench_runs  = 20;

    // =====================================================================
    // (4) cuBLAS SGEMM (FP32)
    // =====================================================================
    float t_cublas_fp32 = benchmark_kernel([&]() {
        CHECK_CUBLAS(cublasSgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N,
            d_A, K,
            &beta,
            d_C, N));
    }, warmup_runs, bench_runs);
    std::cout << "cuBLAS  FP32 avg time: " << t_cublas_fp32 << " ms\n";
    CHECK_CUDA(cudaMemcpy(h_C_cublas_fp32.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // =====================================================================
    // (5) cuBLASLt FP32 (Matmul)
    // =====================================================================
    cublasLtMatmulDesc_t opDesc32; CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc32, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasLtMatrixLayout_t A32, B32, C32;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&A32, CUDA_R_32F, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&B32, CUDA_R_32F, N, K, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&C32, CUDA_R_32F, N, M, N));

    float t_cublasLt_fp32 = benchmark_kernel([&]() {
        CHECK_CUBLAS(cublasLtMatmul(
            cublasLt_handle, opDesc32,
            &alpha,
            d_B, B32,
            d_A, A32,
            &beta,
            d_C, C32,
            d_C, C32,
            nullptr, nullptr, 0, 0));
    }, warmup_runs, bench_runs);
    std::cout << "cuBLASLt FP32 avg time: " << t_cublasLt_fp32 << " ms\n";
    CHECK_CUDA(cudaMemcpy(h_C_cublasLt_fp32.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // =====================================================================
    // (6) cuBLAS HGEMM (FP16)
    // =====================================================================
    float t_cublas_fp16 = benchmark_kernel([&]() {
        CHECK_CUBLAS(cublasHgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha_h,
            d_B_h, N,
            d_A_h, K,
            &beta_h,
            d_C_h, N));
    }, warmup_runs, bench_runs);
    std::cout << "cuBLAS  FP16 avg time: " << t_cublas_fp16 << " ms\n";

    std::vector<half> h_temp_h(M * N);
    CHECK_CUDA(cudaMemcpy(h_temp_h.data(), d_C_h, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M * N; ++i) h_C_cublas_fp16[i] = __half2float(h_temp_h[i]);

    // =====================================================================
    // (7) cuBLASLt FP16 (Matmul)
    // =====================================================================
    cublasLtMatmulDesc_t opDesc16; CHECK_CUBLAS(cublasLtMatmulDescCreate(&opDesc16, CUBLAS_COMPUTE_16F, CUDA_R_16F));
    cublasLtMatrixLayout_t A16, B16, C16;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&A16, CUDA_R_16F, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&B16, CUDA_R_16F, N, K, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&C16, CUDA_R_16F, N, M, N));

    float t_cublasLt_fp16 = benchmark_kernel([&]() {
        CHECK_CUBLAS(cublasLtMatmul(
            cublasLt_handle, opDesc16,
            &alpha_h,
            d_B_h, B16,
            d_A_h, A16,
            &beta_h,
            d_C_h, C16,
            d_C_h, C16,
            nullptr, nullptr, 0, 0));
    }, warmup_runs, bench_runs);
    std::cout << "cuBLASLt FP16 avg time: " << t_cublasLt_fp16 << " ms\n";

    CHECK_CUDA(cudaMemcpy(h_temp_h.data(), d_C_h, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M * N; ++i) h_C_cublasLt_fp16[i] = __half2float(h_temp_h[i]);

    // =====================================================================
    // (8) 朴素 CUDA Kernel (FP32)
    // =====================================================================
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    float t_naive = benchmark_kernel([&]() {
        naiveMatrixMultiply<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CHECK_CUDA(cudaDeviceSynchronize());
    }, warmup_runs, 5); // naive 较慢，跑 5 次即可
    std::cout << "Naive   FP32 avg time: " << t_naive << " ms\n";
    CHECK_CUDA(cudaMemcpy(h_C_naive.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // =====================================================================
    // (9) 结果校验 & 最大绝对误差 (仅与 Naive 比较)
    // =====================================================================
    auto max_abs_err = [](const std::vector<float>& ref, const std::vector<float>& test) {
        float max_err = 0.0f;
        for (size_t i = 0; i < ref.size(); ++i) {
            float err = fabsf(ref[i] - test[i]);
            if (err > max_err) max_err = err;
        }
        return max_err;
    };

    bool ok_cublas32   = verifyResults(h_C_naive, h_C_cublas_fp32, 1e-2f);
    bool ok_cublasLt32 = verifyResults(h_C_naive, h_C_cublasLt_fp32, 1e-2f);
    bool ok_cublas16   = verifyResults(h_C_naive, h_C_cublas_fp16, 5e-1f);
    bool ok_cublasLt16 = verifyResults(h_C_naive, h_C_cublasLt_fp16, 5e-1f);

    std::cout << "Max abs error (FP16 cuBLAS ) = " << max_abs_err(h_C_naive, h_C_cublas_fp16)   << std::endl;
    std::cout << "Max abs error (FP16 cuBLASLt) = " << max_abs_err(h_C_naive, h_C_cublasLt_fp16) << std::endl;

    std::cout << "cuBLAS   FP32 results " << (ok_cublas32   ? "match" : "NOT match") << " naive result\n";
    std::cout << "cuBLASLt FP32 results " << (ok_cublasLt32 ? "match" : "NOT match") << " naive result\n";
    std::cout << "cuBLAS   FP16 results " << (ok_cublas16   ? "match" : "NOT match") << " naive result (tolerance 0.5)\n";
    std::cout << "cuBLASLt FP16 results " << (ok_cublasLt16 ? "match" : "NOT match") << " naive result (tolerance 0.5)\n";

    // =====================================================================
    // (10) 资源释放
    // =====================================================================
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(opDesc32));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(opDesc16));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(A32)); CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(B32)); CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(C32));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(A16)); CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(B16)); CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(C16));

    CHECK_CUBLAS(cublasLtDestroy(cublasLt_handle));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));

    CHECK_CUDA(cudaFree(d_A));   CHECK_CUDA(cudaFree(d_B));   CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_A_h)); CHECK_CUDA(cudaFree(d_B_h)); CHECK_CUDA(cudaFree(d_C_h));

    return 0;
}
