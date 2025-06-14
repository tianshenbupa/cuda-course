// ------------------------------------------------------------
//  作者：ChatGPT
//  功能：演示如何使用 cuBLAS‑Lt（cublasLtMatmul）完成
//        单精度 (FP32) 与半精度 (FP16) 的矩阵乘法，并与 CPU 结果对比。
//  说明：
//  * 采用列主序（column‑major）布局，符合 cuBLAS 默认约定。
//  * 通过 cublasLtMatrixLayout / cublasLtMatmulDesc 等 API 显式描述矩阵。
//  * 演示如何为 FP32 与 FP16 分别创建计算、布局描述。
//  * 程序在手写 4×4×4 小矩阵上运行，便于手动检验正确性。
//  环境要求：CUDA 11+，支持 Tensor Core 的 NVIDIA GPU。
// ------------------------------------------------------------

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <iomanip>

// ------------------------------------------------------------
// 简易错误检查宏：捕获 CUDA Runtime 与 cuBLAS‑Lt 返回值
// ------------------------------------------------------------
#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t status = (call);                                              \
        if (status != cudaSuccess) {                                              \
            std::cerr << "CUDA error at line " << __LINE__ << ": "            \
                      << cudaGetErrorString(status) << std::endl;                \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

#define CHECK_CUBLAS(call)                                                        \
    do {                                                                          \
        cublasStatus_t status = (call);                                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                    \
            std::cerr << "cuBLAS error at line " << __LINE__ << ": "          \
                      << status << std::endl;                                     \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// ------------------------------------------------------------
// 朴素 CPU 矩阵乘：C = A * B
// 参数说明：
//   A: (M × K)  B: (K × N)  C: (M × N)
// ------------------------------------------------------------
void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ------------------------------------------------------------
// 矩阵打印辅助函数（按行主序遍历并打印）
// ------------------------------------------------------------
void print_matrix(const float* matrix, int rows, int cols, const char* name) {
    std::cout << "Matrix " << name << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2)
                      << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    //--------------------------------------------------------------------------
    // 1. 定义矩阵尺寸 (M × K) * (K × N) = (M × N)
    //--------------------------------------------------------------------------
    const int M = 4, K = 4, N = 4;

    //--------------------------------------------------------------------------
    // 2. 在主机端手动初始化矩阵 A、B
    //--------------------------------------------------------------------------
    float h_A[M * K] = {
        1.0f,  2.0f,  3.0f,  4.0f,
        5.0f,  6.0f,  7.0f,  8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f,14.0f, 15.0f, 16.0f
    };

    // 刻意修改部分元素，确保 A ≠ B 以验证乘法
    float h_B[K * N] = {
        1.0f,  2.0f,  4.0f,  4.0f,  // 将 3.0f 改为 4.0f
        5.0f,  6.0f,  7.0f,  8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        17.0f,18.0f, 19.0f, 20.0f   // 将最后一行改为 17~20
    };

    float h_C_cpu[M * N]      = {0}; // CPU 结果
    float h_C_gpu_fp32[M * N] = {0}; // GPU FP32 结果
    float h_C_gpu_fp16[M * N] = {0}; // GPU FP16（转回 FP32）结果

    // 打印输入矩阵，便于肉眼对比
    print_matrix(h_A, M, K, "A");
    print_matrix(h_B, K, N, "B");

    //--------------------------------------------------------------------------
    // 3. 在 GPU 上分配 FP32 / FP16 内存，并拷贝数据
    //--------------------------------------------------------------------------
    float *d_A_fp32, *d_B_fp32, *d_C_fp32;
    CHECK_CUDA(cudaMalloc(&d_A_fp32, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B_fp32, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C_fp32, M * N * sizeof(float)));

    half *d_A_fp16, *d_B_fp16, *d_C_fp16;
    CHECK_CUDA(cudaMalloc(&d_A_fp16, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_fp16, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C_fp16, M * N * sizeof(half)));

    // 拷贝 FP32 数据到 GPU
    CHECK_CUDA(cudaMemcpy(d_A_fp32, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_fp32, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // 将 FP32 → FP16，并拷贝到 GPU
    std::vector<half> h_A_half(M * K), h_B_half(K * N);
    for (int i = 0; i < M * K; ++i) h_A_half[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_half[i] = __float2half(h_B[i]);

    CHECK_CUDA(cudaMemcpy(d_A_fp16, h_A_half.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_fp16, h_B_half.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));

    //--------------------------------------------------------------------------
    // 4. 创建 cuBLAS‑Lt 句柄 & 矩阵/运算描述符
    //--------------------------------------------------------------------------
    cublasLtHandle_t handle;
    CHECK_CUBLAS(cublasLtCreate(&handle));

    // ---- 4.1  布局描述 (MatrixLayout) ----
    cublasLtMatrixLayout_t matA_fp32, matB_fp32, matC_fp32;
    cublasLtMatrixLayout_t matA_fp16, matB_fp16, matC_fp16;

    // 参数：数据类型、行数 (rows)、列数 (cols)、leading dim (ld)
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matA_fp32, CUDA_R_32F, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matB_fp32, CUDA_R_32F, N, K, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matC_fp32, CUDA_R_32F, N, M, N));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matA_fp16, CUDA_R_16F, K, M, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matB_fp16, CUDA_R_16F, N, K, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&matC_fp16, CUDA_R_16F, N, M, N));

    // ---- 4.2  乘法描述 (MatmulDesc) ----
    cublasLtMatmulDesc_t matmulDesc_fp32, matmulDesc_fp16;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc_fp32, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc_fp16, CUBLAS_COMPUTE_16F, CUDA_R_16F));

    // 设置 A、B 是否转置：此处均为常规 (No‑Transpose)
    cublasOperation_t trans = CUBLAS_OP_N;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp32, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp32, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp16, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmulDesc_fp16, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));

    //--------------------------------------------------------------------------
    // 5. 调用 cublasLtMatmul 进行矩阵乘法
    //--------------------------------------------------------------------------
    const float alpha     = 1.0f;
    const float beta      = 0.0f;
    const half  alpha_h   = __float2half(1.0f);
    const half  beta_h    = __float2half(0.0f);

    // FP32 计算：C = alpha * B * A + beta * C
    CHECK_CUBLAS(cublasLtMatmul(
        handle,
        matmulDesc_fp32,
        &alpha,
        d_B_fp32, matB_fp32,   // B 在左 (列主序 N × K)
        d_A_fp32, matA_fp32,   // A 在右 (列主序 K × M)
        &beta,
        d_C_fp32, matC_fp32,   // 输出 C (N × M)
        d_C_fp32, matC_fp32,   // 工作区复用输出
        nullptr, nullptr, 0, 0));

    // FP16 计算
    CHECK_CUBLAS(cublasLtMatmul(
        handle,
        matmulDesc_fp16,
        &alpha_h,
        d_B_fp16, matB_fp16,
        d_A_fp16, matA_fp16,
        &beta_h,
        d_C_fp16, matC_fp16,
        d_C_fp16, matC_fp16,
        nullptr, nullptr, 0, 0));

    //--------------------------------------------------------------------------
    // 6. 将结果拷回主机，并把 FP16 → FP32 便于比较
    //--------------------------------------------------------------------------
    CHECK_CUDA(cudaMemcpy(h_C_gpu_fp32, d_C_fp32, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<half> h_C_gpu_fp16_half(M * N);
    CHECK_CUDA(cudaMemcpy(h_C_gpu_fp16_half.data(), d_C_fp16, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M * N; ++i) h_C_gpu_fp16[i] = __half2float(h_C_gpu_fp16_half[i]);

    //--------------------------------------------------------------------------
    // 7. CPU 参考结果
    //--------------------------------------------------------------------------
    cpu_matmul(h_A, h_B, h_C_cpu, M, N, K);

    //--------------------------------------------------------------------------
    // 8. 打印并验证
    //--------------------------------------------------------------------------
    print_matrix(h_C_cpu,      M, N, "C (CPU)");
    print_matrix(h_C_gpu_fp32, M, N, "C (GPU FP32)");
    print_matrix(h_C_gpu_fp16, M, N, "C (GPU FP16)");

    bool fp32_match = true, fp16_match = true;
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(h_C_cpu[i] - h_C_gpu_fp32[i]) > 1e-5) fp32_match = false;
        if (std::abs(h_C_cpu[i] - h_C_gpu_fp16[i]) > 1e-2) fp16_match = false; // FP16 误差稍大
    }
    std::cout << "FP32 Results " << (fp32_match ? "match" : "do not match") << std::endl;
    std::cout << "FP16 Results " << (fp16_match ? "match" : "do not match") << std::endl;

    //--------------------------------------------------------------------------
    // 9. 资源释放
    //--------------------------------------------------------------------------
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matA_fp32));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matB_fp32));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matC_fp32));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matA_fp16));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matB_fp16));
    CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(matC_fp16));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc_fp32));
    CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc_fp16));
    CHECK_CUBLAS(cublasLtDestroy(handle));
    CHECK_CUDA(cudaFree(d_A_fp32));
    CHECK_CUDA(cudaFree(d_B_fp32));
    CHECK_CUDA(cudaFree(d_C_fp32));
    CHECK_CUDA(cudaFree(d_A_fp16));
    CHECK_CUDA(cudaFree(d_B_fp16));
    CHECK_CUDA(cudaFree(d_C_fp16));

    return 0;
}
