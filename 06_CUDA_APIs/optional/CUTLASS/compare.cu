/*
 *  文件说明：CUTLASS 与 cuBLAS GEMM 性能对比示例
 *  ------------------------------------------------
 *  功能概述：
 *  1. 随机生成两个 1024×1024 的单精度矩阵 A、B。
 *  2. 分别使用 cuBLAS 和 CUTLASS 在 GPU 上计算 C = A × B（列主存储）。
 *  3. 各自先进行 10 次热身（warm‑up），然后测量一次正式计算的耗时。
 *  4. 将两种方法得到的结果拷回主机，并在容差 1e‑2 内做逐元素比对，验证正确性。
 *  5. 打印两者耗时，观察性能差异。
 *
 *  运行与编译：
 *      nvcc cutlass_vs_cublas_annotated.cpp -o gemm_compare \
 *           -I$CUTLASS_PATH/include -lcublas -lcutlass
 *
 *  依赖：
 *      - CUDA Toolkit (含 cuBLAS)
 *      - CUTLASS 源码(仅头文件库)
 *
 *  注意：确保环境变量 CUTLASS_PATH 指向 CUTLASS 根目录。
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cublas_v2.h>

// CUTLASS 头文件
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/gemm.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/tensor_view_io.h>

/* --------------------- 宏：错误检查 --------------------- */
#define CHECK_CUDA(call)                                                                                 \
    if ((call) != cudaSuccess) {                                                                          \
        std::cerr << "CUDA error at: " << __FILE__ << ":" << __LINE__                                   \
                  << " : " << cudaGetErrorString(call) << std::endl;                                     \
        exit(EXIT_FAILURE);                                                                               \
    }

#define CHECK_CUBLAS(call)                                                                               \
    if ((call) != CUBLAS_STATUS_SUCCESS) {                                                               \
        std::cerr << "cuBLAS error at: " << __FILE__ << ":" << __LINE__ << std::endl;                 \
        exit(EXIT_FAILURE);                                                                               \
    }

/* -------------------------------------------------------- */

// 简单的结果比对函数：逐元素绝对误差 < epsilon 即视为相等
void verify_results(const std::vector<float>& A,
                    const std::vector<float>& B,
                    int rows, int cols) {
    const float epsilon = 1e-2f; // 误差容忍度
    for (int i = 0; i < rows * cols; ++i) {
        if (std::abs(A[i] - B[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i
                      << ": A[i] = " << A[i]
                      << ", B[i] = " << B[i] << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    std::cout << "Outputs match within tolerance." << std::endl;
}

int main() {
    /* -------------------- 基本参数 -------------------- */
    const int m = 1024; // A 行数, C 行数
    const int n = 1024; // B 列数, C 列数
    const int k = 1024; // A 列数, B 行数

    size_t bytes_A = static_cast<size_t>(m) * k * sizeof(float);
    size_t bytes_B = static_cast<size_t>(k) * n * sizeof(float);
    size_t bytes_C = static_cast<size_t>(m) * n * sizeof(float);

    /* -------------------- 主机内存 -------------------- */
    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    std::vector<float> h_C_cublas(m * n, 0.0f);
    std::vector<float> h_C_cutlass(m * n, 0.0f);

    // 用随机数初始化 A, B
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto &v : h_A) v = dist(rng);
    for (auto &v : h_B) v = dist(rng);

    /* -------------------- 设备内存 -------------------- */
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    /* ==================== cuBLAS 部分 ==================== */
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // --------- Warm‑up：先跑 10 次让 GPU 进入稳定状态 ---------
    for (int i = 0; i < 10; ++i) {
        CHECK_CUBLAS(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 m, n, k,
                                 &alpha,
                                 d_A, m,
                                 d_B, k,
                                 &beta,
                                 d_C, m));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // --------- 正式计时 ---------
    auto start = std::chrono::high_resolution_clock::now();
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m));
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> cublas_time = end - start;
    std::cout << "cuBLAS Time: " << cublas_time.count() << " ms" << std::endl;

    // 结果拷回主机
    CHECK_CUDA(cudaMemcpy(h_C_cublas.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    /* ==================== CUTLASS 部分 ==================== */
    using ColumnMajor = cutlass::layout::ColumnMajor;          // 列主存储布局
    using Gemm = cutlass::gemm::device::Gemm<                  // 指定数据类型 & 布局的 GEMM 类
        float, ColumnMajor,
        float, ColumnMajor,
        float, ColumnMajor>;

    Gemm gemm_op;                                              // 创建 GEMM 运算对象

    // 设置 GEMM 参数：问题尺寸 / A / B / C / alpha / beta
    Gemm::Arguments args({m, n, k},        //  Problem size：m × n × k
                         {d_A, m},         //  A 指针及行跨度(lda)
                         {d_B, k},         //  B 指针及行跨度(ldb)
                         {d_C, m},         //  C 指针及行跨度(ldc)
                         {d_C, m},         //  D = C (in‑place)
                         {alpha, beta});

    // --------- Warm‑up：同样 10 次 ---------
    for (int i = 0; i < 10; ++i) {
        cutlass::Status status = gemm_op(args);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "CUTLASS GEMM failed: "
                      << cutlassGetStatusString(status) << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // --------- 正式计时 ---------
    start = std::chrono::high_resolution_clock::now();
    cutlass::Status status = gemm_op(args);
    CHECK_CUDA(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> cutlass_time = end - start;

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed: "
                  << cutlassGetStatusString(status) << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "CUTLASS Time: " << cutlass_time.count() << " ms" << std::endl;

    // 结果拷回主机
    CHECK_CUDA(cudaMemcpy(h_C_cutlass.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    /* -------------------- 结果验证 -------------------- */
    verify_results(h_C_cublas, h_C_cutlass, m, n);

    /* -------------------- 资源释放 -------------------- */
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
