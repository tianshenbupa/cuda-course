// 专为小尺寸手写矩阵演示而设计的示例程序
// 使用 CPU 计算、cuBLAS 单精度 (SGEMM) 与半精度 (HGEMM) 三种方式进行矩阵乘法
// 运行环境：NVIDIA GPU + CUDA Toolkit + cuBLAS

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// -------------------------
// 定义矩阵尺寸 (M x K) * (K x N) = (M x N)
// -------------------------
#define M 3  // A 的行数、C 的行数
#define K 4  // A 的列数、B 的行数
#define N 2  // B 的列数、C 的列数

// -------------------------
// 错误检查宏：方便定位 CUDA 运行时与 cuBLAS API 的返回状态
// -------------------------
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}

// -------------------------
// 打印矩阵的辅助宏（按行主序）
// -------------------------
#undef PRINT_MATRIX
#define PRINT_MATRIX(mat, rows, cols) \
    for (int i = 0; i < rows; i++) { \
        for (int j = 0; j < cols; j++) \
            printf("%8.3f ", mat[i * cols + j]); \
        printf("\n"); \
    } \
    printf("\n");

// -------------------------
// CPU 端的朴素矩阵乘法实现，方便与 GPU 结果对比
// C = A * B
// -------------------------
void cpu_matmul(float *A, float *B, float *C) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

int main() {
    // -------------------------
    // 1. 初始化主机端 (CPU) 数据
    //    矩阵 A: (M x K)
    //    矩阵 B: (K x N)
    // -------------------------
    float A[M * K] = {
        1.0f,  2.0f,  3.0f,  4.0f,
        5.0f,  6.0f,  7.0f,  8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    };

    float B[K * N] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f
    };

    float C_cpu[M * N];       // CPU 结果
    float C_cublas_s[M * N];  // cuBLAS SGEMM 结果 (float32)
    float C_cublas_h[M * N];  // cuBLAS HGEMM 结果 (float16 → float32)

    // -------------------------
    // 2. CPU 参考结果
    // -------------------------
    cpu_matmul(A, B, C_cpu);

    // -------------------------
    // 3. 创建 cuBLAS 句柄并分配 GPU 内存
    // -------------------------
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    // 拷贝数据到 GPU
    CHECK_CUDA(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // -------------------------
    // 4. cuBLAS SGEMM (float32)
    //    C = alpha * B * A + beta * C
    //    注意：cuBLAS 采用列主序，参数顺序与常规数学公式略有不同
    // -------------------------
    float alpha = 1.0f, beta = 0.0f;
    // 此处使用 (N x M) 输出，因为 cuBLAS 按列主序存储
    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置
        N, M, K,                   // C (N x M)
        &alpha,
        d_B, N,                   // B: (N x K) 列主序步长 N
        d_A, K,                   // A: (K x M) 列主序步长 K
        &beta,
        d_C, N                    // C: (N x M) 列主序步长 N
    ));
    CHECK_CUDA(cudaMemcpy(C_cublas_s, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // -------------------------
    // 5. cuBLAS HGEMM (float16)
    //    先将 A、B 转为半精度，计算后再转回 float32 以便打印
    // -------------------------
    half *d_A_h, *d_B_h, *d_C_h;
    CHECK_CUDA(cudaMalloc(&d_A_h, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_h, K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C_h, M * N * sizeof(half)));

    // 主机端临时 half 缓冲区
    half A_h[M * K], B_h[K * N];
    for (int i = 0; i < M * K; i++) {
        A_h[i] = __float2half(A[i]);
    }
    for (int i = 0; i < K * N; i++) {
        B_h[i] = __float2half(B[i]);
    }

    // 拷贝到 GPU
    CHECK_CUDA(cudaMemcpy(d_A_h, A_h, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_h, B_h, K * N * sizeof(half), cudaMemcpyHostToDevice));

    // half 精度的 alpha / beta
    __half alpha_h = __float2half(1.0f);
    __half beta_h  = __float2half(0.0f);

    CHECK_CUBLAS(cublasHgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha_h,
        d_B_h, N,  // 注意步长依旧是列主序
        d_A_h, K,
        &beta_h,
        d_C_h, N
    ));

    // 拷回并转 float 以便打印
    half C_h[M * N];
    CHECK_CUDA(cudaMemcpy(C_h, d_C_h, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    for (int i = 0; i < M * N; i++) {
        C_cublas_h[i] = __half2float(C_h[i]);
    }

    // -------------------------
    // 6. 打印结果对比
    // -------------------------
    printf("Matrix A (%dx%d):\n", M, K);
    PRINT_MATRIX(A, M, K);

    printf("Matrix B (%dx%d):\n", K, N);
    PRINT_MATRIX(B, K, N);

    printf("CPU Result (%dx%d):\n", M, N);
    PRINT_MATRIX(C_cpu, M, N);

    printf("cuBLAS SGEMM Result (%dx%d):\n", M, N);
    PRINT_MATRIX(C_cublas_s, M, N);

    printf("cuBLAS HGEMM Result (%dx%d):\n", M, N);
    PRINT_MATRIX(C_cublas_h, M, N);

    // -------------------------
    // 7. 资源释放
    // -------------------------
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_A_h));
    CHECK_CUDA(cudaFree(d_B_h));
    CHECK_CUDA(cudaFree(d_C_h));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}
