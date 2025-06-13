#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// 宏：CUDA 错误检查封装，方便定位问题
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

// 模板函数：统一处理 CUDA API 的返回值
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        // 打印错误信息并立即退出程序
        fprintf(stderr,
                "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file,
                line,
                static_cast<unsigned int>(err),
                cudaGetErrorString(err),
                func);
        exit(EXIT_FAILURE);
    }
}

// ------------------------ CUDA 核函数 ------------------------
// 功能：执行向量加法 C = A + B
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    // 计算当前线程对应的数据索引
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    // ------------------------ 基本参数设置 ------------------------
    int numElements = 50000;                       // 向量长度
    size_t size    = numElements * sizeof(float);  // 内存大小（字节）

    // ------------------------ 主机端指针 ------------------------
    float *h_A, *h_B, *h_C;                        // Host memory

    // ------------------------ 设备端指针 ------------------------
    float *d_A, *d_B, *d_C;                        // Device memory

    // ------------------------ CUDA 流 ------------------------
    cudaStream_t stream1, stream2;                 // 两个异步流，用于数据传输与计算重叠

    // ------------------------ 分配主机端内存 ------------------------
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // ------------------------ 初始化主机端数据 ------------------------
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;  // [0,1) 随机数
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // ------------------------ 分配设备端内存 ------------------------
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, size));

    // ------------------------ 创建 CUDA 流 ------------------------
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));

    // ------------------------ 异步拷贝数据：Host -> Device ------------------------
    // A 使用 stream1，B 使用 stream2，实现传输并行
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2));

    // ------------------------ 计算网格/线程块配置 ------------------------
    int threadsPerBlock = 256;
    int blocksPerGrid   = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // ------------------------ 启动核函数（stream1） ------------------------
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C, numElements);

    // ------------------------ 异步拷贝结果：Device -> Host ------------------------
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream1));

    // ------------------------ 同步 CUDA 流 ------------------------
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

    // ------------------------ 结果校验 ------------------------
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // ------------------------ 释放资源 ------------------------
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
