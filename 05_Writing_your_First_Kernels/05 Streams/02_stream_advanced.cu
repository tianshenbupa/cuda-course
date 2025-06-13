#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <math.h>

// -----------------------------------------------------------------------------
// 宏：封装 CUDA API 返回值检查，若有错误则打印详细信息并退出
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

// 模板函数：统一处理不同 CUDA 函数的返回类型（cudaError_t 或 CUresult）
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
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

// -----------------------------------------------------------------------------
// Kernel 1：每个元素乘以 2
__global__ void kernel1(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

// Kernel 2：每个元素加 1
__global__ void kernel2(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

// -----------------------------------------------------------------------------
// CUDA 流回调函数：当附加到的流中所有先前操作完成后被调用
void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("Stream callback: Operation completed\n");
}

int main(void) {
    // ------------------------ 基本参数与指针 ------------------------
    const int N = 1000000;                        // 元素数量
    size_t size = N * sizeof(float);              // 所需字节数

    float *h_data = nullptr;  // Host（CPU）数据（锁页内存）
    float *d_data = nullptr;  // Device（GPU）数据

    cudaStream_t stream1, stream2;                // 两条 CUDA 流
    cudaEvent_t event;                            // 事件，用于跨流同步

    // ------------------------ 分配主机/设备内存 ------------------------
    // 使用锁页(Pinned)内存可显著提高异步 H↔D 传输带宽
    CHECK_CUDA_ERROR(cudaMallocHost(&h_data, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, size));

    // ------------------------ 初始化主机数据 ------------------------
    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // ------------------------ 创建具有不同优先级的 CUDA 流 ------------------------
    int leastPriority  = 0;   // 值越大，优先级越低
    int greatestPriority = 0; // 值越小，优先级越高
    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

    // stream1：最低优先级，用于前置计算
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastPriority));

    // stream2：最高优先级，用于后续计算和数据回传
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestPriority));

    // ------------------------ 创建事件（默认 flag=0） ------------------------
    CHECK_CUDA_ERROR(cudaEventCreate(&event));

    // ------------------------ Stream1：异步拷贝 → Kernel1 ------------------------
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream1));

    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    kernel1<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, N);

    // 在 stream1 记录事件（当拷贝与 kernel1 均完成时，事件触发）
    CHECK_CUDA_ERROR(cudaEventRecord(event, stream1));

    // ------------------------ Stream2：等待事件 → Kernel2 ------------------------
    // 让 stream2 在事件完成前阻塞，确保 kernel2 使用最新数据
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream2, event, 0));

    kernel2<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_data, N);

    // 在 stream2 中添加回调，所有排队操作都完成后触发
    CHECK_CUDA_ERROR(cudaStreamAddCallback(stream2, myStreamCallback, NULL, 0));

    // 将结果异步拷贝回主机（仍在 stream2 中）
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream2));

    // ------------------------ 同步两个流，确保 GPU 任务全部结束 ------------------------
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

    // ------------------------ 结果校验 ------------------------
    for (int i = 0; i < N; ++i) {
        float expected = (static_cast<float>(i) * 2.0f) + 1.0f;
        if (fabsf(h_data[i] - expected) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // ------------------------ 资源清理 ------------------------
    CHECK_CUDA_ERROR(cudaFreeHost(h_data));
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
    CHECK_CUDA_ERROR(cudaEventDestroy(event));

    return 0;
}
