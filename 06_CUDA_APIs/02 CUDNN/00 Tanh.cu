#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* 宏：检查 CUDA 调用是否成功 */
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

/* 宏：检查 cuDNN 调用是否成功 */
#define CHECK_CUDNN(call) { \
    cudnnStatus_t err = call; \
    if (err != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudnnGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

/* 简单的 CUDA kernel：计算 tanh 激活 */
__global__ void naiveTanhKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = tanhf(input[idx]);
    }
}

/* CPU 端的 tanh 计算（用作验证） */
float cpuTanh(float x) {
    return tanhf(x);
}

/* 初始化输入数据（范围 -1 到 1 的随机数） */
void initializeData(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

/* 比较 CPU 与 GPU 结果是否一致 */
bool verifyResults(float* cpu_output, float* gpu_output, int size, float tolerance = 1e-5) {
    for (int i = 0; i < size; ++i) {
        if (fabs(cpu_output[i] - gpu_output[i]) > tolerance) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, cpu_output[i], gpu_output[i]);
            return false;
        }
    }
    return true;
}

int main() {
    /* 设置张量维度（NCHW），此规模下 cuDNN 预期会优于手写 kernel */
    const int batch_size = 256;
    const int channels = 32;
    const int height = 224;
    const int width = 224;
    const int tensor_size = batch_size * channels * height * width;

    /* 申请主机端内存 */
    float *h_input, *h_output_naive, *h_output_cudnn, *h_output_cpu;
    h_input = (float*)malloc(tensor_size * sizeof(float));
    h_output_naive = (float*)malloc(tensor_size * sizeof(float));
    h_output_cudnn = (float*)malloc(tensor_size * sizeof(float));
    h_output_cpu = (float*)malloc(tensor_size * sizeof(float));

    /* 初始化输入数据 */
    initializeData(h_input, tensor_size);

    /* 申请设备端内存 */
    float *d_input, *d_output_naive, *d_output_cudnn;
    CHECK_CUDA(cudaMalloc(&d_input, tensor_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_naive, tensor_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_cudnn, tensor_size * sizeof(float)));

    /* 将输入数据拷贝到 GPU */
    CHECK_CUDA(cudaMemcpy(d_input, h_input, tensor_size * sizeof(float), cudaMemcpyHostToDevice));

    /* 创建 CUDA 事件用于计时 */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    /* 预热与基准测试参数 */
    const int num_warmup = 10;
    const int num_benchmark = 100;
    float naive_times[num_benchmark];
    float cudnn_times[num_benchmark];

    /* 配置网格与线程块 */
    dim3 block(256);
    dim3 grid((tensor_size + block.x - 1) / block.x);

    /* 预热：运行 naive kernel */
    for (int i = 0; i < num_warmup; ++i) {
        naiveTanhKernel<<<grid, block>>>(d_input, d_output_naive, tensor_size);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    /* 基准测试：naive kernel */
    for (int i = 0; i < num_benchmark; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        naiveTanhKernel<<<grid, block>>>(d_input, d_output_naive, tensor_size);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&naive_times[i], start, stop));
    }

    /* cuDNN 相关设置 */
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t input_descriptor;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch_size, channels, height, width));

    cudnnActivationDescriptor_t activation_descriptor;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
    CHECK_CUDNN(cudnnSetActivationDescriptor(activation_descriptor, CUDNN_ACTIVATION_TANH,
                                             CUDNN_PROPAGATE_NAN, 0.0));

    float alpha = 1.0f, beta = 0.0f;

    /* 预热：cuDNN tanh 前向计算 */
    for (int i = 0; i < num_warmup; ++i) {
        CHECK_CUDNN(cudnnActivationForward(cudnn, activation_descriptor, &alpha, input_descriptor, d_input,
                                           &beta, input_descriptor, d_output_cudnn));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    /* 基准测试：cuDNN tanh */
    for (int i = 0; i < num_benchmark; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDNN(cudnnActivationForward(cudnn, activation_descriptor, &alpha, input_descriptor, d_input,
                                           &beta, input_descriptor, d_output_cudnn));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&cudnn_times[i], start, stop));
    }

    /* 计算平均执行时间 */
    float avg_naive_time = 0.0f, avg_cudnn_time = 0.0f;
    for (int i = 0; i < num_benchmark; ++i) {
        avg_naive_time += naive_times[i];
        avg_cudnn_time += cudnn_times[i];
    }
    avg_naive_time /= num_benchmark;
    avg_cudnn_time /= num_benchmark;

    /* 将结果拷贝回主机内存 */
    CHECK_CUDA(cudaMemcpy(h_output_naive, d_output_naive, tensor_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_cudnn, d_output_cudnn, tensor_size * sizeof(float), cudaMemcpyDeviceToHost));

    /* CPU 端计算 tanh 结果（用于对比） */
    for (int i = 0; i < tensor_size; ++i) {
        h_output_cpu[i] = cpuTanh(h_input[i]);
    }

    /* 验证 GPU 结果正确性 */
    bool naive_correct = verifyResults(h_output_cpu, h_output_naive, tensor_size);
    bool cudnn_correct = verifyResults(h_output_cpu, h_output_cudnn, tensor_size);

    /* 打印性能数据与准确性检查 */
    printf("Tensor size: %d x %d x %d x %d\n", batch_size, channels, height, width);
    printf("Average Naive CUDA kernel time: %.3f ms\n", avg_naive_time);
    printf("Average cuDNN activation time: %.3f ms\n", avg_cudnn_time);
    printf("Speedup: %.2fx\n", avg_naive_time / avg_cudnn_time);
    printf("Naive kernel results correct: %s\n", naive_correct ? "Yes" : "No");
    printf("cuDNN results correct: %s\n", cudnn_correct ? "Yes" : "No");

    /* 资源释放 */
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output_naive));
    CHECK_CUDA(cudaFree(d_output_cudnn));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_descriptor));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(activation_descriptor));
    CHECK_CUDNN(cudnnDestroy(cudnn));
    free(h_input);
    free(h_output_naive);
    free(h_output_cudnn);
    free(h_output_cpu);

    return 0;
}
